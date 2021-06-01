import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from utils.config import K_FOLDS_TRAIN, K_FOLDS_TEST, BLOCK_FOLDS_TRAIN, BLOCK_FOLDS_TEST
from utils.nn_preprocessing import generate_train_val_set_kfold, build_adj_matrix_to_tensor, build_feature_nodes, define_scalings, scheduler_glstm
from typing import TypeVar, List, Dict
import spektral as sp

DataFrame = TypeVar('DataFrame')
Array = TypeVar('Array')
Model = TypeVar('Model')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--tp', type=float, default=0.8)
parser.add_argument('--ts', type=float, default=0.5)
parser.add_argument('--gcn', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--synthetic', type=bool, default=False)


def build_model(
        X_train_price_timesteps: Array,
        X_train_sales_timesteps: Array,
        X_train: Array) -> Model:
    """
    Build the base model 2X-GLSTM

    :param X_train_price_timesteps: The price timesteps
    :param X_train_sales_timesteps: The sales timesteps
    :param X_train: The rest of the features
    """
    n_steps_price = len(X_train_price_timesteps[0])
    n_steps_sales = len(X_train_sales_timesteps[0])
    n_features = 2

    # INPUT
    price_input = tf.keras.Input(shape=(n_steps_price, n_features), name='price_input')
    sales_input = tf.keras.Input(shape=(n_steps_sales, n_features), name='sales_input')
    X_input = tf.keras.Input(shape=(X_train.shape[1],), name='X_input')
    adj_input_price = tf.keras.Input(shape=(220, 220), name='adj_input_price')
    adj_input_sales = tf.keras.Input(shape=(220, 220), name='adj_input_sales')
    nodes_price_feat = tf.keras.Input(shape=(220, 8), name='price_nodes')
    nodes_sales_feat = tf.keras.Input(shape=(220, 8), name='sales_nodes')

    # PRICE LSTM LAYER
    price_LSTM_layer = tf.keras.layers.LSTM(50, activation='tanh')(price_input)
    GNN_price = sp.layers.GCNConv(args.gcn, activation='relu')([nodes_price_feat, adj_input_price])
    # GNN_price = sp.layers.GCNConv(10, activation='relu')([GNN_price, adj_input_price])
    GNN_price = tf.keras.layers.Flatten()(GNN_price)
    price_LSTM_layer = tf.keras.layers.concatenate([price_LSTM_layer, GNN_price])
    # price_LSTM_layer = tf.keras.layers.BatchNormalization()(price_LSTM_layer)
    # price_LSTM_layer = tf.keras.layers.Dense(100, activation='relu')(price_LSTM_layer)
    X_price = tf.keras.layers.Dense(1, name='price_LSTM_output')(price_LSTM_layer)

    # QUANTITY LSTM LAYER
    sales_LSTM_layer = tf.keras.layers.LSTM(50, activation='tanh')(sales_input)
    GNN_quantity = sp.layers.GCNConv(args.gcn, activation='relu')([nodes_sales_feat, adj_input_sales])
    # GNN_quantity = sp.layers.GCNConv(10, activation='relu')([GNN_quantity, adj_input_sales])
    GNN_quantity = tf.keras.layers.Flatten()(GNN_quantity)
    sales_LSTM_layer = tf.keras.layers.concatenate([sales_LSTM_layer, GNN_quantity])
    # sales_LSTM_layer = tf.keras.layers.BatchNormalization()(sales_LSTM_layer)
    # sales_LSTM_layer = tf.keras.layers.Dense(100, activation='relu')(sales_LSTM_layer)
    X_sales = tf.keras.layers.Dense(1, name='sales_LSTM_output')(sales_LSTM_layer)

    # CONCAT
    X = tf.keras.layers.concatenate([X_price, X_sales, X_input])
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001))(X)
    # X = tf.keras.layers.Dropout(0.5)(X)
    X = tf.keras.layers.Dense(200, activation='relu')(X)

    # OUTPUT
    price_output = tf.keras.layers.Dense(1, name='price_output')(X)
    sales_output = tf.keras.layers.Dense(1, name='sales_output')(X)

    model = tf.keras.Model(
        inputs=[price_input, sales_input, X_input,
                adj_input_price, adj_input_sales,
                nodes_price_feat, nodes_sales_feat],
        outputs=[price_output, sales_output]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=args.lr),
        loss={
            "price_output": tf.keras.losses.MeanSquaredError(),
            "sales_output": tf.keras.losses.MeanSquaredError()
        }
    )
    return model


def glstm_kfold(
        df: DataFrame,
        reference_list_price: List[Dict],
        reference_list_sales: List[Dict]) -> DataFrame:
    """
    Evaluate hyperparameters using cross validation

    :param df: The training set
    :param reference_list_price: A list containing all series of price
    :param reference_list_price: A list containing all series of sales
    """
    results = list()
    loss = list()
    val_loss = list()

    for fold in K_FOLDS_TRAIN.keys():
        train_years = K_FOLDS_TRAIN[fold]
        val_years = K_FOLDS_TEST[fold + 1]
        reference_price = reference_list_price[fold]
        reference_sales = reference_list_sales[fold]

        X_train_price_timesteps, X_train_sales_timesteps, \
            X_train, y_train_price, y_train_sales, \
            X_val_price_timesteps, X_val_sales_timesteps, \
            X_val, y_val_price, y_val_sales = generate_train_val_set_kfold(df, train_years=train_years, val_years=val_years)
        df_train = df[df['Year'].isin(train_years)]
        df_val = df[df['Year'].isin(val_years)]
        adj_input_price_train = build_adj_matrix_to_tensor(reference_price, df_train.shape[0], args.tp)
        adj_input_sales_train = build_adj_matrix_to_tensor(reference_sales, df_train.shape[0], args.ts)
        adj_input_price_val = build_adj_matrix_to_tensor(reference_price, df_val.shape[0], args.tp)
        adj_input_sales_val = build_adj_matrix_to_tensor(reference_sales, df_val.shape[0], args.ts)
        groups_price_feat_train, groups_sales_feat_train = build_feature_nodes(reference_price, reference_sales, X_train.shape[0])
        groups_price_feat_val, groups_sales_feat_val = build_feature_nodes(reference_price, reference_sales, X_val.shape[0])

        model = build_model(X_train_price_timesteps, X_train_sales_timesteps, X_train)
        history = model.fit(
            x={
                "price_input": X_train_price_timesteps,
                "sales_input": X_train_sales_timesteps,
                "X_input": X_train,
                "adj_input_price": adj_input_price_train,
                "adj_input_sales": adj_input_sales_train,
                "nodes_price_feat": groups_price_feat_train,
                "nodes_sales_feat": groups_sales_feat_train
            },
            y={"price_output": y_train_price, "sales_output": y_train_sales},
            validation_data=(
                {
                    "price_input": X_val_price_timesteps,
                    "sales_input": X_val_sales_timesteps,
                    "X_input": X_val,
                    "adj_input_price": adj_input_price_val,
                    "adj_input_sales": adj_input_sales_val,
                    "nodes_price_feat": groups_price_feat_val,
                    "nodes_sales_feat": groups_sales_feat_val
                },
                {"price_output": y_val_price, "sales_output": y_val_sales}),
            epochs=args.epochs,
            batch_size=32,
            verbose=1
        )

        loss.append(history.history['loss'][-1])
        val_loss.append(history.history['val_loss'][-1])

        print(f'Fold {fold} is done..')

    results.append([np.mean(loss), np.mean(val_loss), np.std(loss), np.std(val_loss)])

    results = pd.DataFrame(results)
    col_names = ['loss', 'val_loss', 'std_loss', 'std_val_loss']
    results.columns = col_names
    return results


def generate_synthetic_training_set(
        df: DataFrame,
        reference_list_price: List[Dict],
        reference_list_sales: List[Dict]) -> Array:
    """
    Build the synthetic training set

    :param df: The training set
    :param reference_list_price: A list containing all series of price
    :param reference_list_price: A list containing all series of sales
    """
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler_glstm, verbose=0)
    synthetic_training_set_price = list()
    synthetic_training_set_sales = list()
    for fold in BLOCK_FOLDS_TRAIN.keys():
        train_years = BLOCK_FOLDS_TRAIN[fold]
        val_years = BLOCK_FOLDS_TEST[fold + 1]
        reference_price = reference_list_price[fold]
        reference_sales = reference_list_sales[fold]

        X_train_price_timesteps, X_train_sales_timesteps, \
            X_train, y_train_price, y_train_sales, \
            X_val_price_timesteps, X_val_sales_timesteps, \
            X_val, y_val_price, y_val_sales = generate_train_val_set_kfold(df, train_years=train_years, val_years=val_years)
        scaler_price, scaler_sales = define_scalings(df, train_years)

        df_train = df[df['Year'].isin(train_years)]
        df_val = df[df['Year'].isin(val_years)]
        adj_input_price_train = build_adj_matrix_to_tensor(reference_price, df_train.shape[0], args.tp)
        adj_input_sales_train = build_adj_matrix_to_tensor(reference_sales, df_train.shape[0], args.ts)
        adj_input_price_val = build_adj_matrix_to_tensor(reference_price, df_val.shape[0], args.tp)
        adj_input_sales_val = build_adj_matrix_to_tensor(reference_sales, df_val.shape[0], args.ts)
        groups_price_feat_train, groups_sales_feat_train = build_feature_nodes(reference_price, reference_sales, X_train.shape[0])
        groups_price_feat_val, groups_sales_feat_val = build_feature_nodes(reference_price, reference_sales, X_val.shape[0])

        model = build_model(X_train_price_timesteps, X_train_sales_timesteps, X_train)
        model.fit(
            x={
                "price_input": X_train_price_timesteps,
                "sales_input": X_train_sales_timesteps,
                "X_input": X_train,
                "adj_input_price": adj_input_price_train,
                "adj_input_sales": adj_input_sales_train,
                "nodes_price_feat": groups_price_feat_train,
                "nodes_sales_feat": groups_sales_feat_train
            },
            y={"price_output": y_train_price, "sales_output": y_train_sales},
            epochs=10,
            batch_size=32,
            callbacks=[lr_callback],
            verbose=0
        )

        y_pred = model.predict([X_val_price_timesteps, X_val_sales_timesteps, X_val, adj_input_price_val, adj_input_sales_val, groups_price_feat_val, groups_sales_feat_val])
        synthetic_training_set_price.append(scaler_price.inverse_transform(y_pred[0]))
        synthetic_training_set_sales.append(scaler_sales.inverse_transform(y_pred[1]))
        print(f'Fold {fold} is done..')
    return synthetic_training_set_price, synthetic_training_set_sales


if __name__ == '__main__':
    args = parser.parse_args()
    df = pd.read_csv('../data/train/train_nn_target1.csv', sep=';')

    with open('../data/clean/reference_price_to_2016.pkl', 'rb') as file:
        reference_price_16 = pickle.load(file)
    with open('../data/clean/reference_price_to_2014.pkl', 'rb') as file:
        reference_price_14 = pickle.load(file)
    with open('../data/clean/reference_price_to_2012.pkl', 'rb') as file:
        reference_price_12 = pickle.load(file)
    with open('../data/clean/reference_price_to_2010.pkl', 'rb') as file:
        reference_price_10 = pickle.load(file)
    with open('../data/clean/total_sales_to_2016.pkl', 'rb') as file:
        reference_sales_16 = pickle.load(file)
    with open('../data/clean/total_sales_to_2014.pkl', 'rb') as file:
        reference_sales_14 = pickle.load(file)
    with open('../data/clean/total_sales_to_2012.pkl', 'rb') as file:
        reference_sales_12 = pickle.load(file)
    with open('../data/clean/total_sales_to_2010.pkl', 'rb') as file:
        reference_sales_10 = pickle.load(file)

    reference_list_price = [
        reference_price_10,
        reference_price_12,
        reference_price_14,
        reference_price_16
    ]
    reference_list_sales = [
        reference_sales_10,
        reference_sales_12,
        reference_sales_14,
        reference_sales_16
    ]
    if args.synthetic:
        synthetic_training_set_price, synthetic_training_set_sales = generate_synthetic_training_set()
        np.save('../data/train/synthetic_training_set_price_glstm.npy', synthetic_training_set_price)
        np.save('../data/train/synthetic_training_set_sales_glstm.npy', synthetic_training_set_sales)
    else:
        results = glstm_kfold(df, reference_list_price, reference_list_sales)
        print(results)
