import argparse
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
from utils.nn_preprocessing import input_nn, build_adj_matrix_to_tensor, build_feature_nodes, scheduler_glstm
from utils.config import CATEGORICAL_FEATURES
from sklearn.preprocessing import MinMaxScaler
from typing import TypeVar, List, Dict
import spektral as sp

DataFrame = TypeVar('DataFrame')
Array = TypeVar('Array')
Model = TypeVar('Model')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--ensemble', type=bool, default=False)
parser.add_argument('--gcn', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--tp', type=float, default=0.8)
parser.add_argument('--ts', type=float, default=0.05)


def build_model(
        X_price_timesteps: Array,
        X_sales_timesteps: Array,
        X: Array,
        y_price: Array,
        y_sales: Array,
        groups_price_feat: Array,
        groups_sales_feat: Array,
        adj_input_price: Array,
        adj_input_sales: Array) -> Model:
    """
    Return the base model 2X-GLSTM

    :param X_price_timesteps: An array with price timesteps
    :param X_sales_timesteps: An array with sales timesteps
    :param X: An array with features
    :param y_price: The price targets
    :param y_sales: The sales targets
    :param groups_price_feat: The node features of prices
    :param groups_sales_feat: The node features of sales
    :param adj_input_price: The adjacency matrix of prices
    :param adj_input_sales: The adjacency matrix of sales
    """
    n_steps_price = len(X_price_timesteps[0])
    n_steps_sales = len(X_sales_timesteps[0])
    n_features = 2

    # INPUT
    price_input = tf.keras.Input(shape=(n_steps_price, n_features), name='price_input')
    sales_input = tf.keras.Input(shape=(n_steps_sales, n_features), name='sales_input')
    X_input = tf.keras.Input(shape=(X.shape[1],), name='X_input')
    adj_input_price = tf.keras.Input(shape=(220, 220), name='adj_input_price')
    adj_input_sales = tf.keras.Input(shape=(220, 220), name='adj_input_sales')
    nodes_price_feat = tf.keras.Input(shape=(220, 8), name='price_nodes')
    nodes_sales_feat = tf.keras.Input(shape=(220, 8), name='sales_nodes')

    # PRICE LSTM LAYER
    price_LSTM_layer = tf.keras.layers.LSTM(50, activation='tanh')(price_input)
    GNN_price = sp.layers.GCNConv(10, activation='relu')([nodes_price_feat, adj_input_price])
    GNN_price = tf.keras.layers.Flatten()(GNN_price)
    price_LSTM_layer = tf.keras.layers.concatenate([price_LSTM_layer, GNN_price])
    X_price = tf.keras.layers.Dense(1, name='price_LSTM_output')(price_LSTM_layer)

    # QUANTITY LSTM LAYER
    sales_LSTM_layer = tf.keras.layers.LSTM(50, activation='tanh')(sales_input)
    GNN_quantity = sp.layers.GCNConv(10, activation='relu')([nodes_sales_feat, adj_input_sales])
    GNN_quantity = tf.keras.layers.Flatten()(GNN_quantity)
    sales_LSTM_layer = tf.keras.layers.concatenate([sales_LSTM_layer, GNN_quantity])
    X_sales = tf.keras.layers.Dense(1, name='sales_LSTM_output')(sales_LSTM_layer)

    # CONCAT
    X = tf.keras.layers.concatenate([X_price, X_sales, X_input])
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001))(X)
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
        optimizer=tf.keras.optimizers.Adam(lr=0.0001),
        loss={
            "price_output": tf.keras.losses.MeanSquaredError(),
            "sales_output": tf.keras.losses.MeanSquaredError()
        }
    )
    return model


def train_base_model(
        X_price_timesteps: Array,
        X_sales_timesteps: Array,
        X: Array,
        y_price: Array,
        y_sales: Array,
        groups_price_feat: Array,
        groups_sales_feat: Array,
        adj_input_price: Array,
        adj_input_sales: Array) -> object:
    """
    Trains the base model 2X-GLSTM
    """

    model = build_model(
        X_price_timesteps,
        X_sales_timesteps,
        X,
        y_price,
        y_sales,
        groups_price_feat,
        groups_sales_feat,
        adj_input_price,
        adj_input_sales
    )
    model.fit(
        x={
            "price_input": X_price_timesteps,
            "sales_input": X_sales_timesteps,
            "X_input": X,
            "adj_input_price": adj_input_price,
            "adj_input_sales": adj_input_sales,
            "price_nodes": groups_price_feat,
            "sales_nodes": groups_sales_feat
        },
        y={"price_output": y_price, "sales_output": y_sales},
        epochs=10,
        batch_size=32,
        verbose=1,
        callbacks=[lr_callback]
    )
    model.save('../models/2x_lstm.h5')


def train_ensemble(
        X_price_timesteps: Array,
        X_sales_timesteps: Array,
        X: Array,
        y_price: Array,
        y_sales: Array,
        groups_price_feat: Array,
        groups_sales_feat: Array,
        adj_input_price: Array,
        adj_input_sales: Array):
    """
    Trains an ensemble of the base models 2X-GLSTM
    """
    for run in range(6):
        model = build_model(
            X_price_timesteps,
            X_sales_timesteps,
            X,
            y_price,
            y_sales,
            groups_price_feat,
            groups_sales_feat,
            adj_input_price,
            adj_input_sales
        )
        model.fit(
            {"price_input": X_price_timesteps,
             "sales_input": X_sales_timesteps,
             "X_input": X,
             "adj_input_price": adj_input_price,
             "adj_input_sales": adj_input_sales,
             "price_nodes": groups_price_feat,
             "sales_nodes": groups_sales_feat},
            {"price_output": y_price,
             "sales_output": y_sales},
            epochs=7,
            batch_size=32,
            verbose=0,
            callbacks=[lr_callback]
        )

        print(f"Finished 7 epochs of run {run}..")

        for epoch in range(5):
            model.fit(
                x={
                    "price_input": X_price_timesteps,
                    "sales_input": X_sales_timesteps,
                    "X_input": X,
                    "adj_input_price": adj_input_price,
                    "adj_input_sales": adj_input_sales,
                    "price_nodes": groups_price_feat,
                    "sales_nodes": groups_sales_feat
                },
                y={"price_output": y_price, "sales_output": y_sales},
                epochs=1,
                batch_size=32,
                verbose=0,
                callbacks=[lr_callback]
            )

            model.save(f'../src/models/N_2XGLSTM/glstm_r{run}_v{epoch + 1}.h5')

            print(f"Saved model {epoch + 1} of run {run}..")

        print(f"Finshed run {run}..")


if __name__ == '__main__':
    args = parser.parse_args()
    df = pd.read_csv('../data/train/train_nn_target1.csv', sep=';')
    with open('../data/clean/reference_price_to_2018.pkl', 'rb') as file:
        reference_price = pickle.load(file)
    with open('../data/clean/total_sales_to_2018.pkl', 'rb') as file:
        total_sales = pickle.load(file)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler_glstm, verbose=0)

    features_to_scale = list(feature for feature in df.columns if feature not in CATEGORICAL_FEATURES)
    scaler = MinMaxScaler((0, 1))
    df.loc[:, features_to_scale] = scaler.fit_transform(df.loc[:, features_to_scale])
    X_price_timesteps, X_sales_timesteps, X, y_price, y_sales = input_nn(df=df)
    adj_input_price = build_adj_matrix_to_tensor(reference_price, df.shape[0], 0.8)
    adj_input_sales = build_adj_matrix_to_tensor(total_sales, df.shape[0], 0.5)
    groups_price_feat, groups_sales_feat = build_feature_nodes(reference_price, total_sales, X.shape[0])

    if args.ensemble:
        train_ensemble(
            X_price_timesteps,
            X_sales_timesteps,
            X,
            y_price,
            y_sales,
            groups_price_feat,
            groups_sales_feat,
            adj_input_price,
            adj_input_sales
        )
    else:
        train_base_model(
            X_price_timesteps,
            X_sales_timesteps,
            X,
            y_price,
            y_sales,
            groups_price_feat,
            groups_sales_feat,
            adj_input_price,
            adj_input_sales
        )
