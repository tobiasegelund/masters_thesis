import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from utils.config import K_FOLDS_TRAIN, K_FOLDS_TEST, BLOCK_FOLDS_TRAIN, BLOCK_FOLDS_TEST
from utils.nn_preprocessing import generate_train_val_set_kfold, define_scalings, scheduler_lstm
from typing import TypeVar

DataFrame = TypeVar('DataFrame')
Array = TypeVar('Array')
Model = TypeVar('Model')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--dense', type=int, default=200)
parser.add_argument('--lstm', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--l1', type=float, default=0.001)
parser.add_argument('--synthetic', type=bool, default=False)


def build_model(
        X_train_price_timesteps: Array,
        X_train_sales_timesteps: Array,
        X_train: Array,
        optimizer: str) -> Model:
    """
    Return the base model 2X-LSTM

    :param X_train_price_timesteps: The timesteps of prices
    :param X_train_sales_timesteps: The timesteps of sales
    :param X_train: The rest of the features
    """
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr=args.lr),
    if optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(lr=args.lr)
    n_steps_price = len(X_train_price_timesteps[0])
    n_steps_quantity = len(X_train_sales_timesteps[0])
    n_features = 2

    price_input = tf.keras.Input(shape=(n_steps_price, n_features), name='price_input')
    sales_input = tf.keras.Input(shape=(n_steps_quantity, n_features), name='sales_input')
    X_input = tf.keras.Input(shape=(X_train.shape[1],), name='X_input')

    price_LSTM_layer = tf.keras.layers.LSTM(args.lstm, activation='tanh')(price_input)
    quantity_LSTM_layer = tf.keras.layers.LSTM(args.lstm, activation='tanh')(sales_input)

    X_price = tf.keras.layers.Dense(1, name='price_LSTM_output')(price_LSTM_layer)
    X_quantity = tf.keras.layers.Dense(1, name='quantity_LSTM_output')(quantity_LSTM_layer)

    X = tf.keras.layers.concatenate([X_price, X_quantity, X_input])
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Dense(args.dense, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(args.l1))(X)
    X = tf.keras.layers.Dense(args.dense, activation='relu')(X)

    price_output = tf.keras.layers.Dense(1, name='price_output')(X)
    sales_output = tf.keras.layers.Dense(1, name='sales_output')(X)

    model = tf.keras.Model(
        inputs=[price_input, sales_input, X_input],
        outputs=[price_output, sales_output]
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "price_output": tf.keras.losses.MeanSquaredError(),
            "sales_output": tf.keras.losses.MeanSquaredError()
        }
    )
    return model


def lstm_kfold(df: DataFrame) -> DataFrame:
    """
    Evaluate the hyperparameters using cross validation

    :param df: The training set
    """
    results = list()
    loss = list()
    val_loss = list()

    for fold in K_FOLDS_TRAIN.keys():
        train_years = K_FOLDS_TRAIN[fold]
        val_years = K_FOLDS_TEST[fold + 1]

        X_train_price_timesteps, X_train_sales_timesteps, \
            X_train, y_train_price, y_train_sales, \
            X_val_price_timesteps, X_val_sales_timesteps, \
            X_val, y_val_price, y_val_sales = generate_train_val_set_kfold(df, train_years=train_years, val_years=val_years)

        model = build_model(X_train_price_timesteps, X_train_sales_timesteps, X_train, optimizer=args.optimizer)
        history = model.fit(
            x={"price_input": X_train_price_timesteps, "sales_input": X_train_sales_timesteps, "X_input": X_train},
            y={"price_output": y_train_price, "sales_output": y_train_sales},
            validation_data=({"price_input": X_val_price_timesteps, "sales_input": X_val_sales_timesteps, "X_input": X_val},
                             {"price_output": y_val_price, "sales_output": y_val_sales}),
            epochs=30,
            batch_size=32,
            verbose=0
        )

        loss.append(history.history['loss'][-1])
        val_loss.append(history.history['val_loss'][-1])

        print(f'Fold {fold} is done..')
    results.append([np.mean(loss), np.mean(val_loss), np.std(loss), np.std(val_loss)])

    results = pd.DataFrame(results)
    col_names = ['loss', 'val_loss', 'std_loss', 'std_val_loss']
    results.columns = col_names
    return results


def generate_synthetic_training_set() -> Array:
    """
    Build the synthetic training set
    """
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler_lstm, verbose=0)
    synthetic_training_set_price = list()
    synthetic_training_set_sales = list()
    for fold in BLOCK_FOLDS_TRAIN.keys():
        train_years = BLOCK_FOLDS_TRAIN[fold]
        val_years = BLOCK_FOLDS_TEST[fold + 1]

        X_train_price_timesteps, X_train_sales_timesteps, \
            X_train, y_train_price, y_train_sales, \
            X_val_price_timesteps, X_val_sales_timesteps, \
            X_val, y_val_price, y_val_sales = generate_train_val_set_kfold(df, train_years=train_years, val_years=val_years)
        scaler_price, scaler_sales = define_scalings(df, train_years)

        model = build_model(X_train_price_timesteps, X_train_sales_timesteps, X_train, optimizer=args.optimizer)
        model.fit(
            x={"price_input": X_train_price_timesteps, "sales_input": X_train_sales_timesteps, "X_input": X_train},
            y={"price_output": y_train_price, "sales_output": y_train_sales},
            epochs=30,
            batch_size=32,
            callbacks=[lr_callback],
            verbose=0
        )

        y_pred = model.predict([X_val_price_timesteps, X_val_sales_timesteps, X_val])
        synthetic_training_set_price.append(scaler_price.inverse_transform(y_pred[0]))
        synthetic_training_set_sales.append(scaler_sales.inverse_transform(y_pred[1]))
        print(f'Fold {fold} is done..')
    return synthetic_training_set_price, synthetic_training_set_sales


if __name__ == '__main__':
    args = parser.parse_args()
    df = pd.read_csv('../data/train/train_nn_target1.csv', sep=';')

    if args.synthetic:
        synthetic_training_set_price, synthetic_training_set_sales = generate_synthetic_training_set()
        np.save('../data/train/synthetic_training_set_price_lstm.npy', synthetic_training_set_price)
        np.save('../data/train/synthetic_training_set_sales_lstm.npy', synthetic_training_set_sales)
    else:
        results = lstm_kfold(df)
        print(results)
