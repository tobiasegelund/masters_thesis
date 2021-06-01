import argparse
import tensorflow as tf
import pandas as pd
from utils.nn_preprocessing import input_nn, scheduler_lstm
from utils.config import CATEGORICAL_FEATURES
from sklearn.preprocessing import MinMaxScaler
from typing import TypeVar, List, Dict

DataFrame = TypeVar('DataFrame')
Array = TypeVar('Array')

parser = argparse.ArgumentParser()
parser.add_argument('--ensemble', type=bool, default=False)


def build_model(X_price_timesteps: Array, X_sales_timesteps: Array, X: Array) -> object:
    """
    Return the base model 2X-GLSTM

    :param X_price_timesteps: An array with price timesteps
    :param X_sales_timesteps: An array with sales timesteps
    :param X: An array with features
    """
    n_steps_price = len(X_price_timesteps[0])
    n_steps_sales = len(X_sales_timesteps[0])
    n_features = 2

    price_input = tf.keras.Input(shape=(n_steps_price, n_features), name='price_input')
    sales_input = tf.keras.Input(shape=(n_steps_sales, n_features), name='sales_input')
    X_input = tf.keras.Input(shape=(X.shape[1],), name='X_input')

    price_LSTM_layer = tf.keras.layers.LSTM(50, activation='tanh')(price_input)
    sales_LSTM_layer = tf.keras.layers.LSTM(50, activation='tanh')(sales_input)

    X_price = tf.keras.layers.Dense(1, name='price_LSTM_output')(price_LSTM_layer)
    X_sales = tf.keras.layers.Dense(1, name='sales_LSTM_output')(sales_LSTM_layer)

    X = tf.keras.layers.concatenate([X_price, X_sales, X_input])
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001))(X)
    X = tf.keras.layers.Dense(200, activation='relu')(X)

    price_output = tf.keras.layers.Dense(1, name='price_output')(X)
    sales_output = tf.keras.layers.Dense(1, name='sales_output')(X)

    model = tf.keras.Model(
        inputs=[price_input, sales_input, X_input],
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


def train_base_model(X_price_timesteps, X_sales_timesteps, X, y_price, y_sales):
    """
    Trains the base model 2X-GLSTM
    """
    model = build_model(X_price_timesteps, X_sales_timesteps, X)
    model.fit(
        x={
            "price_input": X_price_timesteps,
            "sales_input": X_sales_timesteps,
            "X_input": X
        },
        y={
            "price_output": y_price,
            "sales_output": y_sales
        },
        epochs=30,
        batch_size=32,
        callbacks=[lr_callback]
    )
    model.save('../models/2x_lstm.h5')


def train_ensemble(X_price_timesteps, X_sales_timesteps, X, y_price, y_sales):
    """
    Trains an ensemble of the base models 2X-GLSTM
    """
    for run in range(6):
        model = build_model(X_price_timesteps, X_sales_timesteps, X)
        EPOCHS = 25
        model.fit(
            x={"price_input": X_price_timesteps, "sales_input": X_sales_timesteps, "X_input": X},
            y={"price_output": y_price, "sales_output": y_sales},
            epochs=EPOCHS,
            batch_size=32,
            callbacks=[lr_callback],
            verbose=0
        )

        for epoch in range(5):
            model.fit(
                x={"price_input": X_price_timesteps, "sales_input": X_sales_timesteps, "X_input": X},
                y={"price_output": y_price, "sales_output": y_sales},
                epochs=1,
                batch_size=32,
                callbacks=[lr_callback],
                verbose=0
            )
            model.save(f'../models/N_2XLSTM/2x_lstm_r{run}_v{epoch + 1}.h5')

            print(f"Saved model {epoch + 1} of run {run}..")

        print(f"Finshed run {run}..")


if __name__ == '__main__':
    args = parser.parse_args()
    df = pd.read_csv('../data/train/train_nn_target1.csv', sep=';')
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler_lstm, verbose=0)

    features_to_scale = list(feature for feature in df.columns if feature not in CATEGORICAL_FEATURES)
    scaler = MinMaxScaler((0, 1))
    df.loc[:, features_to_scale] = scaler.fit_transform(df.loc[:, features_to_scale])
    X_price_timesteps, X_sales_timesteps, X, y_price, y_sales = input_nn(df=df)
    if args.ensemble:
        train_ensemble(X_price_timesteps, X_sales_timesteps, X, y_price, y_sales)
    else:
        train_base_model(X_price_timesteps, X_sales_timesteps, X, y_price, y_sales)
