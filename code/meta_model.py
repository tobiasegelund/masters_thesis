import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import TypeVar
from sklearn.preprocessing import MinMaxScaler
from utils.nn_preprocessing import scheduler_meta

parser = argparse.ArgumentParser()
parser.add_argument('--units', type=int, default=50)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.0001)

DataFrame = TypeVar('DataFrame')
Array = TypeVar('Numpy Array')
Model = TypeVar('Model')


def build_model(X_train: Array) -> Model:
    """
    Build the meta model

    :param X_train: The input features
    """
    input_ = tf.keras.Input(shape=(X_train.shape[1], ))

    layer = tf.keras.layers.Dense(args.units, activation='relu')(input_)
    layer = tf.keras.layers.BatchNormalization()(layer)

    price_output = tf.keras.layers.Dense(1, name='price_output')(layer)
    quantity_output = tf.keras.layers.Dense(1, name='quantity_output')(layer)

    model = tf.keras.Model(
        inputs=[input_],
        outputs=[price_output, quantity_output]
    )
    model.compile(
        loss={
            "price_output": tf.keras.losses.MeanSquaredError(),
            "quantity_output": tf.keras.losses.MeanSquaredError()
        },
        optimizer=tf.keras.optimizers.Adam(lr=args.lr)
    )
    return model


def build_synthetic_training_set() -> DataFrame:
    """
    Complete the synthetic training set
    """
    lgbm_price_train = np.load('../src/data/stack/lgbm_price_train.npy')
    lstm_price_train = np.load('../src/data/stack/lstm_price_train.npy')
    glstm_price_train = np.load('../src/data/stack/glstm_price_train.npy')

    lgbm_sales_train = np.load('../src/data/stack/lgbm_sales_train.npy')
    lstm_sales_train = np.load('../src/data/stack/lstm_sales_train.npy')
    glstm_sales_train = np.load('../src/data/stack/glstm_sales_train.npy')

    df = pd.read_csv('../data/train/train_nn_target1.csv', sep=';')
    df = df[~df['Year'].isin([2007, 2008, 2009, 2010])]
    df_train = pd.DataFrame(
        {
            'LGBM price': lgbm_price_train,
            'LSTM price': lstm_price_train,
            'GLSTM price': glstm_price_train,
            'LGBM sales': lgbm_sales_train,
            'LSTM sales': lstm_sales_train,
            'GLSTM sales': glstm_sales_train
        }
    )
    df = df[[
        't+1',
        't+1_quantity',
        'Time',
        'Substitution Group Name'
    ]].reset_index(drop=True)
    df = pd.concat([df, df_train], axis=1)
    return df


def generate_train_set(df: DataFrame) -> Array:
    """
    Prepare the synthetic training set

    :param df: The synthetic training set
    """
    scaler = MinMaxScaler((0, 1))
    y_price = df['t+1']
    y_sales = df['t+1 sales']
    X = df[['LGBM price', 'LGBM sales', 'LSTM price', 'LSTM sales', 'GLSTM price', 'GLSTM sales']]
    X = scaler.fit_transform(X)

    return X, y_price, y_sales


def train_meta_model(df: DataFrame) -> Model:
    """
    Train the meta model using the synthetic training set

    :param df: The synthetic training set
    """
    X_train, y_train_price, y_train_sales = generate_train_set(df)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler_meta, verbose=0)
    model = build_model(X_train)
    model.fit(
        x=X_train,
        y=[y_train_price, y_train_sales],
        epochs=args.epochs,
        verbose=1,
        batch_size=32,
        callbacks=[lr_callback]
    )
    return model


if __name__ == '__main__':
    args = parser.parse_args()
    synthetic_training_set = build_synthetic_training_set()
    model = train_meta_model(synthetic_training_set)
    model.save('../models/meta_model.h5')
