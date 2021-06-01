import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingRegressor
from lightgbm import LGBMRegressor
from typing import TypeVar
from utils.config import PARAMS_PRICE, PARAMS_SALES

DataFrame = TypeVar('DataFrame')
Array = TypeVar('Numpy Array')

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=int, default=1)
parser.add_argument('--estimators', type=int, default=30)


def build_gbm_set(df: DataFrame, category: str) -> Array:
    """
    Build training set to the base model 2X-LGBM

    :param df: The DataFrame with features
    :param category: The category (price or sales)
    """
    if category == 'price':
        importances = np.load("../results/avg_feature_importance_price.npy")
        top_features = (-importances).argsort()[:41]
    elif category == 'sales':
        importances = np.load("../results/avg_feature_importance_sales.npy")
        top_features = (-importances).argsort()[:50]
    else:
        print('Choose either price or sales in category..')

    X = df.iloc[:, 1:]
    del X['Time']
    X = X.values
    y = df.iloc[:, :1].values.reshape(-1)

    Z = X[:, top_features]

    return Z, y


def train_base_model_2X_lgbm(df_price: DataFrame, df_sales: DataFrame) -> object:
    """
    Train the base model 2X-LGBM
    """
    X_price, y_price = build_gbm_set(df_price, 'price')
    X_sales, y_sales = build_gbm_set(df_sales, 'sales')

    gbm_price = LGBMRegressor(**PARAMS_PRICE)
    gbm_sales = LGBMRegressor(**PARAMS_SALES)

    model_price = gbm_price.fit(X_price, y_price)
    model_sales = gbm_sales.fit(X_sales, y_sales)

    return model_price, model_sales


def train_network_2X_lgbm(df: DataFrame, n_estimators: int) -> object:
    """
    Train an ensemble of the base model 2X-LGBM
    """
    X_price, y_price = build_gbm_set(df, 'price')
    X_sales, y_sales = build_gbm_set(df, 'sales')

    gbm_price = LGBMRegressor(**PARAMS_PRICE)
    gbm_sales = LGBMRegressor(**PARAMS_SALES)
    regr_price = BaggingRegressor(base_estimator=gbm_price, n_estimators=n_estimators).fit(X_price, y_price)
    regr_sales = BaggingRegressor(base_estimator=gbm_sales, n_estimators=n_estimators).fit(X_sales, y_sales)

    return regr_price, regr_sales


if __name__ == '__main__':
    args = parser.parse_args()
    df_price = pd.read_csv(f'../data/train/train_gbm_price_target{args.target}.csv', sep=';')
    df_sales = pd.read_csv(f'../data/train/train_gbm_sales_target{args.target}.csv', sep=';')
    base_model_price, base_model_sales = train_base_model_2X_lgbm(df_price, df_sales)
    network_model_price, network_model_sales = train_network_2X_lgbm(df_price, df_sales)

    with open(f'../models/base_model_lgbm_price_target{args.target}.pkl', 'wb') as file:
        pickle.dump(base_model_price, file)

    with open(f'../models/base_model_lgbm_sales_target{args.target}.pkl', 'wb') as file:
        pickle.dump(base_model_sales, file)

    with open(f'../models/network_lgbm_price_target{args.target}.pkl', 'wb') as file:
        pickle.dump(network_model_price, file)

    with open(f'../models/network_lgbm_sales_target{args.target}.pkl', 'wb') as file:
        pickle.dump(network_model_sales, file)
