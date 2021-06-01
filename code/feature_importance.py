import argparse
import pandas as pd
import numpy as np
from typing import TypeVar, Dict
from lightgbm import LGBMRegressor
from utils.split_time_series import SplitOnYears
from utils.config import K_FOLDS, PARAMS_PRICE, PARAMS_SALES

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default='price')

DataFrame = TypeVar('DataFrame')
Array = TypeVar('Numpy Array')


def average_feature_importance(df: DataFrame, params: Dict) -> Array:
    """
    Returns the average feature importance

    :param df: The training set
    :param params: The hyperparameters in 2X-LGBM
    """
    kf = SplitOnYears(K_FOLDS)
    kf.get_indices(df, varname='Year')
    feature_importance = np.array(0)
    ctr = 1

    X = df.iloc[:, 1:]
    y = df.iloc[:, :1].values.reshape(-1)
    del X['Time']
    X = X.values
    for train_index, test_index in kf.split(block_split=False):
        X_train, _ = X[train_index], X[test_index]
        y_train, _ = y[train_index], y[test_index]
        gbm = LGBMRegressor(**params)
        model = gbm.fit(X_train, y_train)
        feature_importance = feature_importance + model.feature_importances_
        ctr += 1

    return feature_importance / ctr


if __name__ == '__main__':
    args = parser.parse_args()
    if args.category == 'price':
        df = pd.read_csv('../data/train/train_gbm_price_target1.csv', sep=';')
        params = PARAMS_PRICE
    if args.category == 'sales':
        df = pd.read_csv('../data/train/train_gbm_sales_target1.csv', sep=';')
        params = PARAMS_SALES
    avg_feature_importance = average_feature_importance(df=df, params=params)
    np.save(f'../results/avg_feature_importance_{args.category}.npy', avg_feature_importance)
