import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
from utils.split_time_series import SplitOnYears
from utils.config import K_FOLDS, LEARNING_RATES, MAX_DEPTHS, MIN_DATA_IN_LEAFS, ESTIMATORS, NUM_LEAVESS, BLOCK_FOLDS, PARAMS_PRICE, PARAMS_SALES
from typing import TypeVar, List, Dict

DataFrame = TypeVar('DataFrame')
Array = TypeVar('Numpy Array')

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default='price')
parser.add_argument('--target', type=int, default=1)
parser.add_argument('--synthetic', type=bool, default=False)


def lgbm_kfold(df: DataFrame) -> DataFrame:
    """
    Evaluate the hyperparameters using cross validation

    :param df: The training set
    """
    kf = SplitOnYears(K_FOLDS)
    kf.get_indices(df, varname='Year')
    results = list()

    X = df.iloc[:, 1:]
    y = df.iloc[:, :1].values.reshape(-1)
    del X['Time']
    X = X.values

    for min_data_in_leaf in MIN_DATA_IN_LEAFS:
        for max_depth in MAX_DEPTHS:
            for num_leaves in NUM_LEAVESS:
                for learning_rate in LEARNING_RATES:
                    for estimator in ESTIMATORS:
                        mae_results = list()
                        mse_results = list()

                        params = {
                            "boosting_type": 'gbdt',
                            "n_estimators": estimator,
                            "learning_rate": learning_rate,
                            "max_depth": max_depth,
                            'metric': 'rmse',
                            'num_leaves': num_leaves,
                            'min_child_samples': min_data_in_leaf
                        }

                        for train_index, test_index in kf.split(block_split=False):
                            X_train, X_test = X[train_index], X[test_index]
                            y_train, y_test = y[train_index], y[test_index]
                            gbm = LGBMRegressor(**params)
                            model = gbm.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            mae_results.append(mean_absolute_error(y_test, y_pred))
                            mse_results.append(mean_squared_error(y_test, y_pred))

                        results.append([
                            np.mean(mse_results),
                            np.std(mse_results),
                            np.mean(mae_results),
                            np.std(mae_results),
                            learning_rate,
                            estimator,
                            max_depth,
                            num_leaves,
                            min_data_in_leaf])

                        print(f"Completed learning rate: {learning_rate}, estimators: {estimator}, max depth: {max_depth}, number of leaves: {num_leaves}, mininum number of leaves: {min_data_in_leaf}")
    results = pd.DataFrame(results)
    results.columns = ['MSE', 'MSE_STD', 'MAE', 'MAE_STD', 'learning_rate',
                       'estimators', "max_depth", "num_leaves", "min_data_in_leaf"]
    return results


def generate_synthetic_training_set(category: str) -> Array:
    """
    Build the synthetic training set

    :param category: The category (price or sales)
    """
    kf = SplitOnYears(BLOCK_FOLDS)
    kf.get_indices(df, varname='Year')

    if category == 'price':
        params = PARAMS_PRICE
    else:
        params == PARAMS_SALES

    X = df.iloc[:, 1:]
    y = df.iloc[:, :1].values.reshape(-1)
    del X['Time']
    X = X.values
    synthetic_training_set = list()

    for train_index, test_index in kf.split(block_split=False):
        X_train, X_test = X[train_index], X[test_index]
        y_train, _ = y[train_index], y[test_index]
        gbm = LGBMRegressor(**params)
        model = gbm.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        synthetic_training_set.append(y_pred)

    synthetic_training_set = np.array(synthetic_training_set).reshape(-1)
    return synthetic_training_set


if __name__ == '__main__':
    args = parser.parse_args()
    print(f'Category is {args.category}..')
    if args.category == 'price':
        df = pd.read_csv(f'../data/train/train_gbm_price_target{args.target}.csv', sep=';')
    if args.category == 'sales':
        df = pd.read_csv(f'../data/train/train_gbm_sales_target{args.target}.csv', sep=';')
    if args.synthetic:
        synthetic_training_set = generate_synthetic_training_set(category=args.category)
        np.save('../data/train/synthetic_training_set_{args.category}_lgbm.npy', synthetic_training_set)
    else:
        results = lgbm_kfold(df=df)
        results.to_csv(f'../results/lgbm_hyperparamters_{args.category}_target{args.target}.csv', sep=';', index=False)
