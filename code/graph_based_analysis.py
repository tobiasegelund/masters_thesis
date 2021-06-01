import argparse
import pickle
import pandas as pd
import numpy as np
from utils.config import K_FOLDS_TRAIN, K_FOLDS_TEST, THRESHOLDS
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.graphs import cliques_in_group
from typing import TypeVar, Dict, List

DataFrame = TypeVar('DataFrame')
Array = TypeVar('Numpy Array')

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default='price')


def graph_analysis(df: DataFrame, train_years: List[int], val_years: List[int], reference: Dict) -> List[float]:
    """
    Evaluate the graph-based analysis

    :param df: The DataFrame
    :param train_years: The training years
    :param val_years: The validation years
    :reference: The series with respect to the df
    """
    X_train = df[~df['Year'].isin(train_years)].iloc[:, 1:]
    X_val = df[df['Year'].isin(val_years)].iloc[:, 1:]
    y_train = df[~df['Year'].isin(train_years)].loc[:, ['t+1', 'Substitution Group Name']]
    y_val = df[df['Year'].isin(val_years)].loc[:, ['t+1', 'Substitution Group Name']]
    del X_train['Time']
    del X_val['Time']
    mse_threshold = list()
    mae_threshold = list()

    for threshold in THRESHOLDS:
        mse_groups = list()
        mae_groups = list()

        gbm = LGBMRegressor()
        cliques = cliques_in_group(reference, threshold)

        for group in cliques.keys():
            clique = cliques.get(group)

            Z_train = X_train[X_train['Substitution Group Name'].isin(clique)].reset_index(drop=True).values
            Z_val = X_val[X_val['Substitution Group Name'].isin([group])].reset_index(drop=True).values
            y_train_group = y_train[y_train['Substitution Group Name'].isin(clique)].reset_index(drop=True)
            y_val_group = y_val[y_val['Substitution Group Name'].isin([group])].reset_index(drop=True)

            del y_train_group['Substitution Group Name']
            del y_val_group['Substitution Group Name']

            y_train_group = y_train_group.values.reshape(-1)
            y_val_group = y_val_group.values.reshape(-1)

            model = gbm.fit(Z_train, y_train_group)
            y_pred = model.predict(Z_val)
            mse_groups.append(mean_squared_error(y_pred, y_val_group))
            mae_groups.append(mean_absolute_error(y_pred, y_val_group))

        mse_threshold.append(np.mean(mse_groups))
        mae_threshold.append(np.mean(mae_groups))

        print(f"Threshold {threshold} is done..")

    return mse_threshold, mae_threshold


def graph_k_fold() -> DataFrame:
    """
    Evaluate the graph-based analysis using cross validation
    """
    reference_list = [
        reference_10,
        reference_12,
        reference_14,
        reference_16
    ]

    overall_mse = np.zeros(len(THRESHOLDS))
    overall_mae = np.zeros(len(THRESHOLDS))
    for fold in K_FOLDS_TRAIN.keys():
        train_years = K_FOLDS_TRAIN[fold]
        val_years = K_FOLDS_TEST[fold + 1]
        reference = reference_list[fold]

        mse_threshold, mae_threshold = graph_analysis(df=df, train_years=train_years, val_years=val_years, reference=reference)
        overall_mse = overall_mse + mse_threshold
        overall_mae = overall_mae + mae_threshold
        print(f'Fold {fold} is done..')

    overall_mse = overall_mse / len(K_FOLDS_TRAIN.keys())
    overall_mae = overall_mae / len(K_FOLDS_TRAIN.keys())
    results = pd.DataFrame({"threhold": THRESHOLDS, "mse": overall_mse, "mae": overall_mae})
    return results


if __name__ == '__main__':
    args = parser.parse_args()
    if args.category == 'price':
        df = pd.read_csv('../data/train/train_gbm_price_target1.csv', sep=';')
        with open('../data/clean/reference_price_to_2016.pkl', 'rb') as file:
            reference_16 = pickle.load(file)
        with open('../data/clean/reference_price_to_2014.pkl', 'rb') as file:
            reference_14 = pickle.load(file)
        with open('../data/clean/reference_price_to_2012.pkl', 'rb') as file:
            reference_12 = pickle.load(file)
        with open('../data/clean/reference_price_to_2010.pkl', 'rb') as file:
            reference_10 = pickle.load(file)
    if args.category == 'sales':
        df = pd.read_csv('../data/train/train_gbm_price_target1.csv', sep=';')
        with open('../data/clean/total_sales_to_2016.pkl', 'rb') as file:
            reference_16 = pickle.load(file)
        with open('../data/clean/total_sales_to_2014.pkl', 'rb') as file:
            reference_14 = pickle.load(file)
        with open('../data/clean/total_sales_to_2012.pkl', 'rb') as file:
            reference_12 = pickle.load(file)
        with open('../data/clean/total_sales_to_2010.pkl', 'rb') as file:
            reference_10 = pickle.load(file)

    results = graph_k_fold()
    results.to_csv(f'../results/graph_analysis_{args.category}.csv', sep=';', index=False)
