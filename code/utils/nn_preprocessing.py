import pandas as pd
import numpy as np
from typing import TypeVar, List, Dict
from sklearn.preprocessing import MinMaxScaler
from .config import CATEGORICAL_FEATURES, NOT_INCLUDE_FEATURES
from .graphs import find_adj_matrix
import random

DataFrame = TypeVar('DataFrame')
Array = TypeVar('Array')


def embed_timesteps(arr: Array) -> Array:
    """
    Masking missing timesteps with 1 if the value is missing, otherwise 0

    :param arr: The array to mask
    """
    new_timesteps = list()
    for time_step in arr:
        temp_list = list()
        for val in time_step:
            if np.isnan(val):
                temp_list.append([random.random(), 1])
            else:
                temp_list.append([val, 0])

        new_timesteps.append(temp_list)

    return np.array(new_timesteps)


def random_missing_columns(df: DataFrame) -> Array:
    """
    Randomise a DataFrame columns' missing values by a value between zero and one

    :param df: A pandas DataFrame
    """
    import random
    df_ = df.copy()
    for col in df_.columns:
        data = df_[col]
        mask = data.isnull()
        samples = list(random.random() for i in range(mask.sum()))
        data[mask] = samples

    return df_


def build_adj_matrix_to_tensor(reference: Dict, size: int, threshold: float) -> Array:
    """
    Returns a adjacency matrix in tensorformat to match the number of observations

    :param reference: The dictionary with each series
    :param size: The size of the tensor / training or validation set
    :param threshold: The threshold to determine whether there is a path between two vertices
    """
    df_reference = pd.DataFrame(reference)
    adj_matrix_price = find_adj_matrix(df_reference.corr().values, 0.8)
    adj_matrix_price_ = np.array(list(adj_matrix_price for i in range(size)))
    return np.array(adj_matrix_price_)


def build_feature_nodes(reference_price: Dict, reference_sales: Dict, size: int) -> Array:
    """
    Build features nodes as input to 2X-GLSTM

    :param reference_price: The reference prices of each substitution group
    :param reference_sales: The total sales for each substitution group
    :param size: The size of the array to engage as input in 2X-GLSTM
    """
    scaler_nodes_price = MinMaxScaler((0, 1))
    scaler_nodes_sales = MinMaxScaler((0, 1))
    df_reference_price = pd.DataFrame(reference_price)
    df_reference_price.loc[:, :] = scaler_nodes_price.fit_transform(df_reference_price.loc[:, :])
    groups_price_feat = df_reference_price.describe().append(pd.DataFrame(df_reference_price.kurtosis()).to_dict()[0], ignore_index=True)
    groups_price_feat = np.array(list(groups_price_feat.values[1:, :] for i in range(size)))

    df_reference_sales = pd.DataFrame(reference_sales)
    df_reference_sales.loc[:, :] = scaler_nodes_sales.fit_transform(df_reference_sales.loc[:, :])
    groups_sales_feat = df_reference_sales.describe().append(pd.DataFrame(df_reference_sales.kurtosis()).to_dict()[0], ignore_index=True)
    groups_sales_feat = np.array(list(groups_sales_feat.values[1:, :] for i in range(size)))
    groups_price_feat = groups_price_feat.reshape(size, 220, 8)
    groups_sales_feat = groups_sales_feat.reshape(size, 220, 8)

    return groups_price_feat, groups_sales_feat


def define_scalings(df: DataFrame, train_years: List) -> object:
    """
    Returns two MinMaxScaler objects on price and sales respectively

    :param df: The training DataFrame
    :param train_years: A list with training years
    """
    scaler_price = MinMaxScaler((0, 1))
    scaler_sales = MinMaxScaler((0, 1))
    train_df = df[df['Year'].isin(train_years)]
    scaler_price.fit(train_df['t+1'].values.reshape(-1, 1))
    scaler_sales.fit(train_df['t+1 sales'].values.reshape(-1, 1))
    return scaler_price, scaler_sales


def input_nn(df: DataFrame) -> Array:
    """
    Returns the price timesteps, the sales timesteps and rest of the features in input format to NN

    :param df: The DataFrame to convert
    """
    features = list(feature for feature in df.columns if feature not in NOT_INCLUDE_FEATURES)
    X = df.loc[:, features]
    del X['Time']
    y_price = df.loc[:, ['t+1']]
    y_sales = df.loc[:, ['t+1 sales']]

    # timesteps
    X_price_timesteps = df.loc[:, [
        't-25',
        't-12',
        't-3',
        't-2',
        't-1',
        't']].values

    X_sales_timesteps = df.loc[:, [
        't-25 sales',
        't-12 sales',
        't-3 sales',
        't-2 sales',
        't-1 sales',
        't sales']].values

    # fill nans
    X = random_missing_columns(X)

    X_price_timesteps = embed_timesteps(X_price_timesteps)
    X_sales_timesteps = embed_timesteps(X_sales_timesteps)

    return X_price_timesteps, X_sales_timesteps, X, y_price, y_sales


def generate_train_val_set_kfold( df: DataFrame, train_years: List, val_years: List) -> Array:
    """
    Returns the price timesteps, the sales timesteps and rest of the features in input format to NN
    to both training and validation

    :param df: The DataFrame to convert
    :param train_years: The training years
    :param val_years: The validation years
    """
    scaler = MinMaxScaler((0, 1))

    features_to_scale = list(feature for feature in df.columns if feature not in CATEGORICAL_FEATURES)
    X_features = list(feature for feature in df.columns if feature not in NOT_INCLUDE_FEATURES)

    train = df[df['Year'].isin(train_years)]
    val = df[df['Year'].isin(val_years)]
    train_scaled = train.copy()
    val_scaled = val.copy()
    train_scaled.loc[:, features_to_scale] = scaler.fit_transform(train.loc[:, features_to_scale])
    val_scaled.loc[:, features_to_scale] = scaler.transform(val.loc[:, features_to_scale])
    X_train = train_scaled.loc[:, X_features]
    X_val = val_scaled.loc[:, X_features]

    # targets
    y_train_price = train_scaled.loc[:, ['t+1']].values.reshape(-1)
    y_train_sales = train_scaled.loc[:, ['t+1 sales']].values.reshape(-1)
    y_val_price = val_scaled.loc[:, ['t+1']].values.reshape(-1)
    y_val_sales = val_scaled.loc[:, ['t+1 sales']].values.reshape(-1)

    # timesteps
    X_train_price_timesteps = train_scaled.loc[:, [
        't-25',
        't-12',
        # 't-7',
        # 't-6',
        # 't-5',
        # 't-4',
        't-3',
        't-2',
        't-1',
        't']].values

    X_train_sales_timesteps = train_scaled.loc[:, [
        't-25 sales',
        't-12 sales',
        # 't-7 sales',
        # 't-6 sales',
        # 't-5 sales',
        # 't-4 sales',
        't-3 sales',
        't-2 sales',
        't-1 sales',
        't sales']].values

    X_val_price_timesteps = val_scaled.loc[:, [
        't-25',
        't-12',
        # 't-7',
        # 't-6',
        # 't-5',
        # 't-4',
        't-3',
        't-2',
        't-1',
        't']].values

    X_val_sales_timesteps = val_scaled.loc[:, [
        't-25 sales',
        't-12 sales',
        # 't-7 sales',
        # 't-6 sales',
        # 't-5 sales',
        # 't-4 sales',
        't-3 sales',
        't-2 sales',
        't-1 sales',
        't sales']].values

    # fill nans
    X_train = random_missing_columns(X_train)
    X_val = random_missing_columns(X_val)
    del X_train['Time']
    del X_val['Time']

    X_train_price_timesteps = embed_timesteps(X_train_price_timesteps)
    X_train_sales_timesteps = embed_timesteps(X_train_sales_timesteps)
    X_val_price_timesteps = embed_timesteps(X_val_price_timesteps)
    X_val_sales_timesteps = embed_timesteps(X_val_sales_timesteps)

    return X_train_price_timesteps, X_train_sales_timesteps, \
           X_train, y_train_price, y_train_sales, X_val_price_timesteps, \
           X_val_sales_timesteps, X_val, y_val_price, y_val_sales


def scheduler_glstm(epoch: int, lr: float) -> float:
    """
    Learning rate scheduler for 2X-GLSTM
    """
    if epoch < 3:
        return lr
    elif epoch < 7:
        return 0.00001
    else:
        return 0.000001


def scheduler_lstm(epoch: int, lr: float) -> float:
    """
    Learning rate scheduler for 2X-LSTM
    """
    if epoch < 20:
        return lr
    else:
        return 0.00001


def scheduler_meta(epoch: int, lr: float) -> float:
    """
    Learning rate scheduler for meta model
    """
    if epoch < 20:
        return lr
    elif epoch < 50:
        return 0.00001
    else:
        return 0.000001
