import pickle
import argparse
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import TypeVar, Dict

parser = argparse.ArgumentParser()
parser.add_argument("--target", type=int, default=1)
parser.add_argument("--diff", type=bool, default=True)
parser.add_argument("--lags", type=int, default=7)
parser.add_argument("--nn", type=bool, default=True)

DataFrame = TypeVar('DataFrame')
get_period_ctr = 11  # start week in 2007


def calendar_features() -> DataFrame:
    """
    Build calendar features
    """
    def get_month(s):
        r = datetime.datetime.strptime(s + '-1', "%Y-W%W-%w")
        return r.month

    def get_period(week):
        global get_period_ctr
        get_period_ctr += 1

        if week == 1 or week == 2:
            get_period_ctr = 1

        return get_period_ctr

    dates = market_sales[['Time', 'Year', 'Week']].drop_duplicates().reset_index(drop=True)
    dates['Month'] = dates['Year'].astype(str) + "-W" + dates['Week'].astype(str)
    dates['Month'] = dates['Month'].apply(get_month)
    dates['Period'] = dates['Week']
    dates['Period'] = dates['Period'].apply(get_period)
    dates['period_sin'] = dates['Period']
    dates['period_cos'] = dates['Period']
    dates['period_sin'] = dates['period_sin'].apply(lambda p: np.sin((2 * np.pi * p) / dates['period_sin'].max()))
    dates['period_cos'] = dates['period_cos'].apply(lambda p: np.cos((2 * np.pi * p) / dates['period_cos'].max()))

    return dates


def generate_drug_features(
        category: str,
        reference: Dict,
        second_reference: Dict,
        group: int,
        lags: int,
        target: int,
        diff=True) -> DataFrame:
    """
    Returns a DataFrame with features of an substitution group

    :param category: The category (price or sales)
    :param reference: The series of the category
    :param second_category: The series of the other category
    :param group: The substitution group
    :param lags: The number of lags
    :param target: The Q-ahead-target
    :param diff: Whether to first-order differencing or not
    """
    # assert category == 'price' or category == 'sales'
    if category == 'price':
        second_category = 'sales'
        t_second_category = 't sales'
        drug = np.log(pd.DataFrame(reference[group], columns=['PPP']))
        sub_first = np.log(pd.DataFrame(reference[group], columns=['PPP']))
        sub_second = np.log(pd.DataFrame(second_reference[group], columns=['PPP']) + 1)
    else:
        second_category = 'price'
        t_second_category = 't price'
        drug = np.log(pd.DataFrame(reference[group], columns=['PPP']) + 1)
        sub_first = np.log(pd.DataFrame(reference[group], columns=['PPP']) + 1)
        sub_second = np.log(pd.DataFrame(second_reference[group], columns=['PPP']))

    t_category = f"t+{target}"
    # number of lags (autoregressive)
    col_names = [t_category, 't']
    for lag in range(target, lags + target):
        drug = pd.concat([drug, sub_first['PPP'].shift(lag)], axis=1).reset_index(drop=True)
        if lag > target:
            col_names.append(f't-{lag-target}')

    col_names.append(t_second_category)
    for lag in range(target, lags + target):
        drug = pd.concat([drug, sub_second['PPP'].shift(lag)], axis=1).reset_index(drop=True)
        if lag > target:
            col_names.append(f't-{lag-target} {second_category}')
    drug.columns = col_names

    # seasonal main category
    ref = pd.Series(reference[group], name='t-12')
    drug = pd.concat([drug, ref.shift(target + 13 - 1)], axis=1)
    ref = pd.Series(reference[group], name='t-25')
    drug = pd.concat([drug, ref.shift(target + 26 - 1)], axis=1)

    # seasonal second category
    sec_ref = pd.Series(second_reference[group], name=f't-12 {second_category}')
    drug = pd.concat([drug, sec_ref.shift(target + 13 - 1)], axis=1)
    sec_ref = pd.Series(second_reference[group], name=f't-25 {second_category}')
    drug = pd.concat([drug, sec_ref.shift(target + 26 - 1)], axis=1)

    if diff:
        drug = drug.diff(target)

    # moving average features
    MA_3_second_name = f'MA-3 {second_category}'
    MA_7_second_name = f'MA-7 {second_category}'
    MA_13_second_name = f'MA-13 {second_category}'
    MA_26_second_name = f'MA-26 {second_category}'
    drug['MA-3'] = drug['t'].rolling(window=3, min_periods=1).mean()
    drug['MA-7'] = drug['t'].rolling(window=7, min_periods=1).mean()
    drug['MA-13'] = drug['t'].rolling(window=13, min_periods=1).mean()
    drug['MA-26'] = drug['t'].rolling(window=26, min_periods=1).mean()
    drug[MA_3_second_name] = drug[t_second_category].rolling(window=3, min_periods=1).mean()
    drug[MA_7_second_name] = drug[t_second_category].rolling(window=7, min_periods=1).mean()
    drug[MA_13_second_name] = drug[t_second_category].rolling(window=13, min_periods=1).mean()
    drug[MA_26_second_name] = drug[t_second_category].rolling(window=26, min_periods=1).mean()

    dates = calendar_features()
    # number of companies bidding at each time
    temp = market_sales[market_sales['Substitution Group Name'] == group]
    bidders = list()
    for time in dates['Time']:
        number_of_bidders = len(temp[(temp['PPP'] > 0) & (temp['Time'] == (time - 1))]['Company Name'].unique())
        bidders.append(number_of_bidders)
    series_bidders = pd.Series(bidders, name="number_of_bidders", dtype='int32')
    drug = pd.concat([drug, series_bidders], axis=1)

    # add time, week and year (week and year are only temporarily information)
    drug.loc[:, 'Time'] = drug.index + 1
    drug = pd.merge(drug, dates[['Time', 'Year', 'Month', 'period_sin', 'period_cos']], how='left', on='Time')

    # remove first observation due to lag
    if diff:
        drug = drug.iloc[2 * target:]
    else:
        drug = drug.iloc[1:]

    return drug


def companies_prices_and_sales_on_drug(
        group: str,
        diff=True) -> DataFrame:
    """
    Returns a DataFrame with each suppliers bid in the substitution group

    :param group: The substitution group
    :param diff: Whether to first-order differencing or not
    """
    temp = market_sales[market_sales['Substitution Group Name'] == group]
    temp = temp[(temp['PPP'] > 0)]
    company_names = temp['Company Name'].unique()

    times = list(time for time in range(1, temp['Time'].max() + 1))

    df_prices = pd.DataFrame()
    df_sales = pd.DataFrame()

    for supplier in company_names:
        temp_company = temp[temp['Company Name'] == supplier]

        company_prices = list()
        company_sales = list()
        for time in times:
            temp_time = temp_company[temp_company['Time'] == time]

            if len(temp_time) > 1:
                temp_time_A = temp_time[temp_time['ABC Price'] == 'A']

                if len(temp_time_A) == 1:
                    company_prices.append(temp_time_A.PPP.values[0])
                    company_sales.append(temp_time_A.Quantity.values[0])
                elif len(temp_time_A) > 1:
                    company_prices.append(temp_time_A.PPP.min())
                    company_sales.append(temp_time_A.Quantity.sum())

                else:
                    temp_time_B = temp_time[temp_time['ABC Price'] == 'B']

                    if len(temp_time_B) == 1:
                        company_prices.append(temp_time_B.PPP.values[0])
                        company_sales.append(temp_time_B.Quantity.values[0])
                    elif len(temp_time_B) > 1:
                        company_prices.append(temp_time_B.PPP.min())
                        company_sales.append(temp_time_B.Quantity.sum())

                    else:
                        temp_time_C = temp_time[temp_time['ABC Price'] == 'C']

                        if len(temp_time_C) == 1:
                            company_prices.append(temp_time_C.PPP.values[0])
                            company_sales.append(temp_time_C.Quantity.values[0])
                        elif len(temp_time_C) > 1:
                            temp_idxmin = temp_time_C['PPP'].idxmin(axis="columns")
                            company_prices.append(temp_time_C.loc[temp_idxmin].PPP)
                            company_sales.append(temp_time_C.loc[temp_idxmin].Quantity)

                        else:
                            temp_time_noprice = temp_time[temp_time['ABC Price'] == 'No Price Classification']

                            if len(temp_time_noprice) == 1:
                                company_prices.append(temp_time_noprice.PPP.values[0])
                                company_sales.append(temp_time_noprice.Quantity[0])
                            elif len(temp_time_noprice) > 1:
                                temp_idxmin = temp_time_noprice['PPP'].idxmin(axis="columns")
                                company_prices.append(temp_time_noprice.loc[temp_idxmin].PPP)
                                company_sales.append(temp_time_noprice.loc[temp_idxmin].Quantity)

            else:
                if len(temp_time) == 0:
                    company_prices.append(0)
                    company_sales.append(0)
                else:
                    company_prices.append(temp_time.PPP.values[0])
                    company_sales.append(temp_time.Quantity.values[0])

        company_price_series = pd.Series(company_prices, name=supplier)
        company_sales_series = pd.Series(company_sales, name=supplier)

        df_prices = pd.concat([df_prices, company_price_series], axis=1)
        df_sales = pd.concat([df_sales, company_sales_series], axis=1)

    df_prices[df_prices < 0] = 0
    df_prices = df_prices + 1
    df_prices = np.log(df_prices)

    df_sales[df_sales < 0] = 0
    df_sales = df_sales + 1
    df_sales = np.log(df_sales)

    col_names = df_prices.columns

    if diff:
        df_prices = df_prices.diff(1)
        df_sales = df_sales.diff(1)

    time_series = pd.Series(times, name='Time')
    df_prices = pd.concat([time_series, df_prices], axis=1)
    df_sales = pd.concat([time_series, df_sales], axis=1)

    # rename columns
    price_names = {key: f"{key}" + "_price" for key in col_names}
    sales_names = {key: f"{key}" + "_sales" for key in col_names}
    df_prices = df_prices.rename(columns=price_names)
    df_sales = df_sales.rename(columns=sales_names)

    # TO CORRECT FOR LAGS, SO THEY CAN BE MERGED ON TIME
    df_prices['Time'] = df_prices['Time'] + 1
    df_sales['Time'] = df_sales['Time'] + 1

    if diff:
        df_prices = df_prices.iloc[1:]
        df_sales = df_sales.iloc[1:]

    return df_prices, df_sales


def generate_gbm_set(
        category: str,
        reference: Dict,
        second_reference: Dict,
        lags: int,
        target: int,
        diff=True) -> DataFrame:
    """
    Build a complete set with respect to the category

    :param category: The category (price or sales)
    :param reference: The series of the category
    :param second_category: The series of the other category
    :param lags: The number of lags
    :param target: The Q-ahead-target
    :param diff: Whether to first-order differencing or not
    """
    df = pd.DataFrame()
    ctr = 1

    for group in reference.keys():
        temp_sub = generate_drug_features(
            category=category,
            reference=reference,
            second_reference=second_reference,
            group=group,
            lags=lags,
            target=target,
            diff=diff
        )

        company_prices, supplier_sales = companies_prices_and_sales_on_drug(
            group=group,
            diff=diff
        )

        # merge competitor's price and quantity and information about returned products
        drug = pd.merge(temp_sub, company_prices, how='left', on='Time')
        drug = pd.merge(drug, supplier_sales, how='left', on='Time')

        # add substitution group
        drug['Substitution Group Name'] = group

        # concat to datafarme
        df = pd.concat([df, drug], axis=0)

        # print updates
        if ctr % 50 == 0:
            print(f"{ctr} substitution groups done...")
        ctr += 1

    # add the features tilskud and opbevaring to the dataframe
    df = pd.merge(df, product_features, how='left', on='Substitution Group Name')

    return df


def generate_nn_set(
        target: str,
        gbm_price_set: DataFrame,
        gbm_sales_set: DataFrame) -> DataFrame:
    """
    Build a complete set with both categories

    :param target: The Q-ahead-target
    :param gbm_price_set: The set with respect to price
    :param gbm_sales_set: The set with respect to sales
    """
    col_category = f"t+{target} sales"
    df = gbm_price_set.copy()
    df[col_category] = gbm_sales_set.iloc[:, :1]
    return df


if __name__ == '__main__':
    args = parser.parse_args()
    le_reimbursement = LabelEncoder()
    le_storage = LabelEncoder()
    with open('../data/clean/reference_price_to_2020.pkl', 'rb') as file:
        reference_price = pickle.load(file)

    with open('../data/clean/total_sales_to_2020.pkl', 'rb') as file:
        total_sales = pickle.load(file)

    market_sales = pd.read_csv(
        '../data/clean/market_sales07-20.csv',
        sep=';',
        dtype={'Substitution Group Name': 'object',
               'Nordic Item No': 'object'})
    product_features = pd.read_csv(
        '../data/clean/tilskud_opbevaring.csv',
        sep=';',
        dtype={'Substitution Group Name': 'object'})
    product_features['tilskud'] = le_reimbursement.fit_transform(product_features['tilskud'])
    product_features['opbevaring'] = le_reimbursement.fit_transform(product_features['opbevaring'])

    df_price = generate_gbm_set(
        category='price',
        reference=reference_price,
        second_reference=total_sales,
        target=args.target,
        lags=args.lags,
        diff=args.diff)
    print('Completed price dataframe..')

    df_sales = generate_gbm_set(
        category='sales',
        reference=total_sales,
        second_reference=reference_price,
        target=args.target,
        lags=args.lags,
        diff=args.diff)
    print('Completed sales dataframe..')

    if args.nn:
        df_nn = generate_nn_set(
            target=args.target,
            gbm_price_set=df_price,
            gbm_sales_set=df_sales)
        df_nn[~df_nn['Year'].isin([2019, 2020])].to_csv(f'../data/train/train_nn_target{args.target}.csv', sep=';', index=False)
        df_nn[df_nn['Year'].isin([2019, 2020])].to_csv(f'../data/test/test_nn_target{args.target}.csv', sep=';', index=False)
        print('Saved NN in-sample and out-of-sample set')

    df_price[~df_price['Year'].isin([2019, 2020])].to_csv(f'../data/train/train_gbm_price_target{args.target}.csv', sep=';', index=False)
    df_price[df_price['Year'].isin([2019, 2020])].to_csv(f'../data/test/test_gbm_price_target{args.target}.csv', sep=';', index=False)
    df_sales[~df_sales['Year'].isin([2019, 2020])].to_csv(f'../data/train/train_gbm_sales_target{args.target}.csv', sep=';', index=False)
    df_sales[df_sales['Year'].isin([2019, 2020])].to_csv(f'../data/test/test_gbm_sales_target{args.target}.csv', sep=';', index=False)
    print('Saved gbm in-sample and out-of-sample set')
