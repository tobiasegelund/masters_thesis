import pickle
import pandas as pd
import argparse
from typing import Dict, TypeVar

DataFrame = TypeVar('Dataframe')

parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, default=2020)


def extract_reference_price(df: DataFrame) -> Dict:
    """
    Extract each substitution group reference price over time

    :param df: The market sales data from DLI
    """
    price = dict()
    ctr = 1
    active_substitution_groups = df['Substitution Group Name'].unique()
    for group in active_substitution_groups:
        temp = df[df['Substitution Group Name'] == group]

        temp_prices = list()
        for time in range(1, df['Time'].max() + 1):
            temp_time_a = temp[(temp['ABC Price'] == 'A') & (temp['PPP'] > 0) & (temp['Time'] == time) & (temp['Quantity'] > 0)]

            if temp_time_a.shape[0] > 0:
                temp_prices.append(temp_time_a['PPP'].min())

            else:
                temp_time_noprice = temp[(temp['ABC Price'] == 'No Price Classification') &
                                         (temp['PPP'] > 0) & (temp['Time'] == time) & (temp['Quantity'] > 0)]

                if temp_time_noprice.shape[0] > 0:
                    temp_prices.append(temp_time_noprice['PPP'].min())

                else:
                    temp_time_b = temp[(temp['ABC Price'] == 'B') & (temp['PPP'] > 0) &
                                       (temp['Time'] == time) & (temp['Quantity'] > 0)]

                    if temp_time_b.shape[0] > 0:
                        temp_idxmin = temp_time_b['PPP'].idxmin(axis="columns")
                        temp_prices.append(temp_time_b.loc[temp_idxmin].PPP)

                    else:
                        temp_time_c = temp[(temp['ABC Price'] == 'C') &
                                           (temp['PPP'] > 0) & (temp['Time'] == time) & (temp['Quantity'] > 0)]

                        if temp_time_c.shape[0] > 0:
                            temp_idxmin = temp_time_c['PPP'].idxmin(axis="columns")
                            temp_prices.append(temp_time_c.loc[temp_idxmin].PPP)

                        else:
                            temp_time_a = temp[(temp['ABC Price'] == 'A') & (temp['PPP'] > 0) &
                                               (temp['Time'] == time)]

                            if temp_time_a.shape[0] > 0:
                                temp_prices.append(temp_time_a['PPP'].min())

                            else:
                                temp_time_noprice = temp[(temp['ABC Price'] == 'No Price Classification') &
                                                         (temp['PPP'] > 0) & (temp['Time'] == time)]

                                if temp_time_noprice.shape[0] > 0:
                                    temp_prices.append(temp_time_noprice['PPP'].min())

                                else:
                                    temp_time_b = temp[(temp['ABC Price'] == 'B') & (temp['PPP'] > 0) &
                                                       (temp['Time'] == time)]

                                    if temp_time_b.shape[0] > 0:
                                        temp_idxmin = temp_time_b['PPP'].idxmin(axis="columns")
                                        temp_prices.append(temp_time_b.loc[temp_idxmin].PPP)

                                    else:
                                        temp_time_c = temp[(temp['ABC Price'] == 'C') &
                                                           (temp['PPP'] > 0) & (temp['Time'] == time)]

                                        if temp_time_c.shape[0] > 0:
                                            temp_idxmin = temp_time_c['PPP'].idxmin(axis="columns")
                                            temp_prices.append(temp_time_c.loc[temp_idxmin].PPP)

                                        else:
                                            temp_prices.append(0)

        price[group] = temp_prices

        if ctr % 50 == 0:
            print(f"Completed {ctr} substitution groups")
        ctr += 1
    return price


if __name__ == '__main__':
    args = parser.parse_args()
    market_sales = pd.read_csv(
        '../data/clean/market_sales07-20.csv',
        sep=';',
        dtype={'Substitution Group Name': 'object', 'Nordic Item No': 'object'})

    market_sales = market_sales[market_sales['Year'] < (args.year + 1)]
    reference_price = extract_reference_price(df=market_sales)

    with open(f'../data/clean/reference_price_to_{args.year}.pkl', "wb") as file:
        pickle.dump(reference_price, file)
