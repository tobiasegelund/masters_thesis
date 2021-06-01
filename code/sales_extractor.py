import pickle
import pandas as pd
import argparse
from typing import Dict, TypeVar

DataFrame = TypeVar('Dataframe')

parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, default=2020)


def extract_total_price(df: DataFrame) -> Dict:
    """
    Extract each substitution group total sales over time

    :param df: The market sales data from DLI
    """
    active_substitution_groups = df['Substitution Group Name'].unique()
    sales = dict()
    ctr = 1

    for group in active_substitution_groups:
        temp = df[df['Substitution Group Name'] == group]

        temp_sales = list()
        for time in range(1, df['Time'].max() + 1):
            temp_time = temp[(temp['PPP'] > 0) & (temp['Time'] == time) & (temp['Quantity'] > 0)]

            if temp_time.shape[0] > 0:
                temp_sales.append(temp_time['Quantity'].sum())
            else:
                temp_sales.append(0)

            sales[group] = temp_sales

        if ctr % 50 == 0:
            print(f"Completed {ctr} substitution groups")

        ctr += 1

    return sales


if __name__ == '__main__':
    args = parser.parse_args()
    market_sales = pd.read_csv(
        '../data/clean/market_sales07-20.csv',
        sep=';',
        dtype={'Substitution Group Name': 'object',
               'Nordic Item No': 'object'})

    market_sales = market_sales[market_sales['Year'] <= args.year]
    total_sales = extract_total_price(df=market_sales)

    with open(f'../data/clean/total_sales_to_{args.year}.pkl', "wb") as file:
        pickle.dump(total_sales, file)
