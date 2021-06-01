import pandas as pd
from typing import TypeVar

DataFrame = TypeVar('DataFrame')


def simulate_auction_bid(group: int, model: str):
    """
    Simulate an auction over time for a substitution group

    :param group: The substitution group
    :param model: The choice of model (Simple or LGBM)
    """
    df_group = predictions[predictions['Substitution Group Name'] == group]
    times = df_group['Time'].unique()

    total_sales = 0
    sales = 0
    revenue = 0
    stock = 0
    stock_value = 0

#    model_price = f"Simple price"
    model_price = f"{model} price"
    model_sales = f"{model} sales"
    a_ctr = 0
    b_ctr = 0
    c_ctr = 0
    delivery_failure_ctr = 0
    for time in times:
        df_time = df_group[df_group['Time'] == time]
        true_price = df_time['True price'].values[0]
        true_sales = df_time['True sales'].values[0]

        predicted_price = df_time[model_price].values[0] - 3
        predicted_demand = df_time[model_sales].values[0]

        share_ = 0
        revenue_ = 0
        temp_minimums_takst = 0
        if group in minimums_takst['Substitutionsgruppe'].unique():
            temp_minimums_takst = true_sales / 2

        if true_price > predicted_price:
            if temp_minimums_takst < predicted_demand:
                a_ctr += 1
                if true_sales > predicted_demand:
                    share_ = predicted_demand * 0.73
                    revenue_ = share_ * predicted_price
                else:
                    share_ = true_sales * 0.73
                    revenue_ = share_ * predicted_price

            else:
                share_ = 0
                revenue_ = 0
                delivery_failure_ctr += 1
        else:
            if true_price <= 100:
                B_price = true_price + 5
            elif true_price > 100 and true_price < 400:
                boundry = true_price * 0.15
                B_price = true_price + boundry
            else:
                B_price = true_price + 20

            if predicted_price <= B_price:
                share_ = true_sales * 0.19
                revenue_ = share_ * predicted_price
                b_ctr += 1
            else:
                share_ = true_sales * 0.08
                revenue_ = share_ * predicted_price
                c_ctr += 1

        stock += ((predicted_demand * 0.73) - share_)
        total_sales += true_sales
        sales += share_
        revenue += revenue_
        market_share = sales / total_sales
        stock_value = ((predicted_demand * 0.73) - share_) * true_price
    return sales, revenue, market_share, a_ctr, b_ctr, c_ctr, delivery_failure_ctr, stock, stock_value


def simulate(model: str) -> DataFrame:
    """
    Simulate using the model on the pharmaceutical market

    :param model: The choice of model (Simple or LGBM)
    """
    groups = predictions['Substitution Group Name'].unique()
    sales = list()
    revenue = list()
    market_share = list()
    a_ctr = list()
    b_ctr = list()
    c_ctr = list()
    delivery_failure_ctr = list()
    stock = list()
    stock_value = list()

    for group in groups:
        g_sales, g_revenue, g_market_share, \
            g_win_ctr, g_b_ctr, g_c_ctr, g_delivery_ctr, \
            g_stock, g_stock_value = simulate_auction_bid(group, model)

        sales.append(g_sales)
        revenue.append(g_revenue)
        market_share.append(g_market_share)
        a_ctr.append(g_win_ctr)
        b_ctr.append(g_b_ctr)
        c_ctr.append(g_c_ctr)
        delivery_failure_ctr.append(g_delivery_ctr)
        stock.append(g_stock)
        stock_value.append(g_stock_value)

    method = pd.DataFrame({
        'Group': groups,
        'Sales': sales,
        'Revenue': revenue,
        'Market share': market_share,
        'A prices': a_ctr,
        'B prices': b_ctr,
        'C prices': c_ctr,
        'Delivery failure': delivery_failure_ctr,
        'Stock': stock,
        'Stock value': stock_value
    })

    return method


if __name__ == '__main__':
    predictions = pd.read_csv('../results/predictions.csv', sep=';')
    minimums_takst = pd.read_csv('../data/clean/minimum_supply_groups.csv', sep=';')

    simple_strategy = simulate('Simple')
    network_strategy = simulate('LGBM')

    simple_strategy.to_csv('../results/auction_result_simple.csv', sep=';', index=False)
    network_strategy.to_csv('../results/auction_result_network.csv', sep=';', index=False)
