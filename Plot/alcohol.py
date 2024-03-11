import numpy as np
from matplotlib import pyplot as plt

from Alcohol.load import load_alcohol_table, aggregate_pivot_joint, Keys
from Processing.time_series import TimeSeries


def main():
    data_pd = load_alcohol_table().read(28_000_000)
    price, quantity = aggregate_pivot_joint(data_pd, Keys.ITEM, Keys.PRICE, Keys.QUANTITY)

    ts = TimeSeries(price, quantity)
    longest_ts = ts.get_nth_longest(0)
    top_5 = ts.get_n_longest(10)

    #print(longest_ts)
    for element in top_5:
        print(element[1].astype(int))

    for element in top_5:
        print(np.corrcoef(x=element[0], y=element[1]))


if __name__ == '__main__':
    main()
