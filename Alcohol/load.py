from enum import StrEnum
import warnings

import numpy as np
import numba

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pandas as pd

PATH = '../Data/Iowa_Liquor_Sales.csv'
use_columns = ['Date', 'Store Number', 'Vendor Number', 'Item Number', 'State Bottle Retail', 'Sale (Dollars)', 'Bottles Sold',
               'Volume Sold (Liters)']


class Keys(StrEnum):
    DATE = 'Date'
    STORE = 'Store Number'
    VENDOR = 'Vendor Number'
    ITEM = 'Item Number'
    PRICE = 'State Bottle Retail'
    QUANTITY = 'Bottles Sold'
    VOLUME = 'Volume Sold (Liters)'


def load_alcohol_table(chunksize: int = 1000) -> pd.io.parsers.readers.TextFileReader:
    return pd.read_csv(PATH,
                       # na_filter=True,
                       parse_dates=[Keys.DATE],
                       dtype={Keys.STORE: str, Keys.VENDOR: str, Keys.ITEM: str, Keys.PRICE: float, Keys.QUANTITY: int, Keys.VOLUME: float},
                       usecols=use_columns,
                       chunksize=chunksize)


def aggregate_pivot_joint(df: pd.DataFrame, by_column: Keys, price_column: Keys, quantity_column: Keys) -> (np.ndarray, np.ndarray):
    pivoted_joint = df.pivot_table(index=pd.Grouper(key=Keys.DATE, freq='1ME'), columns=by_column, values=[price_column, quantity_column],
                                   aggfunc={price_column: 'mean', quantity_column: 'sum'}, dropna=True).sort_index(axis=1)
    df_price = pivoted_joint[price_column]
    df_quantity = pivoted_joint[quantity_column]
    return df_price.to_numpy(), df_quantity.to_numpy()


#
# pivoting two times slower than once and splitting
# def aggregate_pivot_price(df: pd.DataFrame, by_column: Keys, price_column: Keys) -> np.ndarray:
#     return df.pivot_table(index=pd.Grouper(key=Keys.DATE, freq='1ME'), columns=by_column, values=price_column,
#                           aggfunc='mean', dropna=True).sort_index(axis=1).to_numpy()
#
#
# def aggregate_pivot_quantity(df: pd.DataFrame, by_column: Keys, quantity_column: Keys) -> np.ndarray:
#     return df.pivot_table(index=pd.Grouper(key=Keys.DATE, freq='1ME'), columns=by_column, values=quantity_column,
#                           aggfunc='sum', dropna=True).sort_index(axis=1).to_numpy()
#

@numba.njit()
def consecutive_pythonic(values: np.ndarray[2, np.dtype[bool]]) -> np.ndarray[2, np.dtype[int]]:
    new = np.zeros_like(values, dtype=np.uint)
    for ix in range(values.shape[0]):
        current_count = 0
        for iy in range(values.shape[1]):
            if values[ix, iy] == values[ix, iy]:
                current_count += 1
                new[ix, iy] = current_count
            else:
                current_count = 0
    return new


def test():
    df_reader = load_alcohol_table()
    df = df_reader.read(5_000_000)
    print(df[Keys.DATE].max(), df[Keys.DATE].min())
    price_item, quantity_item = aggregate_pivot_joint(df, Keys.ITEM, Keys.PRICE, Keys.QUANTITY)
    print(price_item)
    print(quantity_item)
    print(price_item.shape)
    # data between Jan 2012 and Jan 2024 including, so 145 months (should work with > 10 mil input rows)
    # assert price_item.shape[0] == 145

    consecutive = consecutive_pythonic(price_item.transpose()).transpose()
    print(consecutive)
    longest_in_column: np.ndarray = consecutive.argmax(axis=0)
    print(np.sort(longest_in_column)[-9:])


if __name__ == '__main__':
    test()

    # import time
    # start = time.perf_counter()
    # aggregate_pivot_joint(df, Keys.ITEM, Keys.PRICE, Keys.QUANTITY)
    # time1 = time.perf_counter()
    # aggregate_pivot_price(df, Keys.ITEM, Keys.PRICE)
    # aggregate_pivot_quantity(df, Keys.ITEM, Keys.QUANTITY)
    # time2 = time.perf_counter()
    # print(time1 - start)
    # print(time2 - time1)
