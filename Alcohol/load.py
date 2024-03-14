from enum import StrEnum
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pandas as pd

from Processing import TimeSeries

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


def aggregate_pivot_joint(df: pd.DataFrame, by_column: Keys, price_column: Keys, quantity_column: Keys) -> TimeSeries:
    pivoted_joint = df.pivot_table(index=pd.Grouper(key=Keys.DATE, freq='1ME'), columns=by_column, values=[price_column, quantity_column],
                                   aggfunc={price_column: 'mean', quantity_column: 'sum'}, dropna=True).sort_index(axis=1)
    df_price = pivoted_joint[price_column]
    df_quantity = pivoted_joint[quantity_column]
    return TimeSeries(
        price=df_price.to_numpy(),
        quantity=df_quantity.to_numpy(),
        header=df_price.columns.values,
        index=df_price.index.to_numpy(dtype=np.datetime64)
    )


def main():
    PATH = "./Caches/"
    import caching
    data_pd = load_alcohol_table().read(29_000_000)
    ts = aggregate_pivot_joint(data_pd, Keys.ITEM, Keys.PRICE, Keys.QUANTITY)
    caching.save(ts, path=PATH, identifier='0')


if __name__ == '__main__':
    main()
