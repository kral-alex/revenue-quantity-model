import datetime
import warnings
from enum import StrEnum

import numpy
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pandas as pd


class Keys(StrEnum):
    DATE = 'Date'
    STORE = 'Store Number'
    VENDOR = 'Vendor Number'
    ITEM = 'Item Number'
    PRICE = 'State Bottle Retail'
    QUANTITY = 'Bottles Sold'
    VOLUME = 'Volume Sold (Liters)'


using_columns = ['Date', 'Store Number', 'Vendor Number', 'Item Number', 'State Bottle Retail', 'Sale (Dollars)', 'Bottles Sold', 'Volume Sold (Liters)']

PATH = '../Data/Iowa_Liquor_Sales.csv'
print(type(df_reader := pd.read_csv(PATH,
                        #na_filter=True,
                        parse_dates=[Keys.DATE],
                        dtype={Keys.STORE: str, Keys.VENDOR: str, Keys.ITEM: str, Keys.PRICE: float, Keys.QUANTITY: int, Keys.VOLUME: float},
                        usecols=using_columns,
                        chunksize=1000)))

#df2_reader = pandas.read_csv('/Users/alex/Downloads/Iowa_Liquor_Sales.csv', chunksize=1000)

print('head: ', df_reader.read(1).head(0))
df = df_reader.read(1_000_000)
#print(df['Pack'].agg(['mean', 'std', 'min', 'max']))
print(df['Bottles Sold'].agg(['mean', 'std', 'min', 'max']))

print(df.groupby(Keys.ITEM).count().sort_values(by=Keys.DATE, ascending=False).iloc[0:10][Keys.DATE.value])

by_item_month = (df.groupby([Keys.ITEM, pd.Grouper(key=Keys.DATE, freq='1ME')])
                 .agg({Keys.PRICE: 'mean', Keys.QUANTITY: 'sum'})
                 .sort_values(by=Keys.DATE, ascending=True)
                 .sort_values(by=Keys.ITEM, key=lambda c: ([int(r) if r.isdigit() else -1 for r in c]), kind='stable', ascending=True)
                 )

print(by_item_month.head(20))
popular_items = by_item_month[by_item_month[Keys.QUANTITY] > 1000]
print(popular_items.head(20))

print(
    popular_items.groupby(by=Keys.ITEM).size().sort_values(ascending=False).head(20)
)

min_max = by_item_month.groupby(by=Keys.ITEM).agg({Keys.PRICE: ['min', 'max']})#.where(lambda x: x[(Keys.PRICE, 'min')] != x[(Keys.PRICE, 'max')]).notna()
print(min_max)

print(
    min_max[min_max[(Keys.PRICE, 'min')] != min_max[(Keys.PRICE, 'max')]]
)

print(df.groupby([Keys.ITEM, pd.Grouper(key=Keys.DATE, freq='1ME')]).size().sort_values())

print(
    pivoted := df.pivot_table(index=pd.Grouper(key=Keys.DATE, freq='1ME'), columns=Keys.ITEM, values=[Keys.PRICE, Keys.QUANTITY], aggfunc={Keys.PRICE: 'mean', Keys.QUANTITY: 'sum'}, dropna=True).swaplevel(axis=1).sort_index(axis=1)
)
print(
    b := by_item_month.pivot_table(index=Keys.DATE, columns=Keys.ITEM, values=[Keys.PRICE, Keys.QUANTITY], aggfunc={Keys.PRICE: 'mean', Keys.QUANTITY: 'sum'}, dropna=True).swaplevel(axis=1).sort_index(axis=1)
)

assert pivoted.compare(b).empty

print(by_item_month.pivot_table(index=Keys.DATE, columns=Keys.ITEM, values=Keys.PRICE, aggfunc='mean', dropna=True))
print(by_item_month.pivot_table(index=Keys.DATE, columns=Keys.ITEM, values=Keys.QUANTITY, aggfunc='sum', dropna=True))

print(pivoted.index)
date_column = pivoted.index
all_months = pd.date_range(start=date_column.min(), end=date_column.max(), freq='1ME')
print(all_months)
if len(date_column) == len(all_months):
    print((all_months.to_series() == date_column.values).any())
else:
    print('date columns do not match')
