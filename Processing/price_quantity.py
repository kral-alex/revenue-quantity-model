from dataclasses import dataclass


import numpy.typing as npt
import numpy as np

import matplotlib.pyplot
import pandas


@dataclass(frozen=True)
class PriceQuantity:
    price: np.ndarray[1, np.dtype[float]]
    quantity: np.ndarray[1, np.dtype[np.uint]]
    header: str
    index: np.ndarray[1, np.dtype[np.datetime64]]

    def __len__(self):
        return len(self.price)

    def __getitem__(self, val):
        if not isinstance(val, slice):
            raise IndexError(f'Index {val} out of range for shape {self.price.shape} with {len(self.price.shape)} dimensions')
        return PriceQuantity(
            price=self.price[val],
            quantity=self.quantity[val],
            header=self.header,
            index=self.index[val] if self.index is not None else None
        )

    def get_correlation(self):
        return np.corrcoef(self.price, y=self.quantity)[0, 1]

    @classmethod
    def skip_demean_quantity(cls, pq, period: int, add_mean: bool = False):
        new_quantity = pq.quantity
        for i_offset in range(period):
            new_quantity = PriceQuantity.skip_demean(new_quantity, period, i_offset)
        if add_mean:
            new_quantity += np.mean(pq.quantity)
        return cls(
            price=pq.price,
            quantity=new_quantity,
            index=pq.index,
            header=pq.header
        )

    @classmethod
    def shift_price(cls, pq, shift_amount: int):
        new_pq = cls(
            price=np.roll(pq.price, shift_amount),
            quantity=pq.quantity,
            index=pq.index,
            header=pq.header
        )
        if shift_amount >= 0:
            return new_pq[shift_amount:]
        return new_pq[:shift_amount]

    @classmethod
    def bin_price_absolute(cls, pq, bin_size: float):
        return cls(
            price=np.floor(pq.price / bin_size) * bin_size,
            quantity=pq.quantity,
            index=pq.index,
            header=pq.header
        )

    def draw_time_series_scatter_graph(self, ax1: matplotlib.pyplot.Axes, *, label: str = None, dates: bool = True) -> (
            matplotlib.pyplot.Axes, matplotlib.pyplot.Axes
    ):
        if label is None:
            label = f'Price Quantity timeseries {self.header}'
        ax1.set_title(label)
        if dates:
            ax1.set_xlabel("Month")
            price = (self.index, self.price)
            quantity = (self.index, self.quantity)
        else:
            ax1.set_xlabel("Month #")
            price = (self.price,)
            quantity = (self.quantity,)

        ax1.plot(*price, marker='o', linestyle='none')
        ax2 = ax1.twinx()
        ax2.plot(*quantity, marker='o', linestyle='none', color='orange')

        ax1.set_ylabel("price [$]")
        ax2.set_ylabel("quantity")
        return ax1, ax2

    def draw_scatter_graph(self, ax: matplotlib.pyplot.Axes,  *, label: str = None, economic_axis=False, revenue=False,
                           color: str = None) -> matplotlib.pyplot.Axes:
        if label is None:
            label = f'Price Quantity {self.header}'
        ax.set_title(label)

        if economic_axis:
            x = self.quantity
            y = self.price if not revenue else self.quantity * self.price
            x_label = "quantity"
            y_label = "price [$]" if not revenue else "revenue [$]"
        else:
            x = self.price
            y = self.quantity if not revenue else self.quantity * self.price
            x_label = "price [$]"
            y_label = "quantity" if not revenue else "revenue [$]"

        ax.plot(x, y, marker='o', linestyle='none', color=color)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        return ax

    def draw_std_graph(self, ax: matplotlib.pyplot.Axes,  *, label: str = None, economic_axis=False, revenue=False,
                       color: str = None) -> matplotlib.pyplot.Axes:
        if label is None:
            label = f'Price Quantity {self.header}'
        ax.set_title(label)
        pd = pandas.DataFrame(data=[self.price, self.quantity]).transpose()
        pd = pd.groupby(pd[0].round(3))
        pd_mean = pd.agg(func='mean')
        pd_std = pd.agg(func='std')

        if economic_axis:
            x = pd_mean[1]
            y = pd_mean[0] if not revenue else pd_mean[0] * pd_mean[1]
            x_error = pd_std[1] if not revenue else pd_mean[0] * pd_std[1]
            y_error = None
            x_label = "quantity"
            y_label = "price [$]" if not revenue else "revenue [$]"
        else:
            x = pd_mean[0]
            y = pd_mean[1] if not revenue else pd_mean[1] * pd_mean[0]
            x_error = None
            y_error = pd_std[1] if not revenue else pd_mean[0] * pd_std[1]
            x_label = "price [$]"
            y_label = "quantity" if not revenue else "revenue [$]"

        ax.errorbar(x, y, xerr=x_error, yerr=y_error, color=color, capsize=6, marker='o', linestyle='none')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        return ax

    @staticmethod
    def skip_demean(array: npt.ArrayLike, period: int, offset: int = 0) -> npt.NDArray:
        condition = ~(np.arange(offset, len(array) + offset) % period).astype(bool)
        mean = np.mean(array, where=condition)
        return np.where(condition, array - mean, array)

