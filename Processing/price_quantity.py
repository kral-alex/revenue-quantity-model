from dataclasses import dataclass


import numpy.typing as npt
import numpy as np

import matplotlib.pyplot as plt


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
    def skip_demean_quantity(cls, pq, period: int):
        new_quantity = pq.quantity
        for i_offset in range(period):
            new_quantity = PriceQuantity.skip_demean(new_quantity, period, i_offset)
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

    def draw_scatter_graph(self, ax1, *, label=None):
        if label is None:
            label = f'Price Quantity {self.header}'
        ax1.set_title(label)
        ax1.set_xlabel("month #")

        ax1.plot(self.price, marker='o', linestyle='none')
        ax2 = ax1.twinx()
        ax2.plot(self.quantity, marker='o', linestyle='none', color='orange')

        ax1.set_ylabel("price [$]")
        ax2.set_ylabel("quantity")
        return ax1, ax2

    @staticmethod
    def skip_demean(array: npt.ArrayLike, period: int, offset: int = 0) -> npt.NDArray:
        condition = ~(np.arange(offset, len(array) + offset) % period).astype(bool)
        mean = np.mean(array, where=condition)
        return np.where(condition, array - mean, array)

