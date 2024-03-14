from dataclasses import dataclass


import numpy.typing as npt
import numpy as np


@dataclass(frozen=True)
class PriceQuantity:
    price: np.ndarray[2, np.dtype[float]]
    quantity: np.ndarray[2, np.dtype[np.uint]]
    header: np.ndarray[1, np.dtype[str]]
    index: np.ndarray[1, np.dtype[np.datetime64]]

    def __len__(self):
        return len(self.price)

    def __getitem__(self, val):
        if isinstance(val, slice):
            i_slice = val
            h_slice = slice(None)
        elif isinstance(val, tuple) and len(val) == 2:
            h_slice = val[1]
            i_slice = val[0]
        else:
            raise IndexError(f'Index {val} out of range for shape {self.price.shape}')
        return PriceQuantity(
            price=self.price[val],
            quantity=self.quantity[val],
            header=self.header[h_slice] if self.header is not None else None,
            index=self.index[i_slice] if self.index is not None else None
        )

    @classmethod
    def skip_demean_quantity(cls, pq, period: int):
        new_quantity = pq.quantity
        for i_offset in range(period):
            new_quantity = skip_demean(new_quantity, period, i_offset)
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


def skip_demean(array: npt.ArrayLike, period: int, offset: int = 0) -> npt.NDArray:
    condition = ~(np.arange(offset, len(array) + offset) % period).astype(bool)
    mean = np.mean(array, where=condition)
    return np.where(condition, array - mean, array)

