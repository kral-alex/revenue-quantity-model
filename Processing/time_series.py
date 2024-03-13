from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import numba


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


class TimeSeries:
    def __init__(self, pq: PriceQuantity):
        self.pq = pq
        consecutive = TimeSeries.consecutive_axis1(self.pq.price)  # .astype(dtype=np.bool_))
        self._consecutive_ranges = self.get_consecutive_ranges(consecutive)

    def get_nth_longest(self, n) -> PriceQuantity:
        row = self._consecutive_ranges[-1 - n]
        return self.pq[
               row[1]:row[2],
               row[0],
            ]

    def get_n_longest(self, n) -> list[PriceQuantity]:
        longest = []
        for i in range(n):
            longest.append(self.get_nth_longest(i))
        return longest

    @staticmethod
    @numba.njit()
    def consecutive_axis1(values: np.ndarray[2, np.dtype[float]]) -> np.ndarray[2, np.dtype[np.uint]]:
        new = np.zeros_like(values, dtype=np.uint)
        for iy in range(values.shape[1]):
            current_count = 0
            for ix in range(values.shape[0]):
                if values[ix, iy] == values[ix, iy]:
                    current_count += 1
                    new[ix, iy] = current_count
                else:
                    current_count = 0
        return new

    @staticmethod
    def get_consecutive_ranges(consecutive) -> np.ndarray[2, np.dtype[np.uint]]:
        longest_in_column_indices: np.ndarray = consecutive.argmax(axis=0).astype(np.uint)
        longest_in_column = consecutive[
            longest_in_column_indices,
            np.arange(0, longest_in_column_indices.shape[0], dtype=np.uint)
        ]
        sort_by = longest_in_column.argsort()
        return np.concatenate(
            (
                np.expand_dims(np.arange(0, longest_in_column_indices.shape[0], dtype=np.uint), axis=1),
                np.expand_dims((longest_in_column_indices - longest_in_column + 1).astype(np.uint), axis=1),
                np.expand_dims(longest_in_column_indices, axis=1),
                np.expand_dims(longest_in_column, axis=1)
            ),
            axis=1
        )[sort_by, :]


def skip_demean(array: npt.ArrayLike, period: int, offset: int = 0) -> npt.NDArray:
    condition = ~(np.arange(offset, len(array) + offset) % period).astype(bool)
    mean = np.mean(array, where=condition)
    return np.where(condition, array - mean, array)

