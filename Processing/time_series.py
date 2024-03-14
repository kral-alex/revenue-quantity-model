import numpy as np
import numba

from .price_quantity import PriceQuantity


class TimeSeries:
    def __init__(self,
                 price: np.ndarray[2, np.dtype[float]],
                 quantity: np.ndarray[2, np.dtype[np.uint]],
                 header: np.ndarray[1, np.dtype[str]] | None,
                 index: np.ndarray[1, np.dtype[np.datetime64]] | None
                 ):
        self.price: np.ndarray[2, np.dtype[float]] = price
        self.quantity: np.ndarray[2, np.dtype[np.uint]] = quantity
        self.header: np.ndarray[1, np.dtype[str]] | None = header
        self.index: np.ndarray[1, np.dtype[np.datetime64]] | None = index

        consecutive = TimeSeries.consecutive_axis1(self.price)
        self._consecutive_ranges = self.get_consecutive_ranges(consecutive)

    def __getitem__(self, val):
        if isinstance(val, slice):
            return TimeSeries(
                price=self.price[val],
                quantity=self.quantity[val],
                header=self.header,
                index=self.index[val]
            )
        if isinstance(val, tuple) and len(val) == 2:
            r_slice = val[0]
            c_slice = val[1]
            return PriceQuantity(
                price=self.price[val],
                quantity=self.quantity[val],
                header=self.header[c_slice] if self.header is not None else None,
                index=self.index[r_slice] if self.index is not None else None
            )
        else:
            raise IndexError(f'Index {val} out of range for shape {self.price.shape} with {len(self.price.shape)} dimensions')

    def get_nth_longest(self, n) -> PriceQuantity:
        row = self._consecutive_ranges[-1 - n]
        return self[
               row[1]:row[2],
               row[0],
            ]

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
