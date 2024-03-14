import numpy as np
import numba

from .price_quantity import PriceQuantity


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
