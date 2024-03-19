import logging
from enum import StrEnum

import numpy as np

from .time_series import PriceQuantity


class NoPriceChangeError(ValueError):
    pass


logger = logging.getLogger(__name__)

EPSILON = 1e-9


class ModelPQ:
    def __init__(self, pq: PriceQuantity, min_count: int = 1, max_count: int = np.inf):
        self.pq: PriceQuantity = pq
        self.middles: np.ndarray[1, np.dtype[int]] = np.argwhere(abs(np.diff(self.pq.price)) > EPSILON).squeeze(axis=1)
        self.slices = self.get_pq_slices(min_count, max_count)
        self.correlations = [pq_slice.get_correlation() for pq_slice, _ in self.slices]
        self.results = []

    @staticmethod
    def get_range(edge_indices, index: int) -> (int, int, int):
        return edge_indices[index], edge_indices[index + 1], edge_indices[index + 2]

    def get_pq_slices(self, min_count: int, max_count: int) -> list[(PriceQuantity, int)]:
        pq_slices = []
        edge_indices = np.concatenate(([0], self.middles + 1, [len(self.pq) - 1]))
        for i in range(len(self.middles)):
            len_left, middle, len_right = self.get_range(edge_indices, i)
            if min(middle - len_left, len_right - middle) > min_count:
                take_count = min(middle - len_left, len_right - middle, max_count)
                pq_slices.append((self.pq[middle - take_count: middle + take_count], take_count))

        return pq_slices

    def run_models(self):
        for pq_slice, middle in self.slices:
            raw_res = {
                  Models.LastChange: last_change_slope(pq_slice, middle),
                  Models.LastChangeWT: last_change_with_time_slope(pq_slice, middle),
                  Models.Lin: linear_model_slope(pq_slice),
                  Models.LinWT: linear_model_with_time_slope(pq_slice),
                }
            p2 = np.mean(pq_slice[middle:].price)
            q2 = np.mean(pq_slice[middle:].quantity)
            self.results.append(
                {
                    "price": {
                      str(Models.LastChange): raw_res[Models.LastChange],
                      str(Models.LastChangeWT): raw_res[Models.LastChangeWT][1],
                      str(Models.Lin): raw_res[Models.Lin],
                      str(Models.LinWT): raw_res[Models.LinWT][1],
                    },
                    "PED": {
                        str(Models.LastChange): calculate_PED(p2, q2, raw_res[Models.LastChange]),
                        str(Models.LastChangeWT): calculate_PED(p2, q2, raw_res[Models.LastChangeWT][1]),
                        str(Models.Lin): calculate_PED(p2, q2, raw_res[Models.Lin]),
                        str(Models.LinWT): calculate_PED(p2, q2, raw_res[Models.LinWT][1]),
                    },
                    "time": {
                        str(Models.LastChange): 0,
                        str(Models.LastChangeWT): raw_res[Models.LastChangeWT][0],
                        str(Models.Lin): 0,
                        str(Models.LinWT): raw_res[Models.LinWT][0],
                    }
                }
            )
        return self.results


def calculate_PED(price, quantity, dq_by_dp):
    return dq_by_dp * price / quantity


class Models(StrEnum):
    LastChange = 'last_change_slope'
    LastChangeWT = 'last_change_with_time_slope'
    Lin = 'linear_model_slope'
    LinWT = 'linear_model_with_time_slope'


def last_change_slope(pq: PriceQuantity, middle: int) -> float:

    if abs(2 * middle - len(pq)) > 2:
        logger.warning(f'Price change index provided is not in the middle for item {pq.header}')

    d_quantity = np.mean(
        pq.quantity[middle:]
        - pq.quantity[:middle]
    )

    d_price = pq.price[-1] - pq.price[0]

    return d_quantity / d_price


def last_change_with_time_slope(pq,  middle: int) -> (float, float):

    if abs(2 * middle - len(pq)) > 1:
        logger.warning(f'Price change index provided is not in the middle for item {pq.header}')

    half_count = len(pq) // 4
    d_quantity_thin = np.mean(
        pq.quantity[middle: middle + half_count]
        - pq.quantity[middle - half_count: middle]
    )
    d_quantity_wide = np.mean(
        pq.quantity[middle + half_count:]
        - pq.quantity[: middle - half_count]
    )

    d_quantity_t = 0.5 * d_quantity_wide - d_quantity_thin
    d_quantity_p = 2 * d_quantity_thin - 0.5 * d_quantity_wide

    d_price = pq.price[-1] - pq.price[0]

    return d_quantity_t / (2 * half_count), d_quantity_p / d_price


def linear_model_slope(pq: PriceQuantity, max_count: int = None) -> float:
    if max_count:
        pq = pq[-max_count:]

    return float(
        train_linear_regression(
            np.expand_dims(pq.price, axis=1),
            pq.quantity
        )[0]
    )


def linear_model_with_time_slope(pq: PriceQuantity, max_count: int = None) -> (float, float):
    if max_count:
        pq = pq[-max_count:]

    x_train = np.concatenate(
        (
            np.expand_dims(np.arange(0, len(pq.price)), axis=1),
            np.expand_dims(pq.price, axis=1)
        ),
        axis=1
    )
    return tuple(train_linear_regression(x_train, pq.quantity)[0])


def find_change_range(pq: PriceQuantity, n: int = -1) -> (float, float, float):
    change_indices = np.argwhere(abs(np.diff(pq.price)) > EPSILON).squeeze(axis=1)
    if not len(change_indices):
        raise ValueError('No value changes found')

    if n < 0:
        n = len(change_indices) + n

    if len(change_indices) <= n:
        raise ValueError('Index out of range for value changes')

    edge_indices = np.concatenate(([0], change_indices + 1, [len(pq) - 1]))

    i_left = edge_indices[n]

    i_middle = edge_indices[n + 1]

    i_right = edge_indices[n + 2]

    return (i_middle - i_left,
            i_middle,
            i_right - i_middle)


def train_linear_regression(x_train, y_train) -> (np.ndarray[np.dtype[float]], np.ndarray[np.dtype[float]]):
    y_intercept = np.mean(y_train, axis=0)
    coefficients = np.linalg.lstsq(x_train - np.mean(x_train, axis=0), y_train - y_intercept, rcond=None)[0]  # least squares linear regression
    return coefficients, y_intercept
