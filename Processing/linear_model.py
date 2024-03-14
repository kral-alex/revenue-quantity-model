import numpy as np

from .time_series import PriceQuantity


class NotEnoughDataError(ValueError):
    pass


def last_change_slope(pq: PriceQuantity, min_count: int = 1, max_count: int = np.inf) -> float:
    len_left, middle, len_right = find_change_range(pq)
    if min(len_left, len_right) < min_count:
        raise NotEnoughDataError(
            f'Not enough data on first price change.'
            f' Minimum required: {min_count}. Left count: {len_left}. Right count: {len_right}'
        )

    take_count = min(len_left, len_right, max_count)

    d_quantity = np.mean(
        pq.quantity[middle: middle + take_count]
        - pq.quantity[middle - take_count: middle]
    )

    d_price = pq.price[middle] - pq.price[middle - 1]

    return d_quantity / d_price


def last_change_with_time_slope(pq, min_count: int = 2, max_count: int = np.inf) -> (float, float):
    len_left, middle, len_right = find_change_range(pq)
    if min(len_left, len_right) < min_count:
        raise NotEnoughDataError(
            f'Not enough data on first price change. '
            f'Minimum required: {min_count}. Left count: {len_left}. Right count: {len_right}'
        )

    take_count = min(len_left, len_right, max_count)
    half_count = take_count // 2
    d_quantity_thin = np.mean(
        pq.quantity[middle: middle + half_count]
        - pq.quantity[middle - half_count: middle]
    )
    d_quantity_wide = np.mean(
        pq.quantity[middle + half_count: middle + take_count]
        - pq.quantity[middle - take_count: middle - half_count]
    )

    d_quantity_t = 0.5 * d_quantity_wide - d_quantity_thin
    d_quantity_p = 2 * d_quantity_thin - 0.5 * d_quantity_wide

    d_price = pq.price[middle] - pq.price[middle - 1]

    return d_quantity_t / take_count, d_quantity_p / d_price


EPSILON = 1e-9


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


def train_linear_regression(x_train, y_train) -> (np.ndarray[np.dtype[float]], np.ndarray[np.dtype[float]]):
    y_intercept = np.mean(y_train, axis=0)
    coefficients = np.linalg.lstsq(x_train - np.mean(x_train, axis=0), y_train - y_intercept, rcond=None)[0]  # least squares linear regression
    return coefficients, y_intercept
