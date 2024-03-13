import numpy as np

from Processing.time_series import PriceQuantity

EPSILON = 1e-9


class NotEnoughDataError(ValueError):
    pass


# can be made to take asymmetric mean with weights
def last_change_slope(pq: PriceQuantity, min_count: int = 1, max_count: int = np.inf) -> float:
    change_indices = np.argwhere(np.diff(pq.price) > EPSILON).squeeze()
    d_price = pq.price[-1] - pq.price[change_indices[-1]]
    len_left = change_indices[-1] - change_indices[-2]
    len_right = (len(pq) - 1) - change_indices[-1]
    if min(len_left, len_right) < min_count:
        raise NotEnoughDataError(
            f'Not enough data on first price change. Minimum required: {min_count}. Left count: {len_left}. Right count: {len_right}'
        )
    take_count = min(len_left, len_right, max_count)
    middle = change_indices[-1]
    d_quantity = np.mean(pq.quantity[middle : middle + take_count]) - np.mean(pq.quantity[middle - take_count : middle])
    return d_quantity / d_price


def last_change_with_time_slope(pq, min_count: int = 1, max_count: int = np.inf) -> float:
    pass  # TODO


def find_change_range(pq: PriceQuantity, n: int = 0):
    n += 1
    change_indices = np.argwhere(np.diff(pq.price) > EPSILON).squeeze()
    len_left = change_indices[-n] - change_indices[-(1 + n)]
    len_right = (len(pq) - n) - change_indices[-n]
    middle = change_indices[-n]
    return len_left, middle, len_right


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