import numpy as np
import numpy.typing as npt

from Processing.time_series import PriceQuantity


def skip_demean(array: npt.ArrayLike, period: int, offset: int = 0) -> npt.NDArray:
    condition = ~(np.arange(offset, len(array) + offset) % period).astype(bool)
    mean = np.mean(array, where=condition)
    return np.where(condition, array - mean, array)


def skip_demean_quantity(pq: PriceQuantity, period: int) -> PriceQuantity:
    new_quantity = pq.quantity
    for i_offset in range(period):
        new_quantity = skip_demean(new_quantity, period, i_offset)
    return PriceQuantity(price=pq.price, quantity=new_quantity)


def shift_price(pq: PriceQuantity, shift_amount: int) -> PriceQuantity:
    new_price = np.roll(pq.price, shift_amount)
    if shift_amount >= 0:
        return PriceQuantity(price=new_price[shift_amount:], quantity=pq.quantity[shift_amount:])
    return PriceQuantity(price=new_price[:shift_amount], quantity=pq.quantity[:shift_amount])


# def test():
    # array = np.array([1, 2, 1, 4, 1, 6, 1, 4])
    # print(skip_demean(array, 2))
    # print(skip_demean(array, 2, 1))
    # print(skip_demean_quantity(array, 2))


if __name__ == '__main__':
    pass