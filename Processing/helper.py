from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def skip_demean(array: npt.ArrayLike, period: int, offset: int = 0) -> npt.NDArray:
    condition = ~(np.arange(offset, len(array) + offset) % period).astype(bool)
    mean = np.mean(array, where=condition)
    return np.where(condition, array - mean, array)


@dataclass
class PriceQuantity(frozen=True):
    price: np.ndarray[2, np.dtype[float]]
    quantity: np.ndarray[2, np.dtype[np.uint]]
    header: np.ndarray[1, np.dtype[str]]
    index: np.ndarray[1, np.dtype[np.datetime64]]

    def __len__(self):
        return len(self.price)

    def __getitem__(self, val):
        return PriceQuantity(self.price[val], self.quantity[val])

    @classmethod
    def skip_demean_quantity(cls, pq: PriceQuantity, period: int) -> PriceQuantity:
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
    def shift_price(cls, pq: PriceQuantity, shift_amount: int) -> PriceQuantity:
        new_pq = cls(
            price=np.roll(pq.price, shift_amount),
            quantity=pq.quantity,
            index=pq.index,
            header=pq.header
        )
        if shift_amount >= 0:
            return new_pq[shift_amount:]
        return new_pq[:shift_amount]


# def test():
    # array = np.array([1, 2, 1, 4, 1, 6, 1, 4])
    # print(skip_demean(array, 2))
    # print(skip_demean(array, 2, 1))
    # print(skip_demean_quantity(array, 2))


if __name__ == '__main__':
    pass