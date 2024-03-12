import numpy as np

from Processing.time_series import PriceQuantity

PATH = "./Caches/"


def save(pq: PriceQuantity, identifier: str) -> None:
    np.savetxt(f"{PATH}{identifier}_P.csv", pq.price, delimiter=",")
    np.savetxt(f"{PATH}{identifier}_Q.csv", pq.quantity, delimiter=",")


def load(path_p=None, path_q=None) -> PriceQuantity:
    p = np.loadtxt(path_p, delimiter=",")
    q = np.loadtxt(path_q, delimiter=",")

    return PriceQuantity(p, q)

