import os
import logging

import numpy as np

from Processing import TimeSeries

logger = logging.getLogger(__name__)


class IncompatibleDataShape(ValueError):
    pass


def save(ts: TimeSeries, path: os.path, identifier: str) -> None:
    np.savetxt(os.path.join(path, f"{identifier}_P.csv"), ts.price, delimiter=",")
    np.savetxt(os.path.join(path, f"{identifier}_Q.csv"), ts.quantity, delimiter=",")
    np.savetxt(os.path.join(path, f"{identifier}_H.csv"), ts.header, delimiter=",", fmt="%s")
    np.savetxt(os.path.join(path, f"{identifier}_I.csv"), ts.index, delimiter=",", fmt="%s")


def load(
        dir_path: os.path,
        identifier: str,
        *,                                  # following arguments must be keyword
        path_p_override: os.path = None,
        path_q_override: os.path = None,
        path_h_override: os.path = None,
        path_i_override: os.path = None,
) -> TimeSeries:
    path_p = os.path.join(dir_path, f"{identifier}_P.csv") if path_p_override is None else path_p_override
    path_q = os.path.join(dir_path, f"{identifier}_Q.csv") if path_q_override is None else path_q_override
    path_h = os.path.join(dir_path, f"{identifier}_H.csv") if path_h_override is None else path_h_override
    path_i = os.path.join(dir_path, f"{identifier}_I.csv") if path_i_override is None else path_i_override

    p = np.loadtxt(path_p, delimiter=",")
    q = np.loadtxt(path_q, delimiter=",")

    if p.shape != q.shape:
        raise IncompatibleDataShape(f"Price data shape ({p.shape}) does not equal quantity data shape ({q.shape}).")

    try:
        h = np.loadtxt(path_h, delimiter=",", dtype=str)
        if h.shape[0] != p.shape[1]:
            raise IncompatibleDataShape(f"Header length ({h.shape[0]}) does not equal data column count ({p.shape[1]}).")
    except FileNotFoundError:
        logger.warning("Warning: File with header not found.")
        h = None

    try:
        i = np.loadtxt(path_i, delimiter=",", dtype=np.datetime64)
        if i.shape[0] != p.shape[0]:
            raise IncompatibleDataShape(f"Index length ({i.shape[0]}) does not equal data row count ({p.shape[0]}).")
    except FileNotFoundError:
        i = None
        logger.warning("Warning: File with indices not found.")

    return TimeSeries(p, q, h, i)

