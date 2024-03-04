import warnings

import numpy as np
import numba

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pandas as pd


def longest_consecutive_column(values: np.ndarray) -> np.ndarray:
    #consecutive_indices = np.zeros_like(values)
    # values = values.transpose()
    # print(np.isnan(np.roll(values, 1)).cumsum(axis=1))
    # consecutive_indices = np.where(~np.isnan(values),
    #                                pd.DataFrame(values).groupby(np.isnan(np.roll(values, 1)).cumsum(axis=1)).cumcount() + 1,
    #                                0)
    #condition = np.concatenate([[[False]*values.shape[1]], np.isnan(values)])
    condition = np.concatenate([[False], np.isnan(values)])
    print(condition)
    print(idx := np.where(~condition)[0])
    consecutive_indices = np.diff(idx, axis=0, prepend=False) - 1,
    print(consecutive_indices)
    #return consecutive_indices


condition = np.array([True, True, True, True, False, True, True, False, True])
condition2 = np.array([[True, True, True, True, False, True, True, False, True]]*4)
print(condition2)
# print(
#     np.diff(np.where(np.concatenate(([condition[0]],
#                                      condition[:-1] != condition[1:],
#                                      [True])))[0])
# )
#
# print(np.concatenate(([condition[0]],
#                      condition[:-1] != condition[1:],
#                      [True]))
#       )
#
# print(np.concatenate((condition2[:, [0]],
#                      condition2[:, :-1] != condition2[:, 1:],
#                      [[True]]*condition2.shape[0]), axis=1)
#       )
# print(np.concatenate((condition2[:, [0]],
#                      condition2[:, :-1] != condition2[:, 1:],
#                      [[True]]*condition2.shape[0]), axis=1).nonzero()
#       )


@numba.njit()
def consecutive_pythonic(values: np.ndarray[2, np.dtype[bool]]) -> np.ndarray[2, np.dtype[int]]:
    new = np.zeros_like(values, dtype=np.uint)
    for ix in range(values.shape[0]):
        current_count = 0
        for iy in range(values.shape[1]):
            if values[ix, iy]:
                current_count += 1
                new[ix, iy] = current_count
            else:
                current_count = 0
    return new


consecutive = consecutive_pythonic(condition2)
print(consecutive)

# my_array = np.array(
#     [[1, 2, 1, 1, np.nan, 1, np.nan, 2, 1, np.nan]]
# ).transpose()
# my_array3 = np.array(
#     [1, 2, 1, 1, np.nan, 1, np.nan, 2, 1, np.nan]
# )
# #print(my_array)
# my_array2 = np.array([
#     [1, 2, 1, 1, 2, 1],
#     [1, np.nan, 1, 1, np.nan, 1],
#     [1, 2, 1, np.nan, 1, 2],
#     [1, np.nan, 1, np.nan, 1, np.nan]
# ])
#
#print(np.unique((~np.isnan(my_array)).cumsum()[np.isnan(my_array)], return_counts=True)[1])
#
# try:
#     print(longest_consecutive_column(my_array3))
# except ValueError as e:
#     print(e)
# try:
#     print(longest_consecutive_column(my_array2))
# except ValueError as e:
#     print(e)
