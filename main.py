import sys

from matplotlib import pyplot as plt

from Alcohol.caching import load
from Processing import (
    PriceQuantity,
    last_change_slope as lcs,
    last_change_with_time_slope as lcwts,
    linear_model_slope as lms,
    linear_model_with_time_slope as lmwts
)

data_path = sys.argv[1]
amount = int(sys.argv[2])
min_count = 10

ts = load(dir_path=data_path, identifier="0")

top_n = [ts.get_nth_longest(i) for i in range(amount)]

for i, pq in enumerate(top_n):

    pq = PriceQuantity.skip_demean_quantity(pq, 12)
    print(f'{i}.'
          f'\ncorrelation: {pq.get_correlation()} '
          f'\nlast_change_slope: {lcs(pq, min_count=min_count)} ',
          f'\nlinear_model_with_time_slope: {lcwts(pq, min_count=min_count)} ',
          f'\nlinear_model_slope: {lms(pq)} ',
          f'\nlinear_model_with_time_slope: {lmwts(pq)} ',
          )
    pq.draw_scatter_graph(plt.subplots()[1])

plt.show()

