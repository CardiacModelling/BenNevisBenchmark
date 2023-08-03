from framework import optimizer, Algorithm
from scipy.optimize import minimize
import numpy as np


@optimizer
def run_multistart(f, x_max, y_max, maxiter):
    x_best = (0, 0)
    f_x_best = 0

    for _ in range(maxiter):
        x = np.random.uniform([0, 0], [x_max, y_max])

        local_ret = minimize(
            f,
            x,
            bounds=[(0, x_max), (0, y_max)],
            method='Nelder-Mead',
            options={'fatol': 1e-1}
        )
        if local_ret.success and local_ret.fun < f_x_best:
            x_best, f_x_best = local_ret.x, local_ret.fun

    return {
        'x': x_best,
        'z': f_x_best,
    }


multistart = Algorithm(
    'Multistart',
    run_multistart,
    {
        'maxiter': [400],
    },
    version=1,
)
