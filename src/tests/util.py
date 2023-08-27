import nevis
import numpy as np
from scipy.optimize import minimize
from framework import Result

def run_sample_opt(rand_seed=None, init_guess=None):
    f = nevis.linear_interpolant()
    x_max, y_max = nevis.dimensions()

    def wrapper(x):
        height = f(*x)
        heights.append(height)
        points.append(x)
        return -height
    
    points = []
    heights = []

    if init_guess is None:
        np.random.seed(rand_seed)
        a = np.random.rand() * x_max
        b = np.random.rand() * y_max
    else: 
        a, b = init_guess

    result = minimize(
        wrapper,
        x0=(a, b),
        bounds=[(0, x_max), (0, y_max)],
        method='Nelder-Mead',
        options={'xatol': 10, 'fatol': 0.2},
    )

    return np.array(points), np.array(heights), -result.fun, result.x


def make_result(rand_seed=None, init_guess=None, info=None):
    points, _, ret_height, ret_point = run_sample_opt(
        rand_seed=rand_seed, init_guess=init_guess)
    return Result(
        ret_point=ret_point, ret_height=ret_height, points=points, info=info)
