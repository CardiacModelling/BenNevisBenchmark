from math import inf
from scipy.optimize import dual_annealing, minimize
import nevis
import time
import os
import pickle
import numpy as np
from pprint import pprint

SUCCESS_HEIGHT = 1307
MAX_FES = 50000
TRIAL_N = 25
RANDOM_ITER = 20
EPS = 1e-2

        # maxiter=1000, 
        # initial_temp=5230, 
        # restart_temp_ratio=2e-5, 
        # visit=2.62, 
        # accept=-5.0, 
        # maxfun=1e7

def run_dual_annealing(**kwargs):
    f = nevis.linear_interpolant()
    # points = []
    function_values = []
    def wrapper(u):
        x, y = u
        # points.append((x, y))
        z = f(x, y)
        function_values.append(z)
        return -z

    x_max, y_max = nevis.dimensions()
    ret = dual_annealing(
        wrapper, 
        bounds=[(0, x_max), (0, y_max)],
        maxfun=MAX_FES,
        **kwargs
    )

    x, y = ret.x
    z = -ret.fun
    print(ret.message)
    nevis.print_result(x, y, z)
    return function_values


def classify_run(function_values):
    for i, f in enumerate(function_values):
        if f > SUCCESS_HEIGHT:
            return i + 1
    
    return -1


def compute_perfomance(**kwargs):
    success_cnt = 0
    success_fe_sum = 0

    for _ in range(TRIAL_N):
        function_values = run_dual_annealing(**kwargs)
        res = classify_run(function_values)
        if res > 0:
            success_cnt += 1
            success_fe_sum += res
    
    if success_cnt == 0:
        return inf
    
    return (success_fe_sum / success_cnt) * (TRIAL_N / success_cnt) 



def random_search(parameter_space: dict):
    best_performance = inf
    best_params = {}
    for _ in range(RANDOM_ITER):
        params = {}
        for key, value in parameter_space.items():
            params[key] = np.random.choice(value)
        pprint(params)
        performance = compute_perfomance(**params)
        print(performance)
        print()
        if performance < best_performance:
            best_performance, best_params = performance, params
    
    return best_params, best_performance


best_params, best_performance = random_search({
    'maxiter': [1000, 1500, 2000],
    'initial_temp': np.linspace(1e3, 5e4, 1000),
    'restart_temp_ratio': np.logspace(-5, -1, 100),
    # 'visit': np.linspace(1 + EPS, 3, 1000),
    # 'accept': np.logspace(-5, -1e-4, 1000),
})

print(best_params, best_performance)
