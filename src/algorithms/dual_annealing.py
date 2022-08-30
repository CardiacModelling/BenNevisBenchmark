from framework import *
import scipy.optimize
import numpy as np

@optimizer
def run_dual_annealing(f, x_max, y_max, **kwargs):
    ret = scipy.optimize.dual_annealing(
        f, 
        bounds=[(0, x_max), (0, y_max)],
        maxfun=MAX_FES,
        **kwargs
    )

    return {
        'x': ret.x,
        'z': ret.fun,
        'ret_obj': ret,
        'message': ret.message
    }

dual_annealing = Algorithm(
    'Dual Annealing', 
    run_dual_annealing,
    {
        # 'maxiter': np.arange(1500, 3000, 100),
        'maxiter': [2000],
        'initial_temp': np.linspace(2e4, 4e4, 1000),
        'restart_temp_ratio': np.logspace(-5, -3, 100),
        # 'visit': np.linspace(1 + EPS, 3, 1000),
        # 'accept': np.logspace(-5, -1e-4, 1000),
    },
    version=1
)