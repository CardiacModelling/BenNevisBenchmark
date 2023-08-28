import nevis
import numpy as np
from scipy.optimize import minimize
from framework import Result, MAX_FES
from framework import optimizer
import scipy.optimize
import nlopt

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

@optimizer
def run_dual_annealing(
    f, 
    x_max, 
    y_max, 
    rand_seed, 
    init_guess, 
    **params
):
    params['maxiter'] = int(params['maxiter'])
    ret = scipy.optimize.dual_annealing(
        f,
        bounds=[(0, x_max), (0, y_max)],
        maxfun=MAX_FES,
        seed=rand_seed,
        x0=init_guess,
        minimizer_kwargs={
            'method': 'Nelder-Mead',
            'options': {'xatol': 10, 'fatol': 0.2},
        },
        **params
    )
    return {
        'x': ret.x,
        'z': ret.fun,
        'message': str(ret.message),
    }

@optimizer
def run_mlsl(
    f, 
    x_max, 
    y_max, 
    rand_seed, 
    init_guess, 
    population
):
    opt = nlopt.opt(nlopt.G_MLSL, 2)
    nlopt.srand(seed=rand_seed)
    opt.set_lower_bounds([0, 0])
    opt.set_upper_bounds([x_max, y_max])
    opt.set_min_objective(f)

    local_opt = nlopt.opt(nlopt.LN_NELDERMEAD, 2)
    local_opt.set_ftol_abs(0.2)
    local_opt.set_xtol_abs(10)

    opt.set_local_optimizer(local_opt)
    opt.set_population(int(population))
    opt.set_ftol_abs(0.2)
    opt.set_xtol_abs(10)
    opt.set_maxeval(20_000)

    x0, y0 = init_guess
    x, y = opt.optimize([x0, y0])
    z = opt.last_optimum_value()

    return {
        'x': np.array([x, y]),
        'z': z,
    }