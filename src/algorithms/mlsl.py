import nlopt 
from framework import *
import numpy as np

@optimizer
def run_mlsl(f, x_max, y_max):
    opt = nlopt.opt(nlopt.G_MLSL, 2)
    opt.set_lower_bounds([0, 0])
    opt.set_upper_bounds([x_max, y_max])
    opt.set_min_objective(f)

    local_opt = nlopt.opt(nlopt.LN_NELDERMEAD, 2)
    local_opt.set_ftol_abs(1e-1)
    local_opt.set_xtol_abs(1e-1)

    opt.set_local_optimizer(local_opt)
    opt.set_population(10)
    opt.set_ftol_abs(1e-1)
    opt.set_xtol_abs(1e-1)
    opt.set_maxeval(MAX_FES)

    x, y = opt.optimize([1, 1])
    z = opt.last_optimum_value()

    return {
        'x': np.array([x, y]),
        'z': z,
    }



mlsl = Algorithm(
    'MLSL',
    run_mlsl,
    {},
    3
)