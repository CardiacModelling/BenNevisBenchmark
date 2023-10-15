import nlopt
from framework import optimizer, Algorithm, XTOL, FTOL, SUCCESS_HEIGHT
import numpy as np


@optimizer
def run_nelder_mead(
    f,
    x_max,
    y_max,
    rand_seed,
    init_guess,
    trial,
    get_budget,
):
    nlopt.srand(seed=rand_seed)
    local_opt = nlopt.opt(nlopt.LN_NELDERMEAD, 2)
    local_opt.set_ftol_abs(FTOL)
    local_opt.set_xtol_abs(XTOL)
    local_opt.set_stopval(-SUCCESS_HEIGHT)
    local_opt.set_maxeval(get_budget())
    local_opt.set_lower_bounds([0, 0])
    local_opt.set_upper_bounds([x_max, y_max])
    local_opt.set_min_objective(f)
    x0, y0 = init_guess
    x, y = local_opt.optimize([x0, y0])
    z = local_opt.last_optimum_value()

    return {
        'x': np.array([x, y]),
        'z': z,
    }


nelder_mead = Algorithm(
    name='Nelder-Mead',
    version=1,
    func=run_nelder_mead,
)
