import nlopt
from framework import optimizer, MAX_FES, Algorithm, XTOL, FTOL, SUCCESS_HEIGHT
import numpy as np

@optimizer
def run_mlsl(
    f, 
    x_max, 
    y_max, 
    rand_seed, 
    init_guess, 
    population=None,
):
    opt = nlopt.opt(nlopt.G_MLSL, 2)
    nlopt.srand(seed=rand_seed)
    opt.set_lower_bounds([0, 0])
    opt.set_upper_bounds([x_max, y_max])
    opt.set_min_objective(f)
    opt.set_ftol_abs(FTOL)
    opt.set_xtol_abs(XTOL)
    opt.set_maxeval(MAX_FES)
    opt.set_stopval(-SUCCESS_HEIGHT)
    if population is not None:
        opt.set_population(population)

    local_opt = nlopt.opt(nlopt.LN_NELDERMEAD, 2)
    local_opt.set_ftol_abs(FTOL)
    local_opt.set_xtol_abs(XTOL)
    local_opt.set_stopval(-SUCCESS_HEIGHT)

    opt.set_local_optimizer(local_opt)

    x0, y0 = init_guess
    x, y = opt.optimize([x0, y0])
    z = opt.last_optimum_value()

    return {
        'x': np.array([x, y]),
        'z': z,
    }

mlsl = Algorithm(
    name='MLSL', 
    version=1,
    param_space={
        'population': list(range(1, 1001)),
    },
    func=run_mlsl,
)