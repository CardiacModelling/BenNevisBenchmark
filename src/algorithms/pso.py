import pints
from framework import optimizer, SUCCESS_HEIGHT, Algorithm, FTOL
import numpy as np
import optuna

@optimizer
def run_pso(
    f, 
    x_max, 
    y_max, 
    rand_seed, 
    init_guess, 
    trial: optuna.Trial,
    get_budget,
):
    np.random.seed(rand_seed)
    # Create pints error measure
    class Error(pints.ErrorMeasure):
        """
        Turn a height into an error to be minimised.
        """
        def __init__(self, f):
            self.f = f

        def n_parameters(self):
            return 2

        def __call__(self, p):
            return self.f(p)
    
    b = pints.RectangularBoundaries([0, 0], [x_max, y_max])
    width = min(b.range())
    sigma0 = trial.suggest_float('sigma0', width / 20, width / 2)
    r = trial.suggest_float('r', 0, 1)
    population_size = trial.suggest_int('population_size', 1, 1000)
    e = Error(f)
    opt = pints.OptimisationController(
        e,
        x0=init_guess,
        sigma0=sigma0,
        boundaries=b,
        method=pints.PSO,
    )

    opt.optimiser().set_population_size(population_size)
    opt.optimiser().set_local_global_balance(r)
    opt.set_max_evaluations(get_budget())
    opt.set_threshold(-SUCCESS_HEIGHT)
    opt.set_max_unchanged_iterations(10, FTOL)
    opt.set_log_to_screen(False)

    x, z = opt.run()

    return {
        'x': x, 
        'z': z,
    }

pso = Algorithm(
    'pso', 
    run_pso,
    1,
)