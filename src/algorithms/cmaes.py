import pints
from framework import optimizer, SUCCESS_HEIGHT, Algorithm, FTOL
import numpy as np
import optuna


@optimizer
def run_cmaes(
    f,
    x_max,
    y_max,
    rand_seed,
    init_guess,
    trial: optuna.Trial,
    get_budget,
    mark_end_of_iteration,
):
    np.random.seed(rand_seed)
    # Use best found, instead of best guessed
    use_x_best = True
    b = pints.RectangularBoundaries([0, 0], [x_max, y_max])
    width = min(b.range())
    sigma0 = trial.suggest_float('sigma0', width / 20, width / 2)
    population_size = trial.suggest_int('population_size', 4, 100)
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

    e = Error(f)
    x, z = None, float('inf')
    i = 0

    while get_budget() > 0 and z > -SUCCESS_HEIGHT:
        opt = pints.OptimisationController(
            e,
            x0=b.sample() if i > 0 else init_guess,
            sigma0=sigma0,
            boundaries=b,
            method=pints.CMAES,
        )
        opt.optimiser().set_population_size(population_size)
        opt.set_max_unchanged_iterations(10, threshold=FTOL)
        opt.set_max_evaluations(get_budget())
        opt.set_threshold(-SUCCESS_HEIGHT)
        opt.set_f_guessed_tracking(not use_x_best)
        opt.set_log_to_screen(False)
        x1, z1 = opt.run()
        if z1 < z:
            x, z = x1, z1
        i += 1
        mark_end_of_iteration()

    return {
        'x': x,
        'z': z,
    }


cmaes = Algorithm(
    name='CMA-ES',
    func=run_cmaes,
    version=1,
    default_params={
        'sigma0': 1.2e5,
        'population_size': 6,
    }
)
