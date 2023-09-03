import pints
from framework import optimizer, MAX_FES, SUCCESS_HEIGHT, Algorithm
import numpy as np

@optimizer
def run_cmaes(f, x_max, y_max, rand_seed, init_guess, population_size, sigma0_n):

    np.random.seed(rand_seed)
    # Use best found, instead of best guessed
    use_x_best = True

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
    e = Error(f)
    opt = pints.OptimisationController(
        e,
        x0=init_guess,
        sigma0=min(b.range()) / sigma0_n,
        boundaries=b,
        method=pints.CMAES,
    )
    opt.optimiser().set_population_size(population_size)
    opt.set_max_unchanged_iterations(100, threshold=0.2)
    opt.set_max_evaluations(MAX_FES)
    opt.set_threshold(-SUCCESS_HEIGHT)
    opt.set_f_guessed_tracking(not use_x_best)
    opt.set_log_to_screen(False)
    x1, f1 = opt.run()


    return {
        'x': x1,
        'z': f1,
    }


cmaes = Algorithm(
    'CMAES',
    run_cmaes,
    {
        'population_size': (
            list(range(4, 11)) +
            list(range(20, 101, 10)) +
            list(range(150, 2001, 50)) +
            list(range(2100, 3001, 100))
        ),
        'sigma0_n': list(range(2, 21)),
    },
    1
)
