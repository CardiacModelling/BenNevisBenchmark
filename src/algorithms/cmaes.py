import pints
from framework import optimizer, MAX_FES, SUCCESS_HEIGHT, Algorithm


@optimizer
def run_cmaes(f, x_max, y_max, population_size=None):
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
    fes = MAX_FES
    e = Error(f)
    x_best, f_best = None, 0

    while fes > 0:
        x0 = b.sample()
        s0 = min(b.range()) / 2
        opt = pints.OptimisationController(
            e,
            x0=x0,
            sigma0=s0,
            boundaries=b,
            method=pints.CMAES
        )
        if population_size is not None:
            opt.optimiser().set_population_size(population_size)
        # opt.set_max_unchanged_iterations(100, threshold=0.01)
        opt.set_max_evaluations(fes)
        opt.set_threshold(-SUCCESS_HEIGHT)
        opt.set_f_guessed_tracking(not use_x_best)
        opt.set_log_to_screen(False)
        x1, f1 = opt.run()

        if f1 < f_best:
            x_best, f_best = x1, f1

        fes -= opt.evaluations()

    return {
        'x': x_best,
        'z': f_best,
    }


cmaes = Algorithm(
    'CMAES',
    run_cmaes,
    {},
    11
)
