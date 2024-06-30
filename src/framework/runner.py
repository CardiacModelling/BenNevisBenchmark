import nevis
from functools import wraps

from .result import Result
from .config import MAX_FES

f = nevis.linear_interpolant()
f_grad = nevis.linear_interpolant(grad=True)


def optimizer(opt):
    """
    Decorator for optimization algorithms.

    Parameters
    ----------
    opt : function
        The optimization algorithm to decorate. The function should have the
        following signature:

        def opt(f, x_max, y_max, rand_seed, init_guess, trial, get_budget):
            # MINIMIZE f((x, y)) subject to 0 <= x <= x_max and 0 <= y <= y_max
            # f(u, grad=None) returns the function value at u, and modifies
            # grad in place if grad is not None (as used in nlopt)
            return {
                'x': (x_best, y_best),
                'z': z_best,
                'message': 'A message', # optional
                'trajectory': [(x, y), ...], # optional
            }

    Returns
    -------
    function, which takes three arguments: ``rand_seed``, ``init_guess``, and ``trial``,
    and returns a ``Result`` object. ``rand_seed`` is the random seed used for a run,
    ``init_guess`` is the initial guess for a run, and ``trial`` is the ``Trial`` object
    used for specifying the hyper-parameters of an algorithm instance. This function
    should be used as the ``func`` argument of ``Algorithm``.
    """

    @wraps(opt)
    def func(rand_seed, init_guess, trial):
        points = []
        function_values = []
        end_of_iterations = []

        def wrapper(u, grad=None):
            x, y = u
            points.append((x, y))
            if grad is not None and grad.size > 0:
                z, (gx, gy) = f_grad(x, y)
                grad[0] = gx
                grad[1] = gy
            else:
                z = f(x, y)
            function_values.append(z)
            return -z

        def get_budget():
            return MAX_FES - len(function_values)

        def mark_end_of_iteration():
            end_of_iterations.append(len(function_values))

        x_max, y_max = nevis.dimensions()

        res_dict = opt(
            wrapper,
            x_max,
            y_max,
            rand_seed,
            init_guess,
            trial,
            get_budget,
            mark_end_of_iteration,
        )

        x = res_dict['x']
        z = res_dict['z']
        message = res_dict.get('message', '')
        trajectory = res_dict.get('trajectory', [])

        return Result(
            x,
            -z,
            points,
            message,
            function_values,
            trajectory=trajectory,
            end_of_iterations=end_of_iterations,
        )
    return func
