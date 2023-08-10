from framework import optimizer
import scipy.optimize
from framework.algorithm import Algorithm


@optimizer
def run_shgo(f, x_max, y_max, n, iters, sampling_method):
    ret = scipy.optimize.shgo(
        func=f,
        bounds=((0, x_max), (0, y_max)),
        sampling_method=sampling_method,
        n=n,
        iters=iters,
        minimizer_kwargs={
            'method': 'Nelder-Mead',
            'options': {
                'fatol': 1e-1,
            }
        },
    )
    return {
        'x': ret.x,
        'z': ret.fun,
        'message': ret.message
    }


shgo = Algorithm(
    'SHGO',
    run_shgo,
    {
        'sampling_method': ['sobol'],
        'n': [2**11],
        'iters': [1],
    },
    version=1
)
