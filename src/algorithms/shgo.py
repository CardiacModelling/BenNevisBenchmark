from framework import optimizer
from scipy.optimize import shgo


@optimizer
def run_shgo(f, x_max, y_max):
    ret = shgo(
        func=f,
        bounds=((0, x_max), (0, y_max)),
        # sampling_method=sampling,
        sampling_method='sobol',
        n=2**14,
        # iters=3
    )
    return {
        'x': ret.x,
        'z': ret.fun,
        'ret_obj': ret,
        'message': ret.message
    }

