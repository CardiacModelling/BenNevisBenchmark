from framework import optimizer, MAX_FES, Algorithm, SUCCESS_HEIGHT
import scipy.optimize
import numpy as np
import logging


@optimizer
def run_dual_annealing(f, x_max, y_max, rand_seed, init_guess, **kwargs):
    def stopping_criterion(x, z, context):
        # Define your custom stopping condition here
        # For example, stop when the function value is below a threshold
        if z < -SUCCESS_HEIGHT:
            return True
        else:
            return False
    
    def extract_message(message):
        if type(message) == str:
            return message
        if type(message) == list:
            return '' if not message else message[0]
        return str(message)
    try:
    
        ret = scipy.optimize.dual_annealing(
            f,
            bounds=[(0, x_max), (0, y_max)],
            maxfun=MAX_FES,
            seed=rand_seed,
            x0=init_guess,
            minimizer_kwargs={
                'method': 'Nelder-Mead',
                'options': {'xatol': 10, 'fatol': 0.2},
            },
            maxiter=100000000, # we want this to be large enough
            callback=stopping_criterion,
            **kwargs
        )
        return {
            'x': ret.x,
            'z': ret.fun,
            'message': extract_message(ret.message),
        }
    except Exception as e:
        logging.debug(kwargs)
        logging.exception(e)

        return {
            'x': init_guess,
            'z': f(init_guess),
            'message': 'Invalid parameters.'
        }


dual_annealing = Algorithm(
    'Dual Annealing',
    run_dual_annealing,
    {
        'initial_temp': np.logspace(np.log10(0.02), np.log10(5e4), 200),
        'restart_temp_ratio': np.logspace(-6, np.log10(0.9), 200),
        'visit': np.linspace(1.5, 2.9, 100),
        'accept': -np.logspace(np.log10(1.1e-4), np.log10(5), 100),
    },
    version=3
)
