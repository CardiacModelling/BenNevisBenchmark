from framework import optimizer, Algorithm, SUCCESS_HEIGHT, FTOL, XTOL
import scipy.optimize
import logging
import optuna


@optimizer
def run_dual_annealing(
    f, 
    x_max, 
    y_max, 
    rand_seed, 
    init_guess, 
    trial: optuna.Trial,
    get_budget,
):
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

    initial_temp = trial.suggest_float('initial_temp', 0.02, 5e4, log=True)
    restart_temp_ratio = trial.suggest_float('restart_temp_ratio', 1e-6, 0.9, log=True)
    visit = trial.suggest_float('visit', 1.5, 2.9)
    accept = trial.suggest_float('accept', -5, -1.1e-4)

    try:
        ret = scipy.optimize.dual_annealing(
            f,
            bounds=[(0, x_max), (0, y_max)],
            maxfun=get_budget(),
            seed=rand_seed,
            x0=init_guess,
            minimizer_kwargs={
                'method': 'Nelder-Mead',
                'options': {'xatol': XTOL, 'fatol': FTOL},
            },
            maxiter=100000000, # we want this to be large enough
            callback=stopping_criterion,
            initial_temp=initial_temp,
            restart_temp_ratio=restart_temp_ratio,
            visit=visit,
            accept=accept,
        )
        return {
            'x': ret.x,
            'z': ret.fun,
            'message': extract_message(ret.message),
        }
    except Exception as e:
        logging.debug((initial_temp, restart_temp_ratio, visit, accept))
        logging.exception(e)
        return {
            'x': init_guess,
            'z': f(init_guess),
            'message': 'Invalid parameters.'
        }


dual_annealing = Algorithm(
    'Dual Annealing',
    run_dual_annealing,
    version=3
)
