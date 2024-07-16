from framework import optimizer, Algorithm, SUCCESS_HEIGHT, FTOL
import scipy.optimize
import optuna
import numpy as np


@optimizer
def run_differential_evolution(
    f,
    x_max,
    y_max,
    rand_seed,
    init_guess,
    trial: optuna.Trial,
    get_budget,
    mark_end_of_iteration,
):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
    def callback(intermediate_result):
        # Define your custom stopping condition here
        # For example, stop when the function value is below a threshold
        if intermediate_result.fun < -SUCCESS_HEIGHT:
            return True

        if get_budget() <= 0:
            return True

    popsize = trial.suggest_int('popsize', 10, 50)
    recombination = trial.suggest_float('recombination', 0, 1)
    polish = trial.suggest_categorical('polish', [True, False])
    dithering = trial.suggest_categorical('dithering', [True, False])
    mutation_low = trial.suggest_float('mutation_low', 0, 2)
    mutation_high = trial.suggest_float('mutation_high', mutation_low, 2)
    mutation = mutation_low if not dithering else (mutation_low, mutation_high)

    np.random.seed(rand_seed)

    x, z = None, float('inf')
    i = 0
    while get_budget() > 0 and z > -SUCCESS_HEIGHT:
        ret = scipy.optimize.differential_evolution(
            f,
            bounds=[(0, x_max), (0, y_max)],
            strategy='best1bin',
            maxiter=25000000,
            popsize=popsize,
            tol=0,
            atol=FTOL,
            mutation=mutation,
            recombination=recombination,
            seed=rand_seed,
            callback=callback,
            polish=polish,
            x0=init_guess if i == 0 else [
                np.random.random() * x_max, np.random.random() * y_max],
            init='latinhypercube',
        )
        if ret.fun < z:
            x, z = ret.x, ret.fun
        i += 1
        mark_end_of_iteration()

    return {
        'x': x,
        'z': z,
    }


differential_evolution = Algorithm(
    name='Differential Evolution',
    func=run_differential_evolution,
    version=5,
    default_params={
        'popsize': 15,
        'recombination': 0.7,
        'polish': True,
        'dithering': True,
        'mutation_low': 0.5,
        'mutation_high': 1,
    }
)
