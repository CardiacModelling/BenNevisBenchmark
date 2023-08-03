from framework import optimizer, Algorithm
import scipy.optimize
import numpy as np


@optimizer
def run_simulated_annealing(f, x_max, y_max, maxiter, initial_temp,
                            restart_temp_ratio, sigma):

    ret_x = (0, 0)
    ret_z = 0

    for _ in range(maxiter):
        k = 0
        x_k = np.random.uniform((0, 0), (x_max, y_max))
        f_x_k = f(x_k)
        x_best, f_x_best = x_k, f_x_k
        t = float('inf')
        while t > initial_temp * restart_temp_ratio:
            t = initial_temp / (k + 1)
            z_k = np.random.normal(x_k, sigma)
            if z_k[0] < 0:
                z_k[0] = 0
            if z_k[1] < 0:
                z_k[1] = 0
            if z_k[0] > x_max:
                z_k[0] = x_max
            if z_k[1] > y_max:
                z_k[1] = y_max
            f_z_k = f(z_k)
            diff = f_z_k - f_x_k
            if diff < 0 or np.random.rand() < np.exp(-diff / t):
                x_k, f_x_k = z_k, f_z_k

            if f_x_k < f_x_best:
                x_best, f_x_best = x_k, f_x_k

            k += 1

        local_ret = scipy.optimize.minimize(
            f,
            x_best,
            bounds=[(0, x_max), (0, y_max)],
            method='Nelder-Mead',
            options={'fatol': 1e-1}
        )
        if local_ret.success and local_ret.fun < f_x_best:
            x_best, f_x_best = local_ret.x, local_ret.fun

        if f_x_best < ret_z:
            ret_x, ret_z = x_best, f_x_best

    return {
        'x': ret_x,
        'z': ret_z,
    }


simulated_annealing = Algorithm(
    'Simulated Annealing',
    run_simulated_annealing,
    {
        'maxiter': [10],
        'sigma': [2e4],
        'initial_temp': np.linspace(2e4, 4e4, 1000),
        'restart_temp_ratio': np.logspace(-5, -3, 100),
    },
    version=2,
)
