from framework import *
import nevis
from scipy.optimize import dual_annealing
import numpy as np
import matplotlib.pyplot as plt


f = nevis.linear_interpolant()
x_max, y_max = nevis.dimensions()


if __name__ == '__main__':
    def run_dual_annealing(**kwargs):
        points = []
        function_values = []
        def wrapper(u):
            x, y = u
            points.append((x, y))
            z = f(x, y)
            function_values.append(z)
            return -z

        x_max, y_max = nevis.dimensions()
        ret = dual_annealing(
            wrapper, 
            bounds=[(0, x_max), (0, y_max)],
            maxfun=MAX_FES,
            **kwargs
        )

        return Result(
            ret.x,
            -ret.fun,
            points,
            ret.message,
            function_values,
            ret_obj=ret
        )

    algo = Algorithm(
        'Dual Annealing', 
        run_dual_annealing,
        {
            # 'maxiter': np.arange(1500, 3000, 100),
            'maxiter': [2000],
            'initial_temp': np.linspace(2e4, 4e4, 1000),
            'restart_temp_ratio': np.logspace(-5, -3, 100),
            # 'visit': np.linspace(1 + EPS, 3, 1000),
            # 'accept': np.logspace(-5, -1e-4, 1000),
        },
        version=2
    )
    algo.tune_params(
        # measure='par2',
        # mode='min',
        measure='success_rate',
        mode='max'
    )
    algo_instance = algo.best_instance
    print(algo_instance.success_measures())
    algo_instance.print()
    # algo_instance.plot_convergence_graph()
    algo.plot_all()
    # algo.plot_instances(
    #     x_measure='success_rate',
    #     y_measure='avg_height'
    # )
    # plt.show()

    # TODO: time consumed on different parts of the programme