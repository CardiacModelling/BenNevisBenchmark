from framework import *
import nevis
from scipy.optimize import dual_annealing
from scipy.optimize import minimize, shgo
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


f = nevis.linear_interpolant()
x_max, y_max = nevis.dimensions()


def run_shgo():
    points = []
    function_values = []
    def wrapper(u):
        x, y = u
        points.append((x, y))
        z = f(x, y)
        function_values.append(z)
        return -z
    
    x_max, y_max = nevis.dimensions()

    qmc = scipy.stats.qmc.Halton(2)

    def sampling(n, dim):
        return qmc.random(n)

    ret = shgo(
        func=wrapper,
        bounds=((0, x_max), (0, y_max)),
        # sampling_method=sampling,
        sampling_method='sobol',
        n=2**14,
        # iters=3
    )

    return Result(
        ret.x,
        -ret.fun,
        points,
        ret.message,
        function_values,
        ret_obj=ret
    )

def run_random_search():
    points = []
    function_values = []
    def wrapper(u):
        x, y = u
        points.append((x, y))
        z = f(x, y)
        function_values.append(z)
        return -z
    
    x_max, y_max = nevis.dimensions()

    res_x, res_z = np.array([0, 0]), 0

    while len(points) < MAX_FES:
        x = np.random.rand() * x_max
        y = np.random.rand() * y_max

        ret = minimize(
            wrapper,
            np.array([x, y]),
            method='Nelder-Mead',
            bounds=[(0, x_max), (0, y_max)],
            options={
                'xatol': 0.1,
                'fatol': 0.1
            }
        )

        z = -ret.fun

        if z > res_z:
            res_x, res_z = ret.x , z
        

    return Result(
        res_x, 
        res_z,
        points,
        heights=function_values
    )


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


res = run_shgo()
res.print()
print(len(res.points))

res.plot_global()



# algo = Algorithm(
#     'SHGO',
#     run_shgo,
#     {},
#     2
# )
# instance = algo.generate_instance(1)
# instance.run(25)
# instance.plot_convergence_graph_variant()

# algo = Algorithm(
#     'Random Search with Nelder-Mead',
#     run_random_search,
#     {},
#     2
# )



# instance = algo.generate_instance(3)

# instance.run(25)

# instance.plot_convergence_graph_variant()
# instance.plot_convergence_graph()
# plt.show()



# algo = Algorithm(
#     'Dual Annealing', 
#     run_dual_annealing,
#     {
#         # 'maxiter': np.arange(1500, 3000, 100),
#         'maxiter': [2000],
#         'initial_temp': np.linspace(2e4, 4e4, 1000),
#         'restart_temp_ratio': np.logspace(-5, -3, 100),
#         # 'visit': np.linspace(1 + EPS, 3, 1000),
#         # 'accept': np.logspace(-5, -1e-4, 1000),
#     },
#     version=3
# )
# algo.tune_params(
#     # measure='par2',
#     # mode='min',
#     measure='success_rate',
#     mode='max'
# )
# algo_instance = algo.best_instance
# # algo_instance.plot_histogram()
# algo_instance.plot_convergence_graph_variant()
# # print(algo_instance.success_measures())
# # algo_instance.print()
# # algo_instance.plot_convergence_graph()
# # algo.plot_all()
# # algo.plot_instances(
# #     x_measure='success_rate',
# #     y_measure='avg_height'
# # )
# # plt.show()

# # TODO: time consumed on different parts of the programme

