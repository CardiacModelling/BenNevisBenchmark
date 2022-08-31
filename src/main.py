from algorithms import *
from framework import *

algorithm: Algorithm = dual_annealing
# algorithm: Algorithm = mlsl

# instance = algorithm.generate_instance(1)
algorithm.tune_params()
instance = algorithm.best_instance
instance.run(15)
result: Result = list(instance.results)[0]
# result.plot_global()
# result.plot_partial()
# result.generate_kml('mlsl.kml')
# instance.plot_convergence_graph()
# algorithm.tune_params()

# instance: AlgorithmInstance = algorithm.best_instance
# instance.plot_convergence_graph()
# print(instance.success_measures())
instance.plot_convergence_graph_variant()
# instance.plot_histogram()

