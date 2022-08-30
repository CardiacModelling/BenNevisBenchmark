from algorithms import *
from framework import *

algorithm: Algorithm = dual_annealing

algorithm.tune_params()

instance: AlgorithmInstance = algorithm.best_instance
# instance.plot_convergence_graph()
# instance.plot_convergence_graph_variant()
instance.plot_histogram()

