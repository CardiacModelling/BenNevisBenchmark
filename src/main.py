from framework import *
from algorithms import dual_annealing

algorithm = dual_annealing
algorithm.load_instances()

algorithm.tune_params(25)

# instance = algorithm.generate_random_instance()

# instance.run()

instance: AlgorithmInstance = algorithm.best_instance
# instance.load_results(False)

# instance = list(algorithm.instances)[0]
print(instance.performance_measures())

instance.plot_convergence_graph()
instance.plot_stacked_graph()