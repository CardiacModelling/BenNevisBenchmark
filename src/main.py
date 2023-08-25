# from framework import Randomiser

# print(Randomiser.get_init_guess(1))
# print(Randomiser.get_init_guess(2))
# print(Randomiser.get_init_guess(2))
# print(Randomiser.get_init_guess(1))


# from algorithms import dual_annealing, mlsl

# algorithm = dual_annealing
# algorithm.load_instances()
# algorithm.load_instance(1691490753.9132943)
# algorithm.generate_all_instances()

# for instance in algorithm.instances:
#     instance.print()

# algorithm.tune_params(70)

# instance = algorithm.generate_random_instance()

# instance.run()

# instance = list(algorithm.instances)[0]
# instance: AlgorithmInstance = algorithm.best_instance
# instance.load_results(False)
# instance.plot_measure_by_runs(['success_rate', 'success_rate_upper', 'success_rate_lower'], 1000)
# instance.plot_measure_by_runs(['ert_std'], 1000)
# instance.plot_measure_by_runs('ert', 1000)
# instance.plot_measure_by_runs('hv', 1000)
# instance.plot_measure_by_runs('par2', 1000)
# print(instance.performance_measures())

# instance.plot_convergence_graph()
# instance.plot_stacked_graph()

# algorithm.plot_tuning(
#     param_x='initial_temp',
#     param_y='restart_temp_ratio',
#     measure_color='success_rate',
#     measure_area='ert',
#     y_log=True,
#     reverse_area=True
# )

