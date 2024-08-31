from algorithms import mlsl, pso, differential_evolution, nelder_mead_multi, dual_annealing, cmaes
from framework import SaveHandlerJSON

save_handler = SaveHandlerJSON('../result/june/')
algos = [
    mlsl, pso, differential_evolution,
    nelder_mead_multi, dual_annealing, cmaes
]
for algo in algos:
    algo.tune_params(
        db_path='../result/june.db',
        save_handler=save_handler,
        iter_num=100,
        using_restart_results=False,
    )
    ins = algo.best_instance
    ins.run(
        save_handler=save_handler,
        save_partial=False,
        restart=True,
        does_prune=False,
    )
