from algorithms import dual_annealing, mlsl, ipop_cmaes, pso, nelder_mead
from framework import *

save_handler = SaveHandlerJSON('../result/json/')
algo = nelder_mead
algo.tune_params(
    db_path='../result/main.db',
    save_handler=save_handler,
    iter_num=1,
)
