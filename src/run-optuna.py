from algorithms import dual_annealing, mlsl, ipop_cmaes
from framework import *

save_handler = SaveHandlerJSON('../result/json/')
algo = ipop_cmaes
algo.tune_params(
    db_path='../result/main.db',
    save_handler=save_handler,
    iter_num=15,
)