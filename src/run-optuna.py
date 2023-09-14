from algorithms import dual_annealing, mlsl
from framework import *

save_handler = SaveHandlerJSON('mlsl')
algo = mlsl
algo.tune_params(
    db_path='../result/main.db',
    save_handler=save_handler,
    iter_num=20,
)

