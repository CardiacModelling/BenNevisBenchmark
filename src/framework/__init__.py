import os
from .config import MAX_FES, SUCCESS_HEIGHT, RS_ITER_NUM, RUN_NUM, SAVE_PATH
from .result import Result
from .save_handler import SaveHandler
from .algorithm_instance import AlgorithmInstance
from .algorithm import Algorithm
from .runner import optimizer

os.makedirs(SAVE_PATH, exist_ok=True)

del os

__all__ = [
    'MAX_FES',
    'SUCCESS_HEIGHT',
    'RS_ITER_NUM',
    'RUN_NUM',
    'SAVE_PATH',
    'Result',
    'SaveHandler',
    'AlgorithmInstance',
    'Algorithm',
    'optimizer',
]
