import os
from .config import *
from .result import Result
from .save_handler import SaveHandler
from .algorithm_instance import AlgorithmInstance
from .algorithm import Algorithm
from .runner import optimizer

os.makedirs(SAVE_PATH, exist_ok=True)

del os
