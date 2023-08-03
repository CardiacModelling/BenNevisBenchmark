# Maximum number of function evals
MAX_FES = 50000
# Height to reach to be considered a success
SUCCESS_HEIGHT = 1340
# Number of algorithm instances to generate in a hyper-parameter tuning
RS_ITER_NUM = 15
# Number of runs for each algorithm instance by default
RUN_NUM = 10
# Path to save instances and results
SAVE_PATH = '../saved/'

__all__ = [
    'MAX_FES',
    'SUCCESS_HEIGHT',
    'RS_ITER_NUM',
    'RUN_NUM',
    'SAVE_PATH',
]
