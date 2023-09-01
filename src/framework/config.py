from dotenv import load_dotenv
import os
load_dotenv()


# Maximum number of function evals for a single run
MAX_FES = 50_000
# Maximum number of function evals for a single instance
MAX_INSTANCE_FES = 1_000_000
# Height to reach to be considered a success
SUCCESS_HEIGHT = 1340
GARY_SCORE_CONFIG = [(1340, 10), (1310, 7), (1297, 3), (1235, 2), (1215, 1)]
# Number of algorithm instances to generate in a hyper-parameter tuning
RS_ITER_NUM = 15
# Number of runs for each algorithm instance by default
RUN_NUM = 50
# Uri for mongodb connection
MONGODB_URI = os.getenv('MONGODB_URI')
XTOL = 10
FTOL = 0.2

__all__ = [
    'MAX_FES',
    'SUCCESS_HEIGHT',
    'RS_ITER_NUM',
    'RUN_NUM',
    'MONGODB_URI',
    'XTOL',
    'FTOL',
]
