from dotenv import load_dotenv
import os
load_dotenv()

# Maximum number of function evals
MAX_FES = 50000
# Height to reach to be considered a success
SUCCESS_HEIGHT = 1340
# Number of algorithm instances to generate in a hyper-parameter tuning
RS_ITER_NUM = 15
# Number of runs for each algorithm instance by default
RUN_NUM = 50
# Uri for mongodb connection
MONGODB_URI = os.getenv('MONGODB_URI')

__all__ = [
    'MAX_FES',
    'SUCCESS_HEIGHT',
    'RS_ITER_NUM',
    'RUN_NUM',
    'MONGODB_URI',
]
