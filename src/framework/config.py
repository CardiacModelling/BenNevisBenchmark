from dotenv import load_dotenv
import datetime
import os
import logging
load_dotenv()


def setup_logging():
    logger = logging.getLogger("framework")
    logger.setLevel(logging.DEBUG)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"framework_{current_time}.log"
    handler = logging.FileHandler(log_file_name)
    # Create a formatter to define the log message format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add the handler to the logger
    logger.addHandler(handler)
    return logger


# Initialize the logger when this module is imported
logger = setup_logging()

# Maximum number of function evals for a single run
MAX_FES = 50_000
# Maximum number of function evals for a single instance
MAX_INSTANCE_FES = 1_000_000
# Height to reach to be considered a success
SUCCESS_HEIGHT = 1340
# gary score for each height interval
GARY_SCORE_CONFIG = [(1340, 10), (1310, 7), (1297, 3), (1235, 2), (1215, 1)]
# Number of algorithm instances to generate in a hyper-parameter tuning
RS_ITER_NUM = 100
# Uri for mongodb connection, get from .env file
MONGODB_URI = os.getenv('MONGODB_URI')
# xtol and ftol
XTOL = 10
FTOL = 0.2

__all__ = [
    'MAX_FES',
    'SUCCESS_HEIGHT',
    'RS_ITER_NUM',
    'MONGODB_URI',
    'XTOL',
    'FTOL',
    'logger',
]
