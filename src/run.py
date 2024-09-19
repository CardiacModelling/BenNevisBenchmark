import argparse
from algorithms import mlsl, pso, differential_evolution, nelder_mead_multi, \
    dual_annealing, cmaes
from framework import SaveHandlerJSON, AlgorithmInstance
import optuna

# Define the available algorithms as a dictionary
algorithms = {
    'mlsl': mlsl,
    'pso': pso,
    'de': differential_evolution,
    'da': dual_annealing,
    'cmaes': cmaes,
    'nm': nelder_mead_multi,
}

# Set up argument parsing
parser = argparse.ArgumentParser(
    description="Run one or more optimization algorithms.")

# Required argument: Algorithms (one or more)
parser.add_argument(
    'algorithms',
    nargs='+',  # Accepts one or more algorithms
    choices=algorithms.keys(),
    help="Choose one or more algorithms to run (space-separated)"
)

# Optional argument: Test name
parser.add_argument(
    '--test_name',
    default='june',
    help="Specify the test name (affects save path and database file)"
)

# Optional argument: Number of iterations for tuning
parser.add_argument(
    '--iter_num',
    type=int,
    default=100,
    help="Specify the number of iterations for hyperparameter tuning"
)

# Optional argument: Seed for deterministic behavior
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    help="Specify the seed value to ensure deterministic behavior"
)

# Parse the arguments
args = parser.parse_args()

# Use the test name in save paths
save_handler = SaveHandlerJSON(f'../result/{args.test_name}/')
db_path = f'../result/{args.test_name}.db'

# Loop through the selected algorithms
for algo_name in args.algorithms:
    algo = algorithms[algo_name]

    # Check if the algorithm needs hyperparameter tuning
    if algo_name in ['mlsl', 'pso', 'de', 'da', 'cmaes']:
        # Perform hyperparameter tuning with user-specified iter_num and seed
        algo.tune_params(
            db_path=db_path,
            save_handler=save_handler,
            iter_num=args.iter_num,
            using_restart_results=False,
            seed=args.seed,  # Use the seed from command-line args
        )

        # Get the best instance and run the algorithm
        ins = algo.best_instance
        ins.run(
            save_handler=save_handler,
            save_partial=False,
            restart=True,
            does_prune=False,
        )

    elif algo_name == 'nm':
        # No hyperparameter tuning needed for Nelder Mead
        ins = AlgorithmInstance(algorithm=algorithms[algo_name],
                                trial=optuna.trial.FixedTrial({}),
                                instance_index=-1)
        ins.run(
            save_handler=save_handler,
            save_partial=False,
            restart=True,
            does_prune=False,
        )
