import unittest
from framework import Result, Algorithm, SaveHandlerMongo, SaveHandlerJSON
from util import run_dual_annealing, run_mlsl
import filecmp
from parameterized import parameterized


class TestRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.save_handlers = [
            SaveHandlerMongo('unittest2'),
            SaveHandlerJSON('unittest2')
        ]
        self.maxDiff = None

    def tearDown(self) -> None:
        for s in self.save_handlers:
            s.drop_database()

    def test_optimizer(self):
        rand_seed = 100
        init_guess = [5000, 5000]
        result = run_dual_annealing(
            rand_seed=rand_seed,
            init_guess=init_guess,
            maxiter=1000
        )
        # Test if the decorator returns the right type
        self.assertIsInstance(result, Result)

        result_2 = run_dual_annealing(
            rand_seed=rand_seed,
            init_guess=init_guess,
            maxiter=1200,
        )
        # Make sure the params get passed down
        self.assertLess(result.eval_num, result_2.eval_num)
