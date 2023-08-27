import unittest
from framework import Result
from util import run_sample_opt
import logging
import nevis
import numpy as np

# logging.basicConfig(level=logging.DEBUG)


class TestResult(unittest.TestCase):
    def test_unsuccessful_result(self):
        points, heights, ret_height, ret_point = run_sample_opt(rand_seed=1)
        result = Result(ret_point=ret_point,
                        ret_height=ret_height,
                        points=points)
        logging.debug(result.to_dict())
        self.assertAlmostEqual(
            np.max(np.abs(np.array(heights) - result.heights)),
            0
        )
        self.assertFalse(result.is_success)
        self.assertEqual(result.eval_num, 56)
        self.assertEqual(result.len_points, 56)
    

    def test_successful_result(self):
        x, y = nevis.ben().grid
        points, heights, ret_height, ret_point = run_sample_opt(init_guess=(x + 200, y - 200))
        result = Result(ret_point=ret_point,
                        ret_height=ret_height,
                        points=points)
        logging.debug(result.to_dict())
        # logging.debug(heights)
        # logging.debug((x, y))
        self.assertTrue(result.is_success)
        self.assertEqual(result.eval_num, 49)
        self.assertEqual(result.len_points, 68)
