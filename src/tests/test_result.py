import unittest
from framework import Result
from util import run_sample_opt
import logging
import nevis
import numpy as np


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
        self.assertEqual(result.gary_score, 0)
        self.assertLess(result.ret_height, 1000)

    def test_successful_result(self):
        x, y = nevis.ben().grid
        points, heights, ret_height, ret_point = run_sample_opt(
            init_guess=(x + 200, y - 200))
        result = Result(ret_point=ret_point,
                        ret_height=ret_height,
                        points=points)
        logging.debug(result.to_dict())
        # logging.debug(heights)
        # logging.debug((x, y))
        self.assertTrue(result.is_success)
        # if SUCCESS_HEIGHT == 1344:
        #     self.assertEqual(result.eval_num, 49)
        # elif SUCCESS_HEIGHT == 1340:
        self.assertEqual(result.eval_num, 34)
        self.assertEqual(result.len_points, 68)
        self.assertGreater(result.ret_height, 1344)
        self.assertEqual(result.gary_score, 10)

        result.set_info({
            'algorithm_name': 'Nelder-Mead-test',
            'algorithm_version': 2023,
            'instance_index': 4444,
            'result_index': 1,
        })
        self.assertEqual(result.info['instance_index'], 4444)

    def test_partially_successful_result(self):
        x, y = nevis.macdui()[0].coords.grid
        points, heights, ret_height, ret_point = run_sample_opt(
            init_guess=(x + 500, y - 200))
        result = Result(ret_point=ret_point,
                        ret_height=ret_height,
                        points=points)
        logging.debug(result.to_dict())

        self.assertTrue(1309 < result.ret_height < 1310)
        self.assertEqual(result.gary_score, 3)
        self.assertFalse(result.is_success)
        self.assertEqual(result.eval_num, result.len_points)
