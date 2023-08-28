import unittest
from framework import optimizer, MAX_FES, Result, Algorithm
import scipy.optimize
import logging
import numpy as np
from pympler import asizeof
import nlopt
from util import run_dual_annealing, run_mlsl


class TestRunner(unittest.TestCase):
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
    
    def test_algorithm(self):
        algo_name = 'Dual Annealing test'
        algo_version = 123
        algo = Algorithm(
            name=algo_name,
            func=run_dual_annealing,
            param_space={
                'maxiter': [1000, 1200],
                'initial_temp': [2000 * i for i in range(1, 6)],
                'restart_temp_ratio': [0.1 * i for i in range(1, 6)],
            },
            version=algo_version,
        )

        # Test if params <-> instance index works well
        self.assertEqual(algo.param_space_size, 2 * 5 * 5)
        self.assertEqual(
            algo.index_to_params(0),
            {
                'maxiter': 1000,
                'initial_temp': 2000,
                'restart_temp_ratio': 0.1,
            }
        )
        self.assertEqual(
            34, 
            algo.params_to_index(algo.index_to_params(34)),
        )

        ins_1 = algo.generate_instance(23)
        self.assertEqual(ins_1.info, {
            'algorithm_name': algo_name,
            'algorithm_version': algo_version,
            'instance_index': 23,
        })

        params_2 = algo.index_to_params(23)
        self.assertEqual(
            params_2,
            {'maxiter': 1000, 'initial_temp': 10000, 'restart_temp_ratio': 0.4},
        )
        ins_2 = algo.generate_instance_from_params(**params_2)

        # The two methods of generating instances are equivalent
        self.assertEqual(ins_1, ins_2)
        self.assertEqual(list(algo.instance_indices), [23])

        res_1 = ins_1(88)
        res_2 = ins_1(88)
        # The runs are not random when the index is fixed
        self.assertEqual(res_1.to_dict(), res_2.to_dict())

        max_instance_fes = 90000
        ins_1.run(max_instance_fes=max_instance_fes)
        
        s = 0
        for i, result in enumerate(ins_1.results):
            self.assertEqual(i, result.info['result_index'])
            s += result.eval_num
            if i <= len(ins_1.results) - 2:
                self.assertLess(s, max_instance_fes)
            else:
                self.assertGreaterEqual(s, max_instance_fes)

        measures = ins_1.performance_measures()
        self.assertAlmostEqual(measures['success_rate'], 0)

        size_full = asizeof.asizeof(ins_1)
        ins_1.make_results_partial()
        size_partial = asizeof.asizeof(ins_1)
        self.assertGreater(size_full / size_partial, 10)

    
    def test_algorithm_2(self):
        algo_name = 'MLSL test'
        algo_version = 456
        algo = Algorithm(
            name=algo_name, 
            version=algo_version,
            param_space={
                'population': list(range(1, 30))
            },
            func=run_mlsl,
        )
        algo.tune_params(
            iter_num=3,
            max_instance_fes=80_000,
            measure='avg_height',
            mode='max',
            rand_seed=566,
        )

        ins = algo.best_instance

        self.assertEqual(ins.params['population'], 9)

        # print('%%%')
        # print(ins.params)
        # print(ins.info)
        # print(ins.performance_measures())




