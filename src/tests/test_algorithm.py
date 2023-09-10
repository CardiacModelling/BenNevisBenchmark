import unittest
from framework import Result, Algorithm, SaveHandlerMongo, SaveHandlerJSON
from pympler import asizeof
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

        instance_index = 23
        ins_1 = algo.generate_instance(instance_index)
        self.assertEqual(ins_1.info, {
            'algorithm_name': algo_name,
            'algorithm_version': algo_version,
            'instance_index': 23,
        })

        params_2 = algo.index_to_params(instance_index)
        self.assertEqual(
            params_2,
            {'maxiter': 1000, 'initial_temp': 10000, 'restart_temp_ratio': 0.4},
        )
        ins_2 = algo.generate_instance_from_params(**params_2)

        # The two methods of generating instances are equivalent
        self.assertEqual(ins_1, ins_2)
        self.assertEqual(list(algo.instance_indices), [instance_index])

        result_index = 88
        res_1 = ins_1(result_index)
        res_2 = ins_1(result_index)
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
        self.assertAlmostEqual(measures['success_rate'], 0.25)
        self.assertTrue('gary_ert' in measures.keys())

        size_full = asizeof.asizeof(ins_1)
        ins_1.make_results_partial()
        size_partial = asizeof.asizeof(ins_1)
        self.assertGreater(size_full / size_partial, 10)

        default_ins = algo.generate_default_instance()
        self.assertEqual(default_ins.instance_index, -1)
        self.assertEqual(default_ins.params, {})

    @parameterized.expand([0, 1])
    def test_algorithm_2(self, i):
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
            max_instance_fes=30_000,
            rand_seed=566,
            save_handler=self.save_handlers[i],
        )

        ins = algo.best_instance

        # self.assertEqual(ins.params['population'], 1)
        measures_1 = ins.performance_measures()
        self.assertTrue(ins.results_patial)

        ins.run(restart=True, max_instance_fes=30_000)
        self.assertFalse(ins.results_patial)

        measures_2 = ins.performance_measures()
        self.assertEqual(measures_1, measures_2)

        ins.plot_convergence_graph(img_path='/tmp/convergence-graph.png')
        ins.plot_stacked_graph(img_path='/tmp/stacked-graph.png')

        # Mimic when we start the programme again
        algo2 = Algorithm(
            name=algo_name, 
            version=algo_version,
            param_space={
                'population': list(range(1, 30))
            },
            func=run_mlsl,
        )

        # Loading the best instance of an algorithm
        algo2.load_best_instance(save_handler=self.save_handlers[i])
        ins2 = algo2.best_instance
        self.assertEqual(ins.params, ins2.params)
        self.assertEqual(ins.info, ins2.info)
        self.assertEqual(ins.performance_measures(), ins2.performance_measures())

        # Loading all instance indices
        algo2.load_instance_indices(save_handler=self.save_handlers[i])
        self.assertEqual(algo.instance_indices, algo2.instance_indices)

        # Load the results for a single instance
        ins2.load_results(save_handler=self.save_handlers[i])
        self.assertEqual(len(ins2.results), len(ins.results))
        for res, res2 in zip(ins.results, ins2.results):
            self.assertEqual(res.to_dict(), res2.to_dict())
        
        ins2.run(
            restart=True, 
            max_instance_fes=30_000, 
            save_partial=False, 
            save_handler=self.save_handlers[i],
        )

        ins2.plot_convergence_graph(img_path='/tmp/convergence-graph-2.png')
        ins2.plot_stacked_graph(img_path='/tmp/stacked-graph-2.png')

        self.assertTrue(filecmp.cmp(
            '/tmp/convergence-graph.png',
            '/tmp/convergence-graph-2.png',
            shallow=False,
        ))

        self.assertTrue(filecmp.cmp(
            '/tmp/stacked-graph.png',
            '/tmp/stacked-graph-2.png',
            shallow=False,
        ))

        # Suppose we start once again, with results fully stored
        algo3 = Algorithm(
            name=algo_name, 
            version=algo_version,
            param_space={
                'population': list(range(1, 30))
            },
            func=run_mlsl,
        )
        algo3.load_best_instance(
            save_handler=self.save_handlers[i], 
            result_partial=False,
        )
        ins3 = algo3.best_instance
        ins3.plot_convergence_graph(img_path='/tmp/convergence-graph-3.png')
        ins3.plot_stacked_graph(img_path='/tmp/stacked-graph-3.png')
        
        self.assertTrue(filecmp.cmp(
            '/tmp/convergence-graph.png',
            '/tmp/convergence-graph-3.png',
            shallow=False,
        ))

        self.assertTrue(filecmp.cmp(
            '/tmp/stacked-graph.png',
            '/tmp/stacked-graph-3.png',
            shallow=False,
        ))



