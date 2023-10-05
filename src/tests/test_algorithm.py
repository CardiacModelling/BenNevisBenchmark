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
            version=algo_version,
        )





    @parameterized.expand([0, 1])
    def test_algorithm_2(self, i):
        algo_name = 'MLSL test'
        algo_version = 456
        algo = Algorithm(
            name=algo_name, 
            version=algo_version,
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



