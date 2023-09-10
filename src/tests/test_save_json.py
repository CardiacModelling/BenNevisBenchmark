from framework import SaveHandlerJSON
import unittest
from util import make_result

class TestSaveJSON(unittest.TestCase):
    def setUp(self) -> None:
        self.save_handler = SaveHandlerJSON("../json", "unittest")
        self.maxDiff = None
    
    def tearDown(self) -> None:
        self.save_handler.drop_database()
    
    def test_find(self):
        algorithm_info = {
            'algorithm_name': 'Nelder-Mead-test',
            'algorithm_version': 2023,
        }
        instance_info = {
            **algorithm_info,
            'instance_index': 5555,
        }
        result_indices = [1, 10, 100]
        for i in result_indices:
            result = make_result(
                rand_seed=i, 
                info={
                    **instance_info,
                    'result_index': i,
                })
            self.save_handler.save_result(result)
        results = self.save_handler.find_results(instance_info)
        self.assertEqual([r.info['result_index'] for r in results], result_indices)

        intances = self.save_handler.find_instances(algorithm_info)
        self.assertEqual(
            intances, 
            [{**instance_info, 'results_count': len(result_indices)}]
        )


