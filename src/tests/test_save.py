from framework import SaveHandler
import unittest
from util import make_result

class TestSave(unittest.TestCase):
    def setUp(self) -> None:
        self.db_name = "unittest"
        self.save_handler = SaveHandler(self.db_name)
    
    def tearDown(self) -> None:
        self.save_handler.client.drop_database(self.db_name)
    
    def test_save_result(self):
        result = make_result(
            rand_seed=1, 
            info={
                'algorithm_name': 'Nelder-Mead-test',
                'algorithm_version': 2023,
                'instance_index': 4444,
                'result_index': 1,
            })
        self.save_handler.save_result(result)
        doc = self.save_handler.res_collection.find_one(result.info, {"_id": 0})
        self.assertEqual(doc, result.to_dict())
    
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



