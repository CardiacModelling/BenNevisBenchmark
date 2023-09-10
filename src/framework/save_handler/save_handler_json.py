import json
import os
import shutil
from ..result import Result
from ..config import logger
from collections import defaultdict
from .save_handler import SaveHandler

def dict2str(d: dict):
    parts = [f"{key}-{d[key]}" for key in sorted(d.keys())]
    name = "|".join(parts)
    return name

class SaveHandlerJSON(SaveHandler):
    def __init__(self, directory='data', database='test'):
        self.directory = directory
        self.database = database
        self.results_directory = os.path.join(directory, f'{database}_results')
        self.algorithms_file_path = os.path.join(directory, f'{database}_algorithms.json')
        self.ensure_directory_exists()
    
    def enumerate_results(self):
        jsons = os.listdir(self.results_directory)
        for json_path in jsons:
            if not json_path.endswith('.json'):
                continue
            with open(os.path.join(self.results_directory, json_path), 'r') as file:
                res = json.load(file)
                yield res
    
    def load_algorithms(self):
        with open(self.algorithms_file_path, 'r') as file:
            algorithms = json.load(file)
        return algorithms

    def ensure_directory_exists(self):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)
        if not os.path.exists(self.algorithms_file_path):
            with open(self.algorithms_file_path, 'w') as file:
                json.dump({}, file)

    def drop_database(self):
        shutil.rmtree(self.directory)

    def save_result(self, result, partial=True):
        if None in list(result.info.values()):
            logger.warning('Saving a result with incomplete info!')
        
        result_dict = result.to_dict(partial=partial)

        # Create a unique filename based on result.info
        result_filename = os.path.join(
            self.results_directory, 
            dict2str(result.info) + ".json")

        # Save the result data to the new JSON file
        with open(result_filename, 'w') as file:
            json.dump(result_dict, file, indent=4)

    def find_results(self, query, partial=True):
        filtered_results = []
        
        for result_data in self.enumerate_results():
            matches_query = all(result_data[key] == query[key] for key in query)
            if matches_query:
                filtered_result = result_data.copy()
                if partial and 'points' in filtered_result:
                    del filtered_result['points']
                filtered_results.append(Result(**filtered_result))
        
        filtered_results.sort(key=lambda result: result.info['result_index'])
        return filtered_results

    def find_instances(self, query):
        instance_count = defaultdict(int)
        for result_data in self.enumerate_results():
            if all(result_data[key] == query[key] for key in query):
                instance = (
                    ('algorithm_name', result_data['algorithm_name'],),
                    ('algorithm_version', result_data['algorithm_version'],),
                    ('instance_index', result_data['instance_index'],),
                )
                instance_count[instance] += 1

        result_instances = []
        for instance, count in instance_count.items():
            ins_info = dict(instance)
            ins_info['results_count'] = count
            result_instances.append(ins_info)
        return result_instances

    def save_algorithm_best_instance(self, algorithm):
        algorithm_info = algorithm.info
        algorithms = self.load_algorithms()
        algorithms[dict2str(algorithm_info)] = {
            **algorithm_info,
            'best_instance_index': algorithm.best_instance_index
        }
        with open(self.algorithms_file_path, 'w') as file:
            json.dump(algorithms, file, indent=4)

    def load_algorithm_best_instance(self, algorithm, results_partial=True):
        algorithm_info = algorithm.info
        with open(self.algorithms_file_path, 'r') as file:
            algorithms = json.load(file)

        res = algorithms.get(dict2str(algorithm_info), None)
        if res is None or res['best_instance_index'] == -2:
            return
        algorithm.best_instance_index = res['best_instance_index']
        algorithm.best_instance = algorithm.generate_instance(res['best_instance_index'])
        algorithm.best_instance.load_results(save_handler=self, partial=results_partial)

    def load_algorithm_instance_indices(self, algorithm):
        res = self.find_instances(algorithm.info)
        algorithm.instance_indices = set([ins['instance_index'] for ins in res])
