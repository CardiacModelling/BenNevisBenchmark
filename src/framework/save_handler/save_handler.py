class SaveHandler:
    def __init__(self, **kwargs) -> None:
        pass

    def drop_database(self):
        pass

    def save_result(self, result, partial=True):
        pass

    def find_results(self, query, partial=True):
        pass

    def find_instances(self, query):
        pass

    def save_algorithm_best_instance(self, algorithm):
        pass

    def load_algorithm_best_instance(self, algorithm, results_partial=True):
        pass

    def load_algorithm_instance_indices(self, algorithm):
        pass
