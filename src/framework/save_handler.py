import pickle
import os.path
import os
from .algorithm_instance import AlgorithmInstance


class SaveHandler:
    def __init__(self, path):
        """
        Class for saving and loading algorithm instances and results. This
        class is used by ``Algorithm`` to save and load instances and results.

        Parameters
        ----------
        path : string
            The path to the folder where the instances and results of the
            algorithm will be saved.
        """
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def save_instance(self, instance):
        """
        Save an instance of the algorithm.
        """
        instance_dir = os.path.join(self.path, str(instance.hash))
        os.makedirs(instance_dir, exist_ok=True)
        instance_path = os.path.join(instance_dir, 'instance')
        with open(instance_path, 'wb') as file:
            pickle.dump({
                'algorithm': instance.algorithm,
                'params': instance.params,
                'hash': instance.hash,
            }, file)

    def add_result(self, instance, result):
        """
        Add a result to an instance.

        Parameters
        ----------
        instance : AlgorithmInstance
            The instance to add the result to.
        result : Result
            The result to add.
        """
        instance_dir = os.path.join(self.path, str(instance.hash))
        os.makedirs(instance_dir, exist_ok=True)
        result_path = os.path.join(instance_dir, str(result.time))
        with open(result_path, 'wb') as file:
            pickle.dump(result, file)

    def load_results(self, instance_hash):
        """
        Load all the results of an instance.

        Parameters
        ----------
        instance_hash : int
            The hash of the instance.

        Returns
        -------
        list of Result
        """
        instance_dir = os.path.join(self.path, str(instance_hash))
        if not os.path.exists(instance_dir):
            return set()

        results = set()
        for result_path in os.listdir(instance_dir):
            if result_path == 'instance':
                continue
            with open(os.path.join(instance_dir, result_path), 'rb') as file:
                result = pickle.load(file)
                results.add(result)
        return results

    def load_instance(self, instance_hash):
        """
        Load an instance.

        Parameters
        ----------
        instance_hash : int
            The hash of the instance.

        Returns
        -------
        AlgorithmInstance
        """

        instance_dir = os.path.join(self.path, str(instance_hash))

        instance_path = os.path.join(instance_dir, 'instance')
        with open(instance_path, 'rb') as file:
            instance_dict = pickle.load(file)

        instance = AlgorithmInstance(
            **instance_dict,
            save_handler=self,
        )

        return instance

    def get_all_instances(self):
        """
        Get all instances of the algorithm.

        Returns
        -------
        list of AlgorithmInstance
        """
        instances = []
        for instance_hash in os.listdir(self.path):
            instances.append(self.load_instance(instance_hash))
        return instances
