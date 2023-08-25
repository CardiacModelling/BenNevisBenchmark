import pickle
from .algorithm_instance import AlgorithmInstance
from .result import Result
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson.binary import Binary
from .config import MONGODB_URI

# Create a new client and connect to the server
client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client.test_algorithm_2


def make_binary(object):
    return Binary(pickle.dumps(object))


class SaveHandler:
    def __init__(self):
        """
        Class for saving and loading algorithm instances and results. 
        """

    def save_instance(self, instance):
        """
        Save an algorithm instance.
        """
        collection = db['instances']
        algo = instance.algorithm
        row = {
            'algorithm_name': algo.name,
            'algorithm_version': algo.version,
            'params': dict((key, float(value))
                           for key, value in instance.params.items()),
            'hash': instance.hash,
        }
        query = {"hash": instance.hash}
        collection.update_one(query, {"$set": row}, upsert=True)
        

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

        collection = db['results']
        result_dict = result.to_dict()
        result_dict['instance_hash'] = instance.hash
        collection.insert_one(result_dict)

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

        collection = db['results']
        projection = {
            '_id': 0,
            'ret_point': 1,
            'ret_height': 1,
            'rand_seed': 1,
            'init_guess': 1,
            'is_success': 1,
            'eval_num': 1,
            'message': 1,
            'create_time': 1,
            'len_points': 1,
        }

        query_results = collection.find({
            "instance_hash": instance_hash
        }, projection)

        results = set()
        for row in query_results:
            result = Result(**row)
            results.add(result)
        return results

    def load_instance(self, algorithm, instance_hash):
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

        collection = db['instances']
        row = collection.find_one({'hash': instance_hash})
        instance = AlgorithmInstance(
            algorithm=algorithm,
            params=row['params'],
            hash=row['hash'],
            save_handler=self,
        )
        return instance

    def get_all_instances(self, algorithm):
        """
        Get all instances of the algorithm.

        Returns
        -------
        list of AlgorithmInstance
        """

        collection = db['instances']
        rows = collection.find({
            'algorithm_name': algorithm.name,
            'algorithm_version': algorithm.version,
        })
        instances = []
        for row in rows:
            instances.append(AlgorithmInstance(
                algorithm=algorithm,
                params=row['params'],
                hash=row['hash'],
                save_handler=self,
            ))
        return instances
