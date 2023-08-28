from .algorithm_instance import AlgorithmInstance
from .result import Result
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from .config import MONGODB_URI
import logging



class SaveHandler:
    def __init__(self, database='test'):
        # Create a new client and connect to the server
        self.client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
        # Send a ping to confirm a successful connection
        try:
            self.client.admin.command('ping')
            logging.info("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            logging.exception(e)
        self.db = self.client[database]
        self.res_collection = self.db['results']
        self.algo_collection = self.db['algorithms']
        
    def save_result(self, result):
        if None in list(result.info.values()):
            logging.warning('Saving a result with incomplete info!')
        result_dict = result.to_dict()
        self.res_collection.update_one(result.info, {"$set": result_dict}, upsert=True)
    
    def find_results(self, query):
        docs = self.res_collection.find(query, {"_id": 0})
        return [Result(**doc) for doc in docs]
    
    def find_instances(self, query):
        pipeline = [
            {
                "$match": query
            },
            {
                "$group": {
                    "_id": {
                        "algorithm_name": "$algorithm_name",
                        "algorithm_version": "$algorithm_version",
                        "instance_index": "$instance_index"
                    },
                    "count": {"$sum": 1},
                }
            },
            {
                "$replaceRoot": {
                    "newRoot": {
                        "algorithm_name": "$_id.algorithm_name",
                        "algorithm_version": "$_id.algorithm_version",
                        "instance_index": "$_id.instance_index",
                        "results_count": "$count",
                    }
                }
            }
        ]

        return list(self.res_collection.aggregate(pipeline))

    def save_algorithm(self, algorithm):
        algorithm_info = {
            'algorithm_name': algorithm.name,
            'algorithm_version': algorithm.version,
        }