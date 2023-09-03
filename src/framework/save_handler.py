from .result import Result
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from .config import MONGODB_URI, logger


class SaveHandler:
    def __init__(self, database='test'):
        # Create a new client and connect to the server
        self.client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
        # Send a ping to confirm a successful connection
        try:
            self.client.admin.command('ping')
            logger.info(f"Pinged your deployment at {MONGODB_URI}.\nYou successfully connected to MongoDB!")
        except Exception as e:
            logger.debug(MONGODB_URI)
            logger.exception(e)
        self.db = self.client[database]
        self.res_collection = self.db['results']
        self.algo_collection = self.db['algorithms']
        self.database = database
    
    def drop_database(self):
        # self.client.drop_database(self.database)
        self.res_collection.drop()
        self.algo_collection.drop()
        
    def save_result(self, result):
        if None in list(result.info.values()):
            logger.warning('Saving a result with incomplete info!')
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

    def save_algorithm_best_instance(self, algorithm):
        self.algo_collection.update_one(
            algorithm.info,
            {"$set": {
                **algorithm.info, 
                'best_instance_index': algorithm.best_instance_index,
            }},
            upsert=True,
        )
    
    def load_algorithm_best_instance(self, algorithm):
        doc = self.algo_collection.find_one(algorithm.info)
        if doc is None or doc['best_instance_index'] == -1: 
            return
        algorithm.best_instance_index = doc['best_instance_index']
        algorithm.best_instance = algorithm.generate_instance(algorithm.best_instance_index)
        algorithm.best_instance.load_partial_results(save_handler=self)

    def load_algorithm_instance_indices(self, algorithm):
        res = self.find_instances(algorithm.info)
        algorithm.instance_indices = set([ins['instance_index'] for ins in res])