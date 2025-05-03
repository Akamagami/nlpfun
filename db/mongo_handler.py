import json
import os
from pymongo import MongoClient

# Determine the relative path to the configuration file
config_path = os.path.join(os.path.dirname(__file__), "localconfig.json")

# Load the MongoDB configuration from the JSON file
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found: {config_path}")

with open(config_path, "r") as config_file:
    config = json.load(config_file)

class MongoHandler:
    def __init__(self, uri=config["MONGO_URI"], db_name=config["DB_NAME"]):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.training_data_collection = self.db["training_data"]
        self.models_collection = self.db["models"]

    def add_training_data(self, data):
        """
        Adds training data to the MongoDB collection.
        """
        self.training_data_collection.insert_one(data)

    def add_training_data_bulk(self, data):
        """
        Adds multiple training data entries to the MongoDB collection.
        """
        self.training_data_collection.insert_many(data)

    def get_all_training_data(self):
        """
        Retrieves all training data from the MongoDB collection.
        """
        return list(self.training_data_collection.find({}, {"_id": 0}))

    def save_model(self, model_version, model_path):
        """
        Saves a model version with its file path.
        """
        self.models_collection.insert_one({
            "modelVersion": model_version,
            "modelPath": model_path
        })

    def get_model(self, model_version):
        """
        Retrieves a model by its version.
        """
        return self.models_collection.find_one({"modelVersion": model_version}, {"_id": 0})

    def get_all_model_versions(self):
        """
        Retrieves all model versions from the MongoDB collection.
        """
        models = self.models_collection.find({}, {"_id": 0, "modelVersion": 1})
        return [model["modelVersion"] for model in models]

    def get_last_model(self):
        """
        Retrieves the last model version from the MongoDB collection.
        """
        return self.models_collection.find_one(sort=[("modelVersion", -1)])