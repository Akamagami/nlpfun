from flask import Flask, request, jsonify
from db.mongo_handler import MongoHandler
from spaCy.training_handler import train_model, extract_entities
import os
import threading
import logging

app = Flask(__name__)

# Initialize MongoDB handler
mongo_handler = MongoHandler()

# Directory to save models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

@app.route('/trainingData', methods=['POST'])
def add_training_data():
    """
    Endpoint to add new training data.
    Allows multiple training data entries to be sent in one request.
    """
    data = request.get_json()
    if not data or not isinstance(data, list):
        return jsonify({"error": "Invalid request format. Expected a list of training data."}), 400

    # Validate and add each training data entry
    for entry in data:
        if "text" not in entry or "entities" not in entry:
            return jsonify({"error": "Each training data entry must contain 'text' and 'entities'."}), 400

    # Insert all training data into MongoDB
    mongo_handler.add_training_data_bulk(data)

    return jsonify({"message": "Training data added successfully"}), 200


@app.route('/startTraining', methods=['POST'])
def start_training():
    """
    Endpoint to start training a new spaCy model based on the last available version.
    """
    data = request.get_json()
    if not data or "labels" not in data:
        return jsonify({"error": "Invalid request format"}), 400

    # Retrieve the last model version
    last_model = mongo_handler.get_last_model()
    init_tok2vec = None
    if last_model:
        last_model_path = os.path.join("spaCy", "models", last_model["modelVersion"])
        init_tok2vec = os.path.join(last_model_path, "model-last", "tok2vec.bin")

    # Determine the new model version
    current_model_version = f"v{mongo_handler.models_collection.count_documents({}) + 1}.0.0"

    # Start the training process in a separate thread
    def train_in_background():
        try:
            # Train the new model
            model_path = train_model(current_model_version, mongo_handler, init_tok2vec)

            # Save the new model metadata in MongoDB
            mongo_handler.save_model(current_model_version, model_path)
            logging.info(f"Model training completed: {current_model_version}")
        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")

    # Start the background thread
    training_thread = threading.Thread(target=train_in_background)
    training_thread.start()

    # Return 200 OK immediately
    return jsonify({"message": "Model training started", "modelVersion": current_model_version}), 200


@app.route('/extractEntities', methods=['POST'])
def extract_entities_endpoint():
    """
    Endpoint to extract entities from a given text using a specific model version.
    """
    data = request.get_json()
    if not data or "text" not in data or "labels" not in data or "modelVersion" not in data:
        return jsonify({"error": "Invalid request format"}), 400

    text = data["text"]
    labels = data["labels"]
    model_version = data["modelVersion"]

    # Fetch the model metadata from MongoDB
    model = mongo_handler.get_model(model_version)
    if not model:
        return jsonify({"error": "Model version not found"}), 404

    # Always use the 'model-best' directory for the specified model version
    model_path = os.path.join(model["modelPath"], "model-best")

    # Use the training handler to extract entities
    try:
        entities = extract_entities(model_path, text, labels)
        # Include the actual values in the response
        response_entities = [
            {
                "start": entity["start"],
                "end": entity["end"],
                "label": entity["label"],
                "value": text[entity["start"]:entity["end"]]
            }
            for entity in entities
        ]
    except Exception as e:
        return jsonify({"error": f"Failed to extract entities: {str(e)}"}), 500

    return jsonify({"entities": response_entities}), 200


@app.route('/getTrainingData', methods=['GET'])
def get_training_data():
    """
    Endpoint to get all training data for specific labels provided in the request body.
    """
    data = request.get_json()
    if not data or "labels" not in data:
        return jsonify({"error": "Invalid request format"}), 400

    labels = data["labels"]
    all_training_data = mongo_handler.get_all_training_data()

    # Filter training data by the specified labels
    filtered_data = [
        entry for entry in all_training_data
        if any(entity["label"] in labels for entity in entry["entities"])
    ]

    return jsonify(filtered_data), 200


@app.route('/getAllTrainingData', methods=['GET'])
def get_all_training_data():
    """
    Endpoint to get all training data without any filtering.
    """
    all_training_data = mongo_handler.get_all_training_data()
    return jsonify(all_training_data), 200


@app.route('/getModelVersions', methods=['GET'])
def get_model_versions():
    """
    Endpoint to retrieve all available model versions.
    """
    try:
        # Fetch all model versions using the MongoDB handler
        model_versions = mongo_handler.get_all_model_versions()
        return jsonify({"modelVersions": model_versions}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve model versions: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)