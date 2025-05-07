import os
import spacy
from spacy.cli.train import train
from spacy.util import ensure_path
from spacy.tokens import DocBin
import logging
from spacy.training.loggers import console_logger
from spacy.training import Corpus
from spacy.training import Example
import random
from spacy import displacy
from spacy.training import Example
import threading


# Set the log level to INFO (or DEBUG for more detailed logs)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Dynamically determine the base directory of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Directory to save spaCy-related files
SPACY_DIR = os.path.join(BASE_DIR, "spaCy")
MODEL_DIR = os.path.join(SPACY_DIR, "models")
DATA_DIR = os.path.join(SPACY_DIR, "data")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Change the current working directory to SPACY_DIR
os.chdir(SPACY_DIR)

# Paths for base and generated config files
BASE_CONFIG_PATH = os.path.join(SPACY_DIR, "base_config.cfg")  # Updated path
CONFIG_PATH = os.path.join(SPACY_DIR, "config.cfg")  # Updated path

# Paths for training and development data
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "training_data.spacy")
DEV_DATA_PATH = os.path.join(DATA_DIR, "dev_data.spacy")


def convert_to_spacy_format(data, output_path):
    """
    Converts training data into spaCy's .spacy format.

    Args:
        data (list): List of training data from MongoDB.
        output_path (str): Path to save the .spacy file.
    """
    nlp = spacy.blank("en")  # Create a blank English pipeline
    doc_bin = DocBin()  # Container for serialized Doc objects

    for entry in data:
        text = entry["text"]
        entities = entry["entities"]

        # Replace plain `{}` placeholders with named placeholders like `{placeholder0}`
        placeholder_count = text.count("{}")
        if placeholder_count != len(entities):
            logging.warning(f"Skipping entry: Mismatch between placeholders and entities in text: {text}")
            continue

        for i in range(placeholder_count):
            text = text.replace("{}", f"{{placeholder{i}}}", 1)

        # Build a dictionary of entity values for str.format
        entity_map = {}
        for i, entity in enumerate(entities):
            if "value" in entity and "label" in entity:
                entity_map[f"placeholder{i}"] = entity["value"]
            else:
                logging.warning(f"Skipping invalid entity: {entity} in text: {text}")

        # Replace placeholders with entity values using str.format
        try:
            formatted_text = text.format(**entity_map)
        except KeyError as e:
            logging.warning(f"Skipping entry: Placeholder mismatch in text: {text}. Missing key: {e}")
            continue

        # Calculate start and end positions for each entity
        spans = []
        for i, entity in enumerate(entities):
            value = entity_map.get(f"placeholder{i}")
            if value:
                start = formatted_text.find(value)
                if start == -1:
                    logging.warning(f"Skipping entity: Value '{value}' not found in text: {formatted_text}")
                    continue
                end = start + len(value)
                spans.append((start, end, entity["label"]))

        # Check for overlapping spans
        spans.sort(key=lambda x: x[0])  # Sort spans by start position
        valid_spans = []
        for i, (start, end, label) in enumerate(spans):
            if i > 0 and start < valid_spans[-1][1]:  # Overlap detected
                logging.warning(f"Skipping overlapping entity: [{start}, {end}, {label}] in text: {formatted_text}")
                continue
            valid_spans.append((start, end, label))

        # Create a spaCy Doc object
        doc = nlp.make_doc(formatted_text)
        ents = []
        for start, end, label in valid_spans:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                logging.warning(f"Skipping entity: [{start}, {end}, {label}] in text: {formatted_text}")
                continue
            ents.append(span)
        doc.ents = ents
        doc_bin.add(doc)

    # Save the processed data and log it
    doc_bin.to_disk(output_path)
    logging.info(f"Training data saved to {output_path}")


def prepare_training_data(mongo_handler, labels):
    """
    Prepares training and development data from MongoDB, ensuring that examples of other labels are included.

    Args:
        mongo_handler (MongoHandler): Instance of the MongoDB handler.
        labels (list): List of labels to focus on for the current training.
    """
    # Fetch all training data for the specified labels
    focused_data = mongo_handler.get_training_data_by_labels(labels)

    # Fetch a small number of examples for other labels
    all_labels = mongo_handler.get_all_labels()  # Assuming this method retrieves all unique labels in the dataset
    other_labels = [label for label in all_labels if label not in labels]
    additional_data = []
    for label in other_labels:
        # Fetch all data for the label and shuffle it
        label_data = mongo_handler.get_training_data_by_labels([label])
        random.shuffle(label_data)
        # Add up to 5 examples for the label
        additional_data.extend(label_data[:5])

    # Combine focused data with additional examples
    all_data = focused_data + additional_data

    # Shuffle the combined data
    random.shuffle(all_data)

    # Save every third entry to development data
    dev_data = all_data[::3]

    # Save all other entries to training data
    train_data = [entry for i, entry in enumerate(all_data) if i % 3 != 0]

    # Log the number of training and development entries
    logging.info(f"Number of training entries: {len(train_data)}")
    logging.info(f"Number of development entries: {len(dev_data)}")
    logging.info(f"Number of additional examples added: {len(additional_data)}")

    # Convert data to .spacy format
    convert_to_spacy_format(train_data, TRAIN_DATA_PATH)
    convert_to_spacy_format(dev_data, DEV_DATA_PATH)


def generate_config(mongo_handler, best_score, target_score, last_model_path=None):
    """
    Generates the spaCy configuration file and adjusts the learning rate dynamically.

    Args:
        mongo_handler (MongoHandler): Instance of the MongoDB handler to fetch the last model.
        best_score (float): The current best F1 score of the model.
        target_score (float): The target F1 score for early stopping.
        last_model_path (str): Path to the last model to use as the base for training.
    """
    # Generate the base configuration file
    os.system(f"python -m spacy init fill-config {BASE_CONFIG_PATH} {CONFIG_PATH}")

    # Read the generated config file
    with open(CONFIG_PATH, "r") as file:
        config = file.read()

    # Use the provided last model path or fetch the last model from MongoDB
    if last_model_path:
        logging.info(f"Using last model from previous iteration: {last_model_path}")
    else:
        last_model = mongo_handler.get_last_model()
        if last_model:
            last_model_path = f"models/{last_model['modelVersion']}/model-best"
            logging.info(f"Using last model from MongoDB: {last_model_path}")
        else:
            logging.warning("No previous model found. Training will start from scratch.")

    if last_model_path:
        # Convert the absolute path to a relative path and replace backslashes with forward slashes
        relative_last_model_path = os.path.relpath(last_model_path, SPACY_DIR).replace("\\", "/")

        # Set the NER source to the last model
        config = config.replace("source = \"en_core_web_md\"", f"source = \"{relative_last_model_path}\"")

        # Set the vectors to the last model's vectors
        config = config.replace("vectors = \"en_core_web_md\"", f"vectors = \"{relative_last_model_path}\"")

        # Set init_tok2vec to the model-last directory
        config = config.replace("init_tok2vec = null", f"init_tok2vec = \"{relative_last_model_path}\"")

    # Use relative paths for train and dev data
    train_data_path = "data/training_data.spacy"
    dev_data_path = "data/dev_data.spacy"
    config = config.replace("train = null", f"train = {repr(train_data_path)}")
    config = config.replace("dev = null", f"dev = {repr(dev_data_path)}")

    # Adjust the learning rate based on the current best score and target score
    if best_score is None or target_score is None:
        raise ValueError("best_score and target_score must not be None")
    learning_rate = max(0.001, min(0.2, (target_score - best_score) * 2))  # Clamp between 0.001 and 0.2
    logging.info(f"Setting learning rate to: {learning_rate}")

    # Update the learn_rate only if it exists in the config
    if "learn_rate" in config:
        config = config.replace("learn_rate = 0.2", f"learn_rate = {learning_rate}")

    # Write the updated configuration back to the file
    with open(CONFIG_PATH, "w") as file:
        file.write(config)


def train_model(current_model_version, mongo_handler, labels, target_score=0.97):
    """
    Prepares data, generates config, trains a new spaCy model, evaluates it, and saves it.

    Args:
        current_model_version (str): The version of the new model.
        mongo_handler (MongoHandler): Instance of the MongoDB handler.
        labels (list): List of labels to focus on for the current training.
        target_score (float): The target F1 score for early stopping.

    Returns:
        str: Path to the trained model.
    """
    # Ensure target_score is a float
    target_score = float(target_score)

    # Prepare training and development data
    prepare_training_data(mongo_handler, labels)

    # Define the output directory for the trained model
    output_dir = os.path.join(MODEL_DIR, current_model_version)
    os.makedirs(output_dir, exist_ok=True)

    best_score = 0.0
    iteration = 0
    last_model_path = None  # Initialize the last model path

    while best_score < target_score:
        iteration += 1
        logging.info(f"Starting training iteration {iteration}...")

        # Update the configuration file with the latest model information and adjusted learning rate
        generate_config(mongo_handler, best_score, target_score, last_model_path)

        # If the Iteration count is modulo 10 regenerate te training data
        if iteration % 10 == 0:
            logging.info("Regenerating training data...")
            prepare_training_data(mongo_handler, labels)

        # Construct the spacy train command with paths to training and dev data
        train_command = (
            f"python -m spacy train {CONFIG_PATH} --output {output_dir} "
            f"--paths.train {os.path.abspath(TRAIN_DATA_PATH)} "
            f"--paths.dev {os.path.abspath(DEV_DATA_PATH)}"
        )

        # Log the command for debugging
        logging.info(f"Running training command: {train_command}")

        # Execute the training command
        exit_code = os.system(train_command)

        # Check if the training was successful
        if exit_code != 0:
            raise RuntimeError(f"Training failed with exit code {exit_code}")

        # Update the last model path to the current output directory
        last_model_path = os.path.join(output_dir, "model-best")

        # Load the best model and evaluate it
        nlp = spacy.load(last_model_path)

        # Load the development data from the .spacy file
        dev_data_path = os.path.abspath(DEV_DATA_PATH)
        with open(dev_data_path, "rb") as f:
            doc_bin = DocBin().from_bytes(f.read())
        dev_docs = list(doc_bin.get_docs(nlp.vocab))

        # Convert the development data to Example objects

        examples = [Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}) for doc in dev_docs]

        # Evaluate the model on the development set
        scores = nlp.evaluate(examples)

        # Calculate a weighted score using ents_f, ents_p, and ents_r
        ents_f = scores.get("ents_f", 0.0)  # F1 score
        ents_p = scores.get("ents_p", 0.0)  # Precision
        ents_r = scores.get("ents_r", 0.0)  # Recall
        logging.info(f"Evaluation results: F1: {ents_f}, Precision: {ents_p}, Recall: {ents_r}")
        # Calculate the weighted score
        weighted_score = (0.5 * ents_f) + (0.25 * ents_p) + (0.25 * ents_r)
        logging.info(f"Weighted score: {weighted_score}")

        # Adjust the best score based on the weighted score
        if weighted_score > best_score:
            logging.info(f"New best weighted score achieved: {weighted_score} (F1: {ents_f}, P: {ents_p}, R: {ents_r})")
            best_score = weighted_score
        else:
            difference = best_score - weighted_score
            adjustment = (difference * 0.5) + 0.01
            best_score -= adjustment
            logging.info(f"Current weighted score ({weighted_score}) is lower than the previous best score ({best_score + adjustment}).")
            logging.info(f"Lowering best score by 50% of the difference ({adjustment}). New best score: {best_score}")

        # Stop training if the target score is reached
        if best_score >= target_score:
            logging.info(f"Target F1 score of {target_score} reached. Stopping training.")
            break
        else:
            logging.info(f"Starting a new iteration as the target score of {target_score} has not been reached.")

    # Log the final model and results
    logging.info(f"Training concluded. Final model loaded from: {last_model_path}")
    logging.info(f"Final evaluation results: {scores}")

    return output_dir


def extract_entities(model_path, text, labels):
    """
    Extracts entities from the given text using the specified model.

    Args:
        model_path (str): Path to the trained spaCy model.
        text (str): The input text to process.
        labels (list): List of labels to filter the extracted entities.

    Returns:
        list: A list of extracted entities with their start, end, and label.
    """
    logging.info(f"Loading model from: {model_path}")
    
    # Load the trained spaCy model
    nlp = spacy.load(model_path)

    # log all available labels in the model
    logging.info(f"Available labels in the model: {nlp.get_pipe('ner').labels}")

    logging.info(f"Processing text: {text}")

    # Process the text
    doc = nlp(text)

    # Log the entities found in the text
    logging.info(f"Entities found: {[(ent.text, ent.label_) for ent in doc.ents]}")

    # Extract entities matching the specified labels
    entities = [
        {"start": ent.start_char, "end": ent.end_char, "label": ent.label_, "text": ent.text}
        for ent in doc.ents if ent.label_ in labels
    ]

    # Log the extracted entities
    if entities:
        logging.info(f"Extracted entities: {entities}")
    else:
        logging.info("No entities found matching the specified labels.")

    html = displacy.render(doc, style="ent", page=True)
    output_path = "../static/displacy_visualization.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logging.info(f"Visualization saved to {output_path}")

    return entities