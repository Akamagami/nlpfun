import os
import spacy
from spacy.cli.train import train
from spacy.util import ensure_path
from spacy.tokens import DocBin
import logging
from spacy.training.loggers import console_logger

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

        # Create a spaCy Doc object
        doc = nlp.make_doc(formatted_text)
        ents = []
        for start, end, label in spans:
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
    logging.info(f"Saved training data: {data}")


def prepare_training_data(mongo_handler):
    """
    Prepares training and development data from MongoDB.

    Args:
        mongo_handler (MongoHandler): Instance of the MongoDB handler.
    """
    all_data = mongo_handler.get_all_training_data()

    # Split data into training and development sets (80/20 split)
    split_index = int(len(all_data) * 0.8)
    train_data = all_data[:split_index]
    dev_data = all_data[split_index:]

    # Convert data to .spacy format
    convert_to_spacy_format(train_data, TRAIN_DATA_PATH)
    convert_to_spacy_format(dev_data, DEV_DATA_PATH)


def generate_config(mongo_handler):
    """
    Generates the spaCy configuration file.

    Args:
        mongo_handler (MongoHandler): Instance of the MongoDB handler to fetch the last model.
    """
    # Generate the base configuration file
    os.system(f"python -m spacy init fill-config {BASE_CONFIG_PATH} {CONFIG_PATH}")

    # Read the generated config file
    with open(CONFIG_PATH, "r") as file:
        config = file.read()

    # Retrieve the last model's path
    last_model = mongo_handler.get_last_model()
    if last_model:
        # Use relative paths for the last model
        last_model_path = f"models/{last_model['modelVersion']}/model-last"

        # Set the NER source to the last model
        config = config.replace("source = \"en_core_web_lg\"", f"source = \"{last_model_path}\"")

        # Set the vectors to the last model's vectors
        config = config.replace("vectors = \"en_core_web_lg\"", f"vectors = \"{last_model_path}\"")

        # Set init_tok2vec to the model-last directory
        config = config.replace("init_tok2vec = null", f"init_tok2vec = \"{last_model_path}\"")
    else:
        logging.warning("No previous model found. Training will start from scratch.")

    # Use relative paths for train and dev data
    train_data_path = "data/training_data.spacy"
    dev_data_path = "data/dev_data.spacy"
    config = config.replace("train = null", f"train = {repr(train_data_path)}")
    config = config.replace("dev = null", f"dev = {repr(dev_data_path)}")

    # Write the updated configuration back to the file
    with open(CONFIG_PATH, "w") as file:
        file.write(config)


def train_model(current_model_version, mongo_handler, init_tok2vec=None):
    """
    Prepares data, generates config, trains a new spaCy model, and saves it.

    Args:
        current_model_version (str): The version of the new model.
        mongo_handler (MongoHandler): Instance of the MongoDB handler.
        init_tok2vec (str): Path to the tok2vec weights of the last model (if available).

    Returns:
        str: Path to the trained model.
    """
    # Prepare training and development data
    prepare_training_data(mongo_handler)

    # Generate the configuration file
    generate_config(mongo_handler)

    # Define the output directory for the trained model
    output_dir = os.path.join(MODEL_DIR, current_model_version)
    os.makedirs(output_dir, exist_ok=True)

    # Construct the spacy train command
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

    logging.info(f"Processing text: {text}")

    # Process the text
    doc = nlp(text)

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

    return entities