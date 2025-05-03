# API Documentation

## Overview
Training Service API

### 1. Endpoint Name
**Method:** `POST`  
**URL:** `/trainingData`  
**Description:** Send new Training Data to the Service. Service creates the the Label if not present. 

#### Request
- **Body:**
```json
{
    "text": "Example {} with many {}.",
    "entities": [
        {
            "value": "text",
            "label": "NOUN"
        },
        {
            "value": "placeholders",
            "label": "noun"
        }
    ]
}
```
#### Response
- **Status Code:** `200 OK`

### 2. Start Training
**Method:** `POST`  
**URL:** `/startTraining`  
**Description:** Initiates training with a defined set of labels and returns the name of the newly created model version.

#### Request
- **Body:**
```json
{
    "labels": ["NOUN", "VERB", "ADJECTIVE"]
}
```

#### Response
- **Status Code:** `200 OK`  
- **Body:**
```json
{
    "modelVersion": "v1.0.1"
}
```

### 3. Extract Entities
**Method:** `POST`  
**URL:** `/extractEntities`  
**Description:** Processes a given text using a specified model version and a set of labels, returning the positions of the identified entities along with their respective labels.

#### Request
- **Body:**
```json
{
    "text": "The quick brown fox jumps over the lazy dog.",
    "labels": ["NOUN", "VERB"],
    "modelVersion": "v1.0.1"
}
```

#### Response
- **Status Code:** `200 OK`  
- **Body:**
```json
{
    "entities": [
        {
            "start": 10,
            "end": 15,
            "label": "NOUN"
        },
        {
            "start": 16,
            "end": 19,
            "label": "NOUN"
        }
    ]
}
```

### Note

In a real scenario you would need an endpoint to check wether a model is available, but for this you will just have to pray that the training is already done.

When starting the training it uses all training data it has for the labels. It does not remove or untrain old labels.