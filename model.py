import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import xgboost as xgb

# Define the path to the pre-trained DistilBERT model and tokenizer
MODEL_PATH = os.path.join(os.getcwd(), 'bert_model')

# Initialize variables for tokenizer and model
tokenizer = None
model = None

# Load DistilBERT tokenizer and model
try:
    print(f"Attempting to load DistilBERT tokenizer and model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # Load AutoModelForSequenceClassification as instructed, ensuring output_hidden_states and return_dict are set
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, output_hidden_states=True, return_dict=False)
    model.eval()  # Set model to evaluation mode
    device = torch.device("cpu")  # Use CPU for inference
    model.to(device)
    print("DistilBERT model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading DistilBERT model or tokenizer: {e}")
    tokenizer = None
    model = None

# Initialize variable for XGBoost model
xgboost_model = None

# Load XGBoost model
try:
    print("Attempting to load XGBoost model from: xgboost_model.json")
    xgboost_model = xgb.XGBClassifier()
    xgboost_model.load_model('xgboost_model.json')
    print("XGBoost model loaded successfully.")
except Exception as e:
    print(f"Error loading XGBoost model: {e}")
    xgboost_model = None

def get_cls_embedding(text):
    """
    Generates the CLS token embedding for a given text using the loaded DistilBERT model.
    """
    if tokenizer is None or model is None:
        raise RuntimeError("DistilBERT model or tokenizer not loaded. Cannot get CLS embedding.")

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=tokenizer.model_max_length)
    # Move inputs to the appropriate device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # Perform a forward pass through the model
        # outputs will be a tuple: (logits, hidden_states_tuple) because return_dict=False and output_hidden_states=True
        outputs = model(**inputs)
        # Extract the last layer's hidden states (outputs[1][-1]) and the CLS token (index 0)
        cls_embedding = outputs[1][-1][:, 0, :].cpu().numpy()
    return cls_embedding

def predict(text):
    """
    Predicts the class and probabilities for a given text using CLS embeddings and the XGBoost model.
    """
    if xgboost_model is None:
        raise RuntimeError("XGBoost model not loaded. Cannot make prediction.")

    try:
        # Get the CLS embedding for the input text
        cls_embedding = get_cls_embedding(text)
        
        # Reshape the embedding for XGBoost prediction (expects 2D array: (n_samples, n_features))
        # cls_embedding is already (1, embedding_dim) for a single text, so it's ready.
        prediction = xgboost_model.predict(cls_embedding)
        prediction_proba = xgboost_model.predict_proba(cls_embedding)
        
        # Convert numpy arrays to lists for easier handling
        return prediction.tolist(), prediction_proba.tolist()
    except RuntimeError as re:
        print(f"Prediction failed due to model/tokenizer loading error: {re}")
        return None, None
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None
