import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_scaler(model_path, scaler_path):
    """Load the trained model and scaler."""
    try:
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        logging.info(f"Model loaded from {model_path}")
        logging.info(f"Scaler loaded from {scaler_path}")
        return model, scaler
    except Exception as e:
        logging.error(f"Error loading model or scaler: {str(e)}")
        raise

def validate_model(model, scaler, data):
    """Validate the model and scaler with the provided data."""
    try:
        # Preprocess data
        scaled_data = scaler.transform(data)
        predictions = model.predict(scaled_data)
        logging.info(f"Predictions: {predictions}")
        return predictions
    except Exception as e:
        logging.error(f"Error during model validation: {str(e)}")
        raise

if __name__ == "__main__":
    model_path = os.path.join("models", "drl_model.h5")
    scaler_path = os.path.join("models", "scaler.pkl")
    
    # Example data for validation (replace with actual data)
    data = np.random.rand(10, 4)  # 10 samples, 4 features each
    
    try:
        model, scaler = load_model_and_scaler(model_path, scaler_path)
        validate_model(model, scaler, data)
    except Exception as e:
        logging.error(f"Validation failed: {str(e)}")
