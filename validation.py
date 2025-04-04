import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
import logging
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_scaler(model_path, scaler_path):
    """Load the trained model and scaler."""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model = tf.keras.models.load_model(model_path)
        logging.info(f"Model loaded from {model_path}")
        logging.info(f"Model summary: {model.summary()}")
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
            
        scaler = joblib.load(scaler_path)
        logging.info(f"Scaler loaded from {scaler_path}")
        
        return model, scaler
    except Exception as e:
        logging.error(f"Error loading model or scaler: {str(e)}")
        raise

def validate_model(model, scaler, data, labels=None):
    """Validate the model and scaler with the provided data."""
    try:
        # Check data validity
        if data is None or len(data) == 0:
            raise ValueError("Empty or None data provided for validation")
            
        logging.info(f"Validating model with data shape: {data.shape}")
        
        # Preprocess data
        try:
            scaled_data = scaler.transform(data)
        except Exception as scale_error:
            logging.warning(f"Scaling error: {str(scale_error)}. Attempting to proceed with unscaled data.")
            scaled_data = data
            
        # Make predictions
        predictions = model.predict(scaled_data)
        logging.info(f"Predictions shape: {predictions.shape}")
        
        # If we have labels, calculate performance metrics
        if labels is not None and len(labels) == len(data):
            # Convert predictions to binary classes if needed
            if predictions.shape[1] > 1:  # Multi-class output
                pred_classes = np.argmax(predictions, axis=1)
            else:  # Binary output
                pred_classes = (predictions > 0.5).astype(int)
                
            # Calculate metrics
            accuracy = accuracy_score(labels, pred_classes)
            precision = precision_score(labels, pred_classes, average='weighted')
            recall = recall_score(labels, pred_classes, average='weighted')
            f1 = f1_score(labels, pred_classes, average='weighted')
            
            logging.info(f"Validation Metrics:")
            logging.info(f"  Accuracy:  {accuracy:.4f}")
            logging.info(f"  Precision: {precision:.4f}")
            logging.info(f"  Recall:    {recall:.4f}")
            logging.info(f"  F1 Score:  {f1:.4f}")
            
            return predictions, {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        return predictions
    except Exception as e:
        logging.error(f"Error during model validation: {str(e)}")
        raise

def main():
    model_path = os.path.join("models", "drl_model.h5")
    scaler_path = os.path.join("models", "scaler.pkl")
    
    # Load test data or generate synthetic data
    # In a real scenario, load your test data from a file
    feature_count = 40  # Match the feature count from ml_models.py
    sample_count = 20
    
    logging.info(f"Generating synthetic test data with {sample_count} samples and {feature_count} features")
    data = np.random.rand(sample_count, feature_count)
    
    # Generate some random binary labels for test purposes
    labels = np.random.randint(0, 2, size=sample_count)
    
    try:
        model, scaler = load_model_and_scaler(model_path, scaler_path)
        results = validate_model(model, scaler, data, labels)
        logging.info("Validation completed successfully")
        return results
    except Exception as e:
        logging.error(f"Validation failed: {str(e)}")
        return None

if __name__ == "__main__":
    main()