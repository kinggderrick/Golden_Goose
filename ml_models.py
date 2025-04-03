import tensorflow as tf
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler

class GoldenGooseModel:
    """ML model for signal confirmation with confidence scores"""
    
    def __init__(self, model_path="models/drl_model.h5", scaler_path="models/scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.load_model()
        
    def load_model(self):
        """Load the model and scaler with error handling"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
            else:
                # Fallback to simplified model if file doesn't exist
                self.create_fallback_model()
                
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            else:
                self.scaler = StandardScaler()
        except Exception as e:
            print(f"Model loading error: {str(e)}")
            self.create_fallback_model()
            self.scaler = StandardScaler()
    
    def create_fallback_model(self):
        """Create a simple model as fallback"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(2, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        self.model = model
    
    def preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and prepare features for model input"""
        # Extract relevant features
        features = np.column_stack([
            df['rsi'].values[-10:],          # Last 10 RSI values
            df['macd'].values[-10:],         # Last 10 MACD values
            df['h4_rsi'].values[-5:],        # Last 5 4H RSI values
            df['close'].pct_change().values[-10:],  # Last 10 price changes
            df['volume'].values[-5:] / df['volume'].mean()  # Volume normalization
        ])
        
        # Handle potential NaN values
        features = np.nan_to_num(features, nan=0.0)
        
        # Reshape for model input
        return features.reshape(1, -1)
    
    def predict_signal(self, features: np.ndarray) -> dict:
        """Generate signal with confidence scores"""
        if self.model is None:
            return {'buy_confidence': 0.0, 'sell_confidence': 0.0}
        
        try:
            # Dimension check and adjustment
            if features.shape[1] != 10 and self.model.input_shape[1] == 10:
                # Pad or truncate to match expected input shape
                padded = np.zeros((features.shape[0], 10))
                padded[:, :min(features.shape[1], 10)] = features[:, :min(features.shape[1], 10)]
                features = padded
                
            # Get predictions
            pred = self.model.predict(features, verbose=0)
            
            # Format output as confidence scores
            return {
                'buy_confidence': float(pred[0][0]),
                'sell_confidence': float(pred[0][1]) if pred.shape[1] > 1 else 1 - float(pred[0][0])
            }
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return {'buy_confidence': 0.0, 'sell_confidence': 0.0}