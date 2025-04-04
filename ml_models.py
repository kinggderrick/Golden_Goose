import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ConfidenceScaler:
    """Custom feature engineering for confidence tiers"""
    def __init__(self):
        self.volatility_baseline = None
        
    def fit(self, X, y=None):
        self.volatility_baseline = np.mean(X[:, 3])  # Index 3 = volatility
        return self
        
    def transform(self, X):
        X_new = X.copy()
        X_new[:, 0] /= 100  # RSI
        X_new[:, 1] *= 100  # MACD
        X_new[:, 3] /= self.volatility_baseline  # Volatility
        return X_new

class GoldenGooseModel:
    def __init__(self, model_path="models/drl_model.h5"):
        self.model = self._load_model(model_path)
        self.scaler = Pipeline([
            ('confidence', ConfidenceScaler()),
            ('standard', StandardScaler())
        ])

    def _load_model(self, path):
        try:
            return tf.keras.models.load_model(path)
        except:
            return self._create_fallback_model()

    def _create_fallback_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(40,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(2, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """Feature engineering pipeline"""
        features = np.column_stack([
            df['rsi'].values[-10:],
            df['macd'].values[-10:],
            df['h4_rsi'].values[-5:],
            df['daily_vol'].values[-5:],
            df['volume'].values[-5:] / df['volume'].mean()
        ])
        return self.scaler.fit_transform(features.reshape(1, -1))

    def predict_signal(self, features: np.ndarray) -> dict:
        """Return confidence scores for both directions"""
        pred = self.model.predict(features, verbose=0)
        return {
            'buy_confidence': float(pred[0][0]),
            'sell_confidence': float(pred[0][1])
        }