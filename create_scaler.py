import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from features import VolatilityAdjuster
from sklearn.preprocessing import RobustScaler

class ScalerFactory:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.volatility_window = 30  # Days for volatility baseline

    def create_adaptive_scaler(self):
        """Create regime-aware scaler with confidence tiers"""
        df = self.fetch_market_data()
        
        pipeline = Pipeline([
            ('robust_scaler', RobustScaler()),
            ('volatility_adjuster', VolatilityAdjuster())
        ])
        
        pipeline.fit(
            df[['close', 'high', 'low', 'volume']],
            volatility_adjuster__volatility=df['volatility']
        )
        
        # Save versioned scaler
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, f"models/scaler_{timestamp}.pkl")
        
        return pipeline

    def fetch_market_data(self):
        """Fetch data with volatility regimes"""
        # Implement your data fetching logic
        return pd.DataFrame({
            'close': np.random.normal(1800, 50, 1000),
            'high': np.random.normal(1810, 50, 1000),
            'low': np.random.normal(1790, 50, 1000),
            'volume': np.random.poisson(1000, 1000),
            'volatility': np.random.uniform(0.5, 2.0, 1000)
        })