import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class VolatilityAdjuster(BaseEstimator, TransformerMixin):
    """Custom feature engineering for market regimes"""
    def __init__(self, lookback=50):
        self.lookback = lookback
        self.vol_scale_ = None

    def fit(self, X, y=None, volatility=None):
        if volatility is None:
            raise ValueError("VolatilityAdjuster requires volatility data")
        self.vol_scale_ = np.mean(volatility[-self.lookback:])
        return self

    def transform(self, X):
        if self.vol_scale_ is None:
            raise RuntimeError("VolatilityAdjuster not fitted")
        return X / self.vol_scale_

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, **fit_params)
        return self.transform(X)
