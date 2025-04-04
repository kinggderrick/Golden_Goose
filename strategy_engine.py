import talib
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from ml_models import GoldenGooseModel
from sklearn.preprocessing import StandardScaler

class HybridStrategy:
    def __init__(self, config: dict):
        self.config = config
        self.required_bars = max(20*3, 100)
        self.ml_model = GoldenGooseModel()
        self.confidence_thresholds = config.get('confidence_thresholds', {
            'super': {'ml_prob': 0.8, 'sentiment': 0.7, 'atr_ratio': 1.2},
            'confident': {'ml_prob': 0.6, 'sentiment': 0.4, 'atr_ratio': 1.0}
        })
        self.scaler = StandardScaler()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced indicator calculation with volatility normalization"""
        h4_df = df.resample('4H', on='time').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        })
        h4_df['h4_rsi'] = talib.RSI(h4_df['close'], 14)
        
        df['rsi'] = talib.RSI(df['close'], 14)
        df['upper_bb'], df['middle_bb'], df['lower_bb'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        df['macd'], df['macd_signal'], _ = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['volume_ma'] = talib.SMA(df['volume'], 20)
        
        merged = pd.merge_asof(df, h4_df[['time', 'h4_rsi']], on='time')
        return self._normalize_features(merged)

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features for ML model"""
        features = df[['rsi', 'macd', 'h4_rsi', 'close', 'volume']]
        df[features.columns] = self.scaler.fit_transform(features)
        return df

    def assess_confidence(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate 3-tier confidence levels"""
        features = self.ml_model.preprocess_data(df)
        ml_output = self.ml_model.predict_signal(features)
        
        current = df.iloc[-1]
        atr_ratio = current['atr'] / df['atr'].rolling(30).mean().iloc[-1]
        
        return {
            'ml_confidence': max(ml_output['buy_confidence'], 
                            ml_output['sell_confidence']),
            'sentiment': self._get_market_sentiment(),
            'volatility_ratio': atr_ratio
        }

    def generate_signals(self, df: pd.DataFrame) -> Tuple[bool, bool, str]:
        """Generate signals with confidence tier"""
        confidence = self.assess_confidence(df)
        
        long_entry, short_entry = False, False
        tier = "HOLD"
        
        if (confidence['ml_confidence'] >= self.confidence_thresholds['super']['ml_prob'] and
            abs(confidence['sentiment']) >= self.confidence_thresholds['super']['sentiment'] and
            confidence['volatility_ratio'] >= self.confidence_thresholds['super']['atr_ratio']):
            
            long_entry, short_entry = self._calculate_core_signals(df)
            tier = "SUPER"
            
        elif (confidence['ml_confidence'] >= self.confidence_thresholds['confident']['ml_prob'] or
            (abs(confidence['sentiment']) >= self.confidence_thresholds['confident']['sentiment'] and
            confidence['volatility_ratio'] >= self.confidence_thresholds['confident']['atr_ratio'])):
            
            long_entry, short_entry = self._calculate_core_signals(df)
            tier = "CONFIDENT"
            
        return long_entry, short_entry, tier

    def _calculate_core_signals(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """Original strategy logic"""
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        long_entry = (
            (current['rsi'] < 30) &
            (current['close'] < current['lower_bb']) &
            (current['macd'] > current['macd_signal']) &
            (current['volume'] > current['volume_ma']) &
            (current['h4_rsi'] > 40)
        )
        
        short_entry = (
            (current['rsi'] > 70) &
            (current['close'] > current['upper_bb']) &
            (current['macd'] < current['macd_signal']) &
            (df['volume'].iloc[-3:].mean() > df['volume'].mean()) &
            (current['h4_rsi'] < 60)
        )
        
        return bool(long_entry), bool(short_entry)

    def _get_market_sentiment(self) -> float:
        """Placeholder for sentiment integration"""
        # Will be implemented with your news API
        return np.random.uniform(-1, 1)