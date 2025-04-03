import talib
import pandas as pd
import numpy as np
from typing import Tuple

class HybridStrategy:
    def __init__(self, config: dict):
        self.config = config
        self.required_bars = max(20*3, 100)  # Sufficient for all indicators
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Implements the exact researched strategy indicators"""
        # Multi-timeframe Momentum Component (4H RSI)
        h4_df = df.resample('4H', on='time').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        h4_df['h4_rsi'] = talib.RSI(h4_df['close'], 14)
        
        # Mean Reversion Components
        df['rsi'] = talib.RSI(df['close'], 14)  # 30/70 levels
        df['upper_bb'], df['middle_bb'], df['lower_bb'] = talib.BBANDS(
            df['close'], 
            timeperiod=20, 
            nbdevup=2, 
            nbdevdn=2
        )
        
        # Momentum Confirmation (Volume-Weighted MACD)
        df['macd'], df['macd_signal'], _ = talib.MACD(
            df['close'], 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        df['volume_ma'] = talib.SMA(df['volume'], 20)
        
        # Merge multi-timeframe data
        return pd.merge_asof(df, h4_df[['time', 'h4_rsi']], on='time')

    def generate_signals(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """Executes the researched entry logic"""
        if len(df) < self.required_bars:
            return False, False
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Researched Long Conditions
        long_entry = (
            (current['rsi'] < 30) &                  # Mean reversion oversold
            (current['close'] < current['lower_bb']) & # Bollinger band trigger
            (current['macd'] > current['macd_signal']) & # Momentum confirmation
            (current['volume'] > current['volume_ma'])   # Volume surge
            (current['h4_rsi'] > 40) )                  # Higher timeframe filter
            
        # Researched Short Conditions
        short_entry = (
            (current['rsi'] > 70) &                   # Mean reversion overbought
            (current['close'] > current['upper_bb']) & # Bollinger band trigger
            (current['macd'] < current['macd_signal']) & # Momentum confirmation
            (df['volume'].iloc[-3:].mean() > df['volume'].mean()) & # Volume pattern
            (current['h4_rsi'] < 60) )                  # HTF filter
            
        return bool(long_entry), bool(short_entry)