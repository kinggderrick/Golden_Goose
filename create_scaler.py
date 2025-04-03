import os
import sys
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import argparse
import logging
from datetime import datetime
import json
import shutil
from features import VolatilityAdjuster

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ScalerFactory:
    """Optimized scaler creator with market regime detection"""
    
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.symbol = self.config['symbol']
        self.timeframe = self.config.get('timeframe', 'TIMEFRAME_M15')
        self.num_bars = self.config.get('num_bars', 252*8*2)  # 2 years of 8hr sessions
        
    def _load_config(self, path):
        """Enhanced config validation"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file missing: {path}")
            
        with open(path) as f:
            config = json.load(f).get('mt5', {})
            
        if not config.get('symbol'):
            raise ValueError("Symbol required in config")
            
        return config

    def create_symlink_or_copy(self, src, dst):
        """Intelligent file handling with retries"""
        try:
            os.symlink(src, dst)
            logging.debug(f"Created symlink: {dst} -> {src}")
        except OSError as e:
            if getattr(e, 'winerror', 0) == 1314:  # Admin check
                logging.warning("Insufficient privileges - using file copy")
                for attempt in range(3):
                    try:
                        shutil.copy2(src, dst)
                        logging.info(f"Copied {src} to {dst} (attempt {attempt+1})")
                        break
                    except Exception as copy_error:
                        if attempt == 2: raise
                        time.sleep(1)
            else:
                raise

    def fetch_market_regimes(self):
        """Fetch data with volatility regime detection"""
        try:
            if not mt5.initialize():
                raise ConnectionError(f"MT5 Error: {mt5.last_error()}")
                
            rates = mt5.copy_rates_from_pos(
                self.symbol,
                getattr(mt5, self.timeframe, mt5.TIMEFRAME_M15),
                0, self.num_bars
            )
            
            if rates is None or len(rates) < 1000:
                raise ValueError("Insufficient historical data")
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df[['time', 'open', 'high', 'low', 'close']]
            
            # Market regime detection
            df['volatility'] = (df['high'] - df['low']).rolling(50).mean()
            df['trend'] = df['close'].rolling(200).mean()
            
            return df.dropna()
            
        finally:
            mt5.shutdown()

    def create_adaptive_scaler(self):
        """Create regime-aware scaler"""
        df = self.fetch_market_regimes()
        
        # Segment data by volatility regimes
        high_vol = df[df['volatility'] > df['volatility'].quantile(0.75)]
        low_vol = df[df['volatility'] < df['volatility'].quantile(0.25)]
        
        # Create composite scaler
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('volatility_adjust', VolatilityAdjuster())  # Custom class
        ])
        
        pipeline.fit(
            df[['open', 'high', 'low', 'close']].values,
            volatility_adjust__volatility=df['volatility'].values  # Pass volatility to VolatilityAdjuster
        )
        
        # Versioned output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        scaler_path = f"models/scaler_{timestamp}.pkl"
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, scaler_path)
        
        # Update latest pointer
        latest_path = "models/scaler.pkl"
        if os.path.exists(latest_path):
            os.remove(latest_path)
        self.create_symlink_or_copy(scaler_path, latest_path)
        
        logging.info(f"Created adaptive scaler: {scaler_path}")
        return scaler_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Market-Adaptive Scaler Generator')
    parser.add_argument('--config', default="config.json", help='Path to config file')
    args = parser.parse_args()
    
    try:
        factory = ScalerFactory(args.config)
        factory.create_adaptive_scaler()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)
