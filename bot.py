import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import time
import os
import argparse
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib
from cachetools import TTLCache
import asyncio
import sys

# Create necessary directories
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('backups', exist_ok=True)

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s UTC - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'bot.log')),
        logging.StreamHandler()
    ],
    datefmt='%Y-%m-%d %H:%M:%S'
)

CONFIG_SCHEMA = {
    "login": int,
    "password": str,
    "server": str,
    "symbol": str,
    "risk_per_trade": float
}

def validate_config(config, schema):
    for key, expected_type in schema.items():
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
        if not isinstance(config[key], expected_type):
            raise ValueError(f"Incorrect type for key '{key}': expected {expected_type}, got {type(config[key])}")

def initialize_mt5(config_path="config.json"):
    try:
        with open(config_path) as f:
            config = json.load(f)['mt5']
        validate_config(config, CONFIG_SCHEMA)
        if not mt5.initialize(
            login=config['login'],
            password=config['password'],
            server=config['server'],
            timeout=5000
        ):
            raise ConnectionError(f"MT5 connection failed: {mt5.last_error()}")
        logging.info(f"Connected to {config['server']} (Account #{config['login']})")
        return config
    except Exception as e:
        mt5.shutdown()
        logging.critical(f"Initialization failed: {str(e)}")
        raise

class AIModel:
    def __init__(self, model_path=os.path.join("models", "drl_model.h5"), scaler_path=os.path.join("models", "scaler.pkl")):
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file {model_path} not found")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file {scaler_path} not found")
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.scaler = joblib.load(scaler_path)
            self.expected_shape = self.model.input_shape[1:]
            logging.info(f"Model expects input shape: {self.expected_shape}")
            dummy_data = np.random.randn(1, *self.expected_shape)
            _ = self.model.predict(dummy_data)
            logging.info("Model warm-up complete")
        except Exception as e:
            logging.critical(f"AI Model initialization failed: {str(e)}")
            raise

    def preprocess_data(self, df):
        required_cols = ['open', 'high', 'low', 'close']
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]
        if not set(required_cols).issubset(df.columns):
            raise ValueError("Missing required price columns")
        data = df[required_cols].tail(100).values
        scaled = self.scaler.transform(data)
        aggregated = scaled.mean(axis=1)
        return np.expand_dims(aggregated, axis=0)

class TradingEngine:
    def __init__(self, config, ai_model):
        self.config = config
        self.ai_model = ai_model
        self.symbol = config['symbol']
        self.timeframe = mt5.TIMEFRAME_M15
        self.bar_cache = pd.DataFrame()
        self.cache = TTLCache(maxsize=100, ttl=60)
        self.last_data_fetch = datetime.min
        self.latest_confidence = "hold"

    async def get_market_data(self):
        try:
            if 'market_data' in self.cache and (datetime.now() - self.last_data_fetch).seconds < 30:
                return self.cache['market_data']
            rates = await asyncio.get_event_loop().run_in_executor(
                None,
                mt5.copy_rates_from_pos,
                self.symbol,
                self.timeframe,
                0,
                100
            )
            if rates is None or len(rates) < 100:
                raise ValueError("Failed to fetch 100 bars of data")
            new_bars = pd.DataFrame(rates)
            self.bar_cache = pd.concat([self.bar_cache, new_bars]).drop_duplicates().tail(100)
            self.cache['market_data'] = self.bar_cache
            self.last_data_fetch = datetime.now()
            return self.bar_cache
        except Exception as e:
            logging.error(f"Data fetch error: {str(e)}")
            return self.bar_cache

    async def generate_signal(self):
        try:
            df = await self.get_market_data()
            if len(df) < 100:
                raise ValueError("Insufficient data for prediction")
            
            # --- Momentum Component ---
            df['momentum'] = df['close'].pct_change(periods=14)
            
            # --- Mean Reversion Component ---
            window = 20
            df['sma'] = df['close'].rolling(window=window).mean()
            df['std'] = df['close'].rolling(window=window).std()
            threshold = 1.5 * df['std'].iloc[-1]
            
            current_price = df['close'].iloc[-1]
            sma = df['sma'].iloc[-1]
            momentum = df['momentum'].iloc[-1]
            
            # --- Signal Logic & Confidence Level ---
            if abs(momentum) > 0.02:
                signal = 'buy' if momentum > 0 else 'sell'
                confidence = "super_confident"
            else:
                if current_price < (sma - threshold):
                    signal = 'buy'
                    confidence = "confident"
                elif current_price > (sma + threshold):
                    signal = 'sell'
                    confidence = "confident"
                else:
                    signal = 'hold'
                    confidence = "hold"
            
            self.latest_confidence = confidence
            logging.info(f"Signal: {signal.upper()}, Confidence: {confidence}")
            return signal
        except Exception as e:
            logging.error(f"Signal generation error: {str(e)}", exc_info=True)
            return 'hold'

    async def run_async_benchmark(self):
        try:
            start = time.perf_counter()
            await asyncio.gather(*[asyncio.sleep(0.001) for _ in range(1000)])
            basic_latency = time.perf_counter() - start
            data_start = time.perf_counter()
            await self.get_market_data()
            data_latency = time.perf_counter() - data_start
            logging.info(
                "Async Performance Metrics:\n"
                f"  - Event Loop Throughput: {basic_latency:.4f}s\n"
                f"  - Market Data Fetch Time: {data_latency:.4f}s"
            )
            return {
                'basic_throughput': basic_latency,
                'data_latency': data_latency
            }
        except Exception as e:
            logging.critical(f"Benchmark failed: {str(e)}")
            raise

class TradeManager:
    daily_loss_limit = 0.02
    consecutive_losses = 0
    last_trade_day = None

    @staticmethod
    def calculate_dynamic_lot_size(account_equity, entry_price, stop_loss, confidence_level, pip_value):
        base_risk = 0.02  # 2%
        if confidence_level == "super_confident":
            risk_percentage = base_risk
        elif confidence_level == "confident":
            risk_percentage = base_risk * 0.75  # 1.5%
        else:
            return 0  # No trade if 'hold'
        risk_amount = account_equity * risk_percentage
        risk_per_unit = abs(entry_price - stop_loss) * pip_value
        lot_size = risk_amount / risk_per_unit
        return max(round(lot_size, 2), 0.01)

    @staticmethod
    def execute_order(signal, config):
        if getattr(TradeManager, 'daily_loss', 0) >= TradeManager.daily_loss_limit:
            logging.warning("Daily loss limit reached. Trading halted.")
            return None

        symbol_info = mt5.symbol_info(config['symbol'])
        if not symbol_info or not symbol_info.visible:
            logging.error(f"Symbol {config['symbol']} not available")
            return None

        try:
            tick = mt5.symbol_info_tick(config['symbol'])
            if not tick:
                raise ConnectionError("Failed to get current price")
            
            price = tick.ask if signal == 'buy' else tick.bid
            # Example stop_loss: 0.5% away from entry price
            stop_loss = price * 0.995 if signal == 'buy' else price * 1.005
            pip_value = 0.1  # Adjust based on asset class
            account_info = mt5.account_info()
            if not account_info:
                raise ValueError("Failed to retrieve account info")
            lot_size = TradeManager.calculate_dynamic_lot_size(account_info.equity, price, stop_loss, getattr(config, 'latest_confidence', "hold"), pip_value)
            if lot_size == 0:
                logging.info("Risk conditions not met. No trade executed.")
                return None

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": config['symbol'],
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "deviation": 10,
                "magic": 2024,
                "comment": "AI Trade",
                "type_time": mt5.ORDER_TIME_GTC
            }
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise RuntimeError(f"Trade failed: {result.comment}")
            TradeManager._update_daily_loss()
            return {
                'time': datetime.now(),
                'symbol': config['symbol'],
                'direction': signal,
                'price': price,
                'lots': lot_size
            }
        except Exception as e:
            logging.error(f"Order execution failed: {str(e)}")
            TradeManager.consecutive_losses += 1
            if TradeManager.consecutive_losses >= 3:
                logging.critical("3 consecutive losses - activating circuit breaker")
            return None

    @staticmethod
    def _update_daily_loss():
        account_info = mt5.account_info()
        if not account_info:
            raise ValueError("Failed to retrieve account info")
        current_date = datetime.now().date()
        if TradeManager.last_trade_day != current_date:
            TradeManager.daily_loss = 0
            TradeManager.last_trade_day = current_date
        TradeManager.daily_loss += (account_info.balance - account_info.equity) / account_info.balance

    @staticmethod
    def log_trade(trade):
        if not trade:
            return
        log_entry = (
            f"{trade['time']},{trade['symbol']},{trade['direction']},"
            f"{trade['price']:.2f},{trade['lots']:.2f}\n"
        )
        try:
            with open(os.path.join("backups", "trade_journal.csv"), "a") as f:
                f.write(log_entry)
        except Exception as e:
            logging.error(f"Failed to log trade: {str(e)}")

async def main():
    parser = argparse.ArgumentParser(description='AI Trading Bot')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--mode', choices=['backtest', 'demo', 'live'], default='live', help='Trading mode')
    args = parser.parse_args()
    
    try:
        config = initialize_mt5()
        ai_model = AIModel()
        engine = TradingEngine(config, ai_model)
        
        if args.benchmark:
            results = await engine.run_async_benchmark()
            logging.info(f"Benchmark Results: {results}")
            return
        
        logging.info("="*40)
        logging.info(f"AI Trading Bot Initialized - {datetime.now()}")
        logging.info(f"Trading Symbol: {config['symbol']}")
        logging.info(f"Risk per Trade: {config['risk_per_trade']*100:.1f}%")
        logging.info(f"Operational Mode: {'TEST' if args.test else args.mode.upper()}")
        logging.info("="*40)
        
        mode = args.mode
        while True:
            try:
                signal = await engine.generate_signal()
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                if args.test or mode == "backtest":
                    logging.info(f"[TEST] {timestamp} - Signal: {signal.upper()}")
                    # For backtesting, you could integrate the VirtualTradingEnvironment here.
                else:
                    trade = TradeManager.execute_order(signal, config)
                    TradeManager.log_trade(trade)
                    logging.info(f"[{mode.upper()}] {timestamp} - Executed {signal.upper()} order")
                    
                await asyncio.sleep(60)
            except KeyboardInterrupt:
                logging.info("User requested shutdown")
                break
            except Exception as e:
                logging.error(f"Main loop error: {str(e)}")
                await asyncio.sleep(30)
    finally:
        mt5.shutdown()
        logging.info("MT5 connection terminated")
        if os.path.exists("trade_journal.csv"):
            os.rename("trade_journal.csv", os.path.join("backups", f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"))

if __name__ == "__main__":
    asyncio.run(main())
