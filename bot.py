import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import asyncio
import json
import logging
import time
import sys
import os
from datetime import datetime, timedelta
from strategy_engine import HybridStrategy
from ml_models import GoldenGooseModel
from risk_manager import PropRiskManager, RiskLimitExceeded

# Configure logging with rotation
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s UTC - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bot.log'),
        logging.StreamHandler()
    ]
)

class TradingEngine:
    def __init__(self, config: dict):
        self.config = config
        self.strategy = HybridStrategy(config)
        self.ai_model = GoldenGooseModel()
        self.risk_manager = PropRiskManager(config)
        self.symbol = config['symbol']
        self.timeframe = getattr(mt5, config.get('timeframe', 'TIMEFRAME_M15'), mt5.TIMEFRAME_M15)
        self.last_signal_time = datetime.now() - timedelta(hours=1)
        self.retry_count = 0
        self.max_retries = 5
        self.signal_cooldown = timedelta(minutes=15)  # Prevent signal spamming

    async def fetch_market_data(self) -> pd.DataFrame:
        """Robust data fetcher with error handling"""
        for attempt in range(self.max_retries):
            try:
                rates = mt5.copy_rates_from_pos(
                    self.symbol,
                    self.timeframe,
                    0,
                    300  # Enough for 4H resampling
                )
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    return df
                logging.warning(f"Empty data received, retrying ({attempt+1}/{self.max_retries})")
            except Exception as e:
                logging.error(f"Data fetch error: {str(e)}")
            
            # Exponential backoff
            await asyncio.sleep(2 ** attempt)
            
        self.retry_count += 1
        if self.retry_count >= 3:
            logging.critical("Multiple consecutive data fetch failures")
            # Force reconnection
            mt5.shutdown()
            time.sleep(5)
            self._initialize_mt5()
            self.retry_count = 0
            
        # Return empty DataFrame as fallback
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    
    def _initialize_mt5(self) -> bool:
        """Initialize MT5 connection with credentials"""
        return mt5.initialize(
            login=self.config.get('login'),
            password=self.config.get('password'),
            server=self.config.get('server')
        )

    async def generate_signal(self) -> str:
        """Hybrid strategy decision engine with cooling period"""
        # Check if enough time passed since last signal
        now = datetime.now()
        if now - self.last_signal_time < self.signal_cooldown:
            return 'hold'  # Still in cooldown period
            
        try:
            # Fetch and process data
            raw_data = await self.fetch_market_data()
            if raw_data.empty:
                logging.warning("No data available for signal generation")
                return 'hold'
                
            processed_data = self.strategy.calculate_indicators(raw_data)
            
            # Technical signals
            long_strat, short_strat = self.strategy.generate_signals(processed_data)
            
            # AI confirmation
            ai_input = self.ai_model.preprocess_data(processed_data)
            ai_pred = self.ai_model.predict_signal(ai_input)
            
            # Log prediction confidence
            logging.info(f"AI Confidence - Buy: {ai_pred['buy_confidence']:.2f}, Sell: {ai_pred['sell_confidence']:.2f}")
            
            # Signal generation with thresholds
            if long_strat and ai_pred['buy_confidence'] > 0.7:
                self.last_signal_time = now
                return 'buy'
            elif short_strat and ai_pred['sell_confidence'] > 0.65:
                self.last_signal_time = now
                return 'sell'
                
            return 'hold'
            
        except Exception as e:
            logging.error(f"Signal generation error: {str(e)}", exc_info=True)
            return 'hold'

class TradeExecutor:
    def __init__(self, config: dict):
        self.config = config
        self.risk_manager = PropRiskManager(config)
        self.symbol = config['symbol']
        self.execution_retries = 3

    def execute_trade(self, signal: str) -> dict:
        """Prop firm compliant trade execution with retry logic"""
        for attempt in range(self.execution_retries):
            try:
                # Get latest price
                tick = mt5.symbol_info_tick(self.symbol)
                if not tick:
                    raise ValueError(f"Invalid price data for {self.symbol}")
                    
                price = tick.ask if signal == 'buy' else tick.bid
                
                # Calculate stop loss with minimum distance check
                symbol_info = mt5.symbol_info(self.symbol)
                if not symbol_info:
                    raise ValueError(f"Symbol info not available for {self.symbol}")
                    
                # Ensure minimum stop level compliance
                min_stop_level = symbol_info.point * symbol_info.trade_stops_level
                
                if signal == 'buy':
                    stop_loss = price * 0.995
                    # Ensure minimum distance
                    if price - stop_loss < min_stop_level:
                        stop_loss = price - min_stop_level
                else:
                    stop_loss = price * 1.005
                    # Ensure minimum distance
                    if stop_loss - price < min_stop_level:
                        stop_loss = price + min_stop_level
                
                # Calculate position size
                lot_size = self.risk_manager.calculate_position_size(price, stop_loss)
                
                # Define take profit with 1:2 risk-reward
                take_profit = price * 1.02 if signal == 'buy' else price * 0.98
                
                # Prepare trade request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": lot_size,
                    "type": mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL,
                    "price": price,
                    "deviation": 20,
                    "sl": stop_loss,
                    "tp": take_profit,
                    "magic": 2024,
                    "comment": "GoldenGoose Trade",
                    "type_time": mt5.ORDER_TIME_GTC
                }
                
                # Execute trade
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    error_message = f"Trade failed: {result.comment} (code: {result.retcode})"
                    if attempt < self.execution_retries - 1:
                        logging.warning(f"{error_message} - Retrying ({attempt+1}/{self.execution_retries})")
                        time.sleep(1)
                        continue
                    raise RuntimeError(error_message)
                    
                # Update risk manager
                self.risk_manager.update_trade_history(0)  # Initial profit is 0
                
                # Return trade info
                return {
                    'time': datetime.now(),
                    'symbol': self.symbol,
                    'direction': signal,
                    'price': price,
                    'lots': lot_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
                
            except Exception as e:
                if attempt == self.execution_retries - 1:
                    logging.error(f"Trade execution failed: {str(e)}", exc_info=True)
                    raise
                time.sleep(1)
                
        return {
            'time': datetime.now(),
            'symbol': self.symbol,
            'direction': 'failed',
            'error': 'Execution failed after retries'
        }

async def main():
    """Main bot execution loop with better error handling"""
    try:
        # Load configuration
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        if not os.path.exists(config_path):
            logging.critical(f"Config file not found: {config_path}")
            return
            
        with open(config_path) as f:
            full_config = json.load(f)
            config = full_config.get('mt5', {})
        
        # Validate config
        required_keys = ['symbol', 'risk_per_trade']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            logging.critical(f"Missing required configuration: {', '.join(missing_keys)}")
            return
        
        # Initialize MT5
        if not mt5.initialize(
            login=config.get('login'),
            password=config.get('password'),
            server=config.get('server')
        ):
            logging.critical(f"MT5 initialization failed: {mt5.last_error()}")
            return
            
        # Log account information
        account_info = mt5.account_info()
        if account_info:
            logging.info(f"Connected to account {account_info.login} on {config.get('server')}")
            logging.info(f"Balance: {account_info.balance}, Equity: {account_info.equity}")
        
        # Create trading components
        engine = TradingEngine(config)
        executor = TradeExecutor(config)
        
        # Main trading loop
        logging.info("GoldenGoose trading bot started")
        
        while True:
            try:
                # Generate trading signal
                signal = await engine.generate_signal()
                
                if signal != 'hold':
                    logging.info(f"Signal generated: {signal}")
                    trade = executor.execute_trade(signal)
                    logging.info(f"Executed {trade['direction']} @ {trade.get('price', 0):.2f}")
                
                # Regular status updates
                if datetime.now().minute % 15 == 0:  # Every 15 minutes
                    account_info = mt5.account_info()
                    if account_info:
                        logging.info(f"Account status - Balance: {account_info.balance}, Equity: {account_info.equity}")
                
                # Wait for next cycle
                await asyncio.sleep(60)
                
            except RiskLimitExceeded as e:
                logging.warning(f"Risk limit reached: {str(e)}")
                # Wait longer when risk limits are hit
                await asyncio.sleep(3600)  # 1 hour cooldown
                
            except Exception as e:
                logging.error(f"Trading cycle error: {str(e)}", exc_info=True)
                await asyncio.sleep(300)  # 5 min cooldown on errors
                
    except KeyboardInterrupt:
        logging.info("Bot shutdown requested by user")
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        mt5.shutdown()
        logging.info("MT5 connection closed")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.critical(f"Bot crashed: {str(e)}", exc_info=True)
        sys.exit(1)