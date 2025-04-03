import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import asyncio
import json
import logging
from datetime import datetime
from strategy_engine import HybridStrategy
from ml_models import GoldenGooseModel
from risk_manager import PropRiskManager

# Configure logging
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
        self.timeframe = mt5.TIMEFRAME_M15

    async def fetch_market_data(self) -> pd.DataFrame:
        """Robust data fetcher with error handling"""
        for _ in range(3):
            rates = mt5.copy_rates_from_pos(
                self.symbol,
                self.timeframe,
                0,
                300  # Enough for 4H resampling
            )
            if rates is not None:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                return df
            await asyncio.sleep(1)
        raise ConnectionError("Failed to fetch market data")

    async def generate_signal(self) -> str:
        """Hybrid strategy decision engine"""
        raw_data = await self.fetch_market_data()
        processed_data = self.strategy.calculate_indicators(raw_data)
        long_strat, short_strat = self.strategy.generate_signals(processed_data)
        
        ai_input = self.ai_model.preprocess_data(processed_data)
        ai_pred = self.ai_model.predict_signal(ai_input)
        
        if long_strat and ai_pred['buy_confidence'] > 0.7:
            return 'buy'
        elif short_strat and ai_pred['sell_confidence'] > 0.65:
            return 'sell'
        return 'hold'

class TradeExecutor:
    def __init__(self, config: dict):
        self.config = config
        self.risk_manager = PropRiskManager(config)

    def execute_trade(self, signal: str) -> dict:
        """Prop firm compliant trade execution"""
        tick = mt5.symbol_info_tick(self.config['symbol'])
        if not tick:
            raise ValueError("Invalid price data")
            
        price = tick.ask if signal == 'buy' else tick.bid
        stop_loss = price * 0.995 if signal == 'buy' else price * 1.005
        lot_size = self.risk_manager.calculate_position_size(price, stop_loss)
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config['symbol'],
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": 20,
            "sl": stop_loss,
            "tp": price * 1.02 if signal == 'buy' else price * 0.98,
            "magic": 2024,
            "comment": "GoldenGoose Trade",
            "type_time": mt5.ORDER_TIME_GTC
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Trade failed: {result.comment}")
            
        return {
            'time': datetime.now(),
            'symbol': self.config['symbol'],
            'direction': signal,
            'price': price,
            'lots': lot_size
        }

async def main():
    with open('config.json') as f:
        config = json.load(f)['mt5']
    
    mt5.initialize()
    engine = TradingEngine(config)
    executor = TradeExecutor(config)
    
    try:
        while True:
            signal = await engine.generate_signal()
            if signal != 'hold':
                trade = executor.execute_trade(signal)
                logging.info(f"Executed {trade['direction']} @ {trade['price']:.2f}")
            await asyncio.sleep(60)
            
    finally:
        mt5.shutdown()
        logging.info("Shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())