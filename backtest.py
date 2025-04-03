# backtest.py - FINAL VERSION
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from strategy_engine import HybridStrategy
from ml_models import GoldenGooseModel
from risk_manager import PropRiskManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest.log'),
        logging.StreamHandler()
    ]
)

class GoldenBacktester:
    def __init__(self, config: dict, initial_balance: float = 100000):
        self.strategy = HybridStrategy(config)
        self.ai_model = GoldenGooseModel()
        self.risk_manager = PropRiskManager(config)
        self.balance = initial_balance
        self.equity = initial_balance
        self.trade_history = []
        self.max_drawdown = 0

    def execute_trade(self, signal: str, price: float):
        """Prop firm compliant trade execution"""
        stop_loss = price * 0.995 if signal == 'buy' else price * 1.005
        position_size = self.risk_manager.calculate_position_size(price, stop_loss)
        
        if signal == 'buy':
            self.balance -= price * position_size
            self.trade_history.append({
                'entry': price,
                'size': position_size,
                'direction': 'long',
                'sl': stop_loss,
                'tp': price * 1.02  # 1:2 RR
            })
        elif signal == 'sell':
            self.balance -= price * position_size * 0.05  # Margin
            self.trade_history.append({
                'entry': price,
                'size': position_size,
                'direction': 'short',
                'sl': stop_loss,
                'tp': price * 0.98  # 1:2 RR
            })

    def update_positions(self, current_price: float):
        """Update equity and check stops/targets"""
        self.equity = self.balance
        for trade in self.trade_history:
            if trade['direction'] == 'long':
                value = (current_price - trade['entry']) * trade['size']
                if current_price <= trade['sl'] or current_price >= trade['tp']:
                    self.equity += value
                    self.trade_history.remove(trade)
            else:
                value = (trade['entry'] - current_price) * trade['size']
                if current_price >= trade['sl'] or current_price <= trade['tp']:
                    self.equity += value
                    self.trade_history.remove(trade)

        # Track max drawdown
        drawdown = (self.equity - initial_balance) / initial_balance
        self.max_drawdown = min(self.max_drawdown, drawdown)

    def run_backtest(self, data_path: str) -> dict:
        """Complete strategy validation"""
        df = pd.read_csv(data_path, parse_dates=['time'])
        results = []
        
        for i in range(300, len(df)):  # 300 bars minimum for indicators
            window = df.iloc[i-300:i]
            processed = self.strategy.calculate_indicators(window)
            long, short = self.strategy.generate_signals(processed)
            ai_input = self.ai_model.preprocess_data(processed)
            ai_pred = self.ai_model.predict_signal(ai_input)
            
            current_price = window.iloc[-1]['close']
            
            if long and ai_pred['buy_confidence'] > 0.7:
                self.execute_trade('buy', current_price)
            elif short and ai_pred['sell_confidence'] > 0.65:
                self.execute_trade('sell', current_price)
                
            self.update_positions(current_price)
            results.append({
                'timestamp': window.iloc[-1]['time'],
                'balance': self.balance,
                'equity': self.equity,
                'drawdown': self.max_drawdown
            })
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    import json
    with open('config.json') as f:
        config = json.load(f)
    
    backtester = GoldenBacktester(config)
    results = backtester.run_backtest('historical_data.csv')
    results.to_csv('backtests/latest_results.csv', index=False)
    print(f"Max Drawdown: {backtester.max_drawdown:.2%}")
    print(f"Final Equity: ${backtester.equity:,.2f}")