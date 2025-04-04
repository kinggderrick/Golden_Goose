# backtest.py - FINAL VERSION
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from strategy_engine import HybridStrategy
from ml_models import GoldenGooseModel
from risk_manager import PropRiskManager
import copy

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
        self.initial_balance = initial_balance  # Store initial balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.trade_history = []
        self.active_trades = []  # Separate list for active trades
        self.max_drawdown = 0
        self.trades_taken = 0
        self.profitable_trades = 0

    def execute_trade(self, signal: str, price: float):
        """Prop firm compliant trade execution"""
        stop_loss = price * 0.995 if signal == 'buy' else price * 1.005
        # Allow customization of risk:reward through config
        tp_multiplier = 1.02 if signal == 'buy' else 0.98  # Default 1:2 RR
        
        position_size = self.risk_manager.calculate_position_size(price, stop_loss)
        
        # Ensure we don't overleverage
        max_allowed_size = self.balance * 0.2 / price  # Max 20% balance per trade
        position_size = min(position_size, max_allowed_size)
        
        trade = {
            'entry': price,
            'size': position_size,
            'direction': 'long' if signal == 'buy' else 'short',
            'sl': stop_loss,
            'tp': price * tp_multiplier,
            'open_time': datetime.now()
        }
        
        # Update balance (accounting for margin)
        if signal == 'buy':
            self.balance -= price * position_size * 0.05  # Using consistent margin model
        elif signal == 'sell':
            self.balance -= price * position_size * 0.05  # Margin
            
        self.active_trades.append(trade)
        self.trades_taken += 1
        
        return trade

    def update_positions(self, current_price: float, current_time=None):
        """Update equity and check stops/targets"""
        self.equity = self.balance
        closed_trades = []
        
        for trade in self.active_trades:
            # Calculate current trade value
            if trade['direction'] == 'long':
                value = (current_price - trade['entry']) * trade['size']
                is_stopped = current_price <= trade['sl']
                is_target = current_price >= trade['tp']
            else:  # short
                value = (trade['entry'] - current_price) * trade['size']
                is_stopped = current_price >= trade['sl']
                is_target = current_price <= trade['tp']
                
            # Check if trade should be closed
            if is_stopped or is_target:
                # Close the trade
                trade['exit_price'] = current_price
                trade['exit_time'] = current_time if current_time else datetime.now()
                trade['profit'] = value
                trade['successful'] = is_target  # Was target hit?
                
                # Update account
                self.balance += trade['entry'] * trade['size'] * 0.05  # Return margin
                self.balance += value  # Add/subtract profit/loss
                
                if value > 0:
                    self.profitable_trades += 1
                    
                # Add to trade history and mark for removal
                self.trade_history.append(copy.deepcopy(trade))
                closed_trades.append(trade)
            else:
                # Add unrealized P/L to equity
                self.equity += value
        
        # Remove closed trades from active trades
        for trade in closed_trades:
            self.active_trades.remove(trade)

        # Track max drawdown
        drawdown = (self.equity - self.initial_balance) / self.initial_balance
        self.max_drawdown = min(self.max_drawdown, drawdown)

    def run_backtest(self, data_path: str) -> dict:
        """Complete strategy validation"""
        df = pd.read_csv(data_path, parse_dates=['time'])
        results = []
        
        # Initialize counters for performance metrics
        trades_taken = 0
        win_count = 0
        
        for i in range(300, len(df)):  # 300 bars minimum for indicators
            window = df.iloc[i-300:i]
            processed = self.strategy.calculate_indicators(window)
            long, short = self.strategy.generate_signals(processed)
            ai_input = self.ai_model.preprocess_data(processed)
            ai_pred = self.ai_model.predict_signal(ai_input)
            
            current_price = window.iloc[-1]['close']
            current_time = window.iloc[-1]['time']
            
            # Execute trades based on signals
            if long and ai_pred['buy_confidence'] > 0.7:
                self.execute_trade('buy', current_price)
            elif short and ai_pred['sell_confidence'] > 0.65:
                self.execute_trade('sell', current_price)
                
            # Update positions with current price and timestamp
            self.update_positions(current_price, current_time)
            
            # Record state
            results.append({
                'timestamp': current_time,
                'balance': self.balance,
                'equity': self.equity,
                'drawdown': self.max_drawdown,
                'open_trades': len(self.active_trades)
            })
            
        # Calculate final performance metrics
        win_rate = self.profitable_trades / self.trades_taken if self.trades_taken > 0 else 0
        profit_factor = abs(sum(t['profit'] for t in self.trade_history if t['profit'] > 0)) / \
                       abs(sum(t['profit'] for t in self.trade_history if t['profit'] < 0) or 1)
                       
        summary = {
            'max_drawdown': self.max_drawdown,
            'final_equity': self.equity,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trades_taken': self.trades_taken
        }
        
        logging.info(f"Backtest summary: {summary}")
        return pd.DataFrame(results), summary

if __name__ == "__main__":
    import json
    import os
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    os.makedirs('backtests', exist_ok=True)
    
    with open('config.json') as f:
        config = json.load(f)
    
    backtester = GoldenBacktester(config)
    results, summary = backtester.run_backtest('historical_data.csv')
    
    # Save detailed results
    results.to_csv('backtests/latest_results.csv', index=False)
    
    # Save trade history
    trade_df = pd.DataFrame(backtester.trade_history)
    if not trade_df.empty:
        trade_df.to_csv('backtests/latest_trades.csv', index=False)
    
    # Print summary
    print(f"Max Drawdown: {summary['max_drawdown']:.2%}")
    print(f"Final Equity: ${summary['final_equity']:,.2f}")
    print(f"Win Rate: {summary['win_rate']:.2%}")
    print(f"Profit Factor: {summary['profit_factor']:.2f}")
    print(f"Trades Taken: {summary['trades_taken']}")