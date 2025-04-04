import pandas as pd
import numpy as np
import logging
import backtrader as bt
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

class GoldenBacktester(bt.Strategy):
    params = (
        ('sma1_period', 50),
        ('sma2_period', 200),
        ('initial_balance', 100000),
        ('max_leverage', 0.2),
        ('risk_reward_ratio', 2),
    )

    def __init__(self):
        self.strategy = HybridStrategy(config)
        self.ai_model = GoldenGooseModel()
        self.risk_manager = PropRiskManager(config)
        self.sma1 = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma1_period)
        self.sma2 = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma2_period)
        self.crossover = bt.indicators.CrossOver(self.sma1, self.sma2)
        self.trade_history = []
        self.active_trades = []
        self.max_drawdown = 0
        self.trades_taken = 0
        self.profitable_trades = 0
        self.balance = self.params.initial_balance
        self.equity = self.params.initial_balance

    def next(self):
        current_price = self.data.close[0]
        current_time = self.data.datetime.datetime(0)

        # Execute trades based on signals
        if self.crossover > 0 and self.ai_model.predict_signal(self.data) == 'buy':
            self.execute_trade('buy', current_price)
        elif self.crossover < 0 and self.ai_model.predict_signal(self.data) == 'sell':
            self.execute_trade('sell', current_price)

        # Update positions with current price and timestamp
        self.update_positions(current_price, current_time)

    def execute_trade(self, signal: str, price: float):
        """Prop firm compliant trade execution"""
        stop_loss = price * 0.995 if signal == 'buy' else price * 1.005
        tp_multiplier = 1.02 if signal == 'buy' else 0.98  # Default 1:2 RR

        position_size = self.risk_manager.calculate_position_size(price, stop_loss)

        # Ensure we don't overleverage
        max_allowed_size = self.broker.get_cash() * self.params.max_leverage / price  # Max leverage
        position_size = min(position_size, max_allowed_size)

        if signal == 'buy':
            self.buy(size=position_size)
        elif signal == 'sell':
            self.sell(size=position_size)

        trade = {
            'entry': price,
            'size': position_size,
            'direction': 'long' if signal == 'buy' else 'short',
            'sl': stop_loss,
            'tp': price * tp_multiplier,
            'open_time': datetime.now()
        }

        self.active_trades.append(trade)
        self.trades_taken += 1

    def update_positions(self, current_price: float, current_time=None):
        """Update equity and check stops/targets"""
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
                self.broker.get_cash() += value  # Add/subtract profit/loss

                if value > 0:
                    self.profitable_trades += 1

                # Add to trade history and mark for removal
                self.trade_history.append(copy.deepcopy(trade))
                closed_trades.append(trade)

        # Remove closed trades from active trades
        for trade in closed_trades:
            self.active_trades.remove(trade)

        # Track max drawdown
        equity = self.broker.get_cash()
        drawdown = (equity - self.params.initial_balance) / self.params.initial_balance
        self.max_drawdown = min(self.max_drawdown, drawdown)

    def stop(self):
        # Log final performance metrics
        logging.info(f"Max Drawdown: {self.max_drawdown:.2%}")
        logging.info(f"Final Equity: ${self.broker.get_cash():,.2f}")
        logging.info(f"Win Rate: {self.profitable_trades / self.trades_taken:.2%}")
        logging.info(f"Profit Factor: {sum(t['profit'] for t in self.trade_history if t['profit'] > 0) / abs(sum(t['profit'] for t in self.trade_history if t['profit'] < 0) or 1):.2f}")
        logging.info(f"Trades Taken: {self.trades_taken}")

if __name__ == '__main__':
    import os
    import json
    import backtrader as bt

    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    os.makedirs('backtests', exist_ok=True)

    with open('config.json') as f:
        config = json.load(f)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(GoldenBacktester, config=config)

    # Load data
    data = bt.feeds.YahooFinanceData(dataname='historical_data.csv')
    cerebro.adddata(data)

    # Set initial cash
    cerebro.broker.set_cash(config.get('initial_balance', 100000))

    # Set commission
    cerebro.broker.set_commission(commission=0.001)

    # Set slippage
    cerebro.broker.set_slippage_perc(0.001)

    # Set position size
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    # Run the backtest
    cerebro.run()

    # Plot the results
    cerebro.plot()
