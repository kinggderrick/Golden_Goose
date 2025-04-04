import pandas as pd
import numpy as np
import datetime
import logging

class VirtualTradingEnvironment:
    def __init__(self, historical_data_path):
        # Load historical data (CSV must contain a 'datetime' column and 'open', 'high', 'low', 'close' prices)
        self.data = pd.read_csv(historical_data_path, parse_dates=['datetime'])
        self.data.sort_values('datetime', inplace=True)
        self.current_index = 0
        self.trades = []
    
    def get_next_bar(self):
        if self.current_index < len(self.data):
            bar = self.data.iloc[self.current_index]
            self.current_index += 1
            return bar
        else:
            return None

    def simulate_order(self, trade):
        # Simulate an order (paper trade)
        self.trades.append(trade)
        logging.info(f"Simulated trade: {trade}")

    def run_backtest(self, signal_function):
        """
        Runs a backtest using the provided signal_function.
        The signal_function should take a DataFrame slice and return 'buy', 'sell', or 'hold'.
        """
        logging.info("Starting backtest...")
        while True:
            bar = self.get_next_bar()
            if bar is None:
                break
            # Use the latest 100 bars for signal generation
            current_slice = self.data.iloc[max(0, self.current_index - 100):self.current_index]
            signal = signal_function(current_slice)
            if signal in ['buy', 'sell']:
                trade = {
                    'datetime': bar['datetime'],
                    'signal': signal,
                    'price': bar['close']  # Using close price as execution price
                }
                self.simulate_order(trade)
        logging.info("Backtest complete.")
        return self.analyze_performance()

    def analyze_performance(self):
        # Simple analysis: count trades per signal type
        report = {}
        for trade in self.trades:
            report[trade['signal']] = report.get(trade['signal'], 0) + 1
        logging.info(f"Performance Report: {report}")
        return report

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Replace 'historical_data.csv' with your actual historical data file path
    env = VirtualTradingEnvironment("historical_data.csv")
    
    def example_signal(df):
        # Example signal: If current price is below 20-period SMA, then 'buy', else 'sell'
        if len(df) < 20:
            return 'hold'
        sma = df['close'].rolling(window=20).mean().iloc[-1]
        price = df['close'].iloc[-1]
        return 'buy' if price < sma else 'sell'
    
    performance_report = env.run_backtest(example_signal)
    print(performance_report)
