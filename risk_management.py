import numpy as np
import pandas as pd

class RiskManagement:
    def __init__(self, data, risk_tolerance=0.02, lookback_period=14):
        """
        Initializes the RiskManagement class with market data, risk tolerance, and lookback period.

        :param data: DataFrame containing historical price data with 'Close' prices.
        :param risk_tolerance: Percentage of account equity to risk on each trade (default is 2%).
        :param lookback_period: Number of periods to calculate volatility (default is 14).
        """
        self.data = data
        self.risk_tolerance = risk_tolerance
        self.lookback_period = lookback_period
        self.calculate_volatility()

    def calculate_volatility(self):
        """
        Calculates the rolling standard deviation of returns to estimate market volatility.
        """
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Returns'].rolling(window=self.lookback_period).std() * np.sqrt(252)  # Annualized volatility

    def calculate_position_size(self, account_equity, entry_price):
        """
        Calculates the position size based on account equity, entry price, and estimated volatility.

        :param account_equity: Total equity available for trading.
        :param entry_price: Price at which the trade is entered.
        :return: Position size (number of units to trade).
        """
        volatility_estimate = self.data['Volatility'].iloc[-1]
        dollar_risk_per_trade = account_equity * self.risk_tolerance
        stop_loss_distance = volatility_estimate * entry_price  # Stop-loss set at 1x volatility
        position_size = dollar_risk_per_trade / stop_loss_distance
        return position_size

    def set_stop_loss(self, entry_price):
        """
        Sets a stop-loss price based on the entry price and estimated volatility.

        :param entry_price: Price at which the trade is entered.
        :return: Stop-loss price.
        """
        volatility_estimate = self.data['Volatility'].iloc[-1]
        stop_loss_distance = volatility_estimate * entry_price
        stop_loss_price = entry_price - stop_loss_distance  # For long positions
        return stop_loss_price

    def set_take_profit(self, entry_price):
        """
        Sets a take-profit price based on the entry price and estimated volatility.

        :param entry_price: Price at which the trade is entered.
        :return: Take-profit price.
        """
        volatility_estimate = self.data['Volatility'].iloc[-1]
        take_profit_distance = volatility_estimate * entry_price
        take_profit_price = entry_price + take_profit_distance  # For long positions
        return take_profit_price
