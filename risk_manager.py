import MetaTrader5 as mt5
import numpy as np

class PropRiskManager:
    def __init__(self, config: dict):
        self.config = config
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.max_daily_loss = -0.05
        self.max_drawdown = -0.10

    def calculate_position_size(self, entry_price: float, stop_price: float) -> float:
        """ATR-based dynamic sizing with 2% risk rule"""
        atr = self._calculate_atr(14)
        risk_amount = self._get_risk_amount()
        return round(risk_amount / (atr * 100), 2)

    def _calculate_atr(self, period: int) -> float:
        rates = mt5.copy_rates_from_pos(
            self.config['symbol'], 
            mt5.TIMEFRAME_H1, 
            0, 
            period+14
        )
        true_ranges = [max(high-low, abs(high-prev_close), abs(low-prev_close)) 
                      for (high, low, prev_close) in zip(rates['high'], rates['low'], rates['close'].shift(1))]
        return np.mean(true_ranges[-period:])

    def _get_risk_amount(self) -> float:
        """Adaptive risk based on performance streak"""
        base_risk = self.config['risk_per_trade']
        equity = mt5.account_info().equity
        
        if self.consecutive_losses >= 2:
            return min(base_risk * 0.5 * equity, 0.01 * equity)
        if self.consecutive_wins >= 3:
            return min(base_risk * 1.5 * equity, 0.03 * equity)
        return base_risk * equity

    def update_trade_history(self, profit: float):
        """Track performance streaks and drawdown"""
        if profit > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        current_drawdown = (mt5.account_info().equity - mt5.account_info().balance) / mt5.account_info().balance
        if current_drawdown < self.max_drawdown:
            raise RiskLimitExceeded(f"Max drawdown {self.max_drawdown*100}% breached")