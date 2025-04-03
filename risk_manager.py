import MetaTrader5 as mt5
import numpy as np
import pandas as pd

class RiskLimitExceeded(Exception):
    """Exception raised when risk limits are breached"""
    pass

class PropRiskManager:
    def __init__(self, config: dict):
        self.config = config
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.max_daily_loss = -0.05
        self.max_drawdown = -0.10
        self.daily_trades = []
        self.max_daily_trades = 5  # Prop firm compliance

    def calculate_position_size(self, entry_price: float, stop_price: float) -> float:
        """ATR-based dynamic sizing with 2% risk rule"""
        # Calculate pip distance between entry and stop
        pip_distance = abs(entry_price - stop_price)
        if pip_distance == 0:
            return 0.01  # Minimum position size as safety
            
        # Get risk amount
        risk_amount = self._get_risk_amount()
        
        # Calculate position size based on risk
        position_size = risk_amount / pip_distance
        
        # Validate against min/max sizes
        symbol_info = mt5.symbol_info(self.config['symbol'])
        if symbol_info:
            min_lot = symbol_info.volume_min
            max_lot = min(symbol_info.volume_max, 5.0)  # Cap at 5 lots for prop compliance
            position_size = max(min_lot, min(position_size, max_lot))
        
        return round(position_size, 2)

    def _calculate_atr(self, period: int) -> float:
        try:
            rates = mt5.copy_rates_from_pos(
                self.config['symbol'], 
                mt5.TIMEFRAME_H1, 
                0, 
                period+1
            )
            
            if rates is None or len(rates) < period:
                return 0.001 * self._get_current_price()  # Fallback to 0.1% of price
                
            df = pd.DataFrame(rates)
            df['prev_close'] = df['close'].shift(1)
            df['tr'] = df.apply(
                lambda x: max(
                    x['high'] - x['low'],
                    abs(x['high'] - x['prev_close']),
                    abs(x['low'] - x['prev_close'])
                ),
                axis=1
            )
            
            return df['tr'].mean()
        except Exception as e:
            print(f"ATR calculation error: {str(e)}")
            return 0.001 * self._get_current_price()  # Fallback to 0.1% of price

    def _get_current_price(self) -> float:
        """Get current price with fallback"""
        tick = mt5.symbol_info_tick(self.config['symbol'])
        if tick:
            return (tick.bid + tick.ask) / 2
        return 1000.0  # Arbitrary fallback for gold

    def _get_risk_amount(self) -> float:
        """Adaptive risk based on performance streak"""
        try:
            base_risk = self.config.get('risk_per_trade', 0.02)  # Default 2%
            account_info = mt5.account_info()
            
            if account_info is None:
                return 500.0  # Fallback risk amount
                
            equity = account_info.equity
            
            # Daily drawdown check
            if self._calculate_daily_drawdown() <= self.max_daily_loss:
                raise RiskLimitExceeded(f"Daily loss limit {self.max_daily_loss*100}% reached")
            
            # Adaptive risk based on streak
            if self.consecutive_losses >= 2:
                return min(base_risk * 0.5 * equity, 0.01 * equity)
            if self.consecutive_wins >= 3:
                return min(base_risk * 1.5 * equity, 0.03 * equity)
                
            return base_risk * equity
        except RiskLimitExceeded:
            raise
        except Exception as e:
            print(f"Risk calculation error: {str(e)}")
            return 500.0  # Fallback risk amount

    def _calculate_daily_drawdown(self) -> float:
        """Calculate intraday drawdown"""
        try:
            # Get today's closed positions
            from datetime import datetime
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Filter today's positions
            today_profits = sum(trade.profit for trade in self.daily_trades)
            
            account_info = mt5.account_info()
            if account_info is None:
                return 0.0
                
            # Calculate drawdown percentage
            starting_balance = account_info.balance - today_profits
            current_equity = account_info.equity
            
            if starting_balance == 0:
                return 0.0
                
            return (current_equity - starting_balance) / starting_balance
        except Exception as e:
            print(f"Daily drawdown calculation error: {str(e)}")
            return 0.0

    def update_trade_history(self, profit: float):
        """Track performance streaks and drawdown"""
        from datetime import datetime
        
        # Add to daily trades
        self.daily_trades.append({
            'time': datetime.now(),
            'profit': profit
        })
        
        # Clean old trades (keep only today)
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.daily_trades = [t for t in self.daily_trades if t['time'] >= today]
        
        # Update streaks
        if profit > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        # Check max daily trades
        if len(self.daily_trades) >= self.max_daily_trades:
            raise RiskLimitExceeded(f"Maximum daily trades limit ({self.max_daily_trades}) reached")

        # Check drawdown
        try:
            account_info = mt5.account_info()
            if account_info:
                current_drawdown = (account_info.equity - account_info.balance) / account_info.balance
                if current_drawdown < self.max_drawdown:
                    raise RiskLimitExceeded(f"Max drawdown {self.max_drawdown*100}% breached")
        except RiskLimitExceeded:
            raise
        except Exception as e:
            print(f"Drawdown check error: {str(e)}")