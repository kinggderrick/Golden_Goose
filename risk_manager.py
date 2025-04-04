import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('risk.log'), logging.StreamHandler()]
)

class RiskLimitExceeded(Exception):
    """Custom exception for risk threshold breaches"""
    pass

class PropRiskManager:
    """Advanced risk manager with 3-tier confidence system"""
    
    def __init__(self, config: dict):
        self.config = config
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.max_daily_loss = -0.05  # -5%
        self.max_drawdown = -0.10    # -10%
        self.daily_trades = []
        self.max_daily_trades = 5    # Prop firm limit
        self.confidence_level = "CONFIDENT"  # Default tier
        self.risk_params = {
            "SUPER": {"risk_pct": 0.018, "rr_ratio": 3.0, "sl_mult": 1.5},
            "CONFIDENT": {"risk_pct": 0.008, "rr_ratio": 2.0, "sl_mult": 2.0},
            "HOLD": {"risk_pct": 0.0, "rr_ratio": 0.0, "sl_mult": 0.0}
        }

    def set_confidence_level(self, tier: str):
        """Set current market confidence tier
        Args:
            tier (str): SUPER/CONFIDENT/HOLD
        """
        self.confidence_level = tier.upper()
        logging.info(f"Risk level set to: {self.confidence_level}")

    def calculate_position_size(self, entry_price: float, 
                              stop_price: float,
                              balance: float) -> Tuple[float, float, float]:
        """Calculate tier-based position size with dynamic stops
        
        Args:
            entry_price: Proposed entry price
            stop_price: Initial stop price
            balance: Current account balance
            
        Returns:
            tuple: (lot_size, adjusted_sl, take_profit)
        """
        try:
            params = self.risk_params.get(self.confidence_level, 
                                        self.risk_params["HOLD"])
            
            if params["risk_pct"] <= 0:
                return 0.0, 0.0, 0.0

            # Calculate risk-adjusted position size
            risk_amount = balance * params["risk_pct"]
            atr = self._calculate_atr(period=14)
            price = self._get_current_price()
            
            # Calculate dynamic stop loss
            sl_distance = atr * params["sl_mult"]
            adjusted_sl = entry_price - sl_distance if entry_price > stop_price \
                        else entry_price + sl_distance
                        
            # Calculate lot size
            pip_value = 1.0  # Default for XAUUSD
            if self.config['symbol'] in ['EURUSD', 'GBPUSD']:
                pip_value = 0.0001
                
            lot_size = round(risk_amount / (sl_distance / pip_value), 2)
            
            # Apply symbol constraints
            lot_size = self._apply_lot_limits(lot_size)
            
            # Calculate take profit
            tp_distance = sl_distance * params["rr_ratio"]
            take_profit = entry_price + tp_distance if entry_price > stop_price \
                        else entry_price - tp_distance

            return lot_size, adjusted_sl, take_profit

        except Exception as e:
            logging.error(f"Position calc error: {str(e)}")
            return 0.0, 0.0, 0.0

    def _apply_lot_limits(self, lot_size: float) -> float:
        """Apply exchange/symbol lot size constraints"""
        symbol_info = mt5.symbol_info(self.config['symbol'])
        if not symbol_info:
            return round(lot_size, 2)
            
        min_lot = symbol_info.volume_min
        max_lot = min(symbol_info.volume_max, 5.0)  # Hard cap
        return np.clip(lot_size, min_lot, max_lot)

    def _calculate_atr(self, period: int) -> float:
        """Calculate current ATR with fallback"""
        try:
            rates = mt5.copy_rates_from_pos(
                self.config['symbol'], 
                mt5.TIMEFRAME_H1, 
                0, 
                period+1
            )
            
            if not rates or len(rates) < period:
                return 0.001 * self._get_current_price()
                
            df = pd.DataFrame(rates)
            return talib.ATR(df['high'], df['low'], df['close'], period)[-1]
            
        except Exception as e:
            logging.warning(f"ATR calc failed: {str(e)}")
            return 0.001 * self._get_current_price()

    def _get_current_price(self) -> float:
        """Get current price with redundancy"""
        try:
            tick = mt5.symbol_info_tick(self.config['symbol'])
            return (tick.bid + tick.ask) / 2
        except:
            return mt5.symbol_info(self.config['symbol']).last

    def update_trade_history(self, profit: float):
        """Update performance metrics after trade"""
        self.daily_trades.append({
            'time': pd.Timestamp.now(),
            'profit': profit,
            'confidence': self.confidence_level
        })
        
        # Maintain 24h rolling window
        self.daily_trades = [t for t in self.daily_trades 
                           if pd.Timestamp.now() - t['time'] < pd.Timedelta(hours=24)]
        
        # Update streaks
        if profit > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
        self._enforce_risk_limits()

    def _enforce_risk_limits(self):
        """Check all risk boundaries"""
        if len(self.daily_trades) >= self.max_daily_trades:
            raise RiskLimitExceeded(f"Max daily trades ({self.max_daily_trades}) reached")
            
        drawdown = self._calculate_drawdown()
        if drawdown < self.max_drawdown:
            raise RiskLimitExceeded(f"Max drawdown ({self.max_drawdown*100}%) breached")
            
        daily_pnl = sum(t['profit'] for t in self.daily_trades)
        if daily_pnl < self.max_daily_loss * self.config.get('balance', 10000):
            raise RiskLimitExceeded("Daily loss limit reached")

    def _calculate_drawdown(self) -> float:
        """Calculate current portfolio drawdown"""
        balance = self.config.get('balance', 10000)
        equity = balance + sum(t['profit'] for t in self.daily_trades)
        return (equity - balance) / balance

# Example usage
if __name__ == "__main__":
    config = {'symbol': 'XAUUSD', 'balance': 10000}
    risk_mgr = PropRiskManager(config)
    
    # Super confident trade
    risk_mgr.set_confidence_level("SUPER")
    lot_size, sl, tp = risk_mgr.calculate_position_size(1800, 1795, 10000)
    print(f"SUPER: {lot_size} lots, SL: {sl}, TP: {tp}")
    
    # Confident trade
    risk_mgr.set_confidence_level("CONFIDENT")
    lot_size, sl, tp = risk_mgr.calculate_position_size(1800, 1795, 10000)
    print(f"CONFIDENT: {lot_size} lots, SL: {sl}, TP: {tp}")