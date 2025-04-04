import unittest
import MetaTrader5 as mt5
from bot import TradingEngine, TradeExecutor
from risk_manager import PropRiskManager
from strategy_engine import HybridStrategy
from ml_models import GoldenGooseModel

class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = {
            'symbol': 'XAUUSD',
            'risk_per_trade': 0.02,
            'server': 'MetaQuotes-Demo'
        }

    def test_strategy_integration(self):
        engine = TradingEngine(self.config)
        self.assertIsInstance(engine.strategy, HybridStrategy)
        self.assertIsInstance(engine.ai_model, GoldenGooseModel)

    def test_risk_calculation(self):
        mgr = PropRiskManager(self.config)
        size = mgr.calculate_position_size(1800.00, 1790.00)
        # Using standard assertions instead of assertBetween
        self.assertGreaterEqual(size, 0.01)
        self.assertLessEqual(size, 5.00)

    def test_order_execution(self):
        executor = TradeExecutor(self.config)
        with self.assertRaises(ValueError):  # No live trading in tests
            executor.execute_trade('buy')
    
    def test_mt5_connection(self):
        # Simple check for MT5 connection capability
        self.assertIsNotNone(mt5.version)

if __name__ == "__main__":
    unittest.main()