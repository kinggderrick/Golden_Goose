import unittest
import MetaTrader5 as mt5
from bot import TradingEngine, TradeExecutor
from risk_manager import PropRiskManager
from strategy_engine import HybridStrategy

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
        self.assertBetween(size, 0.01, 5.00)

    def test_order_execution(self):
        executor = TradeExecutor(self.config)
        with self.assertRaises(ValueError):  # No live trading in tests
            executor.execute_trade('buy')

if __name__ == "__main__":
    unittest.main()