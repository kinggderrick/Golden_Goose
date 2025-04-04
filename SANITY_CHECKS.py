import unittest
import pandas as pd
from strategy_engine import HybridStrategy
from risk_manager import PropRiskManager

class ConfidenceSystemTest(unittest.TestCase):
    def test_confidence_tiers(self):
        strategy = HybridStrategy({
            'confidence_thresholds': {
                'super': {'ml_prob': 0.75, 'sentiment': 0.6, 'vol_ratio': 1.1},
                'confident': {'ml_prob': 0.55, 'sentiment': 0.35, 'vol_ratio': 0.9}
            }
        })
        
        # Mock data for SUPER confidence
        df = pd.DataFrame({
            'rsi': [25], 'lower_bb': [1700], 'macd': [2.5],
            'daily_vol': [1.8], 'volume': [1500], 'h4_rsi': [45]
        })
        
        _, _, tier = strategy.generate_signals(df)
        self.assertEqual(tier, "SUPER")

    def test_risk_adjustments(self):
        risk_mgr = PropRiskManager({'symbol': 'XAUUSD'})
        
        # Test SUPER tier
        risk_mgr.set_confidence_level("SUPER")
        size, _, _ = risk_mgr.calculate_position_size(1800, 1790, 10000)
        self.assertGreaterEqual(size, 0.5)
        
        # Test CONFIDENT tier
        risk_mgr.set_confidence_level("CONFIDENT")
        size, _, _ = risk_mgr.calculate_position_size(1800, 1790, 10000)
        self.assertLessEqual(size, 2.0)