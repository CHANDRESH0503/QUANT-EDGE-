# tests/test_risk.py
# Tests for position sizing, exit engine, circuit breaker
# Risk management bugs cost real money — test everything twice

import unittest
import sqlite3
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from risk.capital_mode   import CapitalMode
from risk.position_sizer import PositionSizer
from risk.circuit_breaker import CircuitBreaker


class TestCapitalMode(unittest.TestCase):

    def test_small_capital_detection(self):
        self.assertEqual(CapitalMode.detect(10_000),  "SMALL")
        self.assertEqual(CapitalMode.detect(49_999),  "SMALL")

    def test_growing_capital_detection(self):
        self.assertEqual(CapitalMode.detect(50_000),  "GROWING")
        self.assertEqual(CapitalMode.detect(199_999), "GROWING")

    def test_full_capital_detection(self):
        self.assertEqual(CapitalMode.detect(200_000), "FULL")
        self.assertEqual(CapitalMode.detect(1_000_000),"FULL")

    def test_config_returns_all_keys(self):
        for mode_cap in [10_000, 100_000, 500_000]:
            cfg = CapitalMode.get_config(mode_cap)
            for key in ["mode","max_risk_pct","max_positions","min_conf"]:
                self.assertIn(key, cfg, f"Key {key} missing for capital={mode_cap}")

    def test_small_risk_lower_than_full(self):
        small_risk = CapitalMode.MODES["SMALL"]["max_risk_pct"]
        full_risk  = CapitalMode.MODES["FULL"]["max_risk_pct"]
        self.assertLess(small_risk, full_risk)

    def test_max_risk_amount(self):
        risk = CapitalMode.max_risk_amount(100_000)
        self.assertGreater(risk, 0)
        self.assertLessEqual(risk, 100_000 * 0.02)


class TestPositionSizer(unittest.TestCase):

    def setUp(self):
        self.sizer = PositionSizer()

    def test_basic_long_calculation(self):
        result = self.sizer.calculate(
            signal="LONG", capital=100_000,
            current_price=1842.0, atr=28.4,
            capital_mode="FULL",
        )
        self.assertEqual(result["signal"], "LONG")
        self.assertGreater(result["shares"], 0)
        self.assertLess(result["stop_loss"], 1842.0)
        self.assertGreater(result["target"], 1842.0)
        self.assertGreaterEqual(result["reward_risk"], 2.0)

    def test_basic_short_calculation(self):
        result = self.sizer.calculate(
            signal="SHORT", capital=100_000,
            current_price=1842.0, atr=28.4,
            capital_mode="FULL",
        )
        self.assertEqual(result["signal"], "SHORT")
        self.assertGreater(result["stop_loss"], 1842.0)
        self.assertLess(result["target"], 1842.0)

    def test_risk_amount_within_limit(self):
        result = self.sizer.calculate(
            signal="LONG", capital=100_000,
            current_price=1842.0, atr=28.4,
            capital_mode="FULL",
        )
        max_risk = 100_000 * CapitalMode.MODES["FULL"]["max_risk_pct"]
        self.assertLessEqual(result["risk_amount"], max_risk * 1.01)

    def test_position_cap_40pct_capital(self):
        result = self.sizer.calculate(
            signal="LONG", capital=100_000,
            current_price=1842.0, atr=2.0,   # tiny ATR → huge shares → cap kicks in
            capital_mode="FULL",
        )
        self.assertLessEqual(result["position_value"], 100_000 * 0.41)

    def test_size_multiplier_reduces_position(self):
        full = self.sizer.calculate(
            signal="LONG", capital=100_000,
            current_price=1842.0, atr=28.4,
            size_multiplier=1.0, capital_mode="FULL",
        )
        half = self.sizer.calculate(
            signal="LONG", capital=100_000,
            current_price=1842.0, atr=28.4,
            size_multiplier=0.5, capital_mode="FULL",
        )
        self.assertLessEqual(half["shares"], full["shares"])

    def test_zero_size_multiplier_returns_zero_shares(self):
        result = self.sizer.calculate(
            signal="LONG", capital=100_000,
            current_price=1842.0, atr=28.4,
            size_multiplier=0.0, capital_mode="FULL",
        )
        self.assertEqual(result["shares"], 0)

    def test_small_capital_gets_fewer_shares(self):
        small = self.sizer.calculate(
            signal="LONG", capital=20_000,
            current_price=1842.0, atr=28.4,
            capital_mode="SMALL",
        )
        full_cap = self.sizer.calculate(
            signal="LONG", capital=500_000,
            current_price=1842.0, atr=28.4,
            capital_mode="FULL",
        )
        self.assertLessEqual(small["shares"], full_cap["shares"])

    def test_handles_zero_price(self):
        result = self.sizer.calculate(
            signal="LONG", capital=100_000,
            current_price=0, atr=28.4,
        )
        self.assertEqual(result["shares"], 0)

    def test_handles_zero_atr(self):
        result = self.sizer.calculate(
            signal="LONG", capital=100_000,
            current_price=1842.0, atr=0,
        )
        self.assertEqual(result["shares"], 0)


class TestCircuitBreaker(unittest.TestCase):

    def setUp(self):
        self.cb = CircuitBreaker()

    def test_normal_conditions_ok(self):
        allowed, status = self.cb.check(
            capital_mode="FULL", monthly_dd_pct=-0.01,
            consecutive_losses=0,
        )
        self.assertTrue(allowed)
        self.assertEqual(status["level"], "OK")

    def test_monthly_halt_full_mode(self):
        allowed, status = self.cb.check(
            capital_mode="FULL", monthly_dd_pct=-0.12,
            consecutive_losses=0,
        )
        self.assertFalse(allowed)
        self.assertEqual(status["level"], "HALT")

    def test_monthly_pause_full_mode(self):
        allowed, status = self.cb.check(
            capital_mode="FULL", monthly_dd_pct=-0.11,
            consecutive_losses=0,
        )
        self.assertFalse(allowed)
        self.assertIn(status["level"], ("PAUSE", "HALT"))

    def test_small_capital_lower_threshold(self):
        # SMALL halts at -8% vs FULL at -15%
        allowed_small, _ = self.cb.check(
            capital_mode="SMALL", monthly_dd_pct=-0.09,
            consecutive_losses=0,
        )
        allowed_full, _  = self.cb.check(
            capital_mode="FULL", monthly_dd_pct=-0.09,
            consecutive_losses=0,
        )
        self.assertFalse(allowed_small, "SMALL should halt at -9%")
        self.assertTrue(allowed_full,   "FULL should not halt at -9%")

    def test_three_consecutive_losses_pause(self):
        allowed, status = self.cb.check(
            capital_mode="FULL", monthly_dd_pct=-0.01,
            consecutive_losses=3,
        )
        self.assertFalse(allowed)
        self.assertIn(status["level"], ("PAUSE", "HALT"))

    def test_five_consecutive_losses_halt(self):
        allowed, status = self.cb.check(
            capital_mode="FULL", monthly_dd_pct=-0.01,
            consecutive_losses=5,
        )
        self.assertFalse(allowed)
        self.assertEqual(status["level"], "HALT")

    def test_high_anomaly_pauses(self):
        allowed, status = self.cb.check(
            capital_mode="FULL", monthly_dd_pct=-0.01,
            consecutive_losses=0, anomaly_severity="HIGH",
        )
        self.assertFalse(allowed)

    def test_low_model_accuracy_halts(self):
        allowed, status = self.cb.check(
            capital_mode="FULL", monthly_dd_pct=-0.01,
            consecutive_losses=0, model_accuracy_10=0.38,
        )
        self.assertFalse(allowed)
        self.assertEqual(status["level"], "HALT")

    def test_warn_level_still_allows_trading(self):
        allowed, status = self.cb.check(
            capital_mode="FULL", monthly_dd_pct=-0.05,
            consecutive_losses=1,
        )
        self.assertTrue(allowed)
        self.assertEqual(status["level"], "WARN")

    def test_size_multiplier_zero_on_halt(self):
        _, status = self.cb.check(
            capital_mode="FULL", monthly_dd_pct=-0.20,
            consecutive_losses=0,
        )
        self.assertEqual(status["size_multiplier"], 0.0)

    def test_size_multiplier_reduced_on_warn(self):
        _, status = self.cb.check(
            capital_mode="FULL", monthly_dd_pct=-0.05,
            consecutive_losses=0,
        )
        self.assertLess(status["size_multiplier"], 1.0)

    def test_status_contains_all_keys(self):
        _, status = self.cb.check(
            capital_mode="FULL", monthly_dd_pct=-0.02,
            consecutive_losses=1,
        )
        for key in ["level","trading_allowed","size_multiplier",
                    "monthly_dd_pct","consecutive_losses","warnings"]:
            self.assertIn(key, status)


class TestPositionSizerRewardRisk(unittest.TestCase):
    """Dedicated R:R ratio tests — this is the most important metric."""

    def setUp(self):
        self.sizer = PositionSizer()

    def test_reward_risk_at_least_minimum(self):
        for mode in ["SMALL", "GROWING", "FULL"]:
            result = self.sizer.calculate(
                signal="LONG", capital=100_000,
                current_price=1842.0, atr=28.4,
                capital_mode=mode,
            )
            self.assertGreaterEqual(
                result["reward_risk"],
                PositionSizer.REWARD_RISK,
                f"R:R below minimum for {mode} mode",
            )

    def test_stop_always_below_entry_for_long(self):
        result = self.sizer.calculate(
            signal="LONG", capital=100_000,
            current_price=1842.0, atr=28.4,
        )
        self.assertLess(result["stop_loss"], result["entry"])

    def test_target_always_above_entry_for_long(self):
        result = self.sizer.calculate(
            signal="LONG", capital=100_000,
            current_price=1842.0, atr=28.4,
        )
        self.assertGreater(result["target"], result["entry"])

    def test_stop_always_above_entry_for_short(self):
        result = self.sizer.calculate(
            signal="SHORT", capital=100_000,
            current_price=1842.0, atr=28.4,
        )
        self.assertGreater(result["stop_loss"], result["entry"])

    def test_target_always_below_entry_for_short(self):
        result = self.sizer.calculate(
            signal="SHORT", capital=100_000,
            current_price=1842.0, atr=28.4,
        )
        self.assertLess(result["target"], result["entry"])


if __name__ == "__main__":
    unittest.main(verbosity=2)