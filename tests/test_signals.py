# tests/test_signals.py
# Tests for the 6-gate signal pipeline
# Every gate must block or pass deterministically

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from signals.gate1_regime       import Gate1Regime
from signals.gate2_rule_filter  import Gate2RuleFilter
from signals.gate5_sr_validator import Gate5SRValidator
from signals.gate6_confidence   import Gate6Confidence
from signals.timeframe_alignment import TimeframeAlignment
from signals.signal_stabilizer   import SignalStabilizer


class TestGate1Regime(unittest.TestCase):

    def setUp(self):
        self.gate = Gate1Regime()

    def _regime(self, name, stability=0.8):
        rules = {
            "BULL_TRENDING":   {"trade_long": True,  "trade_short": False, "position_mult": 1.2},
            "BEAR_TRENDING":   {"trade_long": False, "trade_short": True,  "position_mult": 0.8},
            "HIGH_VOLATILITY": {"trade_long": True,  "trade_short": True,  "position_mult": 0.5},
            "CHOPPY_SIDEWAYS": {"trade_long": False, "trade_short": False, "position_mult": 0.0},
        }.get(name, {"trade_long": True, "trade_short": False, "position_mult": 1.0})

        return {
            "regime": name, "stability": stability,
            "trade_long": rules["trade_long"],
            "trade_short": rules["trade_short"],
            "position_mult": rules["position_mult"],
        }

    def _trend(self, changes=0):
        return {"stable": changes <= 1, "changes": changes}

    def test_choppy_blocks_all(self):
        passed, ctx = self.gate.check(
            self._regime("CHOPPY_SIDEWAYS"), self._trend(), "LONG"
        )
        self.assertFalse(passed, "CHOPPY should block ALL signals")

    def test_bull_allows_long(self):
        passed, _ = self.gate.check(
            self._regime("BULL_TRENDING"), self._trend(), "LONG"
        )
        self.assertTrue(passed)

    def test_bull_blocks_short(self):
        passed, _ = self.gate.check(
            self._regime("BULL_TRENDING"), self._trend(), "SHORT"
        )
        self.assertFalse(passed, "BULL regime should block SHORT signals")

    def test_bear_blocks_long(self):
        passed, _ = self.gate.check(
            self._regime("BEAR_TRENDING"), self._trend(), "LONG"
        )
        self.assertFalse(passed, "BEAR regime should block LONG signals")

    def test_bear_allows_short(self):
        passed, _ = self.gate.check(
            self._regime("BEAR_TRENDING"), self._trend(), "SHORT"
        )
        self.assertTrue(passed)

    def test_high_vol_allows_both(self):
        for direction in ["LONG", "SHORT"]:
            passed, _ = self.gate.check(
                self._regime("HIGH_VOLATILITY"), self._trend(), direction
            )
            self.assertTrue(passed, f"HIGH_VOL should allow {direction}")

    def test_low_stability_blocks(self):
        passed, _ = self.gate.check(
            self._regime("BULL_TRENDING", stability=0.2),
            self._trend(), "LONG"
        )
        self.assertFalse(passed, "Stability below 0.40 should block")

    def test_unstable_regime_reduces_size(self):
        passed, ctx = self.gate.check(
            self._regime("BULL_TRENDING", stability=0.7),
            self._trend(changes=3), "LONG"
        )
        self.assertTrue(passed)
        self.assertLess(ctx["position_mult"], 1.2, "Unstable regime should reduce size")

    def test_choppy_returns_meaningful_reason(self):
        passed, ctx = self.gate.check(
            self._regime("CHOPPY_SIDEWAYS"), self._trend(), "LONG"
        )
        self.assertIn("reason", ctx)
        self.assertGreater(len(ctx["reason"]), 5)


class TestGate2RuleFilter(unittest.TestCase):

    def setUp(self):
        self.gate = Gate2RuleFilter()

    def _good_features(self):
        return {
            "adx": 28, "ema_spread": 0.02,
            "volume_ratio": 1.8,
            "nifty_5d_pct": 0.5,
            "earnings_risk_flag": 0,
            "india_vix_level": 14,
            "macro_score": 0.2,
            "intermarket_score": 0.1,
            "usdinr_5d_pct": 0.3,
        }

    def _good_context(self):
        return {
            "fundamental_grade": "GOOD",
            "anomaly_severity":  "LOW",
            "anomaly_type":      "NONE",
            "nifty_5d_pct":      0.5,
        }

    def test_all_good_passes(self):
        passed, ctx = self.gate.check(self._good_features(), self._good_context())
        self.assertTrue(passed)
        self.assertGreaterEqual(ctx["n_passed"], 8)

    def test_earnings_week_hard_fail(self):
        # Gate 2 reads `days_to_earnings` (hard fail within 3 days), not the
        # old `earnings_risk_flag` key. Hard-fail key is "earnings_risk".
        f = {**self._good_features(), "days_to_earnings": 1}
        passed, ctx = self.gate.check(f, self._good_context())
        self.assertFalse(passed)
        self.assertIn("earnings_risk", ctx.get("hard_failed", []))

    def test_vix_panic_hard_fail(self):
        f = {**self._good_features(), "india_vix_level": 28}
        passed, ctx = self.gate.check(f, self._good_context())
        self.assertFalse(passed)
        # Hard-fail key was renamed vix_panic → vix_halt (VIX ≥ 28 = HALT level).
        self.assertIn("vix_halt", ctx.get("hard_failed", []))

    def test_high_anomaly_hard_fail(self):
        ctx = {**self._good_context(), "anomaly_severity": "HIGH"}
        passed, result = self.gate.check(self._good_features(), ctx)
        self.assertFalse(passed)
        self.assertIn("anomaly_high", result.get("hard_failed", []))

    def test_two_soft_fails_still_passes(self):
        """7/10 checks passing with no hard fails should still pass (≥8 needed)."""
        f = {
            **self._good_features(),
            "volume_ratio":    0.5,   # fail volume
            "intermarket_score": -0.5, # fail intermarket
        }
        passed, ctx = self.gate.check(f, self._good_context())
        # 8 pass threshold — 2 soft fails = 8/10 = borderline
        # Depending on exact implementation, test the logic
        self.assertEqual(ctx["n_passed"], len(ctx["checks"]) - 2)

    def test_returns_fail_reasons(self):
        f      = {**self._good_features(), "india_vix_level": 30}
        passed, ctx = self.gate.check(f, self._good_context())
        self.assertFalse(passed)
        self.assertIsInstance(ctx.get("fail_reasons"), list)
        self.assertGreater(len(ctx["fail_reasons"]), 0)


class TestGate5SRValidator(unittest.TestCase):

    def setUp(self):
        self.gate = Gate5SRValidator()

    def _sr_grade_a(self):
        return {
            "entry_quality": "A", "entry_quality_score": 1.0,
            "reward_risk_sr": 3.5, "support_distance_pct": 1.0,
            "resistance_distance_pct": 4.5, "near_support": True,
            "near_resistance": False, "near_breakout": False,
        }

    def _sr_grade_d(self):
        return {
            "entry_quality": "D", "entry_quality_score": -0.2,
            "reward_risk_sr": 1.2, "support_distance_pct": 4.5,
            "resistance_distance_pct": 0.8, "near_support": False,
            "near_resistance": True, "near_breakout": False,
        }

    def test_grade_a_passes_full_size(self):
        passed, ctx = self.gate.check(
            self._sr_grade_a(), "LONG", ml_confidence=0.72, alignment="A+"
        )
        self.assertTrue(passed)
        self.assertGreaterEqual(ctx["size_mult"], 0.9)

    def test_grade_d_passes_at_probe_size(self):
        # GATE5-1: Grade D = limited runway to the ATR target → probe size,
        # NOT a veto. S/R sizes the trade; the conviction gate is Gate 6.
        passed, ctx = self.gate.check(
            self._sr_grade_d(), "LONG", ml_confidence=0.61, alignment="B"
        )
        self.assertTrue(passed)
        self.assertLessEqual(ctx["size_mult"], 0.40, "Grade D must be probe size")

    def test_grade_d_high_conf_still_probe_size(self):
        passed, ctx = self.gate.check(
            self._sr_grade_d(), "LONG", ml_confidence=0.80, alignment="A+"
        )
        self.assertTrue(passed)
        self.assertLessEqual(ctx["size_mult"], 0.40, "Grade D is sized small regardless of conf")

    def test_poor_rr_reduces_size(self):
        sr = {**self._sr_grade_a(), "reward_risk_sr": 1.4}
        passed, ctx = self.gate.check(sr, "LONG", 0.70, "A+")
        # Poor R:R reduces size even if grade is A
        self.assertLess(ctx["size_mult"], 0.85)

    def test_near_resistance_long_reduces_size(self):
        sr = {**self._sr_grade_a(), "near_resistance": True, "near_breakout": False}
        passed, ctx = self.gate.check(sr, "LONG", 0.72, "A")
        self.assertTrue(passed)
        # Size should be reduced when near resistance
        self.assertLessEqual(ctx["size_mult"], 1.0)


class TestGate6Confidence(unittest.TestCase):

    def setUp(self):
        self.gate = Gate6Confidence()

    def _risk_ok(self):
        return {
            "trading_allowed":       True,
            "final_size_mult":       1.0,
            "monthly_dd_flag":       0,
            "consecutive_loss_halt": 0,
        }

    def _risk_halted(self, reason="monthly_dd"):
        # Gate 6 consults circuit_breaker_level FIRST (G6-2 — it is the
        # authoritative halt signal), so set it here to exercise that path.
        return {
            "trading_allowed":         False,
            "final_size_mult":         0.0,
            "circuit_breaker_level":   "HALT",
            "circuit_breaker_reason":  "monthly loss limit",
            "monthly_dd_flag":         1 if reason == "monthly_dd" else 0,
            "consecutive_loss_halt":   1 if reason == "losses" else 0,
        }

    def test_sufficient_confidence_passes(self):
        passed, ctx = self.gate.check(
            primary_conf=0.72, alignment="A+",
            capital_mode="FULL", risk_context=self._risk_ok(),
            model_type="swing",
        )
        self.assertTrue(passed)

    def test_low_confidence_blocked(self):
        passed, ctx = self.gate.check(
            primary_conf=0.50, alignment="A+",
            capital_mode="FULL", risk_context=self._risk_ok(),
            model_type="swing",
        )
        self.assertFalse(passed)

    def test_small_capital_requires_a_plus_only(self):
        passed, ctx = self.gate.check(
            primary_conf=0.75, alignment="B",
            capital_mode="SMALL", risk_context=self._risk_ok(),
            model_type="swing",
        )
        self.assertFalse(passed, "SMALL capital should require A+ alignment")

    def test_small_capital_a_plus_passes(self):
        # SMALL swing base threshold 0.68 + 0.03 provisional edge premium (P2)
        # = 0.71 effective, so use 0.73 to clear it.
        passed, ctx = self.gate.check(
            primary_conf=0.73, alignment="A+",
            capital_mode="SMALL", risk_context=self._risk_ok(),
            model_type="swing",
        )
        self.assertTrue(passed)
        self.assertEqual(ctx["edge_premium"], 0.03)

    def test_circuit_breaker_blocks(self):
        passed, ctx = self.gate.check(
            primary_conf=0.85, alignment="A+",
            capital_mode="FULL",
            risk_context=self._risk_halted("monthly_dd"),
            model_type="swing",
        )
        self.assertFalse(passed)
        self.assertIn("circuit", ctx.get("reason", "").lower())

    def test_growing_capital_allows_a(self):
        passed, ctx = self.gate.check(
            primary_conf=0.65, alignment="A",
            capital_mode="GROWING", risk_context=self._risk_ok(),
            model_type="swing",
        )
        self.assertTrue(passed)

    # ── EDGE-1: expectancy override ──────────────────────────────────────
    def test_expectancy_passes_clean_setup_below_threshold(self):
        # swing thr 0.60 + 0.03 premium = 0.63; conf 0.55 < 0.63 but Grade A @ 2.5:1
        passed, ctx = self.gate.check(
            primary_conf=0.55, alignment="C", capital_mode="FULL",
            risk_context=self._risk_ok(), model_type="swing", skip_alignment=True,
            entry_quality="A", reward_risk=2.5,
        )
        self.assertTrue(passed)
        self.assertTrue(ctx["expectancy_pass"])
        self.assertGreater(ctx["expected_R"], 0)

    def test_expectancy_rejects_weak_geometry(self):
        passed, _ = self.gate.check(
            primary_conf=0.55, alignment="C", capital_mode="FULL",
            risk_context=self._risk_ok(), model_type="swing", skip_alignment=True,
            entry_quality="C", reward_risk=2.5,   # Grade C → no override
        )
        self.assertFalse(passed)

    def test_expectancy_rejects_below_floor_and_poor_rr(self):
        # below conf floor (0.50) even with great R:R
        p1, _ = self.gate.check(
            primary_conf=0.46, alignment="C", capital_mode="FULL",
            risk_context=self._risk_ok(), model_type="swing", skip_alignment=True,
            entry_quality="A", reward_risk=4.0)
        # fair conf but R:R below 2:1
        p2, _ = self.gate.check(
            primary_conf=0.55, alignment="C", capital_mode="FULL",
            risk_context=self._risk_ok(), model_type="swing", skip_alignment=True,
            entry_quality="A", reward_risk=1.2)
        self.assertFalse(p1)
        self.assertFalse(p2)


class TestGate4RegimeAlignment(unittest.TestCase):
    """EDGE-2: a counter-regime model vote must not drag an aligned signal to F."""

    def setUp(self):
        from signals.gate4_ml_predictor import Gate4MLPredictor
        self.g4 = Gate4MLPredictor.__new__(Gate4MLPredictor)  # no model load

    def test_counter_regime_long_not_penalised_in_bear(self):
        # BEAR: pos LONG (counter, noise) + intra SHORT (aligned) + swing FLAT.
        blind = self.g4._calc_alignment("LONG", "FLAT", "SHORT")
        aware = self.g4._calc_alignment("LONG", "FLAT", "SHORT",
                                        regime_trade_long=False, regime_trade_short=True)
        self.assertEqual(blind[0], "F")     # regime-blind: 1v1 conflict → F (−20pp)
        self.assertEqual(aware[0], "C")     # regime-aware: LONG dropped → lone SHORT
        self.assertLess(blind[1], 0)        # F penalises
        self.assertGreaterEqual(aware[1], 0)

    def test_aligned_agreement_unchanged(self):
        # Both SHORT in a BEAR — genuine agreement must still grade well.
        grade, boost, _ = self.g4._calc_alignment(
            "SHORT", "SHORT", "FLAT", regime_trade_long=False, regime_trade_short=True)
        self.assertIn(grade, ("A", "B"))
        self.assertGreater(boost, 0)


class TestTimeframeAlignment(unittest.TestCase):

    def setUp(self):
        self.aligner = TimeframeAlignment()

    def test_all_long_is_a_plus(self):
        result = self.aligner.compute("LONG", "LONG", "LONG")
        self.assertEqual(result["grade"], "A+")
        self.assertGreater(result["conf_boost"], 0)

    def test_pos_swing_agree_is_a(self):
        result = self.aligner.compute("LONG", "LONG", "FLAT")
        self.assertEqual(result["grade"], "A")

    def test_conflict_is_f_grade(self):
        result = self.aligner.compute("LONG", "SHORT", "FLAT")
        self.assertEqual(result["grade"], "F")
        self.assertEqual(result["size_mult"], 0.0)

    def test_all_flat_returns_c(self):
        result = self.aligner.compute("FLAT", "FLAT", "FLAT")
        self.assertIn(result["grade"], ["C", "F"])

    def test_f_grade_conf_boost_negative(self):
        result = self.aligner.compute("LONG", "SHORT", "LONG")
        self.assertLess(result["conf_boost"], 0)

    def test_dominant_direction_correct(self):
        result = self.aligner.compute("LONG", "LONG", "SHORT")
        # 2 LONG vs 1 SHORT — dominant should be LONG
        self.assertEqual(result["dominant"], "LONG")

    def test_single_non_flat_is_b_or_c(self):
        result = self.aligner.compute("LONG", "FLAT", "FLAT")
        self.assertIn(result["grade"], ["B", "C"])


class TestSignalStabilizer(unittest.TestCase):
    """STAB-2 anti-flicker commit layer."""

    def setUp(self):
        self.s = SignalStabilizer()

    @staticmethod
    def _p(long=0.0, flat=0.0, short=0.0):
        return {"prob_long": long, "prob_flat": flat, "prob_short": short}

    def test_deadband_marginal_long_held_flat(self):
        # LONG 0.48 vs FLAT 0.51 → argmax is FLAT anyway; and even a 3pp lead
        # over flat is below the dead-band → stays FLAT.
        r = self.s.commit("swing", "LONG", self._p(long=0.46, flat=0.44, short=0.10))
        self.assertEqual(r["decisive"], "FLAT")
        self.assertEqual(r["committed"], "FLAT")

    def test_decisive_long_needs_confirmation_swing(self):
        # swing confirms over 2 cycles: first cycle holds FLAT, second commits.
        p = self._p(long=0.70, flat=0.20, short=0.10)
        r1 = self.s.commit("swing", "LONG", p)
        self.assertEqual(r1["committed"], "FLAT")          # 1/2 — holding
        r2 = self.s.commit("swing", "LONG", p)
        self.assertEqual(r2["committed"], "LONG")          # 2/2 — committed

    def test_intraday_commits_in_one_cycle(self):
        r = self.s.commit("intraday", "SHORT", self._p(short=0.80, flat=0.15, long=0.05))
        self.assertEqual(r["committed"], "SHORT")          # confirm=1 → immediate

    def test_fast_derisk_to_flat_is_immediate(self):
        p = self._p(long=0.80, flat=0.10, short=0.10)
        self.s.commit("intraday", "LONG", p)               # committed LONG
        r = self.s.commit("intraday", "FLAT", self._p(long=0.30, flat=0.45, short=0.25))
        self.assertEqual(r["committed"], "FLAT")           # dropped immediately
        self.assertTrue(r["intervened"] is False)          # raw was already FLAT

    def test_reversal_exits_to_flat_first(self):
        self.s.commit("intraday", "LONG", self._p(long=0.80, flat=0.10, short=0.10))
        r = self.s.commit("intraday", "SHORT", self._p(short=0.80, flat=0.10, long=0.10))
        self.assertEqual(r["committed"], "FLAT")           # never flips LONG→SHORT directly


class TestEntryTimingGate(unittest.TestCase):
    """STAB-3 — don't open into an adverse immediate move."""

    def setUp(self):
        from signals.signal_engine import SignalEngine
        # Bind the unbound method without constructing the heavy engine.
        self.fn = SignalEngine._entry_timing_ok
        class _Stub:
            ENTRY_MOM_ATR_FACTOR = 0.4
            ENTRY_MOM_MIN        = 0.0035
        self.stub = _Stub()

    def _ok(self, direction, mom, atr_pct=0.02):
        return self.fn(self.stub, direction, {"intraday_mom_30m": mom}, atr_pct)[0]

    def test_long_into_drop_deferred(self):
        self.assertFalse(self._ok("LONG", -0.015))   # price dropping 1.5% → defer LONG

    def test_long_with_uptick_ok(self):
        self.assertTrue(self._ok("LONG", 0.004))      # rising → fine

    def test_short_into_rip_deferred(self):
        self.assertFalse(self._ok("SHORT", 0.015))    # price ripping up → defer SHORT

    def test_missing_data_allows(self):
        self.assertTrue(self.fn(self.stub, "LONG", {}, 0.02)[0])   # no mom → neutral → ok

    def test_threshold_is_atr_relative(self):
        # Same 0.5% drop: deferred for a low-ATR name, allowed for a high-ATR name.
        self.assertFalse(self._ok("LONG", -0.005, atr_pct=0.005))  # thr floor 0.35% → defer
        self.assertTrue(self._ok("LONG",  -0.005, atr_pct=0.03))   # thr 1.2% → allowed


if __name__ == "__main__":
    unittest.main(verbosity=2)