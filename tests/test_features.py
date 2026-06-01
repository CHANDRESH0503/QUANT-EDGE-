 # tests/test_features.py
# Tests for the feature engineering layer
# Every feature must be: finite, bounded, deterministic, no lookahead

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from processing.technical        import TechnicalProcessor
from features.technical_features import TechnicalFeatures
from features.sentiment_features import SentimentFeatures
from features.fundamental_features import FundamentalFeatures
from features.macro_features     import MacroFeatures
from features.options_features   import OptionsFeatures
from features.flow_features      import FlowFeatures
from features.calendar_features  import CalendarFeatures
from features.risk_features      import RiskFeatures


def make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic HDFC Bank OHLCV data."""
    np.random.seed(seed)
    dates  = pd.date_range("2023-01-01", periods=n, freq="B")
    price  = 1600.0
    closes = []
    for _ in range(n):
        price *= (1 + np.random.normal(0.0004, 0.012))
        price  = max(100, price)
        closes.append(round(price, 2))
    closes = np.array(closes)
    highs  = closes * (1 + np.abs(np.random.normal(0, 0.005, n)))
    lows   = closes * (1 - np.abs(np.random.normal(0, 0.005, n)))
    opens  = closes * (1 + np.random.normal(0, 0.003, n))
    vols   = np.random.randint(5_000_000, 20_000_000, n).astype(float)
    return pd.DataFrame({
        "Open":   opens, "High": highs,
        "Low":    lows,  "Close": closes, "Volume": vols,
    }, index=dates)


class TestTechnicalProcessor(unittest.TestCase):
    """Technical features must be finite, bounded, and correct."""

    def setUp(self):
        self.df   = make_ohlcv()
        self.proc = TechnicalProcessor()
        self.feat = TechnicalFeatures()

    def test_build_returns_dataframe(self):
        result = self.proc.build_features(self.df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_no_nan_after_extraction(self):
        tech_df  = self.proc.build_features(self.df)
        features = self.feat.extract(tech_df)
        for k, v in features.items():
            self.assertTrue(np.isfinite(float(v)),
                            f"Feature {k} = {v} is not finite")

    def test_rsi_bounded(self):
        tech_df  = self.proc.build_features(self.df)
        features = self.feat.extract(tech_df)
        rsi = float(features.get("rsi_14", 50))
        self.assertGreaterEqual(rsi, 0,   "RSI below 0")
        self.assertLessEqual(rsi, 100,    "RSI above 100")

    def test_bb_pct_b_winsorised(self):
        tech_df  = self.proc.build_features(self.df)
        features = self.feat.extract(tech_df)
        val = float(features.get("bb_pct_b", 0.5))
        self.assertGreaterEqual(val, -0.5, "BB%B below winsorise floor")
        self.assertLessEqual(val,    1.5,  "BB%B above winsorise ceiling")

    def test_supertrend_binary(self):
        tech_df  = self.proc.build_features(self.df)
        features = self.feat.extract(tech_df)
        st = float(features.get("supertrend", 1))
        self.assertIn(st, [-1.0, 1.0], "Supertrend must be -1 or +1")

    def test_no_lookahead(self):
        """Feature at index i must use only data up to index i."""
        n   = len(self.df)
        mid = n // 2
        df_half = self.df.iloc[:mid]
        df_full = self.df

        tech_half = self.proc.build_features(df_half)
        tech_full = self.proc.build_features(df_full)

        feat_half = self.feat.extract(tech_half)
        feat_full_mid = self.feat.extract(tech_full.iloc[:mid])

        # RSI at midpoint should be identical whether computed on half or full data
        rsi_half = float(feat_half.get("rsi_14", 0))
        rsi_mid  = float(feat_full_mid.get("rsi_14", 0))
        self.assertAlmostEqual(rsi_half, rsi_mid, places=3,
                                msg="Lookahead bias detected in RSI")

    def test_minimum_data_requirement(self):
        """Should return empty df with < 30 bars."""
        df_tiny  = make_ohlcv(n=10)
        result   = self.proc.build_features(df_tiny)
        self.assertTrue(result.empty or len(result) < 5,
                        "Should not produce features with < 30 bars")

    def test_all_feature_names_present(self):
        tech_df  = self.proc.build_features(self.df)
        features = self.feat.extract(tech_df)
        for name in TechnicalFeatures.FEATURE_NAMES:
            self.assertIn(name, features,
                          f"Feature {name} missing from extraction")


class TestSentimentFeatures(unittest.TestCase):

    def setUp(self):
        self.feat = SentimentFeatures()

    def _dummy_finbert(self, score=0.3):
        return {
            "finbert_score_24h": score, "finbert_score_72h": score * 0.8,
            "finbert_momentum": 0.05,   "finbert_news_spike": 1.2,
            "finbert_news_count": 5,
        }

    def _dummy_llm(self):
        return {
            "llm_earnings_score": 0.6, "llm_earnings_relevance": 0.8,
            "llm_announcement_score": 0.2, "llm_combined_score": 0.5,
            "llm_earnings_days_ago": 15,
        }

    def _dummy_social(self):
        return {
            "social_ml_score": -0.4,  # contrarian — high bull = negative
            "bull_ratio": 0.70,
        }

    def test_all_features_bounded(self):
        features = self.feat.extract(
            self._dummy_finbert(), self._dummy_llm(), self._dummy_social()
        )
        for k, v in features.items():
            self.assertGreaterEqual(float(v), -1.0, f"{k} below -1")
            self.assertLessEqual(float(v),     1.0, f"{k} above +1")

    def test_social_inverted(self):
        """High bull ratio should produce negative social_contrarian_score."""
        social = {"social_ml_score": -0.7, "bull_ratio": 0.85}
        feat   = self.feat.extract(
            self._dummy_finbert(), self._dummy_llm(), social
        )
        score = float(feat.get("social_contrarian_score", 0))
        self.assertLess(score, 0, "High bull ratio should give negative contrarian score")

    def test_earnings_decay_applied(self):
        """Old earnings (relevance=0) should zero out the earnings score."""
        llm = self._dummy_llm()
        llm["llm_earnings_relevance"] = 0.0
        feat = self.feat.extract(
            self._dummy_finbert(), llm, self._dummy_social()
        )
        score = float(feat.get("llm_earnings_score", 1))
        self.assertAlmostEqual(score, 0.0, places=3,
                               msg="Earnings score should be zero when relevance=0")

    def test_no_nan_with_empty_inputs(self):
        """Empty/zero inputs should return all-zero features without NaN."""
        feat = self.feat.extract({}, {}, {})
        for k, v in feat.items():
            self.assertTrue(np.isfinite(float(v)), f"{k} is NaN with empty input")


class TestFundamentalFeatures(unittest.TestCase):

    def setUp(self):
        self.feat = FundamentalFeatures()

    def _hdfc_fundamentals(self):
        return {
            "nim": 4.1, "npa_gross": 1.2, "casa_ratio": 46.0,
            "loan_growth": 17.5, "roe": 17.0, "pb_vs_5yr_avg": 0.95,
            "beat_streak": 4, "nim_trend": 0.3, "npa_trend": 0.4,
            "fundamental_score": 0.6,
        }

    def test_nim_normalisation(self):
        """NIM 4.5% = +1.0, NIM 2.5% = -1.0."""
        feat_high = self.feat.extract({**self._hdfc_fundamentals(), "nim": 4.5})
        feat_low  = self.feat.extract({**self._hdfc_fundamentals(), "nim": 2.5})
        self.assertAlmostEqual(feat_high["nim_norm"], 1.0, places=2)
        self.assertAlmostEqual(feat_low["nim_norm"], -1.0, places=2)

    def test_npa_inverted(self):
        """Lower NPA should give higher score (inverted feature)."""
        feat_low_npa  = self.feat.extract({**self._hdfc_fundamentals(), "npa_gross": 0.5})
        feat_high_npa = self.feat.extract({**self._hdfc_fundamentals(), "npa_gross": 4.0})
        self.assertGreater(feat_low_npa["npa_norm"], feat_high_npa["npa_norm"])

    def test_all_bounded(self):
        feat = self.feat.extract(self._hdfc_fundamentals())
        for k, v in feat.items():
            if isinstance(v, (int, float)):
                self.assertGreaterEqual(float(v), -1.0, f"{k} below -1")
                self.assertLessEqual(float(v),    1.0,  f"{k} above +1")

    def test_beat_streak_cap(self):
        """Beat streak capped at 6 consecutive = 1.0."""
        feat = self.feat.extract({**self._hdfc_fundamentals(), "beat_streak": 10})
        self.assertLessEqual(feat["beat_streak_norm"], 1.0)


class TestRiskFeatures(unittest.TestCase):

    def setUp(self):
        self.feat = RiskFeatures()
        # Paper-trading sets FORCE_CAPITAL_MODE=FULL (config default), which
        # bypasses ₹-based auto-detection. These tests validate the detection
        # bands themselves, so neutralise the override and restore it after.
        from config import TradingConfig
        self._saved_force = TradingConfig.FORCE_CAPITAL_MODE
        TradingConfig.FORCE_CAPITAL_MODE = ""

    def tearDown(self):
        from config import TradingConfig
        TradingConfig.FORCE_CAPITAL_MODE = self._saved_force

    def _make_anomaly(self, detected=False):
        return {
            "is_anomaly": detected,
            "severity":   "HIGH" if detected else "LOW",
            "position_size_mult": 0.5 if detected else 1.0,
        }

    def _make_sr(self, quality="B"):
        return {
            "entry_quality": quality,
            "entry_quality_score": {"A": 1.0, "B": 0.6, "C": 0.2, "D": -0.2}.get(quality, 0.2),
            "reward_risk_sr": 2.5 if quality in ("A","B") else 1.2,
        }

    def _make_meta(self, losing_streak=0):
        return {
            "losing_streak_norm": losing_streak / 5,
            "meta_size_mult": 1.0 if losing_streak == 0 else 0.5,
        }

    def test_three_losses_halts_trading(self):
        meta = self._make_meta(losing_streak=3)
        feat = self.feat.extract(
            capital=100_000,
            anomaly_features=self._make_anomaly(),
            sr_features=self._make_sr(),
            meta_features=meta,
        )
        self.assertEqual(feat["final_size_mult"], 0.0,
                         "3 consecutive losses should halt trading")
        self.assertEqual(feat["trading_allowed"], 0)

    def test_anomaly_reduces_size(self):
        feat = self.feat.extract(
            capital=100_000,
            anomaly_features=self._make_anomaly(detected=True),
            sr_features=self._make_sr(),
            meta_features=self._make_meta(),
        )
        self.assertLessEqual(feat["final_size_mult"], 0.5)

    def test_small_capital_mode(self):
        feat = self.feat.extract(
            capital=20_000,
            anomaly_features=self._make_anomaly(),
            sr_features=self._make_sr(),
            meta_features=self._make_meta(),
        )
        self.assertEqual(feat["capital_mode"], "SMALL")
        self.assertEqual(feat["capital_mode_mult"], 0.5)

    def test_full_capital_mode(self):
        feat = self.feat.extract(
            capital=500_000,
            anomaly_features=self._make_anomaly(),
            sr_features=self._make_sr(),
            meta_features=self._make_meta(),
        )
        self.assertEqual(feat["capital_mode"], "FULL")
        self.assertEqual(feat["capital_mode_mult"], 1.0)

    def test_size_mult_never_negative(self):
        feat = self.feat.extract(
            capital=10_000,
            anomaly_features=self._make_anomaly(detected=True),
            sr_features=self._make_sr(quality="D"),
            meta_features=self._make_meta(losing_streak=5),
        )
        self.assertGreaterEqual(feat["final_size_mult"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)