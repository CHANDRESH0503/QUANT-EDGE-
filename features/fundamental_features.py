# features/fundamental_features.py
# Selects and normalises fundamental features for positional ML model
# Source: processing/fundamental.py (quarterly Screener.in data)
# Only meaningful for positional model — swing/intraday use zero weights

import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class FundamentalFeatures:
    """
    Converts quarterly fundamental data into ML-ready features.

    Senior trader perspective:
    Fundamentals do not predict tomorrow's price.
    They predict the DIRECTION of the trend over the next quarter.
    Use them to filter: strong fundamentals = only take longs.
    Weak fundamentals = only take shorts or stay flat.

    Normalisation approach:
    - NIM: 4.5% = +1.0, 2.5% = -1.0 (bank-specific range)
    - NPA: 0.5% = +1.0, 4.0% = -1.0 (inverted — lower is better)
    - CASA: 50% = +1.0, 30% = -1.0
    - ROE:  20% = +1.0, 10% = -1.0
    - P/B vs avg: 0.7 = +1.0, 1.5 = -1.0 (cheaper = better)
    """

    FEATURE_NAMES = [
        "nim_norm",
        "nim_trend",
        "npa_norm",         # inverted — lower NPA = higher score
        "npa_trend",        # inverted — improving = positive
        "casa_norm",
        "loan_growth_norm",
        "roe_norm",
        "pb_vs_avg_norm",   # cheap = positive
        "beat_streak_norm",
        "fundamental_score",
    ]

    def extract(self, fund_features: Dict) -> Dict:
        """
        Normalise fundamental data to ML-ready features.

        Args:
            fund_features: from FundamentalProcessor.get_fundamental_features()
        """
        nim        = float(fund_features.get("nim", 3.5))
        npa        = float(fund_features.get("npa_gross", 2.0))
        casa       = float(fund_features.get("casa_ratio", 42.0))
        loan_gr    = float(fund_features.get("loan_growth", 12.0))
        roe        = float(fund_features.get("roe", 14.0))
        pb_vs_avg  = float(fund_features.get("pb_vs_5yr_avg", 1.0))
        beat_streak= float(fund_features.get("beat_streak", 0))
        nim_trend  = float(fund_features.get("nim_trend", 0))
        npa_trend  = float(fund_features.get("npa_trend", 0))
        fund_score = float(fund_features.get("fundamental_score", 0))

        # Normalise to -1 to +1
        nim_norm       = self._scale(nim,      2.5, 4.5)
        npa_norm       = self._scale(npa,      4.0, 0.5)   # inverted range
        casa_norm      = self._scale(casa,     30,  52)
        loan_gr_norm   = self._scale(loan_gr,  5,   22)
        roe_norm       = self._scale(roe,      10,  20)
        pb_norm        = self._scale(pb_vs_avg,1.5, 0.7)   # cheaper = better
        beat_norm      = min(1.0, beat_streak / 6.0)       # 6 consecutive = max

        return {
            "nim_norm":          round(nim_norm, 4),
            "nim_trend":         round(self._clip(nim_trend), 4),
            "npa_norm":          round(npa_norm, 4),
            "npa_trend":         round(self._clip(npa_trend), 4),
            "casa_norm":         round(casa_norm, 4),
            "loan_growth_norm":  round(loan_gr_norm, 4),
            "roe_norm":          round(roe_norm, 4),
            "pb_vs_avg_norm":    round(pb_norm, 4),
            "beat_streak_norm":  round(beat_norm, 4),
            "fundamental_score": round(self._clip(fund_score), 4),
        }

    def _scale(self, val: float, low: float, high: float) -> float:
        """Linear scale val from [low, high] to [-1, +1]."""
        if high == low:
            return 0.0
        scaled = 2 * (val - low) / (high - low) - 1
        return round(max(-1.0, min(1.0, scaled)), 4)

    def _clip(self, val: float) -> float:
        v = float(val) if np.isfinite(float(val)) else 0.0
        return max(-1.0, min(1.0, v))