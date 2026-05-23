# features/alternative_features.py
# Google Trends and insider activity as leading fundamental indicators
# Source: alternative_fetcher.py, insider_fetcher.py
# Weekly update cycle — leading indicator of business trajectory

import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class AlternativeFeatures:
    """
    Alternative data features — leading indicators of fundamentals.

    Senior trader perspective:
    Google Trends data for HDFC Bank loan searches predicts
    loan growth 2-3 months before it shows in quarterly results.
    This gives a genuine edge in timing positional trades.

    When 'HDFC home loan' searches are trending up for 6 weeks,
    the next quarterly results will almost certainly show
    strong loan book growth. Position accordingly before results.

    Features are slow-moving (weekly updates) but high conviction.
    Weight them heavily for positional model, less for swing, ignore intraday.
    """

    def extract(self,
                alt_features:     Dict,
                insider_features: Dict) -> Dict:
        """
        Convert alternative and insider data to ML features.

        Args:
            alt_features:     from AlternativeFetcher.get_alternative_features()
            insider_features: from InsiderFetcher.get_insider_features()
        """
        # ── Google Trends demand ─────────────────────────────────
        demand_score   = self._clip(
            float(alt_features.get("demand_trend_score", 0)) * 3
        )
        home_loan_slope= self._clip(
            float(alt_features.get("home_loan_trend", 0)) * 5
        )
        personal_loan  = self._clip(
            float(alt_features.get("personal_loan_trend", 0)) * 5
        )
        credit_card    = self._clip(
            float(alt_features.get("credit_card_trend", 0)) * 5
        )

        # ── Hiring trend ──────────────────────────────────────────
        hiring_score   = self._clip(
            float(alt_features.get("hiring_trend_score", 0)) * 4
        )

        # ── Concern trend (inverted) ──────────────────────────────
        concern_score  = float(alt_features.get("concern_trend_score", 0))
        # concern already inverted in AlternativeFetcher (negative concern = positive)

        # ── Composite alt score ───────────────────────────────────
        alt_score      = self._clip(
            float(alt_features.get("alt_data_score", 0))
        )

        # ── Insider / Promoter ────────────────────────────────────
        insider_score  = self._clip(
            float(insider_features.get("insider_score", 0))
        )
        promoter_trend = insider_features.get("promoter_trend", "STABLE")
        promoter_score = (
             1.0 if promoter_trend == "INCREASING" else
            -1.0 if promoter_trend == "DECREASING" else 0.0
        )
        pledge_pct     = float(insider_features.get("promoter_pledge_pct", 0))
        # High pledge = management financially stressed = negative
        pledge_risk    = self._clip(-pledge_pct / 20)

        dii_holding_qoq= self._clip(
            float(insider_features.get("dii_holding_qoq", 0)) * 5
        )

        return {
            # Google Trends
            "demand_trend":         round(demand_score, 4),
            "home_loan_trend":      round(home_loan_slope, 4),
            "personal_loan_trend":  round(personal_loan, 4),
            "credit_card_trend":    round(credit_card, 4),
            "hiring_trend":         round(hiring_score, 4),
            "concern_trend":        round(self._clip(concern_score), 4),
            "alt_data_score":       round(alt_score, 4),

            # Insider / Ownership
            "insider_score":        round(insider_score, 4),
            "promoter_trend_score": round(promoter_score, 4),
            "pledge_risk":          round(pledge_risk, 4),
            "dii_accumulation":     round(dii_holding_qoq, 4),
        }

    def _clip(self, val: float) -> float:
        try:
            v = float(val)
            return round(max(-1.0, min(1.0, v)), 4) if np.isfinite(v) else 0.0
        except (TypeError, ValueError):
            return 0.0