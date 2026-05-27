# features/sentiment_features.py
# Assembles all sentiment signals into ML-ready features
# Sources: FinBERT (news), LLM (earnings/announcements), Social
# Connected to: finbert_sentiment.py, llm_analyzer.py, social_fetcher.py

import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class SentimentFeatures:
    """
    Consolidates all text-based sentiment into 8 ML features.

    Senior trader perspective:
    News sentiment is a genuine alpha source when used correctly.
    The trend of sentiment matters more than a single day's reading.
    LLM earnings score is the highest quality single event signal.
    Social sentiment is valuable ONLY as a contrarian indicator.

    Feature design principles:
    - All scores normalised to [-1, +1]
    - Trend (3-day) always included alongside level
    - Decay applied to LLM earnings score over 90 days
    - Social sentiment INVERTED (contrarian logic)
    """

    FEATURE_NAMES = [
        "finbert_score_24h",
        "finbert_score_trend",      # improving or deteriorating
        "finbert_momentum",         # today vs yesterday
        "finbert_momentum_3d",      # score_24h − score_72h (alpha 2026-05-26)
        "finbert_news_spike",       # volume anomaly
        "llm_earnings_score",
        "llm_earnings_relevance",   # decays over 90 days
        "llm_announcement_score",
        "llm_combined_score",
        "social_contrarian_score",  # inverted — high bull = bearish
    ]

    def extract(self,
                finbert_features: Dict,
                llm_features:     Dict,
                social_features:  Dict) -> Dict:
        """
        Combine all sentiment sources into final ML feature dict.

        Args:
            finbert_features : from FinBERTSentiment.get_sentiment_features_for_ml()
            llm_features     : from LLMAnalyzer.get_llm_features_for_ml()
            social_features  : from SocialFetcher.get_social_features()
        """
        # ── FinBERT ───────────────────────────────────────────────
        finbert_24h   = self._clip(finbert_features.get("finbert_score_24h", 0))
        finbert_72h   = self._clip(finbert_features.get("finbert_score_72h",  0))
        finbert_trend = self._clip(finbert_24h - finbert_72h, -0.5, 0.5)
        momentum      = self._clip(finbert_features.get("finbert_momentum", 0), -0.5, 0.5)

        # News spike: article count 2.5x normal = important
        spike_raw     = float(finbert_features.get("finbert_news_spike", 1.0))
        news_spike    = min(3.0, spike_raw) / 3.0  # normalise to 0-1

        # ── LLM ───────────────────────────────────────────────────
        earnings_score    = self._clip(llm_features.get("llm_earnings_score", 0))
        earnings_relev    = float(llm_features.get("llm_earnings_relevance", 0))
        announcement_score= self._clip(llm_features.get("llm_announcement_score", 0))
        combined          = self._clip(llm_features.get("llm_combined_score", 0))

        # ── Social (CONTRARIAN) ──────────────────────────────────
        # social_ml_score from SocialFetcher is already inverted
        # (high bull ratio → negative ML score)
        social_raw        = float(social_features.get("social_ml_score", 0.0))
        social_contrarian = self._clip(social_raw)

        features = {
            "finbert_score_24h":       round(finbert_24h, 4),
            "finbert_score_trend":     round(finbert_trend, 4),
            "finbert_momentum":        round(momentum, 4),
            # Alpha (2026-05-26): explicit 3-day sentiment momentum.
            # Same definition as finbert_score_trend; kept as a stable name
            # in the prod feature lists so retrains don't have to migrate.
            "finbert_momentum_3d":     round(finbert_trend, 4),
            "finbert_news_spike":      round(news_spike, 4),
            "llm_earnings_score":      round(earnings_score * earnings_relev, 4),
            "llm_earnings_relevance":  round(earnings_relev, 4),
            "llm_announcement_score":  round(announcement_score, 4),
            "llm_combined_score":      round(combined, 4),
            "social_contrarian_score": round(social_contrarian, 4),
        }

        # Sanity check — all must be finite
        return {k: v if np.isfinite(v) else 0.0 for k, v in features.items()}

    def _clip(self, val: float, lo: float = -1.0, hi: float = 1.0) -> float:
        try:
            v = float(val)
            return max(lo, min(hi, v)) if np.isfinite(v) else 0.0
        except (TypeError, ValueError):
            return 0.0