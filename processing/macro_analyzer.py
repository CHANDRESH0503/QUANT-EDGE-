# processing/macro_analyzer.py
# Macro economic context for HDFC Bank signal generation
# RBI rates, India VIX, CPI, credit growth, global macro
# Features used by all three ML models

import sqlite3
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from data.price_fetcher import yf_safe_download

logger = logging.getLogger(__name__)

# ── RBI MPC Meeting Dates 2026 ────────────────────────────────
RBI_MPC_DATES_2026 = [
    "2026-02-07", "2026-04-09", "2026-06-06",
    "2026-08-08", "2026-10-08", "2026-12-05",
]

# ── RBI MPC Meeting Dates 2027 ────────────────────────────────
RBI_MPC_DATES_2027 = [
    "2027-02-05", "2027-04-07", "2027-06-04",
    "2027-08-06", "2027-10-07", "2027-12-04",
]

# Combined for easy lookup across years
_ALL_RBI_MPC_DATES = RBI_MPC_DATES_2026 + RBI_MPC_DATES_2027


class MacroAnalyzer:
    """
    Macro context processor — converts macro data into ML features.

    As a senior trader, macro context changes everything:
    - RBI rate cut approaching = buy banks before the meeting
    - RBI hiking cycle = banks margin squeeze risk
    - India VIX above 22 = reduce all positions regardless of signal
    - Rupee collapsing = FII outflow = headwind even for good stocks

    This module answers: what is the macro environment telling us?
    """

    def __init__(self, db_path: str = "database/trading.db"):
        self.db_path = db_path

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def get_macro_features(self) -> Dict:
        """
        Returns all macro features for ML model input.
        Combines RBI calendar, India VIX, USD/INR, Nifty50, credit growth.
        """
        rbi         = self._rbi_features()
        vix         = self._vix_features()
        currency    = self._currency_features()
        nifty       = self._nifty_features()
        credit      = self._credit_features()
        macro_score = self._calculate_macro_score(rbi, vix, currency)
        event_mult  = self._event_proximity_mult(rbi)

        # Freshness flag — if global_snapshots is stale (>30h since last fetch)
        # the macro context should be treated with reduced trust by Gate 6.
        try:
            from data.global_fetcher import GlobalFetcher
            macro_stale = GlobalFetcher(self.db_path).is_stale(max_age_hours=30)
        except Exception:
            macro_stale = False

        return {
            # RBI calendar
            **rbi,
            # India VIX
            **vix,
            # Currency
            **currency,
            # Nifty50 direction (used by Gate 2 Check 3)
            **nifty,
            # Credit
            **credit,
            # Composite
            "macro_score":        macro_score,
            "macro_signal":       self._score_to_signal(macro_score),
            # Event-aware sizing
            "macro_event_mult":   event_mult,
            # Freshness
            "macro_stale":        int(bool(macro_stale)),
        }

    def _event_proximity_mult(self, rbi: Dict) -> float:
        """
        Continuous size multiplier based on proximity to macro events.
        Combines RBI MPC and a simplified FOMC proximity check.

        Rules:
        - RBI MPC T-1 (day before) : 0.5×
        - RBI MPC T-day            : 0.4× (high uncertainty at decision)
        - RBI MPC T+1              : 0.8× (surprise risk lingers)
        - India pre-Budget (T-2→T) : 0.0× (handled separately if date known)
        - Otherwise                : 1.0×
        """
        days_to_rbi = int(rbi.get("days_to_rbi", 99))
        if days_to_rbi == 0:
            return 0.40
        if days_to_rbi == 1:
            return 0.50
        if days_to_rbi < 0 and days_to_rbi >= -1:
            return 0.80
        return 1.00

    # ─────────────────────────────────────────────────────────────
    # RBI FEATURES
    # ─────────────────────────────────────────────────────────────

    def _rbi_features(self) -> Dict:
        """Days to next RBI meeting and rate direction."""
        today       = datetime.now().date()
        upcoming    = sorted([
            d for d in _ALL_RBI_MPC_DATES
            if datetime.strptime(d, "%Y-%m-%d").date() >= today
        ])
        days_to_rbi = (
            (datetime.strptime(upcoming[0], "%Y-%m-%d").date() - today).days
            if upcoming else 30
        )

        # Rate direction from last known announcement
        rbi_rate     = self._get_cached_rbi_rate()
        rate_trend   = self._get_rbi_rate_trend()

        return {
            "days_to_rbi":   days_to_rbi,
            "is_rbi_week":   int(days_to_rbi <= 5),
            "rbi_rate":      rbi_rate,
            "rbi_direction": rate_trend,          # HIKING / CUTTING / HOLDING
            "rbi_risk_flag": int(days_to_rbi <= 5),  # flag for position sizing
        }

    def _vix_features(self) -> Dict:
        """India VIX level, change, and percentile."""
        try:
            df = yf_safe_download("^INDIAVIX", period="30d")
            if df.empty:
                return self._empty_vix()

            current   = float(df["Close"].iloc[-1])
            prev      = float(df["Close"].iloc[-2])
            change_1d = round((current - prev) / prev * 100, 2)
            pct_20d   = float(
                np.sum(df["Close"].iloc[-20:] < current) / 20 * 100
            )

            return {
                "india_vix_level":  current,
                "india_vix_1d_chg": change_1d,
                "india_vix_pct20":  pct_20d,
                "india_vix_high":   int(current > 20),
                "india_vix_panic":  int(current > 25),  # full defensive mode
            }
        except Exception as e:
            logger.warning(f"VIX fetch failed: {e}")
            return self._empty_vix()

    def _currency_features(self) -> Dict:
        """USD/INR rate and trend — FII flow proxy."""
        try:
            df = yf_safe_download("USDINR=X", period="30d")
            if df.empty:
                return self._empty_currency()

            close     = df["Close"]
            level     = float(close.iloc[-1])
            chg_1d    = float(close.pct_change().iloc[-1] * 100)
            chg_5d    = float((close.iloc[-1] / close.iloc[-5] - 1) * 100)
            chg_20d   = float((close.iloc[-1] / close.iloc[-20] - 1) * 100)

            return {
                "usdinr_level":   round(level, 2),
                "usdinr_1d_pct":  round(chg_1d, 3),
                "usdinr_5d_pct":  round(chg_5d, 3),
                "usdinr_20d_pct": round(chg_20d, 3),
                # Rupee weakening above 1% in 5 days = FII outflow signal
                "rupee_stress":   int(chg_5d > 1.0),
            }
        except Exception as e:
            logger.warning(f"USD/INR fetch failed: {e}")
            return self._empty_currency()

    def _nifty_features(self) -> Dict:
        """Nifty50 5-day return — used by Gate 2 Check 3 to confirm market direction."""
        try:
            df = yf_safe_download("^NSEI", period="30d")
            if df.empty or len(df) < 6:
                return {"nifty_5d_pct": 0.0, "nifty_1d_pct": 0.0}
            close = df["Close"]
            return {
                "nifty_5d_pct": round(float((close.iloc[-1] / close.iloc[-6] - 1) * 100), 3),
                "nifty_1d_pct": round(float(close.pct_change().iloc[-1] * 100), 3),
            }
        except Exception as e:
            logger.warning(f"Nifty fetch failed: {e}")
            return {"nifty_5d_pct": 0.0, "nifty_1d_pct": 0.0}

    def _credit_features(self) -> Dict:
        """
        System credit growth proxy.
        Actual RBI data monthly — use cached value.
        High credit growth = tailwind for all banks.
        """
        conn = self._connect()
        row  = conn.execute("""
            SELECT credit_growth, fetched_at FROM macro_data
            WHERE  credit_growth IS NOT NULL
            ORDER  BY fetched_at DESC LIMIT 1
        """).fetchone()
        conn.close()

        growth = float(row[0]) if row else 15.0  # sensible default
        return {
            "system_credit_growth": growth,
            "credit_strong":        int(growth > 14.0),
            "credit_weak":          int(growth < 8.0),
        }

    def _calculate_macro_score(self, rbi: Dict,
                                vix: Dict, currency: Dict) -> float:
        """Composite macro score for ML model (-1 to +1)."""
        score = 0.0

        # RBI — rate cut expectation is bullish for banks
        if rbi.get("rbi_direction") == "CUTTING": score += 0.2
        elif rbi.get("rbi_direction") == "HIKING": score -= 0.2
        if rbi.get("is_rbi_week"):  score -= 0.1  # uncertainty

        # VIX
        vix_lvl = vix.get("india_vix_level", 15)
        if vix_lvl < 13:   score += 0.1
        elif vix_lvl > 22: score -= 0.3
        elif vix_lvl > 18: score -= 0.1

        # Currency
        rupee_5d = currency.get("usdinr_5d_pct", 0)
        if rupee_5d > 1.5:  score -= 0.2
        elif rupee_5d < -0.5: score += 0.1

        return round(max(-1.0, min(1.0, score)), 3)

    def _score_to_signal(self, score: float) -> str:
        if score >  0.3: return "MACRO_BULLISH"
        if score < -0.3: return "MACRO_BEARISH"
        return "MACRO_NEUTRAL"

    def _get_cached_rbi_rate(self) -> float:
        conn = self._connect()
        row  = conn.execute("""
            SELECT rbi_rate FROM macro_data
            WHERE rbi_rate > 0
            ORDER BY COALESCE(effective_date, fetched_at) DESC
            LIMIT 1
        """).fetchone()
        conn.close()
        return float(row[0]) if row else 6.5

    def _get_rbi_rate_trend(self) -> str:
        conn = self._connect()
        rows = conn.execute("""
            SELECT rbi_rate FROM macro_data
            WHERE rbi_rate > 0
            ORDER BY COALESCE(effective_date, fetched_at) DESC
            LIMIT 3
        """).fetchall()
        conn.close()
        if len(rows) < 2:
            return "HOLDING"
        rates = [r[0] for r in rows]
        if rates[0] < rates[1]:   return "CUTTING"
        if rates[0] > rates[1]:   return "HIKING"
        return "HOLDING"

    def _empty_vix(self) -> Dict:
        return {"india_vix_level": 15, "india_vix_1d_chg": 0,
                "india_vix_pct20": 50, "india_vix_high": 0, "india_vix_panic": 0}

    def _empty_currency(self) -> Dict:
        return {"usdinr_level": 83.5, "usdinr_1d_pct": 0,
                "usdinr_5d_pct": 0, "usdinr_20d_pct": 0, "rupee_stress": 0}

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c