# processing/earnings_predictor.py
# Predicts likely earnings surprise BEFORE results drop
# Uses pre-earnings price drift, peer results, options implied move
# Connected to: bse_fetcher.py (dates), options_fetcher.py (straddle)

import sqlite3
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from config import get_bank_config


class EarningsPredictor:
    """
    Predicts earnings surprise direction before HDFC Bank reports.

    20yr trader insight:
    The 5 days before earnings tell you what smart money expects.
    If stock drifts up quietly on rising volume before results —
    someone knows the results will beat. I trade this pattern every quarter.

    Features generated:
    - days_to_earnings      : risk management feature (stop adding <5 days)
    - pre_earnings_drift    : 5-day price move before results
    - peer_beat_ratio       : did sector peers beat estimates this quarter?
    - implied_move_pct      : options market expected move
    - earnings_beat_pred    : probability of beat (0 to 1)
    - earnings_risk_flag    : 1 if within 5 days of results

    Connected to:
    - options_fetcher.py: gets implied move from straddle
    - bse_fetcher.py: gets official result dates
    - signal_engine.py: earnings_risk_flag reduces position size
    """

    def __init__(self, db_path: str = "database/trading.db", ticker: str = "HDFCBANK.NS"):
        self.db_path       = db_path
        self.ticker        = ticker
        _cfg               = get_bank_config(ticker)
        self._earnings_dates = _cfg["earnings_dates"]
        self._peer_tickers   = {p: p.replace(".NS", "").replace("BANK", " Bank")
                                 for p in _cfg["peers"]}
        self._setup_db()

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def get_earnings_features(self, df_price: pd.DataFrame,
                               implied_move_pct: float = 0.0) -> Dict:
        """
        Master method — return all earnings-related features.
        Called during daily signal generation.

        Args:
            df_hdfc: HDFC Bank daily OHLCV (from price_fetcher)
            implied_move_pct: from options_fetcher.get_latest()['implied_move_pct']
        """
        days   = self._days_to_next_earnings()
        drift  = self._pre_earnings_drift(df_price, days)
        peers  = self._peer_results_signal()
        hist   = self._historical_earnings_reaction()
        beat_prob = self._estimate_beat_probability(drift, peers, hist)

        # Event-aware position size multiplier: scale linearly near earnings
        # T-day=0 (earnings day), T-1=0.3×, T-2=0.5×, T-3=0.7×, ≥T-4=1.0×
        # Post-event (T+1): restore to 0.8× (uncertainty window)
        if days == 0:
            earnings_size_mult = 0.0
        elif days == 1:
            earnings_size_mult = 0.30
        elif days == 2:
            earnings_size_mult = 0.50
        elif days == 3:
            earnings_size_mult = 0.70
        elif days < 0:
            earnings_size_mult = 0.80   # day after earnings
        else:
            earnings_size_mult = 1.00

        return {
            # Core calendar features
            "days_to_earnings":         days,
            "is_earnings_week":         int(days <= 5),
            "is_earnings_day":          int(days == 0),
            "earnings_risk_flag":       int(days <= 5),
            "earnings_size_mult":       earnings_size_mult,

            # Pre-earnings signals
            "pre_earnings_drift":       round(drift, 4),
            "pre_earnings_bullish":     int(drift > 0.015),
            "pre_earnings_bearish":     int(drift < -0.015),

            # Peer results context
            "peer_beat_ratio":          peers["beat_ratio"],
            "peer_signal":              peers["signal"],

            # Options-based
            "implied_move_pct":         implied_move_pct,

            # Historical reaction patterns
            "hist_avg_gap_up":          hist.get("avg_positive_reaction", 0.0),
            "hist_beat_rate":           hist.get("beat_rate", 0.5),

            # Prediction
            "beat_probability":         round(beat_prob, 3),
            "earnings_strategy":        self._recommend_strategy(days, beat_prob,
                                                                  implied_move_pct),
        }

    def record_earnings_result(self, actual_eps: float,
                                estimated_eps: float,
                                result_date: str) -> None:
        """
        Store actual vs estimated EPS for historical analysis.
        Called after each quarterly result drops.
        Used to calculate beat_rate and improve future predictions.
        """
        surprise_pct = ((actual_eps - estimated_eps) / max(abs(estimated_eps), 0.01)) * 100
        conn = self._connect()
        conn.execute("""
            INSERT OR IGNORE INTO earnings_history
            (result_date, actual_eps, estimated_eps,
             surprise_pct, beat_estimates, created_at)
            VALUES (?,?,?,?,?,?)
        """, (
            result_date, actual_eps, estimated_eps,
            round(surprise_pct, 2), int(actual_eps >= estimated_eps),
            str(datetime.now()),
        ))
        conn.commit()
        conn.close()
        logger.info(f"Earnings recorded: surprise={surprise_pct:.1f}%")

    # ─────────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────────

    def _days_to_next_earnings(self) -> int:
        """
        Days until next scheduled HDFC Bank earnings.

        Priority order:
        1. earnings_calendar DB table (populated by BSEFetcher.sync_earnings_calendar)
        2. Hardcoded HDFC_EARNINGS_DATES fallback list
        3. Default 90 days (safe — no earnings restriction applied)
        """
        today = datetime.now().date()
        today_str = str(today)

        # ── 1. Query earnings_calendar table ─────────────────────────
        try:
            conn = self._connect()
            row = conn.execute("""
                SELECT result_date FROM earnings_calendar
                WHERE result_date >= ?
                AND ticker = ?
                ORDER BY result_date ASC
                LIMIT 1
            """, (today_str, self.ticker)).fetchone()
            conn.close()
            if row:
                next_date = datetime.strptime(str(row[0])[:10], "%Y-%m-%d").date()
                return max(0, (next_date - today).days)
        except Exception as e:
            logger.debug(f"earnings_calendar query failed, using fallback: {e}")

        # ── 2. Config fallback ─────────────────────────────────────────
        upcoming = sorted([
            datetime.strptime(d, "%Y-%m-%d").date()
            for d in self._earnings_dates
            if datetime.strptime(d, "%Y-%m-%d").date() >= today
        ])
        if upcoming:
            return max(0, (upcoming[0] - today).days)

        # ── 3. Default ────────────────────────────────────────────────
        return 90

    def _pre_earnings_drift(self, df: pd.DataFrame, days_to: int) -> float:
        """
        5-day return before earnings.
        Positive drift + volume = smart money accumulating = beat likely.
        This pattern works because institutions leak before results.
        """
        if len(df) < 10 or days_to > 10:
            return 0.0

        # How many days until earnings — look at current drift
        lookback = min(5, days_to + 1)
        if lookback < 2:
            return 0.0

        drift = float(
            (df["Close"].iloc[-1] - df["Close"].iloc[-lookback]) /
            df["Close"].iloc[-lookback]
        )
        return round(drift, 4)

    def _peer_results_signal(self) -> Dict:
        """
        Check if sector peers (ICICI, Axis, Kotak) have beaten estimates.
        Strong peer results = tailwind for HDFC Bank results.
        """
        beats  = 0
        misses = 0

        for ticker, name in self._peer_tickers.items():
            result = self._get_peer_result_from_db(ticker)
            if result is not None:
                if result > 0:
                    beats += 1
                else:
                    misses += 1

        total = beats + misses
        if total == 0:
            return {"beat_ratio": 0.5, "signal": "UNKNOWN"}

        beat_ratio = beats / total
        signal = (
            "SECTOR_STRONG"  if beat_ratio >= 0.67 else
            "SECTOR_WEAK"    if beat_ratio <= 0.33 else
            "SECTOR_MIXED"
        )
        return {"beat_ratio": round(beat_ratio, 2), "signal": signal}

    def _historical_earnings_reaction(self) -> Dict:
        """
        How has HDFC Bank historically reacted to beats vs misses?
        Uses stored earnings history from record_earnings_result().
        """
        conn = self._connect()
        rows = conn.execute("""
            SELECT surprise_pct, beat_estimates FROM earnings_history
            ORDER BY result_date DESC LIMIT 12
        """).fetchall()
        conn.close()

        if not rows:
            return {"beat_rate": 0.62, "avg_positive_reaction": 0.025}

        beats = sum(1 for r in rows if r[1] == 1)
        beat_rate = beats / len(rows)
        positive_reactions = [r[0] / 100 for r in rows if r[1] == 1 and r[0]]
        avg_pos = float(np.mean(positive_reactions)) if positive_reactions else 0.025

        return {
            "beat_rate":               round(beat_rate, 3),
            "avg_positive_reaction":   round(avg_pos, 4),
            "consecutive_beats":       self._count_streak(rows),
        }

    def _count_streak(self, rows) -> int:
        streak = 0
        for r in rows:
            if r[1] == 1:
                streak += 1
            else:
                break
        return streak

    def _estimate_beat_probability(self, drift: float,
                                    peers: Dict,
                                    hist: Dict) -> float:
        """
        Naive Bayes-style beat probability estimate.
        Combines: pre-earnings drift + peer results + historical beat rate.
        """
        base       = hist.get("beat_rate", 0.62)
        drift_adj  = 0.1 if drift > 0.02 else (-0.1 if drift < -0.02 else 0.0)
        peer_adj   = (
            0.08  if peers["signal"] == "SECTOR_STRONG" else
            -0.08 if peers["signal"] == "SECTOR_WEAK"  else 0.0
        )
        return max(0.1, min(0.9, base + drift_adj + peer_adj))

    def _recommend_strategy(self, days: int, beat_prob: float,
                             implied_move: float) -> str:
        if days <= 2:
            return "WAIT_FOR_RESULTS — too close, avoid new positions"
        if days <= 5:
            return "REDUCE_SIZE_50% — earnings risk, cap exposure"
        if days <= 10 and beat_prob > 0.70:
            return "SMALL_LONG_OK — high beat probability, but size small"
        if days <= 10 and beat_prob < 0.40:
            return "AVOID_LONGS — low beat probability approaching"
        return "NORMAL_TRADING — earnings not imminent"

    def _get_peer_result_from_db(self, ticker: str) -> Optional[float]:
        """Check if a peer result is stored in DB (from LLM earnings analysis)."""
        conn = self._connect()
        row  = conn.execute("""
            SELECT surprise_pct FROM earnings_history
            WHERE  result_date >= ? AND ticker = ?
            ORDER  BY result_date DESC LIMIT 1
        """, (str((datetime.now() - timedelta(days=90)).date()), ticker)).fetchone()
        conn.close()
        return float(row[0]) if row else None

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _setup_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS earnings_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                result_date   TEXT    UNIQUE,
                ticker        TEXT    DEFAULT 'HDFCBANK.NS',
                actual_eps    REAL,
                estimated_eps REAL,
                surprise_pct  REAL,
                beat_estimates INTEGER,
                created_at    TEXT
            )
        """)
        conn.commit()
        conn.close()