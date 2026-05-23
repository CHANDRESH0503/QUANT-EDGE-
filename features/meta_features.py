# features/meta_features.py
# Features about the model's own recent performance and signal history
# Source: logs/signal_log.csv, models/saved/
# Meta-features: recent model accuracy, signal streak, ensemble agreement

import numpy as np
import pandas as pd
import sqlite3
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SIGNAL_LOG_PATH = "logs/signal_log.csv"


class MetaFeatures:
    """
    Features about the trading system's own recent behavior.

    20yr trader insight:
    When my system has been wrong 3 times in a row, I reduce size.
    Not because the system is broken — but because either the market
    has shifted regime or I am in a drawdown that needs managing.
    The system should know its own recent track record.

    Also: when all three timeframe models agree (A+ alignment),
    I have 20% higher win rate. That alignment itself is a feature.

    Features:
    - rolling_accuracy_10  : win rate over last 10 signals
    - signal_streak        : consecutive same-direction signals
    - losing_streak        : consecutive losses (risk reduction trigger)
    - tf_alignment_score   : positional + swing + intraday agreement
    - model_confidence_avg : average ML confidence over last 5 signals
    - regime_stability     : has regime been stable or shifting?
    """

    def extract(self,
                positional_signal:  str,
                swing_signal:       str,
                intraday_signal:    str,
                positional_conf:    float,
                swing_conf:         float,
                intraday_conf:      float,
                regime_features:    Dict,
                db_path:            str = "database/trading.db") -> Dict:
        """
        Build meta features combining signal alignment + historical performance.

        Args:
            positional_signal:  "LONG" / "SHORT" / "FLAT"
            swing_signal:       "LONG" / "SHORT" / "FLAT"
            intraday_signal:    "LONG" / "SHORT" / "FLAT"
            positional_conf:    0.0 to 1.0
            swing_conf:         0.0 to 1.0
            intraday_conf:      0.0 to 1.0
            regime_features:    from RegimeDetector.get_regime_features_for_ml()
            db_path:            SQLite path for signal history
        """
        # ── Timeframe alignment ───────────────────────────────────
        signals  = [positional_signal, swing_signal, intraday_signal]
        non_flat = [s for s in signals if s != "FLAT"]

        if not non_flat:
            alignment = 0.0
            direction_score = 0.0
        else:
            # All agree?
            if len(set(non_flat)) == 1:
                alignment = float(len(non_flat)) / 3.0  # 1/3, 2/3, or 1.0
            else:
                alignment = 0.0   # conflict = zero alignment

            # Direction: LONG=+1, SHORT=-1, weighted by count
            longs  = non_flat.count("LONG")
            shorts = non_flat.count("SHORT")
            direction_score = (longs - shorts) / max(len(non_flat), 1)

        # ── Confidence composite ──────────────────────────────────
        confs      = [positional_conf, swing_conf, intraday_conf]
        avg_conf   = float(np.mean(confs))
        conf_std   = float(np.std(confs))
        # High std = models disagree on confidence even if direction matches
        conf_agreement = max(0.0, 1.0 - conf_std)

        # ── Historical performance ────────────────────────────────
        hist = self._get_signal_history(db_path)
        rolling_acc  = hist["rolling_accuracy_10"]
        signal_streak= hist["signal_streak"]
        losing_streak= hist["losing_streak"]
        monthly_pnl  = hist["monthly_pnl_norm"]

        # ── Regime stability ──────────────────────────────────────
        regime_stable = float(regime_features.get("regime_stability", 0.5))
        regime_tradeable = float(regime_features.get("regime_is_tradeable", 1))

        # ── Model health score ────────────────────────────────────
        # When recent accuracy is high AND confidence is high = take full size
        # When losing streak AND low confidence = reduce size significantly
        health = (
            rolling_acc * 0.4 +
            avg_conf    * 0.3 +
            (1 - min(losing_streak, 5) / 5) * 0.3
        )

        return {
            # Alignment
            "tf_alignment_score":   round(alignment, 4),
            "direction_consensus":  round(direction_score, 4),

            # Confidence
            "avg_model_confidence": round(avg_conf, 4),
            "conf_agreement":       round(conf_agreement, 4),

            # Historical performance
            "rolling_accuracy_10":  round(rolling_acc, 4),
            "signal_streak_norm":   round(min(signal_streak, 5) / 5, 4),
            "losing_streak_norm":   round(min(losing_streak, 5) / 5, 4),
            "monthly_pnl_norm":     round(monthly_pnl, 4),

            # Regime
            "regime_stability":     round(regime_stable, 4),
            "regime_tradeable":     round(regime_tradeable, 4),

            # Composite model health
            "model_health_score":   round(self._clip(health * 2 - 1), 4),

            # Position size multiplier from meta signals
            "meta_size_mult":       round(self._position_mult(
                                        losing_streak, rolling_acc, monthly_pnl), 4),
        }

    # ─────────────────────────────────────────────────────────────
    # HISTORICAL PERFORMANCE
    # ─────────────────────────────────────────────────────────────

    def _get_signal_history(self, db_path: str) -> Dict:
        """Load recent signal outcomes from signal log."""
        try:
            return self._from_db(db_path)
        except Exception:
            pass

        try:
            return self._from_csv()
        except Exception:
            return self._default_history()

    def _from_db(self, db_path: str) -> Dict:
        """Read from SQLite signal_outcomes table."""
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT signal, outcome, pnl_pct, created_at
            FROM   signal_outcomes
            ORDER  BY created_at DESC
            LIMIT  30
        """).fetchall()
        conn.close()

        if not rows:
            return self._default_history()

        return self._calculate_history([dict(r) for r in rows])

    def _from_csv(self) -> Dict:
        """Fallback — read from CSV log."""
        if not os.path.exists(SIGNAL_LOG_PATH):
            return self._default_history()

        df = pd.read_csv(SIGNAL_LOG_PATH, nrows=30)
        if df.empty or "outcome" not in df.columns:
            return self._default_history()

        records = df.to_dict("records")
        return self._calculate_history(records)

    def _calculate_history(self, records: List[Dict]) -> Dict:
        """Calculate rolling stats from signal history records."""
        # Rolling accuracy last 10
        last_10 = records[:10]
        wins_10 = sum(1 for r in last_10 if str(r.get("outcome", "")).upper() == "WIN")
        rolling_acc = wins_10 / max(len(last_10), 1)

        # Signal streak — consecutive same signal
        signals    = [r.get("signal", "FLAT") for r in records]
        streak     = 0
        if signals:
            first = signals[0]
            for s in signals:
                if s == first:
                    streak += 1
                else:
                    break

        # Losing streak
        outcomes   = [r.get("outcome", "WIN") for r in records]
        lose_streak= 0
        for o in outcomes:
            if str(o).upper() == "LOSS":
                lose_streak += 1
            else:
                break

        # Monthly PnL
        month_start = datetime.now().replace(day=1)
        month_records = [
            r for r in records
            if r.get("created_at", "") and
            str(r["created_at"])[:7] == month_start.strftime("%Y-%m")
        ]
        monthly_pnl = sum(float(r.get("pnl_pct", 0)) for r in month_records)
        monthly_norm = self._clip(monthly_pnl / 0.10)  # ±10% monthly = ±1.0

        return {
            "rolling_accuracy_10": rolling_acc,
            "signal_streak":       streak,
            "losing_streak":       lose_streak,
            "monthly_pnl_norm":    monthly_norm,
        }

    def _default_history(self) -> Dict:
        """Neutral defaults when no history available."""
        return {
            "rolling_accuracy_10": 0.55,
            "signal_streak":       0,
            "losing_streak":       0,
            "monthly_pnl_norm":    0.0,
        }

    def _position_mult(self, losing_streak: int,
                        rolling_acc: float, monthly_pnl: float) -> float:
        """
        Reduce position size when system is underperforming.
        This is the most important risk management meta-feature.
        """
        mult = 1.0
        # Losing streak reduction
        if losing_streak >= 3:
            mult *= 0.5
        elif losing_streak == 2:
            mult *= 0.75

        # Rolling accuracy
        if rolling_acc < 0.40:
            mult *= 0.5
        elif rolling_acc < 0.50:
            mult *= 0.75

        # Monthly drawdown
        if monthly_pnl < -0.8:   # down 8%+ this month
            mult *= 0.0           # stop trading
        elif monthly_pnl < -0.5:  # down 5%
            mult *= 0.5

        return round(max(0.0, min(1.5, mult)), 3)

    def _clip(self, val: float) -> float:
        try:
            v = float(val)
            return round(max(-1.0, min(1.0, v)), 4) if np.isfinite(v) else 0.0
        except (TypeError, ValueError):
            return 0.0