# features/risk_features.py
# Risk context features — survive first, profit second
# Sources: portfolio_tracker.py, capital_mode.py, anomaly_detector.py
# These features gate position sizing, not signal direction

import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict

logger = logging.getLogger(__name__)


class RiskFeatures:
    """
    Risk management features that directly control position sizing.

    20yr trader rule #1: Survive first.
    I have seen brilliant traders blow up because they ignored risk.
    The best trading system in the world fails if it cannot
    survive a bad streak. These features are the circuit breakers.

    Features:
    - capital_mode_mult    : SMALL=0.5, GROWING=0.75, FULL=1.0
    - portfolio_heat       : total risk deployed (0 to 1)
    - monthly_dd_pct       : current month drawdown
    - monthly_dd_flag      : hit monthly loss limit?
    - anomaly_size_mult    : anomaly detected → reduce size
    - sr_quality_score     : support/resistance entry quality
    - consecutive_loss_halt: hard stop after 3 losses
    - final_size_mult      : combined position size multiplier

    The final_size_mult is fed directly to risk/capital_mode.py
    to calculate actual share quantity for each trade.
    """

    # Legacy fallback thresholds — kept for callsites that don't pass
    # capital. Live path reads mode-aware values from `CircuitBreaker.THRESHOLDS`
    # so the reduce/halt levels stay in lock-step across the two modules.
    MONTHLY_HALT_THRESHOLD = -0.06
    MONTHLY_REDUCE_THRESHOLD = -0.04

    def extract(self,
                capital:           float,
                anomaly_features:  Dict,
                sr_features:       Dict,
                meta_features:     Dict,
                db_path:           str = "database/trading.db") -> Dict:
        """
        Calculate all risk context features.

        Args:
            capital:           Current trading capital in ₹
            anomaly_features:  from AnomalyDetector.detect()
            sr_features:       from SupportResistanceEngine.get_sr_features()
            meta_features:     from MetaFeatures.extract()
            db_path:           for portfolio heat calculation
        """
        # ── Capital mode ──────────────────────────────────────────
        # Use CapitalMode.detect() so FORCE_CAPITAL_MODE env override is respected.
        # Paper trading sets FORCE_CAPITAL_MODE=FULL to unlock all 3 timeframes.
        from risk.capital_mode import CapitalMode as _CM
        cap_mode = _CM.detect(capital)
        cap_mult = {"SMALL": 0.5, "GROWING": 0.75, "FULL": 1.0}.get(cap_mode, 1.0)

        # ── Portfolio heat ────────────────────────────────────────
        heat   = self._get_portfolio_heat(db_path, capital)
        # If already >4% risk deployed, reduce new position
        if heat > 0.06:
            heat_mult = 0.0   # fully deployed
        elif heat > 0.04:
            heat_mult = 0.5
        else:
            heat_mult = 1.0

        # ── Monthly drawdown (mode-aware thresholds) ──────────────
        # Mirror CircuitBreaker.THRESHOLDS so the reduce/halt levels stay
        # in lock-step. Without this, RiskFeatures could halt at -6% while
        # CircuitBreaker only paused, or vice-versa — two systems disagreeing
        # is worse than either.
        try:
            from risk.circuit_breaker import CircuitBreaker as _CB
            _cb_thr      = _CB.THRESHOLDS.get(cap_mode, _CB.THRESHOLDS["FULL"])
            halt_thresh  = -abs(_cb_thr["halt"])
            reduce_thresh= -abs(_cb_thr["pause"])
            warn_thresh  = -abs(_cb_thr.get("warn", 0.03))
        except Exception:
            halt_thresh   = self.MONTHLY_HALT_THRESHOLD
            reduce_thresh = self.MONTHLY_REDUCE_THRESHOLD
            warn_thresh   = -0.03

        monthly_dd   = self._get_monthly_drawdown(db_path, capital)
        monthly_flag = int(monthly_dd < halt_thresh)

        if monthly_dd < halt_thresh:
            dd_mult = 0.0    # halt — circuit breaker triggered
        elif monthly_dd < reduce_thresh:
            # Between pause and halt threshold — size halved
            dd_mult = 0.5
        elif monthly_dd < warn_thresh:
            # WARN zone: between warn% and pause% — reduce but stay tradeable.
            # 20yr rule: a trader down 5% in the month should be reducing size,
            # not trading full position. 0.75× reflects a cautious mode, not a halt.
            dd_mult = 0.75
        else:
            dd_mult = 1.0

        # ── Persisted circuit-breaker state ───────────────────────
        # Gate 6 reads `trading_allowed` from here. A breach detected in any
        # cycle persists in `circuit_breaker_state`, so even cycles where the
        # monthly DD math hasn't recomputed yet still see the halt.
        try:
            from risk.circuit_breaker import CircuitBreaker as _CB2
            _cb_state = _CB2(db_path).get_current_state()
            if _cb_state.get("level") in ("PAUSE", "HALT"):
                dd_mult = 0.0
                monthly_flag = 1
        except Exception:
            _cb_state = {"level": "OK"}

        # ── Anomaly ───────────────────────────────────────────────
        anomaly_mult = float(
            anomaly_features.get("position_size_mult", 1.0)
        )

        # ── S/R entry quality ─────────────────────────────────────
        # A=1.0, B=0.75, C=0.5, D=0.25
        sr_quality   = str(sr_features.get("entry_quality", "B"))
        sr_mult_map  = {"A": 1.0, "B": 0.85, "C": 0.65, "D": 0.45}
        sr_mult      = sr_mult_map.get(sr_quality, 0.75)
        sr_score     = float(sr_features.get("entry_quality_score", 0.2))

        # ── Consecutive losses ────────────────────────────────────
        lose_streak  = int(
            meta_features.get("losing_streak_norm", 0) * 5
        )
        if lose_streak >= 3:
            loss_mult = 0.0   # hard halt — mandatory review
        elif lose_streak == 2:
            loss_mult = 0.5
        else:
            loss_mult = 1.0

        # ── Meta model health ─────────────────────────────────────
        meta_mult    = float(meta_features.get("meta_size_mult", 1.0))

        # ── Final composite size multiplier ───────────────────────
        # All multipliers compound: weakest gate dominates
        final_mult   = (
            cap_mult *
            heat_mult *
            dd_mult *
            anomaly_mult *
            sr_mult *
            loss_mult *
            meta_mult
        )
        final_mult   = round(max(0.0, min(1.5, final_mult)), 4)

        # ── ATR-based stop/target params ──────────────────────────
        reward_risk  = float(sr_features.get("reward_risk_sr", 2.5))

        return {
            # Capital context
            "capital_mode":          cap_mode,
            "capital_mode_mult":     round(cap_mult, 4),

            # Portfolio state
            "portfolio_heat":        round(heat, 4),
            "heat_mult":             round(heat_mult, 4),

            # Monthly drawdown
            "monthly_dd_pct":        round(monthly_dd, 4),
            "monthly_dd_flag":       monthly_flag,
            "monthly_dd_mult":       round(dd_mult, 4),

            # Anomaly
            "anomaly_size_mult":     round(anomaly_mult, 4),

            # S/R quality
            "entry_quality":         sr_quality,
            "sr_quality_score":      round(sr_score, 4),
            "sr_mult":               round(sr_mult, 4),

            # Loss management
            "consecutive_loss_halt": int(lose_streak >= 3),
            "loss_mult":             round(loss_mult, 4),

            # Meta
            "meta_size_mult":        round(meta_mult, 4),

            # Reward/Risk
            "reward_risk_ratio":     round(reward_risk, 4),

            # Circuit-breaker passthrough (DB-persisted level)
            "circuit_breaker_level": _cb_state.get("level", "OK"),
            "circuit_breaker_reason":_cb_state.get("reason", ""),

            # MASTER OUTPUT — used directly by risk engine
            "final_size_mult":       final_mult,
            "trading_allowed":       int(final_mult > 0),
        }

    # ─────────────────────────────────────────────────────────────
    # PORTFOLIO STATE
    # ─────────────────────────────────────────────────────────────

    def _get_portfolio_heat(self, db_path: str, capital: float) -> float:
        """
        Total % of capital currently at risk across open positions.
        Each open trade contributes (position_size * stop_pct) / capital.
        """
        try:
            conn = sqlite3.connect(db_path)
            rows = conn.execute("""
                SELECT risk_amount FROM open_trades
                WHERE status = 'OPEN'
            """).fetchall()
            conn.close()

            total_risk = sum(float(r[0]) for r in rows)
            return round(total_risk / max(capital, 1), 4)
        except Exception:
            return 0.0

    def _get_monthly_drawdown(self, db_path: str, capital: float) -> float:
        """
        Current month's P&L as fraction of starting capital.
        Negative = drawdown from month start.
        """
        try:
            month_start = datetime.now().replace(day=1).strftime("%Y-%m-%d")
            conn = sqlite3.connect(db_path)
            row = conn.execute("""
                SELECT SUM(pnl_amount) FROM closed_trades
                WHERE  close_date >= ? AND status = 'CLOSED'
            """, (month_start,)).fetchone()
            conn.close()

            monthly_pnl = float(row[0] or 0)
            return round(monthly_pnl / max(capital, 1), 4)
        except Exception:
            return 0.0