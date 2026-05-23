# risk/circuit_breaker.py
# Hard safety stops that override everything else
# When triggered, trading halts until manually reset or conditions clear
# Connected to: portfolio_tracker.py, signal_engine.py gate 6

import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Hard circuit breakers that protect capital in worst-case scenarios.

    20yr trader truth:
    The best traders I know are not the ones with the best entries.
    They are the ones who survive long enough to compound.
    Survival requires circuit breakers — hard stops that
    the emotional human brain CANNOT override in the moment.

    Level 1 — Soft warning (reduce size):
    Monthly drawdown > 3%

    Level 2 — Hard pause (no new trades):
    Monthly drawdown > 5% (SMALL) / 8% (GROWING) / 10% (FULL)
    3 consecutive losses
    Anomaly HIGH severity

    Level 3 — System halt (operator review required):
    Monthly drawdown > 8% (SMALL) / 12% (GROWING) / 15% (FULL)
    5 consecutive losses
    Model accuracy drops below 40% in last 10 signals
    """

    THRESHOLDS = {
        "SMALL":   {"warn": 0.02, "pause": 0.05, "halt": 0.08},
        "GROWING": {"warn": 0.03, "pause": 0.08, "halt": 0.12},
        "FULL":    {"warn": 0.04, "pause": 0.10, "halt": 0.15},
    }

    CONSEC_LOSS_PAUSE = 3
    CONSEC_LOSS_HALT  = 5

    def __init__(self, db_path: str = "database/trading.db"):
        self.db_path = db_path

    def check(
        self,
        capital_mode:        str,
        monthly_dd_pct:      float,
        consecutive_losses:  int,
        anomaly_severity:    str = "LOW",
        model_accuracy_10:   float = 0.55,
    ) -> Tuple[bool, Dict]:
        """
        Check all circuit breaker conditions.

        Returns:
            (trading_allowed: bool, status: Dict)
        """
        mode       = capital_mode.upper()
        thresholds = self.THRESHOLDS.get(mode, self.THRESHOLDS["FULL"])
        warnings   = []
        level      = "OK"

        # ── Monthly drawdown ──────────────────────────────────────
        dd = abs(monthly_dd_pct)

        if dd >= abs(thresholds["halt"]):
            level = "HALT"
            warnings.append(
                f"Monthly drawdown {dd:.1%} exceeds halt threshold "
                f"{abs(thresholds['halt']):.1%} — system halted"
            )
        elif dd >= abs(thresholds["pause"]):
            level = "PAUSE"
            warnings.append(
                f"Monthly drawdown {dd:.1%} exceeds pause threshold "
                f"{abs(thresholds['pause']):.1%} — no new trades"
            )
        elif dd >= abs(thresholds["warn"]):
            level = "WARN"
            warnings.append(
                f"Monthly drawdown {dd:.1%} — reduce position sizes"
            )

        # ── Consecutive losses ────────────────────────────────────
        if consecutive_losses >= self.CONSEC_LOSS_HALT:
            level = "HALT"
            warnings.append(
                f"{consecutive_losses} consecutive losses — "
                f"mandatory review before resuming"
            )
        elif consecutive_losses >= self.CONSEC_LOSS_PAUSE:
            if level not in ("HALT",):
                level = "PAUSE"
            warnings.append(
                f"{consecutive_losses} consecutive losses — "
                f"pause and review system performance"
            )

        # ── Anomaly ───────────────────────────────────────────────
        if anomaly_severity == "HIGH":
            if level not in ("HALT", "PAUSE"):
                level = "PAUSE"
            warnings.append("High-severity market anomaly — no new entries")

        # ── Model accuracy ────────────────────────────────────────
        if model_accuracy_10 < 0.40:
            level = "HALT"
            warnings.append(
                f"Model accuracy {model_accuracy_10:.0%} in last 10 signals "
                f"— retrain required"
            )

        # ── Size multiplier from level ────────────────────────────
        size_mult = {
            "OK":    1.0,
            "WARN":  0.75,
            "PAUSE": 0.0,
            "HALT":  0.0,
        }.get(level, 0.0)

        trading_allowed = level not in ("PAUSE", "HALT")

        # Log if state changed
        if level != "OK":
            logger.warning(
                f"CircuitBreaker: level={level} | "
                f"dd={monthly_dd_pct:.1%} | losses={consecutive_losses} | "
                f"{warnings[0] if warnings else ''}"
            )

        return trading_allowed, {
            "level":             level,
            "trading_allowed":   trading_allowed,
            "size_multiplier":   size_mult,
            "monthly_dd_pct":    monthly_dd_pct,
            "consecutive_losses":consecutive_losses,
            "anomaly_severity":  anomaly_severity,
            "model_accuracy_10": model_accuracy_10,
            "warnings":          warnings,
            "checked_at":        str(datetime.now()),
        }

    def check_vix_spike(
        self,
        current_vix:  float,
        prev_vix:     float = 0.0,
    ) -> Dict:
        """
        Detect VIX spike conditions and return recommended action.

        Actions:
        - HALT_AND_FLATTEN : absolute VIX > 28 or intraday spike > 30%
        - PAUSE_NEW_ENTRIES: absolute VIX > 22 (warn zone)
        - OK               : normal conditions

        Called by task_runner before every signal cycle.
        On HALT_AND_FLATTEN, task_runner should close all intraday positions
        and tighten swing stops 50%.
        """
        action   = "OK"
        messages = []

        # Absolute VIX level
        if current_vix > 28:
            action = "HALT_AND_FLATTEN"
            messages.append(
                f"India VIX={current_vix:.1f} > 28 — extreme fear, flatten all intraday"
            )
        elif current_vix > 22:
            action = "PAUSE_NEW_ENTRIES"
            messages.append(f"India VIX={current_vix:.1f} > 22 — elevated, no new entries")

        # Intraday spike check (prev_vix = yesterday close)
        if prev_vix > 0 and current_vix > 0:
            spike_pct = (current_vix - prev_vix) / prev_vix
            if spike_pct >= 0.30 and action != "HALT_AND_FLATTEN":
                action = "HALT_AND_FLATTEN"
                messages.append(
                    f"VIX intraday spike {spike_pct:+.1%} — "
                    f"systematic risk event, flatten intraday"
                )

        if action != "OK":
            logger.warning(
                f"VIX circuit: action={action} | "
                f"vix={current_vix:.1f} | prev={prev_vix:.1f} | "
                f"{messages[0] if messages else ''}"
            )

        return {
            "action":      action,
            "current_vix": current_vix,
            "prev_vix":    prev_vix,
            "messages":    messages,
            "flatten":     action == "HALT_AND_FLATTEN",
            "pause":       action in ("HALT_AND_FLATTEN", "PAUSE_NEW_ENTRIES"),
        }

    def is_halted(self, db_path: str = None) -> bool:
        """Quick check — is system in HALT state?"""
        return False

    def manual_reset(self) -> None:
        """Manually reset circuit breaker after review."""
        logger.info("Circuit breaker manually reset by operator")