# dashboard/psychology_tracker.py
# Detects behavioral biases from trade log data
# The most important dashboard module — discipline is the edge

import pandas as pd
import numpy as np
import sqlite3
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger(__name__)

TRADE_LOG  = "logs/trade_log.csv"
SIGNAL_LOG = "logs/signal_log.csv"


class PsychologyTracker:
    """
    Detects trading psychology issues from the trade log.

    20yr trader truth:
    I have watched hundreds of traders with great systems lose money
    because of their own behaviour. The system works — they don't.

    The five most destructive trading biases (in order of damage):
    1. Loss aversion    — holding losers too long, cutting winners too early
    2. Overtrading      — trading out of boredom, not edge
    3. Revenge trading  — doubling down after a loss to "get it back"
    4. Early exit       — panic-selling winners before target
    5. Inconsistent sizing — gut-feeling position sizes instead of formula

    This tracker measures each bias from actual trade data.
    Weekly review is mandatory. Improvement must be measurable.
    """

    THRESHOLDS = {
        "loss_aversion_ratio":  1.5,    # avg loss hold > 1.5× avg win hold
        "overtrade_flat_pct":   0.15,   # > 15% trades with conf < 0.55
        "revenge_hours":        24,     # trade within 24h of a loss
        "early_exit_ratio":     0.60,   # avg win < 60% of target
        "sizing_cv":            0.50,   # position size CV > 50%
    }

    def __init__(
        self,
        db_path: str = "database/trading.db",
    ):
        self.db_path = db_path

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def analyse(self) -> Dict:
        """
        Full psychology analysis from trade and signal logs.
        Called weekly — results included in Sunday report.
        """
        trades  = self._load_trades()
        signals = self._load_signals()

        if len(trades) < 10:
            return {
                "status":  "INSUFFICIENT_DATA",
                "message": "Need 10+ closed trades for psychology analysis",
            }

        biases = {
            "loss_aversion":    self._check_loss_aversion(trades),
            "overtrading":      self._check_overtrading(signals),
            "revenge_trading":  self._check_revenge_trading(trades),
            "early_exit":       self._check_early_exit(trades),
            "inconsistent_sizing": self._check_sizing_consistency(trades),
        }

        flags    = [k for k, v in biases.items() if v.get("detected")]
        severity = self._severity(len(flags))

        return {
            "analysis_date": str(datetime.now()),
            "trades_analysed": len(trades),
            "biases":          biases,
            "flags_detected":  flags,
            "n_flags":         len(flags),
            "severity":        severity,
            "overall_grade":   self._grade(len(flags)),
            "action_items":    self._action_items(biases),
            "summary":         self._summary(biases, len(flags)),
        }

    def format_telegram(self, analysis: Dict) -> str:
        """Format psychology report as Telegram message."""
        if analysis.get("status") == "INSUFFICIENT_DATA":
            return f"🧠 Psychology: Need 10+ trades for analysis"

        grade   = analysis.get("overall_grade", "B")
        flags   = analysis.get("flags_detected", [])
        n       = analysis.get("n_flags", 0)
        biases  = analysis.get("biases", {})

        grade_emoji = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴"}.get(grade, "🟡")

        lines = [
            f"🧠 *Psychology Report*",
            f"━━━━━━━━━━━━━━━━━━━━",
            f"{grade_emoji} Overall Grade: *{grade}*",
            f"Trades analysed: {analysis['trades_analysed']}",
            f"",
        ]

        for bias_name, data in biases.items():
            detected = data.get("detected", False)
            label    = bias_name.replace("_", " ").title()
            emoji    = "⚠️" if detected else "✅"
            value    = data.get("value_str", "")
            lines.append(f"  {emoji} {label}: {value}")

        actions = analysis.get("action_items", [])
        if actions:
            lines += ["", "📋 *Action Items*"]
            for a in actions[:3]:
                lines.append(f"  → {a}")

        lines.append(f"━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────
    # BIAS DETECTORS
    # ─────────────────────────────────────────────────────────────

    def _check_loss_aversion(self, trades: List[Dict]) -> Dict:
        """
        Holding losses longer than winners = loss aversion.
        Detection: avg hold days of losing trades > 1.5× winning trades.
        Fix: Set hard time exits. If it hasn't moved in 3 days, exit.
        """
        wins   = [t for t in trades if float(t.get("pnl_pct", 0)) > 0]
        losses = [t for t in trades if float(t.get("pnl_pct", 0)) <= 0]

        win_hold  = self._avg_hold(wins)
        loss_hold = self._avg_hold(losses)

        if win_hold == 0:
            return {"detected": False, "value_str": "insufficient data"}

        ratio    = loss_hold / win_hold
        detected = ratio > self.THRESHOLDS["loss_aversion_ratio"]

        return {
            "detected":   detected,
            "ratio":      round(ratio, 2),
            "win_hold":   round(win_hold, 1),
            "loss_hold":  round(loss_hold, 1),
            "value_str":  f"Loss hold {loss_hold:.1f}d vs Win hold {win_hold:.1f}d (ratio={ratio:.1f})",
            "message":    (
                f"Holding losses {ratio:.1f}× longer than winners — classic loss aversion"
                if detected else "Hold times balanced ✓"
            ),
        }

    def _check_overtrading(self, signals: List[Dict]) -> Dict:
        """
        Taking trades with low confidence = desperation trading.
        Detection: > 15% of non-FLAT signals have confidence < 55%.
        Fix: Hard minimum confidence gate in signal_engine.
        """
        non_flat = [s for s in signals if s.get("signal") != "FLAT"]
        if not non_flat:
            return {"detected": False, "value_str": "no trades"}

        low_conf = [
            s for s in non_flat
            if float(s.get("confidence", 1)) < 0.55
        ]
        low_rate = len(low_conf) / len(non_flat)
        detected = low_rate > self.THRESHOLDS["overtrade_flat_pct"]

        return {
            "detected":  detected,
            "low_conf_rate": round(low_rate, 3),
            "low_conf_count":len(low_conf),
            "total_signals": len(non_flat),
            "value_str": f"{low_rate:.0%} of trades below 55% confidence",
            "message":   (
                f"{low_rate:.0%} of signals had conf < 55% — reduce threshold"
                if detected else "Signal confidence quality good ✓"
            ),
        }

    def _check_revenge_trading(self, trades: List[Dict]) -> Dict:
        """
        Trading too soon after a loss = revenge trading.
        Detection: Next trade entered < 24 hours after a loss.
        Fix: Mandatory 24-hour cooldown after any losing trade.
        """
        sorted_trades  = sorted(
            trades,
            key=lambda t: str(t.get("close_date", t.get("exit_date", ""))),
        )
        revenge_count  = 0
        for i in range(1, len(sorted_trades)):
            prev = sorted_trades[i - 1]
            curr = sorted_trades[i]
            if float(prev.get("pnl_pct", 0)) <= 0:
                try:
                    prev_exit = pd.to_datetime(
                        prev.get("close_date", prev.get("exit_date"))
                    )
                    curr_entry = pd.to_datetime(
                        curr.get("entry_date", curr.get("opened_at"))
                    )
                    hours_gap  = (curr_entry - prev_exit).total_seconds() / 3600
                    if hours_gap < self.THRESHOLDS["revenge_hours"]:
                        revenge_count += 1
                except Exception:
                    pass

        total    = max(len(trades) - 1, 1)
        rate     = revenge_count / total
        detected = revenge_count >= 2

        return {
            "detected":      detected,
            "revenge_count": revenge_count,
            "rate":          round(rate, 3),
            "value_str":     f"{revenge_count} trades entered < 24h after a loss",
            "message":       (
                f"{revenge_count} revenge trades detected — enforce 24h cooldown"
                if detected else "No revenge trading detected ✓"
            ),
        }

    def _check_early_exit(self, trades: List[Dict]) -> Dict:
        """
        Selling winners before target = letting fear cut profits short.
        Detection: avg winning trade exit < 60% of target distance.
        Fix: Set limit orders at target immediately on entry.
        """
        wins = [t for t in trades if float(t.get("pnl_pct", 0)) > 0]
        if not wins:
            return {"detected": False, "value_str": "no winning trades yet"}

        exit_ratios = []
        for t in wins:
            entry  = float(t.get("entry_price", 0))
            exit_p = float(t.get("exit_price",  0))
            target = float(t.get("target_price", 0))
            if entry > 0 and target > entry:
                achieved = (exit_p - entry) / (target - entry)
                exit_ratios.append(min(achieved, 1.5))

        if not exit_ratios:
            return {"detected": False, "value_str": "target data unavailable"}

        avg_ratio = float(np.mean(exit_ratios))
        detected  = avg_ratio < self.THRESHOLDS["early_exit_ratio"]

        return {
            "detected":   detected,
            "avg_ratio":  round(avg_ratio, 3),
            "value_str":  f"Avg win exits at {avg_ratio:.0%} of target",
            "message":    (
                f"Exiting winners at {avg_ratio:.0%} of target — let winners run"
                if detected else f"Holding to target well ({avg_ratio:.0%}) ✓"
            ),
        }

    def _check_sizing_consistency(self, trades: List[Dict]) -> Dict:
        """
        Inconsistent sizing = gut feeling overriding the formula.
        Detection: CV of position sizes > 50% (excluding size_mult adjustments).
        Fix: Always and only use ATR-based formula. Remove discretion.
        """
        sizes = [
            float(t.get("position_value", 0))
            for t in trades
            if float(t.get("position_value", 0)) > 0
        ]
        if len(sizes) < 5:
            return {"detected": False, "value_str": "need 5+ trades"}

        cv       = float(np.std(sizes) / np.mean(sizes))
        detected = cv > self.THRESHOLDS["sizing_cv"]

        return {
            "detected":    detected,
            "cv":          round(cv, 3),
            "avg_size":    round(float(np.mean(sizes)), 2),
            "std_size":    round(float(np.std(sizes)),  2),
            "value_str":   f"Position size CV = {cv:.0%} (std/mean)",
            "message":     (
                f"Position sizes vary {cv:.0%} — use formula consistently"
                if detected else "Position sizing consistent ✓"
            ),
        }

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _avg_hold(self, trades: List[Dict]) -> float:
        days = []
        for t in trades:
            try:
                entry = pd.to_datetime(t.get("entry_date", t.get("opened_at")))
                exit_ = pd.to_datetime(t.get("close_date", t.get("exit_date")))
                days.append((exit_ - entry).days)
            except Exception:
                pass
        return float(np.mean(days)) if days else 0.0

    def _severity(self, n_flags: int) -> str:
        if n_flags == 0: return "NONE"
        if n_flags == 1: return "LOW"
        if n_flags == 2: return "MEDIUM"
        return "HIGH"

    def _grade(self, n_flags: int) -> str:
        return {0: "A", 1: "B", 2: "C"}.get(n_flags, "D")

    def _action_items(self, biases: Dict) -> List[str]:
        actions = []
        if biases.get("loss_aversion", {}).get("detected"):
            actions.append("Set hard 3-day time exit rule for all swing trades")
        if biases.get("overtrading", {}).get("detected"):
            actions.append("Raise minimum confidence threshold by 5% in gate 6")
        if biases.get("revenge_trading", {}).get("detected"):
            actions.append("Enforce 24-hour mandatory pause after every loss")
        if biases.get("early_exit", {}).get("detected"):
            actions.append("Place limit orders at target immediately on entry — no manual exit")
        if biases.get("inconsistent_sizing", {}).get("detected"):
            actions.append("Use only ATR formula for sizing — remove all discretion")
        return actions

    def _summary(self, biases: Dict, n_flags: int) -> str:
        if n_flags == 0:
            return "No psychological biases detected. System discipline is excellent."
        if n_flags <= 2:
            return f"{n_flags} minor bias(es) detected. Address with action items."
        return (
            f"{n_flags} significant biases active. "
            "Psychology is costing real money. Prioritise correction."
        )

    def _load_trades(self) -> List[Dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT ct.*, ot.target_price, ot.opened_at as entry_date
                FROM   closed_trades ct
                LEFT JOIN open_trades ot ON ot.id = ct.id
                WHERE  ct.status = 'CLOSED'
                ORDER  BY ct.close_date DESC
            """).fetchall()
            conn.close()
            if rows:
                return [dict(r) for r in rows]
        except Exception:
            pass
        if os.path.exists(TRADE_LOG):
            try:
                return pd.read_csv(TRADE_LOG).to_dict("records")
            except Exception:
                pass
        return []

    def _load_signals(self) -> List[Dict]:
        if not os.path.exists(SIGNAL_LOG):
            return []
        try:
            return pd.read_csv(SIGNAL_LOG, nrows=200).to_dict("records")
        except Exception:
            return []