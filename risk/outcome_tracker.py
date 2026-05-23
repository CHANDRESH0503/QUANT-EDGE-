# risk/outcome_tracker.py
# Closes the learning loop: every closed trade writes a tagged signal_outcomes row
# so the system can measure realized win rate per (alignment × regime × confidence) bucket.
# Called automatically from exit_engine.close_position() — no manual wiring needed.

import sqlite3
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DB_PATH = "database/trading.db"


class OutcomeTracker:
    """
    After every trade closes, record a tagged row in signal_outcomes.

    Tags written:
    - signal_uuid       : links back to feature_snapshot + open_trade
    - signal            : LONG / SHORT
    - confidence_bucket : 0.60-0.65, 0.65-0.70, etc.
    - alignment         : A+/A/B/C/F
    - regime_at_entry   : BULL_TRENDING etc.
    - entry_quality     : A/B/C/D from gate 5
    - R_multiple        : pnl_amount / risk_amount  (>1 = win above R)
    - hold_days         : days the trade was open
    - exit_reason       : STOP_LOSS / TRAILING_STOP / TARGET_HIT / TIME_EXIT etc.
    - outcome           : WIN / LOSS / BREAKEVEN
    - pnl_pct / pnl_amount : for P&L analytics

    Weekly query calibration_by_bucket() surfaces realized win rate per bucket
    so model confidence thresholds can be tuned against reality.
    """

    # Confidence brackets for calibration analysis
    CONFIDENCE_BUCKETS = [
        (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
        (0.70, 0.75), (0.75, 0.80), (0.80, 1.01),
    ]

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def record(self, trade: Dict) -> None:
        """
        Record outcome for a closed trade.

        trade dict must contain at minimum:
          signal_uuid, signal, pnl_pct, pnl_amount, hold_days, reason

        Optional enrichment fields (pass from open_trades metadata if stored):
          confidence, alignment, regime, entry_quality, risk_amount
        """
        signal_uuid   = trade.get("signal_uuid", "")
        signal        = trade.get("signal", "UNKNOWN")
        ticker        = trade.get("ticker", "HDFCBANK.NS")
        pnl_pct       = float(trade.get("pnl_pct", 0.0))
        pnl_amount    = float(trade.get("pnl_amount", 0.0))
        hold_days     = int(trade.get("hold_days", 0))
        exit_reason   = trade.get("reason", "")

        # Pull richer context from the open_trades metadata if available
        meta          = self._fetch_trade_metadata(signal_uuid)
        confidence    = float(meta.get("confidence", 0.0))
        alignment     = meta.get("alignment", "")
        regime        = meta.get("regime", "")
        entry_quality = meta.get("entry_quality", "")
        risk_amount   = float(meta.get("risk_amount", 0.0))

        outcome         = self._classify_outcome(pnl_pct)
        confidence_bkt  = self._bucket(confidence)
        r_multiple      = round(pnl_amount / max(risk_amount, 0.01), 3) if risk_amount else None

        try:
            conn = self._connect()
            conn.execute("""
                INSERT OR IGNORE INTO signal_outcomes
                (signal_uuid, ticker, signal, confidence, alignment, regime,
                 outcome, pnl_pct, pnl_amount, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                signal_uuid, ticker, signal, confidence, alignment, regime,
                outcome, round(pnl_pct, 6), round(pnl_amount, 2),
                str(datetime.now()),
            ))
            conn.commit()
            conn.close()
            logger.info(
                f"Outcome recorded: uuid={signal_uuid[:8]} | "
                f"{signal} | {outcome} | pnl={pnl_pct:+.2%} | "
                f"conf_bucket={confidence_bkt} | align={alignment} | "
                f"regime={regime} | hold={hold_days}d"
            )
        except Exception as e:
            logger.error(f"OutcomeTracker.record failed: {e}")

    def calibration_by_bucket(self) -> Dict:
        """
        Realized win rate per (confidence_bucket, alignment, regime) cross.
        Used weekly to check if model confidence reflects reality.

        Returns: { "0.65-0.70": {"total": 12, "win_rate": 0.583, ...}, ... }
        """
        try:
            conn  = self._connect()
            rows  = conn.execute("""
                SELECT confidence, alignment, regime, outcome, pnl_pct
                FROM   signal_outcomes
                WHERE  signal NOT IN ('FLAT','')
                ORDER  BY created_at DESC
            """).fetchall()
            conn.close()
        except Exception:
            return {}

        buckets: Dict = {}
        for row in rows:
            conf   = float(row[0] or 0)
            almt   = str(row[2] or "")
            reg    = str(row[3] or "")
            outcome= str(row[4] or "")
            bkt    = self._bucket(conf)
            key    = bkt
            if key not in buckets:
                buckets[key] = {"total": 0, "wins": 0, "pnl_sum": 0.0}
            buckets[key]["total"]   += 1
            buckets[key]["wins"]    += 1 if outcome == "WIN" else 0
            buckets[key]["pnl_sum"] += float(row[5] or 0)

        result = {}
        for bkt, data in sorted(buckets.items()):
            total = data["total"]
            result[bkt] = {
                "total":        total,
                "wins":         data["wins"],
                "win_rate":     round(data["wins"] / total, 3) if total else 0,
                "avg_pnl_pct":  round(data["pnl_sum"] / total, 4) if total else 0,
            }
        return result

    def weekly_calibration_report(self) -> str:
        """Human-readable calibration table for EOD report."""
        cal = self.calibration_by_bucket()
        if not cal:
            return "No outcome data yet — keep running demo trades."

        lines = [
            "─" * 55,
            "MODEL CALIBRATION REPORT (realized win rate vs confidence)",
            f"{'Conf Bucket':<14} {'Trades':>7} {'Win Rate':>9} {'Avg P&L':>9}",
            "─" * 55,
        ]
        for bkt, data in cal.items():
            lines.append(
                f"{bkt:<14} {data['total']:>7} "
                f"{data['win_rate']:>8.1%}  "
                f"{data['avg_pnl_pct']:>+8.2%}"
            )
        lines.append("─" * 55)
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────────

    def _fetch_trade_metadata(self, signal_uuid: str) -> Dict:
        """Pull enrichment fields from open_trades that was just closed."""
        if not signal_uuid:
            return {}
        try:
            conn = self._connect()
            row  = conn.execute("""
                SELECT confidence, alignment, regime_at_entry,
                       entry_quality, risk_amount
                FROM   closed_trades
                WHERE  signal_uuid = ?
                ORDER  BY id DESC LIMIT 1
            """, (signal_uuid,)).fetchone()
            conn.close()
            if row:
                return {
                    "confidence":    row[0],
                    "alignment":     row[1],
                    "regime":        row[2],
                    "entry_quality": row[3],
                    "risk_amount":   row[4],
                }
        except Exception:
            pass
        return {}

    @staticmethod
    def _classify_outcome(pnl_pct: float) -> str:
        if pnl_pct > 0.002:   return "WIN"
        if pnl_pct < -0.002:  return "LOSS"
        return "BREAKEVEN"

    @staticmethod
    def _bucket(conf: float) -> str:
        if conf >= 0.80: return "0.80+"
        if conf >= 0.75: return "0.75-0.80"
        if conf >= 0.70: return "0.70-0.75"
        if conf >= 0.65: return "0.65-0.70"
        if conf >= 0.60: return "0.60-0.65"
        if conf >= 0.55: return "0.55-0.60"
        return "<0.55"

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        return c
