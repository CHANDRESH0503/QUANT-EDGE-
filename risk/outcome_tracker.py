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
        hold_hours    = trade.get("hold_hours")
        exit_reason   = trade.get("reason", "")
        category      = trade.get("category", trade.get("trade_type", "swing"))

        # Entry context now flows through directly from exit_engine.close_position
        # (it was persisted on open_trades and carried into closed_trades). Only
        # fall back to a DB re-query if a caller passes a thin dict.
        confidence    = trade.get("confidence")
        alignment     = trade.get("alignment")
        regime        = trade.get("regime")
        entry_quality = trade.get("entry_quality")
        regime_match  = trade.get("regime_match")
        r_multiple    = trade.get("r_multiple")
        mfe_pct       = trade.get("mfe_pct")
        mae_pct       = trade.get("mae_pct")

        if confidence is None or alignment is None:
            meta          = self._fetch_trade_metadata(signal_uuid)
            confidence    = confidence    if confidence    is not None else meta.get("confidence", 0.0)
            alignment     = alignment     if alignment     is not None else meta.get("alignment", "")
            regime        = regime        if regime        is not None else meta.get("regime", "")
            entry_quality = entry_quality if entry_quality is not None else meta.get("entry_quality", "")
            if r_multiple is None and meta.get("risk_amount"):
                r_multiple = round(pnl_amount / max(float(meta["risk_amount"]), 0.01), 3)

        confidence      = float(confidence or 0.0)
        outcome         = self._classify_outcome(pnl_pct)
        confidence_bkt  = self._bucket(confidence)
        rm_int          = None if regime_match is None else int(bool(regime_match))

        try:
            conn = self._connect()
            conn.execute("""
                INSERT OR IGNORE INTO signal_outcomes
                (signal_uuid, ticker, signal, category, confidence, alignment,
                 regime, regime_match, entry_quality, outcome, pnl_pct,
                 pnl_amount, r_multiple, hold_hours, exit_reason,
                 mfe_pct, mae_pct, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                signal_uuid, ticker, signal, category, confidence, alignment or "",
                regime or "", rm_int, entry_quality or "", outcome, round(pnl_pct, 6),
                round(pnl_amount, 2), r_multiple, hold_hours, exit_reason,
                mfe_pct, mae_pct, str(datetime.now()),
            ))
            conn.commit()
            conn.close()
            logger.info(
                f"Outcome recorded: uuid={signal_uuid[:8]} | {ticker} {category} | "
                f"{signal} | {outcome} | pnl={pnl_pct:+.2%} "
                f"R={r_multiple if r_multiple is not None else '?'} | "
                f"conf={confidence_bkt} align={alignment or '?'} SR={entry_quality or '?'} | "
                f"{'aligned' if rm_int==1 else 'counter' if rm_int==0 else '?'}"
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
    # ATTRIBUTION — the desk dashboard
    # ─────────────────────────────────────────────────────────────

    def attribution(self, group_by: str = "category") -> Dict:
        """
        Slice realised outcomes by an entry-decision dimension and compute the
        stats a trader actually uses to decide whether an edge is real.

        group_by ∈ {"category", "regime_match", "entry_quality",
                    "alignment", "regime", "ticker"}

        Per group returns: trades, win_rate, profit_factor (gross win / gross
        loss), expectancy_R (mean R-multiple — the single best edge metric),
        avg_pnl_pct, avg_mfe, avg_mae. profit_factor > 1.5 and expectancy_R > 0
        are the green lights; the point of paper trading is to find WHICH slices
        clear that bar before risking capital on them.
        """
        col = {
            "category":      "category",
            "regime_match":  "regime_match",
            "entry_quality": "entry_quality",
            "alignment":     "alignment",
            "regime":        "regime",
            "ticker":        "ticker",
        }.get(group_by, "category")

        try:
            conn = self._connect()
            rows = conn.execute(f"""
                SELECT {col} AS grp, outcome, pnl_pct, pnl_amount,
                       r_multiple, mfe_pct, mae_pct
                FROM   signal_outcomes
                WHERE  signal NOT IN ('FLAT','')
            """).fetchall()
            conn.close()
        except Exception as e:
            logger.warning(f"attribution query failed: {e}")
            return {}

        agg: Dict = {}
        for grp, outcome, pnl_pct, pnl_amt, r_mult, mfe, mae in rows:
            key = self._label_group(group_by, grp)
            a = agg.setdefault(key, {
                "trades": 0, "wins": 0, "gross_win": 0.0, "gross_loss": 0.0,
                "r_sum": 0.0, "r_n": 0, "pnl_sum": 0.0,
                "mfe_sum": 0.0, "mfe_n": 0, "mae_sum": 0.0, "mae_n": 0,
            })
            a["trades"]  += 1
            a["wins"]    += 1 if outcome == "WIN" else 0
            amt = float(pnl_amt or 0.0)
            if amt >= 0: a["gross_win"]  += amt
            else:        a["gross_loss"] += -amt
            a["pnl_sum"] += float(pnl_pct or 0.0)
            if r_mult is not None: a["r_sum"] += float(r_mult); a["r_n"] += 1
            if mfe   is not None: a["mfe_sum"] += float(mfe);  a["mfe_n"] += 1
            if mae   is not None: a["mae_sum"] += float(mae);  a["mae_n"] += 1

        out = {}
        for key, a in agg.items():
            n  = a["trades"]
            gl = a["gross_loss"]
            out[key] = {
                "trades":         n,
                "win_rate":       round(a["wins"] / n, 3) if n else 0.0,
                "profit_factor":  round(a["gross_win"] / gl, 2) if gl > 0
                                  else (float("inf") if a["gross_win"] > 0 else 0.0),
                "expectancy_R":   round(a["r_sum"] / a["r_n"], 3) if a["r_n"] else None,
                "avg_pnl_pct":    round(a["pnl_sum"] / n, 4) if n else 0.0,
                "avg_mfe_pct":    round(a["mfe_sum"] / a["mfe_n"], 4) if a["mfe_n"] else None,
                "avg_mae_pct":    round(a["mae_sum"] / a["mae_n"], 4) if a["mae_n"] else None,
            }
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["trades"]))

    def attribution_report(self, min_trades: int = 1) -> str:
        """Human-readable multi-dimension attribution table for EOD/weekly report."""
        dims = [
            ("By category",      "category"),
            ("Aligned vs counter", "regime_match"),
            ("By S/R entry grade", "entry_quality"),
            ("By alignment grade", "alignment"),
        ]
        sections = ["═" * 64, "OUTCOME ATTRIBUTION  (paper-trading edge by decision)"]
        any_data = False
        for title, dim in dims:
            data = {k: v for k, v in self.attribution(dim).items()
                    if v["trades"] >= min_trades}
            if not data:
                continue
            any_data = True
            sections.append("─" * 64)
            sections.append(title)
            sections.append(
                f"{'group':<16}{'N':>4}{'Win':>7}{'PF':>7}{'ExpR':>7}{'MFE':>8}{'MAE':>8}"
            )
            for k, v in data.items():
                pf = "∞" if v["profit_factor"] == float("inf") else f"{v['profit_factor']:.2f}"
                er = "  -  " if v["expectancy_R"] is None else f"{v['expectancy_R']:+.2f}"
                mfe = "   -  " if v["avg_mfe_pct"] is None else f"{v['avg_mfe_pct']:+.1%}"
                mae = "   -  " if v["avg_mae_pct"] is None else f"{v['avg_mae_pct']:+.1%}"
                sections.append(
                    f"{str(k):<16}{v['trades']:>4}{v['win_rate']:>6.0%}{pf:>7}{er:>7}{mfe:>8}{mae:>8}"
                )
        if not any_data:
            return "No closed-trade outcomes yet — attribution builds as paper trades close."
        sections.append("═" * 64)
        return "\n".join(sections)

    @staticmethod
    def _label_group(group_by: str, raw) -> str:
        """Human label for a group value (esp. regime_match 1/0/None)."""
        if group_by == "regime_match":
            return {1: "aligned", 0: "counter"}.get(
                None if raw is None else int(raw), "unknown")
        return str(raw) if raw not in (None, "") else "—"

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
