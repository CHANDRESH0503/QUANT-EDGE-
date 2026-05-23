
# database/queries.py
# Centralised query repository — all read queries live here
# Never write raw SQL elsewhere in the codebase after setup
# Connected to: dashboard/, risk/, signals/, processing/

import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DB_PATH = "database/trading.db"


class Queries:
    """
    Central repository for all database read queries.

    Design principle:
    Write queries once, use everywhere.
    If the same query appears in two places — it belongs here.
    Every query returns typed Python objects — never raw tuples.

    Categories:
    1. Performance queries  — equity, P&L, win rate
    2. Signal queries       — recent signals, alignment history
    3. Data quality queries — staleness checks, source health
    4. Regime queries       — current and historical regime
    5. Risk queries         — heat, drawdown, open positions
    6. News/Sentiment       — scored articles, LLM cache
    """

    def __init__(self, db_path: str = DB_PATH, default_ticker: str = None):
        self.db_path        = db_path
        self.default_ticker = default_ticker  # None = aggregate all banks

    def _ticker_clause(self, alias: str = "") -> tuple:
        """Return (sql_fragment, params) for optional ticker WHERE filter."""
        col = f"{alias}.ticker" if alias else "ticker"
        if self.default_ticker:
            return f" AND {col} = ?", (self.default_ticker,)
        return "", ()

    # ─────────────────────────────────────────────────────────────
    # PERFORMANCE
    # ─────────────────────────────────────────────────────────────

    def get_total_pnl(self) -> float:
        """Sum of all closed trade P&L."""
        tc, tp = self._ticker_clause()
        row = self._fetchone(
            f"SELECT COALESCE(SUM(pnl_amount),0) FROM closed_trades WHERE status='CLOSED'{tc}",
            tp
        )
        return round(float(row[0]), 2) if row else 0.0

    def get_monthly_pnl(self, year: int = None, month: int = None) -> float:
        """P&L for a specific month (defaults to current month)."""
        now   = datetime.now()
        y     = year  or now.year
        m     = month or now.month
        start = f"{y}-{m:02d}-01"
        end   = f"{y}-{m:02d}-31"
        tc, tp = self._ticker_clause()
        row   = self._fetchone(
            f"SELECT COALESCE(SUM(pnl_amount), 0) FROM closed_trades"
            f" WHERE status='CLOSED' AND close_date BETWEEN ? AND ?{tc}",
            (start, end) + tp
        )
        return round(float(row[0]), 2) if row else 0.0

    def get_performance_by_period(self, days: int = 30) -> Dict:
        """Win rate, profit factor, and trade count for last N days."""
        since  = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        tc, tp = self._ticker_clause()
        rows   = self._fetchall(
            f"SELECT pnl_pct, pnl_amount FROM closed_trades"
            f" WHERE status='CLOSED' AND close_date >= ?{tc}",
            (since,) + tp
        )
        if not rows:
            return {"win_rate": 0, "profit_factor": 0, "n_trades": 0}

        pnls     = [float(r[0]) for r in rows]
        amounts  = [float(r[1]) for r in rows]
        wins     = [p for p in pnls if p > 0]
        losses   = [p for p in pnls if p <= 0]
        gw       = sum(a for a in amounts if a > 0)
        gl       = abs(sum(a for a in amounts if a < 0))

        return {
            "win_rate":      round(len(wins) / len(pnls), 4),
            "profit_factor": round(gw / gl, 3) if gl > 0 else 0,
            "n_trades":      len(rows),
            "avg_win_pct":   round(sum(wins) / len(wins), 4) if wins else 0,
            "avg_loss_pct":  round(sum(losses) / len(losses), 4) if losses else 0,
        }

    def get_consecutive_losses(self) -> int:
        """Current streak of consecutive losing trades (most recent first)."""
        tc, tp = self._ticker_clause()
        rows   = self._fetchall(
            f"SELECT pnl_pct FROM closed_trades WHERE status='CLOSED'{tc}"
            f" ORDER BY close_date DESC LIMIT 20",
            tp
        )
        count = 0
        for r in rows:
            if float(r[0]) <= 0:
                count += 1
            else:
                break
        return count

    def get_equity_curve(self, starting_capital: float = 100_000) -> List[Dict]:
        """Full equity curve as list of {date, equity} dicts."""
        tc, tp = self._ticker_clause()
        rows   = self._fetchall(
            f"SELECT close_date, pnl_amount FROM closed_trades WHERE status='CLOSED'{tc}"
            f" ORDER BY close_date ASC",
            tp
        )
        equity = starting_capital
        curve  = []
        for r in rows:
            equity += float(r[1])
            curve.append({"date": r[0], "equity": round(equity, 2)})
        return curve

    # ─────────────────────────────────────────────────────────────
    # SIGNAL HISTORY
    # ─────────────────────────────────────────────────────────────

    def get_recent_signals(self, limit: int = 20) -> List[Dict]:
        """Most recent signal outcomes for meta-features."""
        tc, tp = self._ticker_clause()
        rows   = self._fetchall(
            f"SELECT signal, confidence, alignment, regime, outcome, pnl_pct, created_at"
            f" FROM signal_outcomes WHERE 1=1{tc} ORDER BY created_at DESC LIMIT ?",
            tp + (limit,)
        )
        return [
            {
                "signal":     r[0], "confidence": r[1],
                "alignment":  r[2], "regime":     r[3],
                "outcome":    r[4], "pnl_pct":    r[5],
                "created_at": r[6],
            }
            for r in rows
        ]

    def get_rolling_win_rate(self, n: int = 10) -> float:
        """Win rate over last N closed trades."""
        tc, tp = self._ticker_clause()
        rows   = self._fetchall(
            f"SELECT pnl_pct FROM closed_trades WHERE status='CLOSED'{tc}"
            f" ORDER BY close_date DESC LIMIT ?",
            tp + (n,)
        )
        if not rows:
            return 0.55
        wins = sum(1 for r in rows if float(r[0]) > 0)
        return round(wins / len(rows), 4)

    def get_signals_by_regime(self) -> Dict:
        """Win rate breakdown by market regime."""
        tc, tp = self._ticker_clause()
        rows   = self._fetchall(
            f"SELECT regime, COUNT(*) as total,"
            f" SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as wins"
            f" FROM signal_outcomes WHERE 1=1{tc} GROUP BY regime",
            tp
        )
        return {
            r[0]: {
                "total":    r[1],
                "win_rate": round(r[2] / r[1], 3) if r[1] > 0 else 0,
            }
            for r in rows if r[0]
        }

    def get_signals_by_alignment(self) -> Dict:
        """Win rate breakdown by alignment grade."""
        tc, tp = self._ticker_clause()
        rows   = self._fetchall(
            f"SELECT alignment, COUNT(*) as total,"
            f" SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as wins"
            f" FROM signal_outcomes WHERE 1=1{tc} GROUP BY alignment",
            tp
        )
        return {
            r[0]: {
                "total":    r[1],
                "win_rate": round(r[2] / r[1], 3) if r[1] > 0 else 0,
            }
            for r in rows if r[0]
        }

    # ─────────────────────────────────────────────────────────────
    # RISK & POSITIONS
    # ─────────────────────────────────────────────────────────────

    def get_open_positions(self) -> List[Dict]:
        """All currently open positions."""
        tc, tp = self._ticker_clause()
        rows   = self._fetchall(
            f"SELECT * FROM open_trades WHERE status='OPEN'{tc} ORDER BY opened_at",
            tp
        )
        return [dict(r) for r in rows]

    def get_portfolio_heat(self, capital: float) -> float:
        """Total risk deployed as fraction of capital."""
        tc, tp = self._ticker_clause()
        row    = self._fetchone(
            f"SELECT COALESCE(SUM(risk_amount),0) FROM open_trades WHERE status='OPEN'{tc}",
            tp
        )
        return round(float(row[0]) / max(capital, 1), 4) if row else 0.0

    def get_monthly_drawdown(self, capital: float) -> float:
        """Current month drawdown as fraction of capital."""
        month_start = datetime.now().replace(day=1).strftime("%Y-%m-%d")
        tc, tp      = self._ticker_clause()
        row         = self._fetchone(
            f"SELECT COALESCE(SUM(pnl_amount), 0) FROM closed_trades"
            f" WHERE status='CLOSED' AND close_date >= ?{tc}",
            (month_start,) + tp
        )
        return round(float(row[0]) / max(capital, 1), 4) if row else 0.0

    # ─────────────────────────────────────────────────────────────
    # REGIME
    # ─────────────────────────────────────────────────────────────

    def get_current_regime(self) -> Optional[Dict]:
        """Most recently detected market regime."""
        row = self._fetchone("""
            SELECT regime, stability, bull_prob, bear_prob,
                   high_vol_prob, choppy_prob, detected_at
            FROM   regime_snapshots
            ORDER  BY detected_at DESC LIMIT 1
        """)
        if not row:
            return None
        return {
            "regime":    row[0], "stability":    row[1],
            "bull_prob": row[2], "bear_prob":    row[3],
            "hv_prob":   row[4], "choppy_prob":  row[5],
            "detected":  row[6],
        }

    def get_regime_history(self, days: int = 10) -> List[Dict]:
        """Recent regime snapshots for stability analysis."""
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        rows  = self._fetchall("""
            SELECT regime, stability, detected_at
            FROM   regime_snapshots
            WHERE  detected_at >= ?
            ORDER  BY detected_at DESC
        """, (since,))
        return [
            {"regime": r[0], "stability": r[1], "detected_at": r[2]}
            for r in rows
        ]

    # ─────────────────────────────────────────────────────────────
    # NEWS & SENTIMENT
    # ─────────────────────────────────────────────────────────────

    def get_sentiment_summary(self, hours: int = 24) -> Dict:
        """Aggregate sentiment from scored news articles."""
        since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        row   = self._fetchone("""
            SELECT COUNT(*),
                   AVG(sentiment_score),
                   MIN(sentiment_score),
                   MAX(sentiment_score),
                   SUM(CASE WHEN sentiment_score > 0.2 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN sentiment_score < -0.2 THEN 1 ELSE 0 END)
            FROM   news
            WHERE  processed=1 AND created_at >= ?
        """, (since,))
        if not row or not row[0]:
            return {"count": 0, "avg": 0, "quality": "NO_NEWS"}
        return {
            "count":      row[0],
            "avg":        round(float(row[1] or 0), 4),
            "min":        round(float(row[2] or 0), 4),
            "max":        round(float(row[3] or 0), 4),
            "pos_count":  row[4] or 0,
            "neg_count":  row[5] or 0,
            "quality":    "GOOD" if row[0] >= 3 else "LOW",
        }

    def get_unprocessed_news(self, limit: int = 50) -> List[Dict]:
        """Articles awaiting FinBERT scoring."""
        rows = self._fetchall("""
            SELECT id, title, summary
            FROM   news
            WHERE  processed=0
            ORDER  BY created_at DESC LIMIT ?
        """, (limit,))
        return [{"id": r[0], "title": r[1], "summary": r[2]} for r in rows]

    def get_latest_llm_earnings(self) -> Optional[Dict]:
        """Most recent LLM earnings analysis result."""
        import json
        row = self._fetchone("""
            SELECT earnings_data, analyzed_at
            FROM   llm_earnings_cache
            ORDER  BY analyzed_at DESC LIMIT 1
        """)
        if not row:
            return None
        try:
            data = json.loads(row[0])
            data["analyzed_at"] = row[1]
            return data
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────
    # DATA QUALITY / STALENESS
    # ─────────────────────────────────────────────────────────────

    def get_data_freshness(self) -> Dict:
        """
        Check how fresh each data source is.
        Alerts if any source is stale beyond its expected update window.
        """
        now = datetime.now()

        def hours_ago(ts_str: Optional[str]) -> Optional[float]:
            if not ts_str:
                return None
            try:
                ts = datetime.strptime(str(ts_str)[:19], "%Y-%m-%d %H:%M:%S")
                return round((now - ts).total_seconds() / 3600, 1)
            except Exception:
                return None

        sources = {
            "global": self._fetchone(
                "SELECT fetched_at FROM global_snapshots ORDER BY fetched_at DESC LIMIT 1"
            ),
            "options": self._fetchone(
                "SELECT fetched_at FROM options_snapshots ORDER BY fetched_at DESC LIMIT 1"
            ),
            "fii": self._fetchone(
                "SELECT fetched_at FROM fii_data ORDER BY fetched_at DESC LIMIT 1"
            ),
            "news": self._fetchone(
                "SELECT created_at FROM news ORDER BY created_at DESC LIMIT 1"
            ),
            "regime": self._fetchone(
                "SELECT detected_at FROM regime_snapshots ORDER BY detected_at DESC LIMIT 1"
            ),
            "fundamentals": self._fetchone(
                "SELECT fetched_at FROM fundamentals ORDER BY fetched_at DESC LIMIT 1"
            ),
        }

        # Expected max staleness in hours
        limits = {
            "global": 12, "options": 1, "fii": 26,
            "news": 1, "regime": 2, "fundamentals": 168,  # 1 week
        }

        result = {}
        for src, row in sources.items():
            ts  = row[0] if row else None
            age = hours_ago(ts)
            result[src] = {
                "last_update": ts,
                "hours_ago":   age,
                "stale":       age is None or age > limits.get(src, 24),
                "limit_hours": limits.get(src, 24),
            }

        stale_sources = [k for k, v in result.items() if v["stale"]]
        result["_summary"] = {
            "all_fresh":     len(stale_sources) == 0,
            "stale_sources": stale_sources,
        }
        return result

    def get_system_stats(self) -> Dict:
        """Quick system stats for startup health check."""
        tables = [
            "news", "bse_announcements", "options_snapshots",
            "fii_data", "global_snapshots", "regime_snapshots",
            "open_trades", "closed_trades", "signal_outcomes",
        ]
        counts = {}
        for t in tables:
            row = self._fetchone(f"SELECT COUNT(*) FROM {t}")
            counts[t] = row[0] if row else 0
        return counts

    # ─────────────────────────────────────────────────────────────
    # WRITE HELPERS
    # ─────────────────────────────────────────────────────────────

    def log_signal_outcome(
        self,
        signal:      str,
        confidence:  float,
        alignment:   str,
        regime:      str,
        outcome:     str,
        pnl_pct:     float,
        pnl_amount:  float,
        signal_uuid: Optional[str] = None,
        ticker:      str           = "HDFCBANK.NS",
    ) -> None:
        """Record signal result after trade closes."""
        conn = self._connect()
        conn.execute("""
            INSERT INTO signal_outcomes
            (signal_uuid, ticker, signal, confidence, alignment, regime,
             outcome, pnl_pct, pnl_amount, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            signal_uuid, ticker, signal, confidence, alignment, regime,
            outcome, pnl_pct, pnl_amount, str(datetime.now()),
        ))
        conn.commit()
        conn.close()

    def latest_outcome(self) -> Optional[Dict]:
        """Most recent signal_outcomes row — used for verification + EOD report."""
        row = self._fetchone("""
            SELECT signal_uuid, signal, confidence, alignment, regime,
                   outcome, pnl_pct, pnl_amount, created_at
            FROM   signal_outcomes
            ORDER  BY id DESC LIMIT 1
        """)
        if not row:
            return None
        return {
            "signal_uuid": row[0], "signal":    row[1],
            "confidence":  row[2], "alignment": row[3],
            "regime":      row[4], "outcome":   row[5],
            "pnl_pct":     row[6], "pnl_amount": row[7],
            "created_at":  row[8],
        }

    def get_outcome_by_signal_uuid(self, signal_uuid: str) -> Optional[Dict]:
        """Look up the outcome for a specific signal — round-trip audit."""
        row = self._fetchone("""
            SELECT signal_uuid, signal, confidence, alignment, regime,
                   outcome, pnl_pct, pnl_amount, created_at
            FROM   signal_outcomes
            WHERE  signal_uuid = ? LIMIT 1
        """, (signal_uuid,))
        if not row:
            return None
        return {
            "signal_uuid": row[0], "signal":    row[1],
            "confidence":  row[2], "alignment": row[3],
            "regime":      row[4], "outcome":   row[5],
            "pnl_pct":     row[6], "pnl_amount": row[7],
            "created_at":  row[8],
        }

    def update_capital_mode(self, capital: float) -> str:
        """Return capital mode based on current equity."""
        if capital < 50_000:   return "SMALL"
        if capital < 200_000:  return "GROWING"
        return "FULL"

    # ─────────────────────────────────────────────────────────────
    # CORE HELPERS
    # ─────────────────────────────────────────────────────────────

    def _fetchone(self, sql: str, params: tuple = ()) -> Optional[tuple]:
        try:
            conn = self._connect()
            row  = conn.execute(sql, params).fetchone()
            conn.close()
            return row
        except Exception as e:
            logger.warning(f"Query error: {e} | SQL: {sql[:60]}")
            return None

    def _fetchall(self, sql: str, params: tuple = ()) -> List:
        try:
            conn = self._connect()
            rows = conn.execute(sql, params).fetchall()
            conn.close()
            return rows
        except Exception as e:
            logger.warning(f"Query error: {e} | SQL: {sql[:60]}")
            return []

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn


# ─────────────────────────────────────────────────────────────
# MODULE-LEVEL CONVENIENCE WRAPPERS
# ─────────────────────────────────────────────────────────────

def latest_outcome() -> Optional[Dict]:
    """Module-level shortcut for quick CLI verification."""
    return Queries().latest_outcome()


def get_outcome_by_signal_uuid(signal_uuid: str) -> Optional[Dict]:
    """Module-level shortcut for round-trip audit."""
    return Queries().get_outcome_by_signal_uuid(signal_uuid)