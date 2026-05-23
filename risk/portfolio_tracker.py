# risk/portfolio_tracker.py
# Tracks portfolio state: equity, drawdown, P&L, open positions
# Connected to: exit_engine.py, circuit_breaker.py, dashboard/performance.py

import sqlite3
import logging
import csv
import os
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger(__name__)

TRADE_LOG = "logs/trade_log.csv"


class PortfolioTracker:
    """
    Tracks all portfolio metrics needed for risk management.

    Provides:
    - Current equity (starting capital + cumulative P&L)
    - Monthly drawdown (for circuit breaker)
    - Open position heat (total risk deployed)
    - Win rate, profit factor, expectancy
    - P&L curve for dashboard

    20yr trader rule:
    What gets measured gets managed.
    If you do not track every trade, you cannot improve.
    The log is more important than the signal.
    """

    def __init__(
        self,
        starting_capital: float = 100_000.0,
        db_path: str = "database/trading.db",
    ):
        self.starting_capital = starting_capital
        self.db_path          = db_path

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def get_current_equity(self) -> float:
        """Starting capital + all closed trade P&L."""
        total_pnl = self._sum_closed_pnl()
        return round(self.starting_capital + total_pnl, 2)

    def get_monthly_pnl(self) -> float:
        """P&L since start of current calendar month."""
        month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0)
        conn        = self._connect()
        row         = conn.execute("""
            SELECT COALESCE(SUM(pnl_amount), 0)
            FROM   closed_trades
            WHERE  close_date >= ? AND status='CLOSED'
        """, (str(month_start),)).fetchone()
        conn.close()
        return round(float(row[0]), 2)

    def get_monthly_dd_pct(self) -> float:
        """Monthly drawdown as fraction of starting capital."""
        return round(self.get_monthly_pnl() / max(self.starting_capital, 1), 4)

    def get_portfolio_heat(self) -> float:
        """Total risk deployed across all open positions (fraction of equity)."""
        equity = self.get_current_equity()
        conn   = self._connect()
        row    = conn.execute("""
            SELECT COALESCE(SUM(risk_amount), 0)
            FROM   open_trades WHERE status='OPEN'
        """).fetchone()
        conn.close()
        total_risk = float(row[0])
        return round(total_risk / max(equity, 1), 4)

    def get_performance_stats(self) -> Dict:
        """Complete performance statistics for dashboard and psychology tracker."""
        conn  = self._connect()
        rows  = conn.execute("""
            SELECT pnl_pct, pnl_amount, signal, exit_reason, close_date
            FROM   closed_trades
            WHERE  status='CLOSED'
            ORDER  BY close_date DESC
        """).fetchall()
        conn.close()

        if not rows:
            return self._empty_stats()

        trades    = [dict(r) for r in rows]
        pnl_pcts  = [float(t["pnl_pct"]) for t in trades]
        wins      = [p for p in pnl_pcts if p > 0]
        losses    = [p for p in pnl_pcts if p <= 0]

        total     = len(trades)
        win_count = len(wins)
        win_rate  = win_count / total

        avg_win   = float(sum(wins)   / len(wins))   if wins   else 0.0
        avg_loss  = float(sum(losses) / len(losses)) if losses else 0.0

        profit_factor = (
            abs(sum(wins)) / abs(sum(losses))
            if losses and sum(losses) != 0 else 0.0
        )
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Consecutive losses (most recent)
        consec_losses = 0
        for t in trades:
            if float(t["pnl_pct"]) <= 0:
                consec_losses += 1
            else:
                break

        # Monthly breakdown
        month_pnl = self.get_monthly_pnl()
        equity    = self.get_current_equity()

        return {
            "total_trades":      total,
            "win_rate":          round(win_rate, 4),
            "avg_win_pct":       round(avg_win, 4),
            "avg_loss_pct":      round(avg_loss, 4),
            "profit_factor":     round(profit_factor, 3),
            "expectancy_pct":    round(expectancy, 4),
            "consecutive_losses":consec_losses,
            "current_equity":    equity,
            "monthly_pnl":       month_pnl,
            "monthly_dd_pct":    self.get_monthly_dd_pct(),
            "portfolio_heat":    self.get_portfolio_heat(),
            "total_pnl":         round(self._sum_closed_pnl(), 2),
            "total_return_pct":  round((equity - self.starting_capital) / self.starting_capital, 4),
        }

    def log_trade(self, trade: Dict) -> None:
        """Append closed trade to CSV log."""
        os.makedirs(os.path.dirname(TRADE_LOG), exist_ok=True)
        write_header = not os.path.exists(TRADE_LOG)
        with open(TRADE_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "date", "signal", "entry", "exit", "shares",
                    "pnl_amount", "pnl_pct", "exit_reason",
                ])
            writer.writerow([
                trade.get("close_date", str(datetime.now())),
                trade.get("signal", ""),
                trade.get("entry_price", 0),
                trade.get("exit_price", 0),
                trade.get("shares", 0),
                trade.get("pnl_amount", 0),
                trade.get("pnl_pct", 0),
                trade.get("exit_reason", ""),
            ])

    def _sum_closed_pnl(self) -> float:
        try:
            conn = self._connect()
            row  = conn.execute(
                "SELECT COALESCE(SUM(pnl_amount),0) FROM closed_trades WHERE status='CLOSED'"
            ).fetchone()
            conn.close()
            return float(row[0])
        except Exception:
            return 0.0

    def _empty_stats(self) -> Dict:
        return {
            "total_trades": 0, "win_rate": 0, "avg_win_pct": 0,
            "avg_loss_pct": 0, "profit_factor": 0, "expectancy_pct": 0,
            "consecutive_losses": 0,
            "current_equity": self.starting_capital,
            "monthly_pnl": 0, "monthly_dd_pct": 0,
            "portfolio_heat": 0, "total_pnl": 0, "total_return_pct": 0,
        }

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c