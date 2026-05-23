# dashboard/performance.py
# Generates the daily performance report for the trader
# Pulls from portfolio_tracker, backtest metrics, and signal log
# Delivered via Telegram at 6 PM daily and on demand

import pandas as pd
import numpy as np
import sqlite3
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger(__name__)

SIGNAL_LOG = "logs/signal_log.csv"
TRADE_LOG  = "logs/trade_log.csv"


class PerformanceDashboard:
    """
    Generates daily and weekly performance reports.

    20yr trader perspective:
    Every day I review the same 6 numbers:
    1. Today's P&L
    2. Month-to-date P&L
    3. Win rate (last 10 trades)
    4. Profit factor (last 20 trades)
    5. Max consecutive losses
    6. Current drawdown from peak

    These 6 numbers tell me everything I need to know
    about whether the system is working or breaking down.
    Anything else is noise.

    The dashboard answers: Is the system healthy? Should I size up or down?
    """

    def __init__(
        self,
        starting_capital: float = 100_000.0,
        db_path:          str   = "database/trading.db",
    ):
        self.starting_capital = starting_capital
        self.db_path          = db_path

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def daily_report(self) -> Dict:
        """
        Full daily performance report.
        Called at 6 PM every trading day.
        """
        trades   = self._load_trades()
        signals  = self._load_signals()
        equity   = self._equity_curve(trades)
        stats    = self._calculate_stats(trades)
        today    = self._today_summary(trades, signals)
        health   = self._system_health(stats, signals)

        return {
            "date":            datetime.now().strftime("%d %b %Y"),
            "today":           today,
            "stats":           stats,
            "equity":          equity,
            "system_health":   health,
            "report_type":     "daily",
            "generated_at":    str(datetime.now()),
        }

    def weekly_report(self) -> Dict:
        """
        Weekly performance report with deeper analysis.
        Called every Sunday at 6 PM.
        """
        daily    = self.daily_report()
        weekly   = self._weekly_breakdown()
        features = self._feature_performance()
        regimes  = self._regime_performance()

        return {
            **daily,
            "weekly_breakdown":    weekly,
            "feature_performance": features,
            "regime_performance":  regimes,
            "report_type":         "weekly",
        }

    def format_telegram(self, report: Dict) -> str:
        """Format report as Telegram message."""
        stats    = report.get("stats", {})
        today    = report.get("today", {})
        health   = report.get("system_health", {})
        equity   = report.get("equity", {})
        date     = report.get("date", "")

        win_rate    = float(stats.get("win_rate_10",  0))
        pf          = float(stats.get("profit_factor_20", 0))
        monthly_pnl = float(today.get("month_pnl", 0))
        monthly_dd  = float(stats.get("monthly_dd_pct", 0))
        consec_loss = int(stats.get("consec_losses", 0))
        cur_equity  = float(equity.get("current", self.starting_capital))
        total_ret   = float(equity.get("total_return_pct", 0))
        heat        = float(stats.get("portfolio_heat", 0))

        health_emoji = {
            "EXCELLENT": "🟢", "GOOD": "🟡",
            "DEGRADED": "🟠", "POOR": "🔴",
        }.get(health.get("grade", "GOOD"), "🟡")

        today_pnl   = float(today.get("today_pnl", 0))
        today_trades= int(today.get("today_trades", 0))
        signals_sent= int(today.get("signals_sent", 0))

        lines = [
            f"📊 *Performance Report — {date}*",
            f"━━━━━━━━━━━━━━━━━━━━",
            f"",
            f"💼 *Portfolio*",
            f"  Equity:       ₹{cur_equity:,.2f}",
            f"  Total Return: {total_ret:+.2%}",
            f"  Month P&L:    ₹{monthly_pnl:+,.2f} ({monthly_dd:+.1%})",
            f"  Today P&L:    ₹{today_pnl:+,.2f}",
            f"  Heat:         {heat:.1%} deployed",
            f"",
            f"📈 *System Performance*",
            f"  Win Rate (10):  {win_rate:.0%}",
            f"  Profit Factor:  {pf:.2f}",
            f"  Consec Losses:  {consec_loss}",
            f"",
            f"📡 *Today*",
            f"  Signals sent:   {signals_sent}",
            f"  Trades taken:   {today_trades}",
            f"",
            f"{health_emoji} System Health: *{health.get('grade', 'UNKNOWN')}*",
        ]

        issues = health.get("issues", [])
        if issues:
            lines.append("")
            for issue in issues[:3]:
                lines.append(f"  ⚠️ {issue}")

        rec = health.get("recommendation", "")
        if rec:
            lines += ["", f"💡 {rec}"]

        lines.append(f"━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────
    # CORE CALCULATIONS
    # ─────────────────────────────────────────────────────────────

    def _calculate_stats(self, trades: List[Dict]) -> Dict:
        """Core stats — the 6 numbers that matter."""
        if not trades:
            return self._empty_stats()

        pnl_pcts   = [float(t.get("pnl_pct", 0)) for t in trades]
        pnl_amounts= [float(t.get("pnl_amount", 0)) for t in trades]

        # Win rate last 10
        last_10    = pnl_pcts[:10]
        wr_10      = sum(1 for p in last_10 if p > 0) / max(len(last_10), 1)

        # Profit factor last 20
        last_20    = pnl_pcts[:20]
        gross_win  = sum(p for p in last_20 if p > 0)
        gross_loss = abs(sum(p for p in last_20 if p < 0))
        pf_20      = gross_win / gross_loss if gross_loss > 0 else 0.0

        # Consecutive losses
        consec = 0
        for p in pnl_pcts:
            if p <= 0: consec += 1
            else: break

        # Monthly DD
        month_start = datetime.now().replace(day=1)
        month_trades= [
            t for t in trades
            if str(t.get("close_date", t.get("exit_date", "")))[:7]
               == month_start.strftime("%Y-%m")
        ]
        monthly_pnl = sum(float(t.get("pnl_amount", 0)) for t in month_trades)
        monthly_dd  = monthly_pnl / max(self.starting_capital, 1)

        # Portfolio heat from DB
        heat = self._portfolio_heat()

        return {
            "win_rate_10":      round(wr_10,    4),
            "profit_factor_20": round(pf_20,    3),
            "consec_losses":    consec,
            "monthly_dd_pct":   round(monthly_dd, 4),
            "portfolio_heat":   round(heat,     4),
            "total_trades":     len(trades),
        }

    def _equity_curve(self, trades: List[Dict]) -> Dict:
        """Build equity curve from trade history."""
        capital = self.starting_capital
        curve   = [{"date": str(datetime.now().date()), "equity": capital}]

        for t in reversed(trades):   # oldest first
            capital += float(t.get("pnl_amount", 0))
            curve.append({
                "date":   str(t.get("close_date", t.get("exit_date", "")))[:10],
                "equity": round(capital, 2),
            })

        curve.reverse()
        final = float(curve[-1]["equity"]) if curve else self.starting_capital
        total_ret = (final - self.starting_capital) / self.starting_capital

        # Peak and drawdown
        equities = [c["equity"] for c in curve]
        peak     = self.starting_capital
        max_dd   = 0.0
        for eq in equities:
            if eq > peak: peak = eq
            dd = (eq - peak) / peak
            if dd < max_dd: max_dd = dd

        return {
            "current":          round(final, 2),
            "starting":         self.starting_capital,
            "total_return_pct": round(total_ret, 4),
            "max_drawdown_pct": round(max_dd, 4),
            "curve":            curve[-90:],  # last 90 data points for chart
        }

    def _today_summary(self, trades: List[Dict], signals: List[Dict]) -> Dict:
        """Today's trading activity summary."""
        today_str    = datetime.now().strftime("%Y-%m-%d")
        today_trades = [
            t for t in trades
            if str(t.get("close_date", t.get("exit_date", "")))[:10] == today_str
        ]
        today_signals= [
            s for s in signals
            if str(s.get("timestamp", ""))[:10] == today_str
        ]
        month_start  = datetime.now().replace(day=1)
        month_trades = [
            t for t in trades
            if str(t.get("close_date", t.get("exit_date", "")))[:7]
               == month_start.strftime("%Y-%m")
        ]

        today_pnl   = sum(float(t.get("pnl_amount", 0)) for t in today_trades)
        month_pnl   = sum(float(t.get("pnl_amount", 0)) for t in month_trades)
        non_flat    = [s for s in today_signals if s.get("signal") != "FLAT"]

        return {
            "today_pnl":     round(today_pnl, 2),
            "today_trades":  len(today_trades),
            "signals_sent":  len(today_signals),
            "live_signals":  len(non_flat),
            "month_pnl":     round(month_pnl, 2),
        }

    def _system_health(self, stats: Dict, signals: List[Dict]) -> Dict:
        """
        Grade overall system health.
        Used to automatically flag if system needs attention.
        """
        issues  = []
        score   = 10   # start full, deduct for problems

        wr      = float(stats.get("win_rate_10",      0))
        pf      = float(stats.get("profit_factor_20", 0))
        consec  = int(stats.get("consec_losses",       0))
        dd      = float(stats.get("monthly_dd_pct",    0))

        if wr < 0.45:
            score -= 3
            issues.append(f"Win rate {wr:.0%} dangerously low (last 10 trades)")
        elif wr < 0.52:
            score -= 1
            issues.append(f"Win rate {wr:.0%} below target (need >52%)")

        if pf < 1.0:
            score -= 3
            issues.append(f"Profit factor {pf:.2f} below 1.0 — losing system")
        elif pf < 1.5:
            score -= 1
            issues.append(f"Profit factor {pf:.2f} below 1.5 target")

        if consec >= 4:
            score -= 3
            issues.append(f"{consec} consecutive losses — system review required")
        elif consec >= 2:
            score -= 1
            issues.append(f"{consec} consecutive losses — monitor closely")

        if dd < -0.06:
            score -= 3
            issues.append(f"Monthly DD {dd:.1%} — circuit breaker approaching")
        elif dd < -0.03:
            score -= 1
            issues.append(f"Monthly DD {dd:.1%} — reduce position sizes")

        # Recent signal quality
        recent_signals = signals[:20]
        flat_rate = sum(1 for s in recent_signals if s.get("signal") == "FLAT") / max(len(recent_signals), 1)
        if flat_rate > 0.90:
            score -= 1
            issues.append("90%+ FLAT rate — market may be choppy")

        if score >= 9:   grade = "EXCELLENT"
        elif score >= 7: grade = "GOOD"
        elif score >= 5: grade = "DEGRADED"
        else:            grade = "POOR"

        recommendations = {
            "EXCELLENT": "System performing well — maintain current approach",
            "GOOD":      "Minor issues noted — monitor but continue",
            "DEGRADED":  "Performance degraded — reduce size 25%, investigate",
            "POOR":      "System breakdown — halt trading, full review required",
        }

        return {
            "grade":           grade,
            "score":           score,
            "issues":          issues,
            "recommendation":  recommendations.get(grade, ""),
        }

    def _weekly_breakdown(self) -> List[Dict]:
        """Performance breakdown by day of week."""
        trades = self._load_trades()
        dow_stats = {}
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

        for t in trades:
            try:
                date = pd.to_datetime(
                    t.get("close_date", t.get("exit_date", ""))
                )
                dow = date.day_name()
                if dow not in dow_stats:
                    dow_stats[dow] = {"wins": 0, "total": 0, "pnl": 0}
                dow_stats[dow]["total"] += 1
                if float(t.get("pnl_pct", 0)) > 0:
                    dow_stats[dow]["wins"] += 1
                dow_stats[dow]["pnl"] += float(t.get("pnl_amount", 0))
            except Exception:
                pass

        result = []
        for day in days:
            d = dow_stats.get(day, {"wins": 0, "total": 0, "pnl": 0})
            result.append({
                "day":      day,
                "win_rate": round(d["wins"] / max(d["total"], 1), 3),
                "trades":   d["total"],
                "total_pnl":round(d["pnl"], 2),
            })
        return result

    def _feature_performance(self) -> Dict:
        """
        Load SHAP feature importance for display.
        Shows trader which features are driving signals.
        """
        try:
            with open("models/evaluation/feature_importance.json") as f:
                return json.load(f)
        except Exception:
            return {}

    def _regime_performance(self) -> Dict:
        """Win rate breakdown by market regime."""
        try:
            trades  = self._load_trades()
            conn    = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            result  = {}
            regimes = ["BULL_TRENDING", "BEAR_TRENDING",
                       "HIGH_VOLATILITY", "CHOPPY_SIDEWAYS"]

            for regime in regimes:
                rows = conn.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                           AVG(pnl_pct) as avg_pnl
                    FROM   signal_outcomes
                    WHERE  regime = ?
                """, (regime,)).fetchone()

                if rows and rows["total"] > 0:
                    result[regime] = {
                        "total":    rows["total"],
                        "win_rate": round(rows["wins"] / rows["total"], 3),
                        "avg_pnl":  round(rows["avg_pnl"] or 0, 4),
                    }
            conn.close()
            return result
        except Exception:
            return {}

    # ─────────────────────────────────────────────────────────────
    # DATA LOADERS
    # ─────────────────────────────────────────────────────────────

    def _load_trades(self) -> List[Dict]:
        """Load closed trades from DB, fallback to CSV."""
        try:
            conn  = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows  = conn.execute("""
                SELECT * FROM closed_trades
                WHERE status='CLOSED'
                ORDER BY close_date DESC
            """).fetchall()
            conn.close()
            if rows:
                return [dict(r) for r in rows]
        except Exception:
            pass

        if os.path.exists(TRADE_LOG):
            try:
                df = pd.read_csv(TRADE_LOG)
                return df.to_dict("records")
            except Exception:
                pass
        return []

    def _load_signals(self) -> List[Dict]:
        """Load signal history from CSV log."""
        if not os.path.exists(SIGNAL_LOG):
            return []
        try:
            df = pd.read_csv(SIGNAL_LOG, nrows=200)
            return df.to_dict("records")
        except Exception:
            return []

    def _portfolio_heat(self) -> float:
        """Total risk deployed as fraction of capital."""
        try:
            conn = sqlite3.connect(self.db_path)
            row  = conn.execute(
                "SELECT COALESCE(SUM(risk_amount),0) FROM open_trades WHERE status='OPEN'"
            ).fetchone()
            conn.close()
            return round(float(row[0]) / max(self.starting_capital, 1), 4)
        except Exception:
            return 0.0

    def _empty_stats(self) -> Dict:
        return {
            "win_rate_10": 0, "profit_factor_20": 0,
            "consec_losses": 0, "monthly_dd_pct": 0,
            "portfolio_heat": 0, "total_trades": 0,
        }