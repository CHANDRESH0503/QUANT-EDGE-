# backtest/metrics.py
# Calculates all performance metrics from a list of trade results
# Used by both engine.py and walk_forward.py
# No external dependencies — pure numpy/pandas

import numpy as np
import pandas as pd
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class BacktestMetrics:
    """
    Computes comprehensive performance statistics from trade results.

    20yr trader minimum acceptance criteria:
    Win rate:        > 52%   (anything below = coin flip territory)
    Profit factor:   > 1.5   (gross wins / gross losses)
    Sharpe ratio:    > 1.0   (risk-adjusted annual return)
    Max drawdown:    < 20%   (survivable without blowing account)
    Total trades:    > 50    (statistical significance)

    If your backtest shows > 70% win rate — SOMETHING IS WRONG.
    Either lookahead bias, overfitting, or survivorship bias.
    Realistic expectation: 55-65% with a well-built system.
    """

    ANNUALISATION = 252   # trading days per year

    def calculate(
        self,
        trades:          List[Dict],
        starting_capital:float = 100_000.0,
        risk_free_rate:  float = 0.065,   # India 10yr gilt ~6.5%
    ) -> Dict:
        """
        Full performance metrics from list of closed trades.

        Args:
            trades:           List of trade dicts with pnl_pct, pnl_amount,
                              entry_date, exit_date, signal
            starting_capital: Starting capital ₹
            risk_free_rate:   Annual risk-free rate (India gilt)
        """
        if not trades or len(trades) < 5:
            logger.warning(f"Insufficient trades for metrics: {len(trades)}")
            return self._empty_metrics()

        # ── Core trade statistics ─────────────────────────────────
        pnl_pcts    = [float(t.get("pnl_pct", 0))    for t in trades]
        pnl_amounts = [float(t.get("pnl_amount", 0)) for t in trades]
        signals     = [t.get("signal", "LONG")       for t in trades]

        wins        = [p for p in pnl_pcts if p > 0]
        losses      = [p for p in pnl_pcts if p <= 0]
        longs       = [t for t in trades   if t.get("signal") == "LONG"]
        shorts      = [t for t in trades   if t.get("signal") == "SHORT"]

        total       = len(trades)
        win_count   = len(wins)
        win_rate    = win_count / total

        avg_win     = float(np.mean(wins))   if wins   else 0.0
        avg_loss    = float(np.mean(losses)) if losses else 0.0
        avg_trade   = float(np.mean(pnl_pcts))

        gross_win   = sum(p for p in pnl_amounts if p > 0)
        gross_loss  = abs(sum(p for p in pnl_amounts if p < 0))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else 0.0

        # ── Equity curve ─────────────────────────────────────────
        equity = [starting_capital]
        for amt in pnl_amounts:
            equity.append(equity[-1] + amt)

        equity_series   = np.array(equity)
        daily_returns   = np.diff(equity_series) / equity_series[:-1]

        # ── Drawdown ──────────────────────────────────────────────
        peak        = np.maximum.accumulate(equity_series)
        drawdown    = (equity_series - peak) / peak
        max_dd      = float(drawdown.min())

        # ── Sharpe ratio ──────────────────────────────────────────
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            rf_daily    = (1 + risk_free_rate) ** (1 / self.ANNUALISATION) - 1
            excess_ret  = daily_returns - rf_daily
            sharpe      = (
                float(excess_ret.mean()) /
                float(excess_ret.std()) *
                np.sqrt(self.ANNUALISATION)
            )
        else:
            sharpe = 0.0

        # ── Calmar ratio ──────────────────────────────────────────
        total_return = (equity_series[-1] - starting_capital) / starting_capital
        calmar       = total_return / abs(max_dd) if max_dd < 0 else 0.0

        # ── Sortino ratio ─────────────────────────────────────────
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 1:
            downside_dev = float(negative_returns.std() * np.sqrt(self.ANNUALISATION))
            annual_ret   = float(np.mean(daily_returns) * self.ANNUALISATION)
            sortino      = (annual_ret - risk_free_rate) / downside_dev
        else:
            sortino = 0.0

        # ── Expectancy ────────────────────────────────────────────
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # ── Consecutive stats ─────────────────────────────────────
        max_consec_wins, max_consec_losses = self._consecutive_stats(pnl_pcts)

        # ── Hold time analysis ────────────────────────────────────
        hold_times  = self._hold_times(trades)

        # ── Direction breakdown ───────────────────────────────────
        long_wr     = (
            sum(1 for t in longs if float(t.get("pnl_pct",0)) > 0) / len(longs)
            if longs else 0.0
        )
        short_wr    = (
            sum(1 for t in shorts if float(t.get("pnl_pct",0)) > 0) / len(shorts)
            if shorts else 0.0
        )

        # ── Pass/Fail assessment ──────────────────────────────────
        grade, issues = self._grade(
            win_rate, profit_factor, sharpe, max_dd, total
        )

        return {
            # Core metrics
            "total_trades":       total,
            "win_count":          win_count,
            "loss_count":         total - win_count,
            "win_rate":           round(win_rate, 4),
            "avg_win_pct":        round(avg_win, 4),
            "avg_loss_pct":       round(avg_loss, 4),
            "avg_trade_pct":      round(avg_trade, 4),
            "profit_factor":      round(profit_factor, 3),
            "expectancy_pct":     round(expectancy, 4),

            # Risk-adjusted
            "sharpe_ratio":       round(sharpe, 3),
            "sortino_ratio":      round(sortino, 3),
            "calmar_ratio":       round(calmar, 3),
            "max_drawdown_pct":   round(max_dd, 4),

            # Capital
            "starting_capital":   starting_capital,
            "ending_capital":     round(float(equity_series[-1]), 2),
            "total_return_pct":   round(total_return, 4),
            "total_pnl_amount":   round(sum(pnl_amounts), 2),

            # Streaks
            "max_consec_wins":    max_consec_wins,
            "max_consec_losses":  max_consec_losses,

            # Direction breakdown
            "long_win_rate":      round(long_wr,  4),
            "short_win_rate":     round(short_wr, 4),
            "long_trades":        len(longs),
            "short_trades":       len(shorts),

            # Hold time
            "avg_hold_days":      hold_times.get("avg", 0),
            "median_hold_days":   hold_times.get("median", 0),

            # Assessment
            "grade":              grade,
            "issues":             issues,
            "is_deployable":      grade in ("A", "B"),

            # Equity curve (for chart)
            "equity_curve":       equity,
        }

    def summary_line(self, metrics: Dict) -> str:
        """One-line summary for quick review."""
        return (
            f"Trades={metrics['total_trades']} | "
            f"WR={metrics['win_rate']:.0%} | "
            f"PF={metrics['profit_factor']:.2f} | "
            f"Sharpe={metrics['sharpe_ratio']:.2f} | "
            f"MaxDD={metrics['max_drawdown_pct']:.1%} | "
            f"Return={metrics['total_return_pct']:.1%} | "
            f"Grade={metrics['grade']}"
        )

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _consecutive_stats(self, pnl_pcts: List[float]):
        max_wins = max_losses = cur_wins = cur_losses = 0
        for p in pnl_pcts:
            if p > 0:
                cur_wins  += 1; cur_losses = 0
            else:
                cur_losses += 1; cur_wins = 0
            max_wins   = max(max_wins,   cur_wins)
            max_losses = max(max_losses, cur_losses)
        return max_wins, max_losses

    def _hold_times(self, trades: List[Dict]) -> Dict:
        days = []
        for t in trades:
            try:
                entry = pd.to_datetime(t.get("entry_date") or t.get("opened_at"))
                exit_ = pd.to_datetime(t.get("exit_date")  or t.get("close_date"))
                days.append((exit_ - entry).days)
            except Exception:
                pass
        if not days:
            return {"avg": 0, "median": 0}
        return {
            "avg":    round(float(np.mean(days)), 1),
            "median": round(float(np.median(days)), 1),
        }

    def _grade(
        self,
        win_rate:      float,
        profit_factor: float,
        sharpe:        float,
        max_dd:        float,
        n_trades:      int,
    ):
        issues  = []
        score   = 0

        if win_rate  >= 0.58: score += 2
        elif win_rate >= 0.52: score += 1
        else: issues.append(f"Win rate {win_rate:.0%} too low (need >52%)")

        if profit_factor >= 2.0: score += 2
        elif profit_factor >= 1.5: score += 1
        else: issues.append(f"Profit factor {profit_factor:.2f} too low (need >1.5)")

        if sharpe >= 1.5: score += 2
        elif sharpe >= 1.0: score += 1
        else: issues.append(f"Sharpe {sharpe:.2f} too low (need >1.0)")

        if abs(max_dd) <= 0.10: score += 2
        elif abs(max_dd) <= 0.20: score += 1
        else: issues.append(f"Max drawdown {max_dd:.1%} too high (need <20%)")

        if n_trades >= 100: score += 1
        elif n_trades >= 50: score += 0
        else: issues.append(f"Only {n_trades} trades — need 50+ for significance")

        if score >= 8:   grade = "A"
        elif score >= 6: grade = "B"
        elif score >= 4: grade = "C"
        else:            grade = "D"

        return grade, issues

    def _empty_metrics(self) -> Dict:
        return {
            "total_trades": 0, "win_rate": 0, "profit_factor": 0,
            "sharpe_ratio": 0, "max_drawdown_pct": 0, "total_return_pct": 0,
            "grade": "D", "is_deployable": False,
            "issues": ["Insufficient trade data"],
            "equity_curve": [],
        }