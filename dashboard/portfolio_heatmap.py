# dashboard/portfolio_heatmap.py
# Visual risk heatmap of portfolio state
# Shows capital deployment, heat, open positions, regime context
# Outputs formatted text tables for Telegram (no matplotlib needed)

import sqlite3
import logging
import os
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


class PortfolioHeatmap:
    """
    Portfolio risk visualisation — text-based for Telegram delivery.

    20yr trader morning routine:
    Before looking at price — look at the portfolio.
    How much is deployed? Where is the risk?
    Is the heat too high? Are stops in sensible places?
    Only after this do I look at new opportunities.

    Heatmap metrics:
    - Capital deployment (% invested)
    - Risk heat (% of capital at risk in open trades)
    - Per-position contribution to heat
    - Regime-adjusted risk limits
    - Portfolio-level stop distance from HALT level
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

    def generate(self) -> Dict:
        """Generate full portfolio heatmap data."""
        positions  = self._get_open_positions()
        equity     = self._current_equity()
        monthly_dd = self._monthly_dd(equity)
        regime     = self._latest_regime()

        total_deployed = sum(float(p.get("position_value", 0)) for p in positions)
        total_risk     = sum(float(p.get("risk_amount",    0)) for p in positions)

        deployed_pct   = total_deployed / max(equity, 1)
        heat_pct       = total_risk     / max(equity, 1)

        # Risk limits based on capital mode
        cap_mode = self._capital_mode(equity)
        limits   = {
            "SMALL":   {"max_heat": 0.04, "max_positions": 2, "halt_dd": -0.05},
            "GROWING": {"max_heat": 0.06, "max_positions": 3, "halt_dd": -0.08},
            "FULL":    {"max_heat": 0.10, "max_positions": 5, "halt_dd": -0.10},
        }.get(cap_mode, {"max_heat": 0.10, "max_positions": 5, "halt_dd": -0.10})

        # Distance to halt level
        halt_dd      = limits["halt_dd"]
        dd_remaining = abs(halt_dd) - abs(monthly_dd)
        pct_to_halt  = max(0.0, dd_remaining / abs(halt_dd))

        # Heat level
        max_heat     = limits["max_heat"]
        heat_used    = heat_pct / max_heat
        heat_bar     = self._bar(heat_used)

        # Per-position details
        pos_details = []
        for p in positions:
            entry   = float(p.get("entry_price",    0))
            current = float(p.get("current_price",  entry))
            stop    = float(p.get("stop_price",     0))
            signal  = p.get("signal", "LONG")
            shares  = int(p.get("shares", 0))

            if signal == "LONG":
                pnl_pct = (current - entry) / entry if entry > 0 else 0
            else:
                pnl_pct = (entry - current) / entry if entry > 0 else 0

            risk_contrib = float(p.get("risk_amount", 0)) / max(equity, 1)

            pos_details.append({
                "signal":       signal,
                "entry":        entry,
                "stop":         stop,
                "pnl_pct":      round(pnl_pct, 4),
                "risk_contrib": round(risk_contrib, 4),
                "trade_type":   p.get("trade_type", "swing"),
                "opened_at":    str(p.get("opened_at", ""))[:10],
            })

        return {
            "equity":           round(equity, 2),
            "monthly_dd_pct":   round(monthly_dd, 4),
            "capital_mode":     cap_mode,
            "regime":           regime,
            "n_positions":      len(positions),
            "deployed_pct":     round(deployed_pct, 4),
            "heat_pct":         round(heat_pct, 4),
            "heat_used_pct":    round(heat_used, 4),
            "heat_bar":         heat_bar,
            "pct_to_halt":      round(pct_to_halt, 4),
            "positions":        pos_details,
            "limits":           limits,
            "alerts":           self._generate_alerts(
                heat_pct, heat_used, monthly_dd,
                halt_dd, len(positions), limits
            ),
            "generated_at":     str(datetime.now()),
        }

    def format_telegram(self, heatmap: Dict) -> str:
        """Format heatmap as clean Telegram message."""
        equity      = float(heatmap.get("equity", 0))
        cap_mode    = heatmap.get("capital_mode", "FULL")
        regime      = heatmap.get("regime", "UNKNOWN").replace("_", " ")
        n_pos       = int(heatmap.get("n_positions", 0))
        deployed    = float(heatmap.get("deployed_pct", 0))
        heat        = float(heatmap.get("heat_pct", 0))
        heat_bar    = heatmap.get("heat_bar", "░░░░░")
        monthly_dd  = float(heatmap.get("monthly_dd_pct", 0))
        pct_to_halt = float(heatmap.get("pct_to_halt", 1))

        # Heat colour
        heat_emoji  = "🟢" if heat < 0.03 else ("🟡" if heat < 0.07 else "🔴")

        lines = [
            f"🗺️ *Portfolio Heatmap*",
            f"━━━━━━━━━━━━━━━━━━━━",
            f"💼 Equity:    ₹{equity:,.2f}  [{cap_mode}]",
            f"📊 Regime:    {regime}",
            f"",
            f"🏦 Deployed:  {deployed:.1%}",
            f"{heat_emoji} Heat:      {heat:.2%}  {heat_bar}",
            f"📅 Month DD:  {monthly_dd:+.1%}",
            f"🛑 To halt:   {pct_to_halt:.0%} remaining",
            f"",
            f"📋 Open Positions: {n_pos}",
        ]

        positions = heatmap.get("positions", [])
        if positions:
            for i, p in enumerate(positions, 1):
                pnl_emoji = "✅" if p["pnl_pct"] >= 0 else "⚠️"
                lines.append(
                    f"  {i}. {p['signal']} {p['trade_type']} | "
                    f"P&L: {p['pnl_pct']:+.1%} | "
                    f"Risk: {p['risk_contrib']:.1%} {pnl_emoji}"
                )
        else:
            lines.append("  No open positions — fully in cash")

        alerts = heatmap.get("alerts", [])
        if alerts:
            lines += [""]
            for a in alerts:
                lines.append(f"  ⚠️ {a}")

        lines.append(f"━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────────

    def _generate_alerts(
        self,
        heat_pct:    float,
        heat_used:   float,
        monthly_dd:  float,
        halt_dd:     float,
        n_positions: int,
        limits:      Dict,
    ) -> List[str]:
        """Generate actionable alerts based on portfolio state."""
        alerts = []
        if heat_used > 0.80:
            alerts.append(
                f"Heat at {heat_used:.0%} of limit — no new positions until exits"
            )
        if monthly_dd < halt_dd * 0.80:
            alerts.append(
                f"Monthly DD {monthly_dd:.1%} nearing halt level {halt_dd:.1%}"
            )
        if n_positions >= limits["max_positions"]:
            alerts.append(
                f"Max positions ({limits['max_positions']}) reached — wait for exit"
            )
        return alerts

    def _bar(self, pct: float, width: int = 10) -> str:
        """Text progress bar for heat visualisation."""
        filled = int(min(1.0, pct) * width)
        empty  = width - filled
        char   = "█" if pct < 0.5 else ("▓" if pct < 0.8 else "░")
        return char * filled + "░" * empty

    def _get_open_positions(self) -> List[Dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM open_trades WHERE status='OPEN'"
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def _current_equity(self) -> float:
        try:
            conn = sqlite3.connect(self.db_path)
            row  = conn.execute(
                "SELECT COALESCE(SUM(pnl_amount),0) FROM closed_trades WHERE status='CLOSED'"
            ).fetchone()
            conn.close()
            return self.starting_capital + float(row[0])
        except Exception:
            return self.starting_capital

    def _monthly_dd(self, equity: float) -> float:
        try:
            month_start = datetime.now().replace(day=1).strftime("%Y-%m-%d")
            conn        = sqlite3.connect(self.db_path)
            row         = conn.execute("""
                SELECT COALESCE(SUM(pnl_amount), 0)
                FROM   closed_trades
                WHERE  status='CLOSED' AND close_date >= ?
            """, (month_start,)).fetchone()
            conn.close()
            return round(float(row[0]) / max(self.starting_capital, 1), 4)
        except Exception:
            return 0.0

    def _latest_regime(self) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            row  = conn.execute(
                "SELECT regime FROM regime_snapshots ORDER BY detected_at DESC LIMIT 1"
            ).fetchone()
            conn.close()
            return row[0] if row else "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    def _capital_mode(self, equity: float) -> str:
        if equity < 50_000:   return "SMALL"
        if equity < 200_000:  return "GROWING"
        return "FULL"