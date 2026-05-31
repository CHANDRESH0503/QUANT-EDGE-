# risk/costs.py
# Single source of truth for trading frictions (audit P0-2).
#
# WHY: paper trades previously booked P&L at the exact signal price with ZERO
# cost — overstating edge by ~20–40 bps/trade and making it impossible to tell
# a real edge from a cost-eaten loser. These constants mirror backtest/engine.py
# so the paper book and the backtest use the SAME assumptions.
#
# Calibrated for NSE equity via a discount broker:
#   brokerage + STT + exchange txn + GST + SEBI + stamp  ≈ 0.10% round-trip
#   slippage (queue/spread/impact)                       ≈ 0.05% per side
#   → ~0.20% total round-trip on entry notional.

TRANSACTION_COST_PCT = 0.001    # 0.10% round-trip brokerage + STT + charges
SLIPPAGE_PCT         = 0.0005   # 0.05% per side
ROUND_TRIP_COST_PCT  = TRANSACTION_COST_PCT + 2 * SLIPPAGE_PCT   # ~0.20%


def round_trip_cost(notional: float) -> float:
    """Total friction (brokerage + taxes + slippage) for a round-trip on
    `notional` (the entry position value). Always non-negative."""
    return abs(float(notional)) * ROUND_TRIP_COST_PCT


def net_after_costs(gross_pnl_amount: float, notional: float) -> tuple:
    """Return (net_pnl_amount, cost_amount) after deducting round-trip cost.

    Costs are charged on the entry notional regardless of win/loss — you pay
    brokerage and slippage on both legs whether the trade wins or not.
    """
    cost = round_trip_cost(notional)
    return round(float(gross_pnl_amount) - cost, 2), round(cost, 2)
