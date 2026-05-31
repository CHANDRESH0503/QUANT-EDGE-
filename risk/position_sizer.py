# risk/position_sizer.py
# ATR-based position sizing — the only correct way to size positions
# Fixed percentage stops ignore volatility and lead to blown accounts

import math
import logging
import sqlite3
from typing import Dict, Optional
from risk.capital_mode import CapitalMode

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Calculates position size using ATR-based risk management.

    20yr trader rule:
    NEVER use a fixed rupee stop (e.g. ₹10 below entry).
    ALWAYS use ATR × multiplier.
    Why: A ₹10 stop on a stock with ₹28 ATR gets hit on random noise.
    A 1.5× ATR stop breathes with the stock's natural volatility.

    Formula:
    stop_distance  = ATR × stop_multiplier
    shares         = floor(risk_amount / stop_distance)
    position_value = shares × entry_price
    target_price   = entry + (stop_distance × reward_risk_ratio)

    Cap: position_value cannot exceed 40% of capital.
    This prevents accidental over-concentration.
    """

    REWARD_RISK = 2.5   # minimum R:R — non-negotiable

    # Drawdown-adaptive multipliers: as DD deepens, risk less
    DD_MULT_TABLE = [
        (-0.06, 0.25),   # DD worse than -6%: quarter size
        (-0.04, 0.50),   # DD -4% to -6%: half size
        (-0.02, 0.75),   # DD -2% to -4%: three-quarter size
        (0.00,  1.00),   # DD ≤ -2% or positive: full size
    ]

    def calculate(
        self,
        signal:          str,
        capital:         float,
        current_price:   float,
        atr:             float,
        size_multiplier: float = 1.0,
        capital_mode:    str   = "FULL",
        monthly_dd_pct:  float = 0.0,
        event_mult:      float = 1.0,
        category:        str   = "swing",
    ) -> Dict:
        """
        Calculate full position parameters.

        Args:
            signal:         "LONG" or "SHORT"
            capital:        Current account capital ₹
            current_price:  Entry price ₹
            atr:            14-period ATR ₹
            size_multiplier:Combined multiplier from all gates
            capital_mode:   "SMALL" / "GROWING" / "FULL"
            category:       "swing" / "positional" / "intraday" — drives ATR mults

        Returns:
            Dict with entry, stop, target, shares, risk_amount etc.
        """
        if current_price <= 0 or atr <= 0:
            return self._empty(signal, current_price)

        cfg        = CapitalMode.MODES.get(capital_mode, CapitalMode.MODES["FULL"])
        cat_cfg    = CapitalMode.CATEGORY_ATR.get(category, {})
        # Category-specific ATR mults take precedence over capital-mode defaults.
        # This ensures intraday gets a 1×ATR stop, positional gets 2.5×ATR stop, etc.
        stop_mult  = cat_cfg.get("stop_atr_mult",  cfg["stop_atr_mult"])
        target_mult= cat_cfg.get("target_atr_mult", cfg["target_atr_mult"])
        max_risk_pct   = cfg["max_risk_pct"]

        # Drawdown-adaptive multiplier (applied before gate multiplier)
        dd_mult = 1.0
        for threshold, mult in self.DD_MULT_TABLE:
            if monthly_dd_pct <= threshold:
                dd_mult = mult
                break

        # Risk amount after size multiplier + DD multiplier + event multiplier
        base_risk      = capital * max_risk_pct
        effective_mult = max(0.0, min(1.5, size_multiplier)) * dd_mult * max(0.0, min(1.0, event_mult))
        risk_amount    = round(base_risk * effective_mult, 2)

        if risk_amount <= 0:
            return self._empty(signal, current_price)

        # ATR-based stop distance
        stop_dist      = round(atr * stop_mult, 2)
        target_dist    = round(atr * target_mult, 2)

        # Shares: integer, at least 1
        raw_shares     = risk_amount / stop_dist
        shares         = max(1, math.floor(raw_shares))

        # Position value cap: 40% of capital
        max_pos_val    = capital * 0.40
        if shares * current_price > max_pos_val:
            shares = max(1, math.floor(max_pos_val / current_price))

        position_value = round(shares * current_price, 2)
        actual_risk    = round(shares * stop_dist, 2)

        # Direction-aware stop and target
        if signal == "LONG":
            stop_price   = round(current_price - stop_dist, 2)
            target_price = round(current_price + target_dist, 2)
        else:  # SHORT
            stop_price   = round(current_price + stop_dist, 2)
            target_price = round(current_price - target_dist, 2)

        reward_risk = round(target_dist / stop_dist, 2)

        result = {
            "signal":         signal,
            "entry":          current_price,
            "stop_loss":      stop_price,
            "target":         target_price,
            "shares":         shares,
            "position_value": position_value,
            "risk_amount":    actual_risk,
            "risk_pct":       round(actual_risk / capital * 100, 3),
            "reward_risk":    reward_risk,
            "stop_distance":  stop_dist,
            "target_distance":target_dist,
            "capital_used_pct":round(position_value / capital * 100, 2),
            "size_multiplier":size_multiplier,
            "dd_mult":        dd_mult,
            "event_mult":     event_mult,
            "effective_mult": round(effective_mult, 3),
            "monthly_dd_pct": monthly_dd_pct,
            "capital_mode":   capital_mode,
            "category":       category,
            "stop_atr_mult":  stop_mult,
            "target_atr_mult":target_mult,
        }

        logger.info(
            f"Position: {signal} {shares} shares @ ₹{current_price:.2f} | "
            f"stop=₹{stop_price:.2f} | target=₹{target_price:.2f} | "
            f"risk=₹{actual_risk:.2f} ({result['risk_pct']:.2f}%) | "
            f"R:R={reward_risk}"
        )
        return result

    def check_exposure(
        self,
        ticker:          str,
        proposed_value:  float,
        capital:         float,
        db_path:         str   = "database/trading.db",
    ) -> Dict:
        """
        Check per-name (40%) and total portfolio (80%) exposure limits.

        Returns:
            {
                "allowed":          True/False,
                "reason":           str,
                "current_exposure": float,   # total open position value
                "ticker_exposure":  float,   # this ticker's open value
            }
        """
        try:
            conn = sqlite3.connect(db_path)
            rows = conn.execute(
                "SELECT ticker, shares, entry_price FROM open_trades WHERE status='OPEN'"
            ).fetchall()
            conn.close()
        except Exception:
            rows = []

        total_val  = sum(float(r[1] or 0) * float(r[2] or 0) for r in rows)
        ticker_val = sum(
            float(r[1] or 0) * float(r[2] or 0)
            for r in rows if r[0] == ticker
        )

        max_per_name  = capital * 0.40
        max_total     = capital * 0.80

        if ticker_val + proposed_value > max_per_name:
            return {
                "allowed":          False,
                "reason":           f"{ticker} would exceed 40% per-name cap",
                "current_exposure": total_val,
                "ticker_exposure":  ticker_val,
            }
        if total_val + proposed_value > max_total:
            return {
                "allowed":          False,
                "reason":           "Total portfolio would exceed 80% exposure cap",
                "current_exposure": total_val,
                "ticker_exposure":  ticker_val,
            }
        return {
            "allowed":          True,
            "reason":           "Within exposure limits",
            "current_exposure": total_val,
            "ticker_exposure":  ticker_val,
        }

    def check_correlation_risk(
        self,
        ticker:          str,
        signal:          str,
        proposed_value:  float,
        capital:         float,
        capital_mode:    str   = "FULL",
        db_path:         str   = "database/trading.db",
    ) -> Dict:
        """
        Concentration guard for the 5 correlated bank universe (audit P0-1).

        The per-name/total caps in check_exposure() are direction-blind and
        treat the banks as independent — but they are ~0.8 correlated to Bank
        Nifty, so 5 LONG positions are one big macro bet, not five picks.
        Two checks:
          1. Same-direction cluster cap — at most `max_same_dir_cluster`
             simultaneous positions in the SAME direction (mode-scaled).
          2. Net-exposure cap — |Σ signed exposure| / capital ≤ MAX_NET_EXPOSURE.
             A hedged pair (one long, one short) nets to ~0 and is fine; five
             same-direction names blow past the net cap and are blocked.

        Returns {"allowed": bool, "reason": str, "cluster": int, "net_pct": float}.
        """
        from risk.capital_mode import CapitalMode
        cfg          = CapitalMode.MODES.get(capital_mode, CapitalMode.MODES["FULL"])
        max_cluster  = int(cfg.get("max_same_dir_cluster", 3))
        max_net      = CapitalMode.MAX_NET_EXPOSURE_PCT

        try:
            conn = sqlite3.connect(db_path)
            rows = conn.execute(
                "SELECT ticker, shares, entry_price, signal "
                "FROM open_trades WHERE status='OPEN'"
            ).fetchall()
            conn.close()
        except Exception:
            rows = []

        # Signed net exposure across the universe (LONG +, SHORT −), incl. proposal.
        def _signed(sig, val):
            return val if str(sig).upper() == "LONG" else -val
        net_signed = sum(_signed(r[3], float(r[1] or 0) * float(r[2] or 0)) for r in rows)
        new_net    = net_signed + _signed(signal, proposed_value)

        # Same-direction cluster count (distinct tickers already in this direction,
        # excluding this ticker so adding a second category on the same name isn't
        # counted as a new correlated bet), + 1 for the proposed name if new.
        same_dir_tickers = {
            r[0] for r in rows
            if str(r[3]).upper() == str(signal).upper() and r[0] != ticker
        }
        proposed_cluster = len(same_dir_tickers) + 1   # this ticker joins the cluster

        if proposed_cluster > max_cluster:
            return {
                "allowed": False,
                "reason":  (f"{signal} cluster cap: {proposed_cluster} correlated "
                            f"banks same-direction > {max_cluster} ({capital_mode})"),
                "cluster": proposed_cluster,
                "net_pct": round(new_net / max(capital, 1), 3),
            }

        if abs(new_net) > max_net * capital:
            return {
                "allowed": False,
                "reason":  (f"net-exposure cap: {abs(new_net)/max(capital,1):.0%} "
                            f"net {'long' if new_net>0 else 'short'} > {max_net:.0%}"),
                "cluster": proposed_cluster,
                "net_pct": round(new_net / max(capital, 1), 3),
            }

        return {
            "allowed": True,
            "reason":  "within concentration limits",
            "cluster": proposed_cluster,
            "net_pct": round(new_net / max(capital, 1), 3),
        }

    def _empty(self, signal: str, price: float) -> Dict:
        return {
            "signal": signal, "entry": price, "stop_loss": 0,
            "target": 0, "shares": 0, "position_value": 0,
            "risk_amount": 0, "risk_pct": 0, "reward_risk": 0,
            "stop_distance": 0, "target_distance": 0,
            "capital_used_pct": 0, "size_multiplier": 0,
        }