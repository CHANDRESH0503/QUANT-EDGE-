# features/options_features.py
# Options market features — where smart money reveals positioning
# Source: options_fetcher.py (NSE options chain)
# PCR is the most reliable smart money signal available for free

import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class OptionsFeatures:
    """
    Converts options chain data into ML-ready features.

    Senior trader perspective:
    Options market is where real money is on the line — not opinions.
    When I see PCR at 1.4, I know institutions bought protection.
    That puts floor under the stock. When PCR is 0.6, too much
    complacency — smart money is not protecting = top is close.

    Features:
    - pcr_norm           : Put-Call Ratio normalised to [-1, +1]
    - pcr_roc_norm       : PCR acceleration (more puts building = warning)
    - max_pain_dist_norm : distance from max pain level
    - iv_percentile_norm : options cheap or expensive
    - implied_move_norm  : expected move size from straddle
    - expiry_proximity   : is it expiry week? (structural change)
    - call_wall_dist     : distance from highest call OI (resistance)
    - put_wall_dist      : distance from highest put OI (support)
    """

    FEATURE_NAMES = [
        "pcr_norm",
        "pcr_roc_norm",
        "max_pain_dist_norm",
        "iv_percentile_norm",
        "implied_move_norm",
        "expiry_proximity",
        "call_wall_dist_norm",
        "put_wall_dist_norm",
    ]

    def extract(self,
                options_data:  Dict,
                expiry_data:   Dict,
                current_price: float) -> Dict:
        """
        Convert raw options data into normalised ML features.

        Args:
            options_data:  from OptionsFetcher.fetch_and_calculate() or get_latest()
            expiry_data:   from OptionsFetcher.get_expiry_features()
            current_price: HDFC Bank current close price
        """
        pcr             = float(options_data.get("pcr", 1.0))
        pcr_roc         = float(options_data.get("pcr_roc", 0.0))
        max_pain        = float(options_data.get("max_pain", current_price))
        iv_pct          = float(options_data.get("iv_percentile", 50.0))
        implied_move    = float(options_data.get("implied_move_pct", 2.0))
        call_wall       = float(options_data.get("max_call_oi_strike", 0))
        put_wall        = float(options_data.get("max_put_oi_strike", 0))
        days_to_expiry  = int(expiry_data.get("days_to_monthly_expiry", 15))

        # ── PCR normalisation ─────────────────────────────────────
        # PCR 0.7 = bearish (-1), PCR 1.0 = neutral (0), PCR 1.3 = bullish (+1)
        pcr_norm     = self._scale(pcr, 0.7, 1.3)

        # PCR acceleration: rising = institutions buying more protection
        pcr_roc_norm = self._clip(-pcr_roc / 0.3)   # invert: rising PCR ROC = bearish

        # ── Max pain ─────────────────────────────────────────────
        if current_price > 0 and max_pain > 0:
            pain_dist = (current_price - max_pain) / current_price
            # Price above max pain = gravity will pull it down
            max_pain_norm = self._clip(-pain_dist / 0.03)
        else:
            max_pain_norm = 0.0

        # ── IV percentile ─────────────────────────────────────────
        # Low IV (<20) = cheap options, expand position = bullish for buying
        # High IV (>80) = expensive options = volatility risk = be careful
        iv_norm = self._scale(iv_pct, 20, 80, invert=True)  # low IV = positive

        # ── Implied move ─────────────────────────────────────────
        # High implied move = high uncertainty = reduce size
        im_norm = self._clip(-implied_move / 5.0)   # 5% straddle = max bearish

        # ── Expiry proximity ──────────────────────────────────────
        # Expiry week = structural change in behavior
        expiry_norm = max(0.0, 1.0 - days_to_expiry / 5) if days_to_expiry <= 5 else 0.0

        # ── OI walls ─────────────────────────────────────────────
        call_dist = 0.0
        put_dist  = 0.0
        if current_price > 0:
            if call_wall > current_price:
                call_dist = self._clip((call_wall - current_price) / current_price / 0.05)
            if put_wall > 0 and put_wall < current_price:
                put_dist = self._clip((current_price - put_wall) / current_price / 0.05)

        return {
            "pcr_norm":           round(pcr_norm, 4),
            "pcr_roc_norm":       round(pcr_roc_norm, 4),
            "max_pain_dist_norm": round(max_pain_norm, 4),
            "iv_percentile_norm": round(iv_norm, 4),
            "implied_move_norm":  round(im_norm, 4),
            "expiry_proximity":   round(expiry_norm, 4),
            "call_wall_dist_norm":round(call_dist, 4),
            "put_wall_dist_norm": round(put_dist, 4),
        }

    def _scale(self, val, lo, hi, invert=False) -> float:
        if hi == lo: return 0.0
        s = 2 * (val - lo) / (hi - lo) - 1
        s = max(-1.0, min(1.0, s))
        return round(-s if invert else s, 4)

    def _clip(self, val: float) -> float:
        try:
            v = float(val)
            return round(max(-1.0, min(1.0, v)), 4) if np.isfinite(v) else 0.0
        except (TypeError, ValueError):
            return 0.0