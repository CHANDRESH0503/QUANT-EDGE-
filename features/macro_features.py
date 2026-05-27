# features/macro_features.py
# Macro and intermarket features for all three ML models
# Source: macro_analyzer.py, intermarket.py, global_fetcher.py
# Most important for positional model; moderate for swing; low for intraday

import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class MacroFeatures:
    """
    Consolidates macro + intermarket signals into ML features.

    Senior trader perspective:
    Macro context changes the probability of every trade.
    A perfect LONG setup in a BEAR macro environment has 40% win rate.
    Same setup in BULL macro has 65% win rate.
    Always know the macro backdrop — it multiplies or divides your edge.

    Key features:
    - india_vix_norm       : fear level normalised
    - rbi_proximity_risk   : approaching MPC meeting = uncertainty
    - dollar_stress        : FII outflow risk
    - yield_headwind       : US yields pulling capital away
    - crude_risk           : inflation = RBI = bank headwind
    - intermarket_score    : composite from intermarket.py
    - rs_vs_banknifty      : HDFC outperforming/underperforming sector
    """

    FEATURE_NAMES = [
        "india_vix_norm",
        "india_vix_trend",
        "rbi_proximity_norm",    # 0=far from RBI, 1=day before
        "rbi_cutting_cycle",     # 1=cutting, 0=holding, -1=hiking
        "dollar_stress",
        "rupee_stress",
        "yield_headwind",
        "crude_risk",
        "macro_score",
        "intermarket_score",
        "rs_vs_banknifty",
        "banks_outperforming_nifty",
        "banknifty_relative_momentum_5d",  # alpha 2026-05-26
        "peer_rank_norm",        # 1=rank1, 0=rank6
    ]

    def extract(self,
                macro_features:       Dict,
                intermarket_features: Dict) -> Dict:
        """
        Combine macro and intermarket into final feature set.

        Args:
            macro_features:       from MacroAnalyzer.get_macro_features()
            intermarket_features: from IntermarketAnalyzer.get_intermarket_features()
        """
        # ── India VIX ────────────────────────────────────────────
        vix       = float(macro_features.get("india_vix_level", 15))
        vix_chg   = float(macro_features.get("india_vix_1d_chg", 0))
        # VIX 10=calm(+1), VIX 30=panic(-1)
        vix_norm  = self._scale(vix, 10, 30, invert=True)
        vix_trend = self._clip(vix_chg / 20)   # normalise daily change

        # ── RBI ───────────────────────────────────────────────────
        days_rbi  = float(macro_features.get("days_to_rbi", 30))
        # 0 days = 1.0 risk, 30+ days = 0.0 risk
        rbi_prox  = max(0.0, 1.0 - min(days_rbi, 30) / 30)
        direction = macro_features.get("rbi_direction", "HOLDING")
        rbi_cycle = 1.0 if direction == "CUTTING" else (-1.0 if direction == "HIKING" else 0.0)

        # ── Currency / Dollar ────────────────────────────────────
        dollar_stress = float(intermarket_features.get("dollar_stress", 0))
        rupee_stress  = float(macro_features.get("rupee_stress", 0))
        usdinr_5d     = float(intermarket_features.get("usdinr_5d_pct", 0))
        rupee_norm    = self._clip(-usdinr_5d / 2)   # weaker rupee = negative

        # ── Bond Yields ───────────────────────────────────────────
        yield_hw  = float(intermarket_features.get("yield_headwind", 0))
        bp_5d     = float(intermarket_features.get("us_10y_bp_5d", 0))
        yield_norm= self._clip(-bp_5d / 30)          # rising 30bp = -1

        # ── Crude ─────────────────────────────────────────────────
        crude_risk= float(intermarket_features.get("crude_risk", 0))
        crude_5d  = float(intermarket_features.get("crude_5d_pct", 0))
        crude_norm= self._clip(-crude_5d / 10)       # crude up 10% = -1

        # ── Intermarket composite ────────────────────────────────
        im_score  = self._clip(intermarket_features.get("intermarket_score", 0))
        macro_sc  = self._clip(macro_features.get("macro_score", 0))

        # ── Relative strength ─────────────────────────────────────
        rs_bn     = self._clip(
            intermarket_features.get("rs_vs_banknifty_5d", 0) / 3
        )
        bn_outperf= float(intermarket_features.get("banks_outperforming", 0))
        peer_rank = int(intermarket_features.get("peer_rank", 3))
        peer_norm = round(1.0 - (peer_rank - 1) / 5, 3)  # rank1=1.0, rank6=0.0

        # Alpha (2026-05-26): banknifty_relative_momentum_5d.
        # Bank-sector excess return over broad market over 5 trading days.
        # Clipped to ±10% (then /5 for [-2, +2] → /2 final → [-1, +1]).
        # Source values come from intermarket._sector_rotation which fetches
        # both indexes on the same period for an apples-to-apples comparison.
        bn_5d_pct    = float(intermarket_features.get("banknifty_5d_pct", 0))
        nf_5d_pct    = float(intermarket_features.get("nifty_5d_pct", 0))
        bn_rel_mom_5d= max(-10.0, min(10.0, bn_5d_pct - nf_5d_pct))
        bn_rel_mom_5d_norm = self._clip(bn_rel_mom_5d / 5.0)

        return {
            "india_vix_norm":            round(vix_norm, 4),
            "india_vix_trend":           round(vix_trend, 4),
            "rbi_proximity_norm":        round(rbi_prox, 4),
            "rbi_cutting_cycle":         round(rbi_cycle, 4),
            "dollar_stress":             round(dollar_stress, 4),
            "rupee_stress":              round(rupee_norm, 4),
            "yield_headwind":            round(yield_norm, 4),
            "crude_risk":                round(crude_norm, 4),
            "macro_score":               round(macro_sc, 4),
            "intermarket_score":         round(im_score, 4),
            "rs_vs_banknifty":           round(rs_bn, 4),
            "banks_outperforming_nifty": round(bn_outperf, 4),
            "banknifty_relative_momentum_5d": round(bn_rel_mom_5d_norm, 4),
            "peer_rank_norm":            round(peer_norm, 4),
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