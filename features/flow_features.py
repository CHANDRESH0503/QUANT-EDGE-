# features/flow_features.py
# Money flow features: FII/DII, bulk deals, insider, order flow
# Source: fii_fetcher.py, insider_fetcher.py, order_flow.py

import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class FlowFeatures:
    """
    Combines all institutional money flow signals.

    20yr trader rule: Follow the money.
    FII flow > RS > technicals in terms of predictive power for large caps.
    When all three agree — that is the highest conviction setup.
    """

    def extract(self,
                fii_features:      Dict,
                insider_features:  Dict,
                order_flow:        Dict) -> Dict:
        """
        Args:
            fii_features:     from FIIFetcher.get_flow_features()
            insider_features: from InsiderFetcher.get_insider_features()
            order_flow:       from OrderFlowAnalyzer.get_order_flow_features()
        """
        # ── FII flow ─────────────────────────────────────────────
        fii_5d    = float(fii_features.get("fii_net_5d_cr", 0))
        dii_3d    = float(fii_features.get("dii_net_3d_cr", 0))
        fii_signal= float(fii_features.get("fii_signal", 0))   # already -1 to +1
        bulk_net  = float(fii_features.get("bulk_net_cr", 0))

        # Alpha: fii_flow_surprise — z-score from fii_fetcher (-3..+3 raw).
        # Normalised here to roughly [-1, +1] for ML by /3.
        fii_surprise_raw = float(fii_features.get("fii_flow_surprise", 0))
        fii_surprise     = self._clip(fii_surprise_raw / 3.0)

        # Normalise FII flow — ±5000 Cr = ±1.0
        fii_5d_norm= self._clip(fii_5d / 5000)
        dii_3d_norm= self._clip(dii_3d / 2000)
        bulk_norm  = self._clip(bulk_net / 200)

        # Confluence: both FII and DII buying = +2
        confluence = fii_features.get("flow_confluence", "MIXED")
        conf_score = 1.0 if confluence == "BOTH_BUYING" else (
                    -1.0 if confluence == "BOTH_SELLING" else 0.0)

        # ── Insider ───────────────────────────────────────────────
        insider_score = self._clip(insider_features.get("insider_score", 0))
        promoter_chg  = self._clip(
            float(insider_features.get("promoter_qoq_change", 0)) * 20
        )
        block_buy     = float(insider_features.get("block_deal_is_buying", False))
        block_5d      = self._clip(
            float(insider_features.get("block_deal_5d_net_cr", 0)) / 500
        )

        # ── Order flow ────────────────────────────────────────────
        of_score      = self._clip(order_flow.get("order_flow_score", 0))
        delta_div_flag= float(order_flow.get("delta_divergence_flag", 0))
        absorption    = float(order_flow.get("absorption_detected", 0))
        inst_bias_val = (
             1.0 if order_flow.get("inst_bias") == "INSTITUTIONAL_BUYING"  else
            -1.0 if order_flow.get("inst_bias") == "INSTITUTIONAL_SELLING" else 0.0
        )

        return {
            "fii_5d_norm":         round(fii_5d_norm, 4),
            "fii_signal":          round(fii_signal, 4),
            "fii_flow_surprise":   round(fii_surprise, 4),  # alpha 2026-05-26
            "dii_3d_norm":         round(dii_3d_norm, 4),
            "bulk_deal_norm":      round(bulk_norm, 4),
            "flow_confluence":     round(conf_score, 4),
            "insider_score":       round(insider_score, 4),
            "promoter_change":     round(promoter_chg, 4),
            "block_5d_norm":       round(block_5d, 4),
            "block_is_buying":     round(block_buy, 4),
            "order_flow_score":    round(of_score, 4),
            "delta_divergence":    round(delta_div_flag, 4),
            "absorption_flag":     round(absorption, 4),
            "institutional_bias":  round(inst_bias_val, 4),
        }

    def _clip(self, val: float) -> float:
        try:
            v = float(val)
            return round(max(-1.0, min(1.0, v)), 4) if np.isfinite(v) else 0.0
        except (TypeError, ValueError):
            return 0.0