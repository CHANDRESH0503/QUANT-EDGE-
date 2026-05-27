# signals/gate6_confidence.py
# Gate 6 — Final confidence threshold and capital mode enforcement
# Last gate before signal is approved
# Connected to: risk/capital_mode.py, features/risk_features.py

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class Gate6Confidence:
    """
    Gate 6: Final confidence threshold and capital mode filter.

    20yr trader rule:
    When the model is uncertain (< 60% confidence), do not trade.
    Uncertainty is information — it tells you the setup is ambiguous.
    Ambiguous setups have near coin-flip win rates.
    Wait for clarity.

    Capital mode enforces additional restrictions:
    SMALL capital → only A+ alignment signals → highest quality only
    This protects beginners from their own eagerness.
    """

    # 2026-05-22 evening: GROWING swing 0.63→0.60 to match FULL. With
    # per-category alignment-bypass live, the 3pp premium GROWING paid
    # for relaxed alignment no longer applies — alignment is already
    # informational across all modes. GROWING still has tighter
    # per-trade risk (1.5% vs FULL's 2.0%), so capital protection holds.
    THRESHOLDS = {
        "swing":      {"SMALL": 0.68, "GROWING": 0.60, "FULL": 0.60},
        "intraday":   {"SMALL": 0.72, "GROWING": 0.68, "FULL": 0.65},
        "positional": {"SMALL": 0.65, "GROWING": 0.60, "FULL": 0.55},
    }

    MIN_ALIGNMENT = {
        "SMALL":   {"A+"},          # strictest — only best signals
        "GROWING": {"A+", "A"},
        "FULL":    {"A+", "A", "B"},
    }

    def check(
        self,
        primary_conf:        float,
        alignment:           str,
        capital_mode:        str,
        risk_context:        Dict,
        model_type:          str   = "swing",
        skip_alignment:      bool  = False,
        data_quality_boost:  float = 0.0,
    ) -> Tuple[bool, Dict]:
        """
        Final confidence and capital mode gate.

        Args:
            primary_conf:    Boosted confidence from gate 4
            alignment:       Alignment grade from gate 4
            capital_mode:    "SMALL" / "GROWING" / "FULL"
            risk_context:    from RiskFeatures dict
            model_type:      "swing" / "positional" / "intraday" — picks threshold
            skip_alignment:  When True, skip the MIN_ALIGNMENT block. Used by the
                             per-category pipeline where each model's signal is
                             evaluated independently — alignment is informational
                             only and must never gate a valid per-category signal.
            data_quality_boost: Additive threshold adjustment from data-quality
                             gate (e.g. +0.05 when feature vector is DEGRADED).
                             Demands stronger conviction when data is sparse.
        """
        # ── Hard circuit breakers first ───────────────────────────
        cb_level = str(risk_context.get("circuit_breaker_level", "OK"))
        if cb_level in ("PAUSE", "HALT"):
            return False, {
                "gate":   6, "passed": False,
                "reason": (
                    f"Circuit breaker {cb_level}: "
                    f"{risk_context.get('circuit_breaker_reason','')}"
                ),
                "circuit_breaker_level": cb_level,
            }

        if not risk_context.get("trading_allowed", True):
            reason = "Risk circuit breaker active"
            if risk_context.get("monthly_dd_flag"):
                reason = "Monthly loss limit hit — trading halted this month"
            elif risk_context.get("consecutive_loss_halt"):
                reason = "3 consecutive losses — mandatory review pause"
            return False, {
                "gate":   6, "passed": False,
                "reason": reason,
            }

        # ── Alignment check for capital mode (skippable) ──────────
        if not skip_alignment:
            allowed_alignments = self.MIN_ALIGNMENT.get(capital_mode, {"A+", "A", "B"})
            if alignment not in allowed_alignments:
                return False, {
                    "gate":        6,
                    "passed":      False,
                    "reason":      (
                        f"{capital_mode} capital mode requires "
                        f"{'/'.join(sorted(allowed_alignments, reverse=True))} alignment, "
                        f"got {alignment}"
                    ),
                    "capital_mode":capital_mode,
                    "alignment":   alignment,
                }

        # ── Confidence threshold (VIX- and data-quality-adaptive) ─
        threshold = self.THRESHOLDS.get(model_type, {}).get(capital_mode, 0.60)
        india_vix = float(risk_context.get("india_vix", 0.0))
        if india_vix > 25:
            threshold = min(0.90, threshold + 0.10)
        elif india_vix > 20:
            threshold = min(0.90, threshold + 0.05)
        # Data-quality boost — DEGRADED feature vectors need stronger conviction.
        if data_quality_boost > 0:
            threshold = min(0.95, threshold + data_quality_boost)
        if primary_conf < threshold:
            return False, {
                "gate":        6,
                "passed":      False,
                "reason":      (
                    f"Confidence {primary_conf:.0%} < "
                    f"{threshold:.0%} threshold "
                    f"({capital_mode} {model_type} mode"
                    f"{', +DQ boost' if data_quality_boost > 0 else ''})"
                ),
                "confidence":  primary_conf,
                "threshold":   threshold,
                "capital_mode":capital_mode,
                "data_quality_boost": data_quality_boost,
            }

        # ── Final size multiplier (all gates combined) ─────────────
        final_mult = float(risk_context.get("final_size_mult", 1.0))

        return True, {
            "gate":          6,
            "passed":        True,
            "confidence":    round(primary_conf, 4),
            "threshold":     round(threshold, 4),
            "vix_adjusted":  india_vix > 20,
            "india_vix":     india_vix,
            "capital_mode":  capital_mode,
            "alignment":     alignment,
            "final_size_mult":final_mult,
            "data_quality_boost": data_quality_boost,
        }