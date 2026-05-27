# signals/gate5_sr_validator.py
# Gate 5 — Support/Resistance entry quality validation
# Entry near support = tight stop, great R:R (LONG)
# Entry near resistance = tight stop, great R:R (SHORT)
# Connected to: processing/support_resistance.py, features/feature_builder.py

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class Gate5SRValidator:
    """
    Gate 5: Entry quality based on support/resistance proximity.

    For LONG: near support + far from resistance = Grade A (tight stop, lots of room)
    For SHORT: near resistance + far from support = Grade A (tight stop above, lots of room below)

    The S/R engine computes entry_quality from a LONG perspective only.
    Gate 5 recomputes quality for SHORT signals using the inverted distance perspective.

    Grade C: reduced size (65%).
    Grade D: reduced size (40%) + requires confidence ≥ category threshold and alignment B or better.
             Category thresholds: positional=68%, swing=65%, intraday=62%.
             20yr rule: longer hold duration in a bad S/R setup = needs proportionally
             more conviction. A positional trade risks 3-4 weeks of adverse drift in a
             Grade D setup; an intraday trade is out by end-of-day regardless.

    SHORT near_breakout: buyers are aggressive at resistance (volume spike).
             Shorting into a potential breakout = swimming against institutional flow.
             Penalised 0.80× even when entry_quality is otherwise good.
    """

    MIN_REWARD_RISK      = 2.0   # soft floor — below this: size reduced
    RR_HARD_BLOCK        = 0.5   # hard floor — below this: signal blocked regardless of confidence.
                                 # R:R < 0.5 means resistance is >2× closer than support for LONGs
                                 # (or support >2× closer than resistance for SHORTs) — the S/R
                                 # geometry makes the trade statistically losing before fees.
    GRADE_SIZE_MAP  = {
        "A":  1.00,
        "B":  0.85,
        "C":  0.65,
        "D":  0.40,
    }

    # Category-aware Grade D confidence threshold.
    # Grade D = far from support (LONG) or near support (SHORT) — poor entry geometry.
    # A positional trade holds 2–4 weeks in a Grade D setup = widest possible risk window.
    # An intraday trade has the same poor geometry but Gate 6 already enforces 65%,
    # making this threshold non-binding for intraday (Gate 6 is tighter).
    #
    # 20yr rule: the worse the entry geometry, the MORE conviction you need.
    # Grade D entries are small probes (0.40× size) — require proportionally higher
    # confidence to justify even that small probe, especially for longer holds.
    GRADE_D_OVERRIDE_CONF = {
        "positional": 0.68,   # 3-4 week hold + bad S/R = needs 68%+ conviction
        "swing":      0.65,   # 3-10 day hold + bad S/R = needs 65%+
        "intraday":   0.62,   # Gate 6 at 65% is the binding gate for intraday D-entries
    }

    def _short_quality(self, sup_dist_pct: float, res_dist_pct: float,
                       res_str: int) -> Tuple[str, float]:
        """
        Recompute entry quality for SHORT direction.
        Near resistance = small res_dist = tight stop above = GOOD.
        Mirrors _entry_quality() in support_resistance.py with distances swapped.
        """
        rr = sup_dist_pct / max(res_dist_pct, 0.1)  # room down / stop up
        if rr >= 3 and res_dist_pct < 1.5 and res_str >= 3:
            quality = "A"
        elif rr >= 2 and res_dist_pct < 2.5:
            quality = "B"
        elif rr >= 1.5:
            quality = "C"
        else:
            quality = "D"
        return quality, round(rr, 2)

    def check(
        self,
        sr_levels:        Dict,
        signal_direction: str,
        ml_confidence:    float,
        alignment:        str,
        per_category_mode:bool = False,
        category:         str  = "swing",
    ) -> Tuple[bool, Dict]:
        """
        Validate entry quality from S/R perspective.

        Args:
            sr_levels:        from FeatureBuilder result['sr_levels']
            signal_direction: "LONG" or "SHORT"
            ml_confidence:    boosted confidence from gate 4
            alignment:        alignment grade from gate 4
            per_category_mode: When True, the Grade-D override drops the
                              {A+, A, B} alignment requirement and relies
                              on confidence alone. Used by the per-category
                              pipeline where a single model's signal is
                              valid on its own — multi-model alignment is
                              informational, never a veto. Gate 6's
                              category-specific threshold still gates the
                              signal downstream.
            category:         "swing" | "positional" | "intraday"
                              Used to select the category-appropriate Grade-D
                              confidence threshold. Positional holds require
                              higher conviction in a Grade-D S/R setup.
        """
        entry_quality   = sr_levels.get("entry_quality", "C")
        reward_risk     = float(sr_levels.get("reward_risk_sr", 1.5))
        sup_dist        = float(sr_levels.get("support_distance_pct", 3.0))
        res_dist        = float(sr_levels.get("resistance_distance_pct", 3.0))
        sup_str         = int(sr_levels.get("support_strength", 1))
        res_str         = int(sr_levels.get("resistance_strength", 1))
        near_support    = bool(sr_levels.get("near_support", False))
        near_resistance = bool(sr_levels.get("near_resistance", False))
        near_breakout   = bool(sr_levels.get("near_breakout", False))

        # S/R engine computes entry_quality from LONG perspective only.
        # For SHORT, near resistance = tight stop = GOOD — recompute with inverted distances.
        if signal_direction == "SHORT":
            entry_quality, reward_risk = self._short_quality(sup_dist, res_dist, res_str)
            logger.debug(
                f"Gate5 SHORT quality recomputed: {entry_quality} "
                f"(sup={sup_dist:.1f}% res={res_dist:.1f}% rr={reward_risk:.2f})"
            )

        # ── Hard R:R floor ────────────────────────────────────────
        # If S/R geometry makes the trade statistically losing (resistance <2×
        # closer for LONG, support <2× closer for SHORT), block regardless of
        # ML confidence. A high-confidence model cannot overcome a trade where
        # you risk ₹5 to potentially make ₹1 from S/R perspective.
        if reward_risk < self.RR_HARD_BLOCK:
            return False, {
                "gate":          5,
                "passed":        False,
                "reason":        (
                    f"S/R R:R {reward_risk:.2f}:1 below hard floor "
                    f"({self.RR_HARD_BLOCK:.1f}:1) — "
                    f"entry geometry statistically losing"
                ),
                "entry_quality": entry_quality,
                "reward_risk":   reward_risk,
                "category":      category,
                "size_mult":     0.0,
            }

        size_mult = self.GRADE_SIZE_MAP.get(entry_quality, 0.5)
        reasons   = []

        # ── LONG-specific checks ──────────────────────────────────
        if signal_direction == "LONG":
            if near_resistance and not near_breakout:
                reasons.append(
                    f"Price near resistance — limited upside "
                    f"({res_dist:.1f}% to resistance)"
                )
                size_mult *= 0.75

            if near_breakout:
                reasons.append("Breakout setup — confirm with volume")
                size_mult *= 1.1

        # ── SHORT-specific checks ─────────────────────────────────
        if signal_direction == "SHORT":
            if near_support:
                reasons.append(
                    f"Price near support — risky short entry "
                    f"({sup_dist:.1f}% to support)"
                )
                size_mult *= 0.75

            if near_breakout:
                # Breakout setups are DANGEROUS for shorts.
                # Volume surge at resistance means BUYERS are aggressive —
                # the stock may be about to break out, not roll over.
                # 20yr rule: never short into a confirmed breakout volume spike.
                reasons.append(
                    f"Breakout volume spike at resistance — high risk for SHORT "
                    f"({res_dist:.1f}% to resistance); buyers aggressive"
                )
                size_mult *= 0.80

        # ── R:R check ─────────────────────────────────────────────
        if reward_risk < self.MIN_REWARD_RISK:
            reasons.append(
                f"Poor reward:risk = {reward_risk:.1f}:1 "
                f"(need {self.MIN_REWARD_RISK:.1f}:1)"
            )
            size_mult *= 0.5

        # ── Grade D override ──────────────────────────────────────
        # In per-category mode: confidence-only gate (alignment is purely
        # informational; Gate 6's category-specific threshold is the real
        # confidence gate). In multi-model mode: also require alignment B+.
        #
        # Threshold is category-aware: positional (3-4 week hold) needs
        # more conviction in a Grade-D setup than intraday (< 1 day).
        if entry_quality == "D":
            grade_d_threshold = self.GRADE_D_OVERRIDE_CONF.get(category, 0.65)
            conf_ok  = ml_confidence >= grade_d_threshold
            align_ok = per_category_mode or alignment in ("A+", "A", "B")
            if conf_ok and align_ok:
                reasons.append(
                    f"Grade D entry — allowing at reduced size "
                    f"(conf={ml_confidence:.0%} ≥ {grade_d_threshold:.0%}"
                    + (f", per-category [{category}]" if per_category_mode else f", align={alignment}")
                    + ")"
                )
                size_mult = 0.40
            else:
                reason_bits = []
                if not conf_ok:
                    reason_bits.append(f"conf {ml_confidence:.0%} < {grade_d_threshold:.0%} [{category}]")
                if not align_ok:
                    reason_bits.append(f"align {alignment} not B/A/A+")
                return False, {
                    "gate":          5,
                    "passed":        False,
                    "reason":        "Grade D entry — " + " · ".join(reason_bits),
                    "entry_quality": entry_quality,
                    "reward_risk":   reward_risk,
                    "category":      category,
                    "size_mult":     0.0,
                }

        # ── Final size cap ────────────────────────────────────────
        size_mult = round(max(0.0, min(1.2, size_mult)), 3)

        return True, {
            "gate":             5,
            "passed":           True,
            "entry_quality":    entry_quality,
            "reward_risk":      round(reward_risk, 2),
            "support_dist_pct": round(sup_dist, 2),
            "resist_dist_pct":  round(res_dist, 2),
            "near_support":     near_support,
            "near_resistance":  near_resistance,
            "near_breakout":    near_breakout,
            "category":         category,
            "size_mult":        size_mult,
            "notes":            reasons,
        }