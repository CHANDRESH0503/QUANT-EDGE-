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

    GATE5-1 (momentum-aligned): the entry_quality grade is now an
    ATR-relative *runway* signal (computed in support_resistance.py for both
    directions) and SIZES the trade — it never vetoes it:
        Grade A → full size · B → 0.85× · C → 0.65× · D → 0.40× (probe).
    Grade D = entering with limited clear runway to the ATR target (a strong
    wall is near). reward_risk is the REAL trade R:R (target 5×ATR / stop
    2×ATR = 2.5:1), so the old hard R:R floor — which fired on a sub-ATR
    noise ratio and blocked ~37% of genuine 2.5:1 trades — no longer
    triggers. The conviction gate is Gate 6's category threshold downstream.

    SHORT near_breakout: buyers are aggressive at resistance (volume spike).
             Shorting into a potential breakout = swimming against institutional flow.
             Penalised 0.80× even when entry_quality is otherwise good.
    """

    MIN_REWARD_RISK      = 2.0   # soft floor — below this: size reduced (defensive;
                                 # prod reward_risk is the real 2.5:1 so this rarely fires)
    RR_HARD_BLOCK        = 0.5   # defensive hard floor for malformed/degenerate R:R only.
                                 # prod reward_risk = real trade R:R (2.5) so this never
                                 # fires on S/R geometry (GATE5-1). Kept as an input guard.
    GRADE_SIZE_MAP  = {
        "A":  1.00,
        "B":  0.85,
        "C":  0.65,
        "D":  0.40,
    }

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

        # GATE5-1: the S/R engine now computes an ATR-relative *runway* grade
        # for BOTH directions. SHORT uses its own grade; reward_risk is the
        # real ATR trade R:R (direction-independent), not a recomputed ratio.
        if signal_direction == "SHORT":
            entry_quality = sr_levels.get("entry_quality_short", entry_quality)
            logger.debug(
                f"Gate5 SHORT runway grade: {entry_quality} (rr={reward_risk:.2f})"
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

        # ── Grade D = probe size, NOT a veto (GATE5-1) ────────────
        # Grade D now means "entering into a near strong wall" — limited
        # runway, so the trade is a small probe (GRADE_SIZE_MAP D = 0.40×,
        # already applied above, further shrunk by any near_* penalty).
        # S/R SIZES the trade; it does not veto it. The conviction gate is
        # Gate 6's category-specific confidence threshold downstream — Gate 5
        # no longer re-imposes its own (that double-counted and, on the old
        # noise grade, blocked ~37% of genuine 2.5:1 trades). Counter-regime
        # + Grade D remains a hard block in signal_engine/backtest (Rule #21)
        # — that's a regime safety on probes, independent of this change.
        if entry_quality == "D":
            reasons.append(
                "Grade D entry (limited runway to ATR target) — probe size"
            )

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