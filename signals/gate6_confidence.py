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

    # Provisional per-category confidence premium (P2 — edge quality).
    # Swing (~0.46–0.60 CV) and positional (~0.49–0.64 CV) have the weakest
    # 3-class model edge in the universe; intraday (~0.67–0.80) is strong. Until
    # the paper-trading attribution loop PROVES a real per-category edge, demand
    # extra conviction on the weak categories — a modest additive premium on top
    # of the base threshold. It applies everywhere Gate 6 runs (live signal_engine
    # AND the offline P0-3 pipeline backtest), so validation reflects live gating.
    #
    # Tradeoff: this slows paper-data collection on swing/positional (fewer trades
    # → slower attribution). Remove (set to {}) or zero a category once attribution
    # shows that category clears the scale-up bar (WR>52, PF>1.5). Folded into the
    # same +0.15pp total threshold cap as the situational boosts.
    PROVISIONAL_EDGE_PREMIUM = {"swing": 0.03, "positional": 0.05, "intraday": 0.0}

    # ── Expectancy acceptance path (EDGE-1, 2026-06-03) ──────────────────────
    # A 20yr trader trades positive EXPECTED VALUE, not certainty. A clean setup
    # (Gate-5 Grade A/B geometry) at fair odds (>=2:1 reward:risk) with the model
    # at least leaning the right way is +EV even below the P(win) threshold:
    #   EV(R) = conf*reward_risk - (1-conf)   >0 well below 60% when rr>=2.
    # This lets the bread-and-butter setups through instead of only the rare
    # high-conviction tail — the weak-AUC swing/positional models (calibrated
    # conf clusters near 0.50) otherwise almost never clear the P(win) bar, so
    # paper-trading never collects the data needed to validate them.
    # Bounded: needs clean geometry AND fair odds AND conf within reach of the
    # (already regime/VIX-adjusted) threshold. Sizing is unchanged — Gate 5's
    # grade multiplier already shrinks Grade-B / weaker entries. Tune via funnel.
    EXPECTANCY_CONF_FLOOR = 0.50   # model must lean the chosen direction
    EXPECTANCY_MIN_RR     = 2.0    # reward:risk at least 2:1
    EXPECTANCY_MAX_GAP    = 0.10   # conf no more than 10pp below threshold
    EXPECTANCY_GRADES     = ("A", "B")

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
        entry_quality:       str   = "C",
        reward_risk:         float = 0.0,
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
        # P2 provisional edge premium — weak-edge categories (swing/positional)
        # demand more conviction until attribution proves edge. See class const.
        edge_premium = self.PROVISIONAL_EDGE_PREMIUM.get(model_type, 0.0)
        if edge_premium > 0:
            threshold = min(0.95, threshold + edge_premium)
        india_vix = float(risk_context.get("india_vix", 0.0))
        # VIX penalty is CATEGORY-AWARE (G6-1). 20yr rule: elevated VIX is the
        # SOURCE of alpha for intraday — the market moves 2–3%/session, exactly
        # the regime where same-session edges pay. Penalising intraday like
        # positional blocks valid trades in the best environment. A 4-week
        # positional hold THROUGH that volatility is the real risk → penalise it
        # most. (vix>25 boost, vix>20 boost) per category:
        _vix_boost = {
            "intraday":   (0.03, 0.00),
            "swing":      (0.10, 0.05),
            "positional": (0.15, 0.08),
        }.get(model_type, (0.10, 0.05))
        if india_vix > 25:
            threshold = min(0.90, threshold + _vix_boost[0])
        elif india_vix > 20:
            threshold = min(0.90, threshold + _vix_boost[1])
        # Data-quality boost — DEGRADED feature vectors need stronger conviction.
        if data_quality_boost > 0:
            threshold = min(0.95, threshold + data_quality_boost)
        # ── Expectancy override (EDGE-1) ──────────────────────────
        # Below the P(win) bar, a clean setup at fair odds is still +EV. Accept
        # iff geometry is A/B, R:R>=2, the model leans the right way (>=floor),
        # and conf is within reach of the threshold.
        expectancy_pass = False
        expected_R      = None
        if primary_conf < threshold:
            expectancy_pass = (
                primary_conf >= self.EXPECTANCY_CONF_FLOOR
                and entry_quality in self.EXPECTANCY_GRADES
                and reward_risk  >= self.EXPECTANCY_MIN_RR
                and primary_conf >= threshold - self.EXPECTANCY_MAX_GAP
            )
            if expectancy_pass:
                expected_R = round(primary_conf * reward_risk - (1 - primary_conf), 3)
                logger.info(
                    f"Gate 6 EXPECTANCY pass: conf {primary_conf:.0%} < thr "
                    f"{threshold:.0%} but Grade {entry_quality} @ {reward_risk:.1f}:1 "
                    f"→ +{expected_R}R expected ({model_type})"
                )
            else:
                return False, {
                    "gate":        6,
                    "passed":      False,
                    "reason":      (
                        f"Confidence {primary_conf:.0%} < "
                        f"{threshold:.0%} threshold "
                        f"({capital_mode} {model_type} mode"
                        # G6-5: the boost stacks counter-regime/HIGH_VOL/instability/
                        # stale-data/DQ — show the magnitude, don't falsely call it "DQ".
                        f"{f', +{data_quality_boost:.0%} thr-adj' if data_quality_boost > 0 else ''}"
                        f"{f', +{edge_premium:.0%} edge-premium' if edge_premium > 0 else ''}"
                        f"{f', +{(_vix_boost[0] if india_vix>25 else _vix_boost[1]):.0%} VIX' if india_vix > 20 else ''}"
                        f"; no expectancy override: Grade {entry_quality} @ {reward_risk:.1f}:1)"
                    ),
                    "confidence":  primary_conf,
                    "threshold":   threshold,
                    "capital_mode":capital_mode,
                    "data_quality_boost": data_quality_boost,
                    "edge_premium": edge_premium,
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
            "edge_premium":  edge_premium,
            "expectancy_pass": expectancy_pass,
            "expected_R":    expected_R,
        }