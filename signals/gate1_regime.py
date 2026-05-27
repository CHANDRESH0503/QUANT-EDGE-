# signals/gate1_regime.py
# Gate 1 — Market regime check. First and most important gate.
# CHOPPY market = all signals blocked. No exceptions.
# Connected to: processing/regime_detector.py, models/train_regime.py

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class Gate1Regime:
    """
    Gate 1: Is the current regime tradeable?

    20yr trader rule:
    The biggest losses of my career came from trading in choppy,
    directionless markets where even perfect signals failed.
    Learning to sit on hands in a sideways market is worth more
    than any indicator. This gate enforces that discipline.

    Hard blocks:
    - CHOPPY_SIDEWAYS → BLOCK ALL. Return FLAT immediately.
    - Stability < 25% → BLOCK. Regime extremely uncertain.

    Soft penalties (size reductions, never vetoes):
    - BULL_TRENDING  → normal sizing (1.2× mult). LONGs are regime-aligned.
                       SHORTs allowed but 0.5× via counter-regime penalty (signal_engine).
    - BEAR_TRENDING  → reduced sizing (0.8× mult). SHORTs are regime-aligned.
                       LONGs allowed but 0.5× via counter-regime penalty (signal_engine).
    - HIGH_VOLATILITY→ half size (0.5×), both directions allowed.

    Direction is NEVER vetoed here — this gate is called with signal_direction=None
    so per-category counter-regime trades pass. The signal_engine applies a 0.5×
    size penalty for counter-regime trades (LONG in BEAR, SHORT in BULL) instead.
    See signal_engine.py: "User directive 4B".

    Stability threshold:
    25–40%: reduce position size (LOW_STAB_MULT).
    Recent regime changes: additional instability mult (INSTABILITY_MULT).
    """

    TRADEABLE = {"BULL_TRENDING", "BEAR_TRENDING", "HIGH_VOLATILITY"}
    BLOCKED   = {"CHOPPY_SIDEWAYS"}

    MIN_STABILITY    = 0.25   # hard block below 25% (regime extremely uncertain)
    LOW_STAB_MULT    = 0.65   # size reduction for stability 25–40% range
    INSTABILITY_MULT = 0.75   # additional size reduction if regime recently changed

    def check(
        self,
        regime_result:  Dict,
        regime_trend:   Dict,
        signal_direction: Optional[str] = None,
    ) -> Tuple[bool, Dict]:
        """
        Check if current regime allows trading.

        Args:
            regime_result:    from RegimeDetector.detect()
            regime_trend:     from RegimeDetector.get_recent_regime_trend()
            signal_direction: "LONG" / "SHORT" — direction-specific veto. When
                              None, only regime/stability are validated (used by
                              the global gate-pass; per-category direction checks
                              re-call this with the actual model direction).

        Returns:
            (passed: bool, context: Dict)
        """
        regime      = regime_result.get("regime", "CHOPPY_SIDEWAYS")
        stability   = float(regime_result.get("stability", 0.5))
        trade_long  = bool(regime_result.get("trade_long",  False))
        trade_short = bool(regime_result.get("trade_short", False))
        pos_mult    = float(regime_result.get("position_mult", 1.0))

        # ── Hard block ────────────────────────────────────────────
        if regime in self.BLOCKED:
            return False, {
                "gate":    1,
                "passed":  False,
                "reason":  f"CHOPPY_SIDEWAYS regime — all signals blocked",
                "regime":  regime,
                "action":  "Stay in cash. Wait for clear trend.",
            }

        # ── Direction filter (only when a direction is given) ─────
        if signal_direction == "LONG" and not trade_long:
            return False, {
                "gate":    1,
                "passed":  False,
                "reason":  f"{regime} — LONG signals not allowed",
                "regime":  regime,
                "action":  "Only SHORT signals allowed in current regime",
            }

        if signal_direction == "SHORT" and not trade_short:
            return False, {
                "gate":    1,
                "passed":  False,
                "reason":  f"{regime} — SHORT signals not allowed",
                "regime":  regime,
                "action":  "Only LONG signals allowed in current regime",
            }

        # ── Stability check ───────────────────────────────────────
        # Hard block only when regime is extremely uncertain (<25%).
        # Between 25–40%, reduce position size instead of blocking entirely —
        # the regime is still the plurality direction, just not firmly established.
        if stability < self.MIN_STABILITY:
            return False, {
                "gate":    1,
                "passed":  False,
                "reason":  f"Regime stability too low: {stability:.0%} (need {self.MIN_STABILITY:.0%})",
                "regime":  regime,
                "action":  "Wait for regime to stabilise",
            }

        low_stability = stability < 0.40

        # ── Instability multiplier (recent regime change) ─────────
        regime_changes = int(regime_trend.get("changes", 0))
        final_mult     = pos_mult
        if regime_changes >= 2:
            final_mult = round(pos_mult * self.INSTABILITY_MULT, 3)
            logger.info(
                f"Gate 1: Regime unstable ({regime_changes} changes) "
                f"— size reduced to {final_mult:.2f}x"
            )
        if low_stability:
            final_mult = round(final_mult * self.LOW_STAB_MULT, 3)
            logger.info(
                f"Gate 1: Low stability ({stability:.0%}) "
                f"— size further reduced to {final_mult:.2f}x"
            )

        return True, {
            "gate":           1,
            "passed":         True,
            "regime":         regime,
            "stability":      stability,
            "low_stability":  low_stability,
            "position_mult":  final_mult,
            "description":    regime_result.get("description", ""),
            "regime_changes": regime_changes,
        }