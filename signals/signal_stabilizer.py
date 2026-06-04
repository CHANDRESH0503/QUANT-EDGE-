# signals/signal_stabilizer.py
# Per-category signal commit layer (anti-flicker) — STAB-2.
# Mirrors the regime anti-whipsaw (RegimeDetector._apply_persistence):
# slow to commit a new direction, fast to cut risk.

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class SignalStabilizer:
    """
    Stabilises the per-category model DECISION so it doesn't flicker
    LONG↔FLAT↔SHORT on marginal/noisy reads. Two mechanisms:

    1. Dead-band — a directional vote only counts if the model's probability
       for that direction beats FLAT by ≥ MARGIN_DELTA. A near-coin-flip
       (e.g. LONG 0.48 vs FLAT 0.51, or LONG 0.30 vs FLAT 0.28) is treated as
       FLAT, so genuinely-uncertain signals never become directional. This is
       the primary lever for day-to-day marginal flips.

    2. Confirmation persistence — committing a NEW direction (out of FLAT)
       requires CONFIRM_CYCLES consecutive decisive reads. Dropping to FLAT,
       or reversing, is IMMEDIATE — we never hold a direction the model no
       longer supports (de-risk fast; Rule #35 / "fast to cut risk").

    State is per category and lives for the engine's lifetime (one engine per
    bank), so it carries across cycles — exactly like the regime's committed
    state. Raw model telemetry is untouched; only the trading decision is
    stabilised.

    Tuning: intraday confirms in 1 cycle (its 30-90 min horizon can't afford a
    multi-cycle entry lag — the dead-band alone de-noises it); swing/positional
    confirm over 2 (they now move on closed daily bars, STAB-1, so confirmation
    costs little and buys stability).
    """

    MARGIN_DELTA   = 0.05                         # min prob[dir] − prob[flat] to be "decisive"
    CONFIRM_CYCLES = {"swing": 2, "positional": 2, "intraday": 1}

    def __init__(self):
        self._committed: Dict[str, str]            = {}   # cat -> committed direction
        self._pending:   Dict[str, str]            = {}   # cat -> direction being confirmed
        self._streak:    Dict[str, int]            = {}   # cat -> consecutive decisive reads

    def _confirm_n(self, category: str) -> int:
        return int(self.CONFIRM_CYCLES.get(category, 2))

    def commit(self, category: str, raw_dir: str, probs: Dict) -> Dict:
        """
        Return the committed (stabilised) direction for this cycle.

        Args:
            raw_dir: the model's argmax this cycle ("LONG"/"SHORT"/"FLAT").
            probs:   dict with prob_long / prob_flat / prob_short.
        Returns dict: {raw, decisive, committed, intervened, reason}.
        """
        pl = float(probs.get("prob_long",  0.0))
        pf = float(probs.get("prob_flat",  0.0))
        ps = float(probs.get("prob_short", 0.0))

        # ── Dead-band: is the raw direction decisive vs FLAT? ─────────
        if raw_dir == "LONG" and (pl - pf) >= self.MARGIN_DELTA:
            decisive = "LONG"
        elif raw_dir == "SHORT" and (ps - pf) >= self.MARGIN_DELTA:
            decisive = "SHORT"
        else:
            decisive = "FLAT"

        committed = self._committed.get(category, "FLAT")
        n         = self._confirm_n(category)

        if decisive == committed:
            committed_now, reason = committed, "stable"
            self._pending[category], self._streak[category] = "", 0

        elif decisive == "FLAT":
            # Fast de-risk: drop to FLAT immediately, never hold a stale signal.
            committed_now = "FLAT"
            reason = "de-risk → FLAT" if committed != "FLAT" else "flat (below dead-band)"
            self._pending[category], self._streak[category] = "", 0

        elif committed != "FLAT":
            # Reversal (LONG↔SHORT): exit to FLAT first, then confirm the new side.
            committed_now = "FLAT"
            self._pending[category], self._streak[category] = decisive, 1
            reason = f"reversal → FLAT first (confirming {decisive} 1/{n})"

        else:
            # From FLAT into a new decisive direction → require confirmation.
            cnt = (self._streak.get(category, 0) + 1
                   if self._pending.get(category) == decisive else 1)
            self._pending[category], self._streak[category] = decisive, cnt
            if cnt >= n:
                committed_now, reason = decisive, f"committed {decisive} ({cnt}/{n})"
                self._pending[category], self._streak[category] = "", 0
            else:
                committed_now = "FLAT"
                reason = f"confirming {decisive} ({cnt}/{n}) — holding FLAT"

        self._committed[category] = committed_now
        return {
            "raw":        raw_dir,
            "decisive":   decisive,
            "committed":  committed_now,
            "intervened": committed_now != raw_dir,
            "reason":     reason,
        }
