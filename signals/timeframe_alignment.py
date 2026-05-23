# signals/timeframe_alignment.py
# Timeframe alignment utility used by gate4 and signal_engine
# Encapsulates alignment logic for reuse

import logging
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)


class TimeframeAlignment:
    """
    Computes alignment grade from three model signals.
    Used by Gate4 and signal_engine for consistent alignment logic.
    """

    GRADE_CONFIG = {
        "A+": {"boost": 0.15, "size_mult": 1.20, "description": "All timeframes agree"},
        "A":  {"boost": 0.10, "size_mult": 1.00, "description": "Positional + Swing agree"},
        "B":  {"boost": 0.05, "size_mult": 0.85, "description": "Two timeframes agree"},
        "C":  {"boost": 0.00, "size_mult": 0.70, "description": "Single direction signal"},
        "F":  {"boost":-0.20, "size_mult": 0.00, "description": "Timeframe conflict — FLAT"},
    }

    def compute(
        self,
        positional_signal: str,
        swing_signal:      str,
        intraday_signal:   str,
    ) -> Dict:
        """
        Compute alignment grade and associated parameters.
        Returns full alignment context dict.
        """
        signals  = [positional_signal, swing_signal, intraday_signal]
        non_flat = [s for s in signals if s != "FLAT"]
        longs    = non_flat.count("LONG")
        shorts   = non_flat.count("SHORT")

        # Conflict — block all
        if longs > 0 and shorts > 0:
            grade = "F"
        # All 3 non-flat and agree
        elif len(non_flat) == 3 and len(set(non_flat)) == 1:
            grade = "A+"
        # Positional and swing agree
        elif (positional_signal == swing_signal and
              positional_signal != "FLAT"):
            grade = "A"
        # Any 2 agree
        elif len(non_flat) >= 2 and len(set(non_flat)) == 1:
            grade = "B"
        # Only 1 non-flat
        elif len(non_flat) == 1:
            grade = "C"
        # All flat
        else:
            grade = "C"

        cfg = self.GRADE_CONFIG[grade]
        # Dominant direction
        dominant = "LONG" if longs >= shorts else ("SHORT" if shorts > longs else "FLAT")

        return {
            "grade":           grade,
            "dominant":        dominant,
            "conf_boost":      cfg["boost"],
            "size_mult":       cfg["size_mult"],
            "description":     cfg["description"],
            "signals":         {"positional": positional_signal,
                                "swing":      swing_signal,
                                "intraday":   intraday_signal},
            "non_flat_count":  len(non_flat),
            "longs":           longs,
            "shorts":          shorts,
        }