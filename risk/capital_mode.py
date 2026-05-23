# risk/capital_mode.py
# Determines trading mode based on capital size
# Small capital = conservative rules. Grows with you automatically.

from typing import Dict

class CapitalMode:
    """
    Auto-adjusting capital mode based on account size.

    20yr trader truth:
    The #1 mistake new traders make is sizing too large.
    A 10% loss on ₹10,000 = ₹1,000 — psychologically devastating.
    Same loss on ₹1L = ₹10,000 — still hurts but survivable.
    Start small. Prove the system. Then scale.

    SMALL  (< ₹50,000):   1% risk, A+ only, swing only
    GROWING (₹50K-₹2L):   1.5% risk, A signals, swing+intraday
    FULL   (> ₹2,00,000):  2% risk, B signals, all timeframes
    """

    # Per-category ATR multipliers — timeframe-aware stop/target geometry.
    # Intraday: tight stop so daily noise doesn't trigger it; close target for same-day exit.
    # Swing: medium stop; hold 3–7 days through normal intraday swings.
    # Positional: wide stop; hold 2–4 weeks, must breathe through daily volatility.
    CATEGORY_ATR = {
        "intraday":   {"stop_atr_mult": 1.0, "target_atr_mult": 2.5},
        "swing":      {"stop_atr_mult": 2.0, "target_atr_mult": 5.0},
        "positional": {"stop_atr_mult": 2.5, "target_atr_mult": 6.5},
    }

    MODES = {
        "SMALL": {
            "max_risk_pct":     0.010,
            "max_positions":    2,
            "min_alignment":    {"A+"},
            "allowed_tf":       ["swing"],
            "min_conf":         0.68,
            "monthly_halt_pct": 0.05,
            "stop_atr_mult":    1.5,
            "target_atr_mult":  4.0,
        },
        "GROWING": {
            "max_risk_pct":     0.015,
            "max_positions":    3,
            "min_alignment":    {"A+", "A"},
            "allowed_tf":       ["swing", "intraday"],
            "min_conf":         0.63,
            "monthly_halt_pct": 0.08,
            "stop_atr_mult":    2.0,
            "target_atr_mult":  5.0,
        },
        "FULL": {
            "max_risk_pct":     0.020,
            "max_positions":    5,
            "min_alignment":    {"A+", "A", "B"},
            "allowed_tf":       ["swing", "intraday", "positional"],
            "min_conf":         0.60,
            "monthly_halt_pct": 0.10,
            "stop_atr_mult":    2.5,
            "target_atr_mult":  6.0,
        },
    }

    @classmethod
    def detect(cls, capital: float) -> str:
        if capital < 50_000:   return "SMALL"
        if capital < 200_000:  return "GROWING"
        return "FULL"

    @classmethod
    def get_config(cls, capital: float) -> Dict:
        mode = cls.detect(capital)
        cfg  = cls.MODES[mode].copy()
        cfg["mode"]    = mode
        cfg["capital"] = capital
        return cfg

    @classmethod
    def max_risk_amount(cls, capital: float) -> float:
        cfg = cls.get_config(capital)
        return round(capital * cfg["max_risk_pct"], 2)