# signals/gate4_ml_predictor.py
# Gate 4 — Run all 3 ML models and check timeframe alignment
# This is the core prediction gate
# Connected to: models/train_all.py, features/feature_builder.py

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

from models.train_swing      import SwingModelTrainer
from models.train_intraday   import IntradayModelTrainer
from models.train_positional import PositionalModelTrainer


class Gate4MLPredictor:
    """
    Gate 4: Run all 3 ML models and compute timeframe alignment.

    20yr trader insight:
    One model saying LONG is a suggestion.
    All three models saying LONG simultaneously is a conviction trade.
    The alignment grade is one of the strongest meta-features
    in the entire system. Never trade F-grade alignment.

    Alignment grades (now INFORMATIONAL — never blocks):
    A+: All 3 non-flat signals agree → +15% confidence, 1.2× size
    A:  Positional + Swing agree     → +10% confidence, 1.0× size
    B:  Any 2 agree                  → +5%  confidence, 0.85× size
    C:  Only 1 non-flat signal       → 0%   confidence, 0.7× size
    F:  Cross-category conflict       → -20% (no blanket block — each model runs Gate 5/6 alone)

    Gate 4 passes if ANY model is non-FLAT. Per-category confidence
    thresholds are enforced downstream in Gate 6 (one call per category).
    """

    MIN_CONF = {
        "swing":      0.60,
        "intraday":   0.65,
        "positional": 0.55,
    }

    ALIGNMENT_BOOST = {
        "A+": 0.15,
        "A":  0.10,
        "B":  0.05,
        "C":  0.00,
        "F": -0.20,
    }

    # F = 0.50 (not 0.00). Under per-category mode alignment is purely
    # informational — a hidden veto via size_mult=0 made any F-grade signal
    # emit with shares=0/stop=0/target=0, which propagated as a "SIGNAL"
    # alert with no actionable levels (real bug found on ICICIBANK
    # swing-SHORT). 0.50 still penalises model disagreement (along with the
    # -20% conf_boost) without nulling out the trade entirely.
    ALIGNMENT_SIZE = {
        "A+": 1.20,
        "A":  1.00,
        "B":  0.85,
        "C":  0.70,
        "F":  0.50,
    }

    def __init__(self, ticker: str = "HDFCBANK.NS"):
        # Priority: prod (25yr + non-zero features) → full (2020+) → technical (baseline)
        self._swing_prod      = SwingModelTrainer(variant="prod",      ticker=ticker)
        self._intraday_prod   = IntradayModelTrainer(variant="prod",   ticker=ticker)
        self._positional_prod = PositionalModelTrainer(variant="prod", ticker=ticker)
        self.swing_model      = SwingModelTrainer(variant="full",      ticker=ticker)
        self.intraday_model   = IntradayModelTrainer(variant="full",   ticker=ticker)
        self.positional_model = PositionalModelTrainer(variant="full", ticker=ticker)
        self._swing_fallback      = SwingModelTrainer(variant="technical",      ticker=ticker)
        self._intraday_fallback   = IntradayModelTrainer(variant="technical",   ticker=ticker)
        self._positional_fallback = PositionalModelTrainer(variant="technical", ticker=ticker)
        self._loaded          = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        # Swing: prefer prod → full → technical
        if self._swing_prod.load():
            self.swing_model = self._swing_prod
        elif not self.swing_model.load():
            self.swing_model = self._swing_fallback
            self.swing_model.load()
        # Technical fallback always loaded — used as rescue when primary is degenerate
        self._swing_fallback.load()

        # Intraday: prefer prod → full → technical
        if self._intraday_prod.load():
            self.intraday_model = self._intraday_prod
        elif not self.intraday_model.load():
            self.intraday_model = self._intraday_fallback
            self.intraday_model.load()
        self._intraday_fallback.load()

        # Positional: prefer prod → full → technical
        if self._positional_prod.load():
            self.positional_model = self._positional_prod
        elif not self.positional_model.load():
            self.positional_model = self._positional_fallback
            self.positional_model.load()
        self._positional_fallback.load()

        self._loaded = True

    # Calibration-collapse threshold. Tightened 0.95 → 0.92 so we catch
    # isotonic-clipped predictions earlier and route to the technical
    # fallback. Lower values would trigger too often on legitimate
    # high-confidence calls; 0.92 isolates the bad cases.
    DEGENERATE_PROB_THRESHOLD = 0.92

    def _is_degenerate(self, pred: Dict) -> bool:
        """True when calibration collapsed — any class probability is ≥ threshold."""
        return (
            pred.get("prob_flat",  0) >= self.DEGENERATE_PROB_THRESHOLD or
            pred.get("prob_long",  0) >= self.DEGENERATE_PROB_THRESHOLD or
            pred.get("prob_short", 0) >= self.DEGENERATE_PROB_THRESHOLD
        )

    def check(
        self,
        feature_vector: Dict,
    ) -> Tuple[bool, Dict]:
        """
        Run all 3 ML models and check alignment.

        Args:
            feature_vector: output of FeatureBuilder.build_all()

        Returns:
            (passed: bool, details: Dict)
        """
        self._ensure_loaded()

        raw = feature_vector.get("raw_features", {})

        # Build model-specific row DataFrames
        swing_row = self._build_row(raw, self.swing_model.features or [])
        intra_row = self._build_row(raw, self.intraday_model.features or [])
        pos_row   = self._build_row(raw, self.positional_model.features or [])

        # Predict — fall back to technical variant if full model output is degenerate
        # (isotonic calibration can collapse to prob=1.0 when features are all-zero)
        swing_pred = self.swing_model.predict(swing_row)
        if self._is_degenerate(swing_pred) and self._swing_fallback.model is not None:
            logger.warning(
                f"Swing full model degenerate (prob_flat={swing_pred.get('prob_flat',0):.3f}) "
                f"— using technical fallback"
            )
            fallback_row = self._build_row(raw, self._swing_fallback.features or [])
            swing_pred = self._swing_fallback.predict(fallback_row)
            swing_pred["used_fallback"] = True

        intra_pred = self.intraday_model.predict(intra_row)
        if self._is_degenerate(intra_pred) and self._intraday_fallback.model is not None:
            logger.warning(
                f"Intraday full model degenerate (prob_flat={intra_pred.get('prob_flat',0):.3f}) "
                f"— using technical fallback"
            )
            fallback_row = self._build_row(raw, self._intraday_fallback.features or [])
            intra_pred = self._intraday_fallback.predict(fallback_row)
            intra_pred["used_fallback"] = True

        pos_pred = self.positional_model.predict(pos_row)
        if self._is_degenerate(pos_pred) and self._positional_fallback.model is not None:
            logger.warning(
                f"Positional full model degenerate (prob_flat={pos_pred.get('prob_flat',0):.3f}) "
                f"— using technical fallback"
            )
            fallback_row = self._build_row(raw, self._positional_fallback.features or [])
            pos_pred = self._positional_fallback.predict(fallback_row)
            pos_pred["used_fallback"] = True

        # Alignment
        alignment, conf_boost, size_mult = self._calc_alignment(
            pos_pred["signal"],
            swing_pred["signal"],
            intra_pred["signal"],
        )

        # Boosted confidences (per-category — used downstream in Gate 6)
        swing_conf_boosted = min(0.95, swing_pred["confidence"] + conf_boost)
        intra_conf_boosted = min(0.95, intra_pred["confidence"] + conf_boost)
        pos_conf_boosted   = min(0.95, pos_pred["confidence"]   + conf_boost)

        # Backward-compat: surface highest-confidence non-FLAT model as "primary"
        # so legacy callers (dashboard, format_signal) keep working. Per-category
        # gating happens in signal_engine — never block here on primary.
        candidates = [
            ("swing",      swing_pred["signal"], swing_conf_boosted),
            ("positional", pos_pred["signal"],   pos_conf_boosted),
            ("intraday",   intra_pred["signal"], intra_conf_boosted),
        ]
        non_flat = [c for c in candidates if c[1] != "FLAT"]

        # Gate 4 passes iff at least one model is non-FLAT. Threshold + S/R
        # are enforced per-category in signal_engine (Gates 5 and 6).
        if not non_flat:
            return False, {
                "gate":              4,
                "passed":            False,
                "reason":            "All three models predict FLAT — no directional edge",
                "alignment":         alignment,
                "conf_boost":        round(conf_boost, 3),
                "size_mult":         size_mult,
                "positional":        pos_pred,
                "swing":             swing_pred,
                "intraday":          intra_pred,
            }

        # Pick highest-confidence non-FLAT model as primary (backward-compat field)
        primary_cat, primary_signal, primary_conf = max(non_flat, key=lambda c: c[2])

        return True, {
            "gate":              4,
            "passed":            True,
            "primary_signal":    primary_signal,
            "primary_conf":      round(primary_conf, 4),
            "primary_category":  primary_cat,
            "alignment":         alignment,
            "conf_boost":        round(conf_boost, 3),
            "size_mult":         size_mult,
            "positional":        pos_pred,
            "swing":             swing_pred,
            "intraday":          intra_pred,
            "swing_conf_boosted":     round(swing_conf_boosted, 4),
            "intra_conf_boosted":     round(intra_conf_boosted, 4),
            "positional_conf_boosted":round(pos_conf_boosted,   4),
        }

    def _calc_alignment(
        self,
        pos_sig: str,
        sw_sig:  str,
        in_sig:  str,
    ) -> Tuple[str, float, float]:
        """Compute alignment grade, confidence boost, and size multiplier."""
        signals  = [pos_sig, sw_sig, in_sig]
        non_flat = [s for s in signals if s != "FLAT"]

        if not non_flat:
            return "C", 0.0, 0.70

        # Conflict check — read multiplier from the table so the F-grade
        # size_mult stays in one place (0.50, not 0.0). Hardcoding 0.0 here
        # silently bypassed the ALIGNMENT_SIZE table and was the root cause
        # of "SIGNAL emitted but stop=0 target=0" alerts.
        if "LONG" in non_flat and "SHORT" in non_flat:
            return "F", self.ALIGNMENT_BOOST["F"], self.ALIGNMENT_SIZE["F"]

        # All 3 agree
        if len(non_flat) == 3 and len(set(non_flat)) == 1:
            grade = "A+"
        # Positional + swing agree (most important pair)
        elif pos_sig == sw_sig and pos_sig != "FLAT":
            grade = "A"
        # Any 2 agree
        elif len(non_flat) >= 2 and len(set(non_flat)) == 1:
            grade = "B"
        else:
            grade = "C"

        return grade, self.ALIGNMENT_BOOST[grade], self.ALIGNMENT_SIZE[grade]

    def _build_row(self, raw: Dict, features: list) -> pd.DataFrame:
        row = {f: raw.get(f, 0.0) for f in features}
        return pd.DataFrame([row])