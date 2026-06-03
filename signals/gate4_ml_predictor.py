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
    in the entire system. ALL grades are INFORMATIONAL — none blocks.

    Alignment grades:
    A+: All 3 non-flat agree         → +15pp conf, 1.20× size
    A:  Positional + Swing agree     → +10pp conf, 1.00× size
    B:  Any 2 agree (same dir)       → +5pp  conf, 0.85× size
    B-: 2 agree, 1 dissents (2-vs-1) → −5pp  conf, 0.75× size
        20yr rule: if swing+positional say LONG but intraday says SHORT,
        the two higher-timeframe models win. Mild penalty, signal proceeds.
        This was previously Grade F which nuked all signals. Missed entries.
    C:  Only 1 non-flat signal       →  0pp  conf, 0.70× size
    F:  1-vs-1 direct conflict       → −20pp conf, 0.50× size
        Only fires when exactly 1 LONG model and 1 SHORT model (no majority).
        Genuine "wait — models fundamentally disagree" signal.

    Gate 4 passes if ANY model is non-FLAT. Per-category thresholds
    enforced in Gate 6 (one call per category, skip_alignment=True).
    """

    # NOTE: MIN_CONF values below are DOCUMENTATION ONLY — not enforced here.
    # Per-category thresholds live in Gate 6 (gate6_confidence.py) and are
    # read from CapitalMode.MODES[cap_mode][timeframe]. Do not add checks here.
    MIN_CONF = {
        "swing":      0.60,   # Gate 6 FULL-mode reference
        "intraday":   0.65,   # Gate 6 FULL-mode reference
        "positional": 0.55,   # Gate 6 FULL-mode reference
    }

    ALIGNMENT_BOOST = {
        "A+":  0.15,
        "A":   0.10,
        "B":   0.05,
        "B-": -0.05,   # 2-vs-1 conflict: majority direction mildly penalised
        "C":   0.00,
        "F":  -0.20,   # 1-vs-1 conflict: genuine "wait" — penalise both sides
    }

    # F = 0.50 (not 0.00). Under per-category mode alignment is purely
    # informational — a hidden veto via size_mult=0 made any F-grade signal
    # emit with shares=0/stop=0/target=0 (real bug on ICICIBANK swing-SHORT).
    # 0.50 still penalises model disagreement without collapsing sizing to zero.
    ALIGNMENT_SIZE = {
        "A+": 1.20,
        "A":  1.00,
        "B":  0.85,
        "B-": 0.75,   # 2-vs-1: between B (0.85) and C (0.70) — cautious but tradeable
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
        regime_trade_long:  bool = True,
        regime_trade_short: bool = True,
    ) -> Tuple[bool, Dict]:
        """
        Run all 3 ML models and check alignment.

        Args:
            feature_vector: output of FeatureBuilder.build_all()
            regime_trade_long/short: which directions the current regime permits.
                A model vote AGAINST the regime (e.g. the LONG-biased positional
                model voting LONG in a BEAR) is noise — it must not count as a
                conflicting vote that drags a regime-aligned signal to Grade F
                (EDGE-2). Defaults True/True = regime-blind (back-compat).

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

        # Alignment — counter-regime votes are neutralised so a noise model
        # (e.g. LONG-biased positional in a BEAR) can't veto an aligned signal.
        alignment, conf_boost, size_mult = self._calc_alignment(
            pos_pred["signal"],
            swing_pred["signal"],
            intra_pred["signal"],
            regime_trade_long=regime_trade_long,
            regime_trade_short=regime_trade_short,
        )

        # Boosted confidences (per-category — used downstream in Gate 6).
        # Floor at 0.0: a large negative boost (F-grade) on a low-confidence
        # model must not produce a negative probability. Ceiling at 0.95:
        # calibrated probabilities should never be treated as certainty.
        swing_conf_boosted = max(0.0, min(0.95, swing_pred["confidence"] + conf_boost))
        intra_conf_boosted = max(0.0, min(0.95, intra_pred["confidence"] + conf_boost))
        pos_conf_boosted   = max(0.0, min(0.95, pos_pred["confidence"]   + conf_boost))

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
        regime_trade_long:  bool = True,
        regime_trade_short: bool = True,
    ) -> Tuple[str, float, float]:
        """
        Compute alignment grade, confidence boost, and size multiplier.

        Conflict handling (20yr trader logic):
        ─── 2-vs-1 (B-) ─────────────────────────────────────────────
        Two models agree, one dissents. The majority has a valid directional
        view; the minority is the outlier (often the intraday model which is
        most susceptible to same-session noise). Mild penalty to all, majority
        direction still proceeds. Previously this fired as Grade F (-0.20),
        which nuked all three models' confidence below Gate 6 thresholds —
        valid LONG signals killed by a single intraday counter-signal.

        ─── 1-vs-1 (F) ──────────────────────────────────────────────
        Exactly one model says LONG and one says SHORT (third is FLAT or
        the two models on different timeframes directly contradict). No
        majority — genuine "models are split, wait for resolution" signal.
        Hard penalty (-0.20) to both. Only very high confidence (>0.80)
        models survive F-grade and pass Gate 6.
        """
        # EDGE-2: a model vote against what the regime permits is noise — treat
        # it as FLAT for GRADING so it neither earns an agreement boost nor
        # triggers a 1-vs-1 (F) / 2-vs-1 (B-) conflict penalty against an aligned
        # signal. The per-category direction/confidence are untouched downstream.
        def _regime_ok(sig: str) -> str:
            if sig == "LONG"  and not regime_trade_long:  return "FLAT"
            if sig == "SHORT" and not regime_trade_short: return "FLAT"
            return sig
        pos_sig, sw_sig, in_sig = _regime_ok(pos_sig), _regime_ok(sw_sig), _regime_ok(in_sig)

        signals  = [pos_sig, sw_sig, in_sig]
        non_flat = [s for s in signals if s != "FLAT"]

        if not non_flat:
            return "C", 0.0, 0.70

        # Direction conflict — split into 2-vs-1 (B-) vs balanced 1-vs-1 (F)
        if "LONG" in non_flat and "SHORT" in non_flat:
            long_votes  = non_flat.count("LONG")
            short_votes = non_flat.count("SHORT")
            if max(long_votes, short_votes) >= 2:
                # 2 agree, 1 dissents — majority signal gets mild penalty and proceeds.
                # Examples: sw+pos LONG, intra SHORT → B-
                #           sw+intra SHORT, pos LONG  → B-
                grade = "B-"
                logger.debug(
                    f"Alignment B-: {long_votes}×LONG vs {short_votes}×SHORT "
                    f"— majority direction mildly penalised (−5pp, 0.75×)"
                )
            else:
                # Balanced 1-vs-1: no majority — genuine conflicted signal
                grade = "F"
                logger.debug(
                    f"Alignment F: 1×LONG vs 1×SHORT (third=FLAT) "
                    f"— direct conflict, both penalised (−20pp, 0.50×)"
                )
            return grade, self.ALIGNMENT_BOOST[grade], self.ALIGNMENT_SIZE[grade]

        # No conflict below this line (all non-flat signals agree)
        # All 3 agree
        if len(non_flat) == 3 and len(set(non_flat)) == 1:
            grade = "A+"
        # Positional + swing agree — the two primary timeframes
        elif pos_sig == sw_sig and pos_sig != "FLAT":
            grade = "A"
        # Any 2 agree (e.g. swing + intraday, or positional + intraday)
        elif len(non_flat) >= 2 and len(set(non_flat)) == 1:
            grade = "B"
        # Only 1 non-flat model
        else:
            grade = "C"

        return grade, self.ALIGNMENT_BOOST[grade], self.ALIGNMENT_SIZE[grade]

    def _build_row(self, raw: Dict, features: list) -> pd.DataFrame:
        row = {f: raw.get(f, 0.0) for f in features}
        return pd.DataFrame([row])