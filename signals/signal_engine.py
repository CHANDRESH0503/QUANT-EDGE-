# signals/signal_engine.py
# Master signal orchestrator — runs all 6 gates in sequence
# Assembles final signal with risk parameters
# Called every 15 minutes by orchestrator.py (one engine per bank).

import logging
import sqlite3
import csv
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from signals.gate1_regime      import Gate1Regime
from signals.signal_stabilizer  import SignalStabilizer
from signals.gate2_rule_filter import Gate2RuleFilter
from signals.gate3_universe_rank import Gate3UniverseRank
from signals.gate4_ml_predictor import Gate4MLPredictor
from signals.gate5_sr_validator import Gate5SRValidator
from signals.gate6_confidence   import Gate6Confidence
from signals.timeframe_alignment import TimeframeAlignment

from features.feature_builder  import FeatureBuilder
from processing.regime_detector import RegimeDetector
from risk.position_sizer        import PositionSizer
from risk.exit_engine           import ExitEngine
from config                     import TradingConfig

SIGNAL_LOG = "logs/signal_log.csv"


class SignalEngine:
    """
    Master signal engine — the brain of the trading system.

    Runs the complete 6-gate pipeline every 15 minutes.
    Only signals that pass ALL gates reach the trader.

    20yr trader perspective:
    In my manual trading days, I had a checklist of 12 items
    before entering any trade. The signal engine IS that checklist,
    running automatically, without emotion, every 15 minutes.
    The gates represent 20 years of hard-earned risk discipline.

    Pipeline:
    FeatureBuilder.build_all()
        → Gate 1: Regime check
        → Gate 2: Rule filter (10 conditions)
        → Gate 3: Universe ranking
        → Gate 4: ML models (3 timeframes)
        → Gate 5: S/R entry quality
        → Gate 6: Confidence + capital mode
        → Risk Engine: position sizing
        → Signal output + logging

    Output signal format:
    {
        "signal":        "LONG" | "SHORT" | "FLAT",
        "confidence":    0.71,
        "entry_price":   1842.35,
        "stop_loss":     1787.20,
        "target_price":  1934.50,
        "shares":        5,
        "risk_amount":   100.0,
        "position_value":9211.75,
        "reward_risk":   2.5,
        "alignment":     "A+",
        "regime":        "BULL_TRENDING",
        "gate_results":  {...},
        "reasons":       [...],
        "generated_at":  "2026-05-01 06:32:11",
    }
    """

    def __init__(
        self,
        ticker:  str   = "HDFCBANK.NS",
        capital: float = 100_000.0,
        db_path: str   = "database/trading.db",
    ):
        self.ticker  = ticker
        self.capital = capital
        self.db_path = db_path

        # Gates
        self.gate1  = Gate1Regime()
        self.gate2  = Gate2RuleFilter()
        self.gate3  = Gate3UniverseRank()
        self.gate4  = Gate4MLPredictor(ticker=ticker)
        self.gate5  = Gate5SRValidator()
        self.gate6  = Gate6Confidence()
        self.aligner= TimeframeAlignment()
        # Anti-flicker commit layer (STAB-2) — per-bank, carries across cycles.
        self.stabilizer = SignalStabilizer()

        # Processors
        self.feature_builder = FeatureBuilder(ticker, capital, db_path)
        self.regime_detector  = RegimeDetector(db_path, ticker=ticker)
        self.position_sizer   = PositionSizer()
        self.exit_engine      = ExitEngine(db_path)

        # State
        self._last_signal   = "FLAT"
        self._last_conf     = 0.0
        self._gate4_cache   = None  # cache previous gate4 predictions

        self._setup_log()
        logger.info(
            f"SignalEngine ready — {ticker} | "
            f"capital=₹{capital:,.0f} | db={db_path}"
        )

    # ─────────────────────────────────────────────────────────────
    # PRIMARY PUBLIC METHOD
    # ─────────────────────────────────────────────────────────────

    def run(
        self,
        force_rebuild: bool = False,
    ) -> Dict:
        """
        Run the complete signal pipeline.
        Returns the final signal dict.
        Called every 15 minutes by task_runner.py.

        Args:
            force_rebuild: Force feature rebuild even if recently built.
        """
        t0           = datetime.now()
        signal_uuid  = str(uuid.uuid4())   # one UUID per signal cycle
        gate_results = {}
        reasons      = []

        logger.info(f"{'─'*50}")
        logger.info(f"SignalEngine.run() [{signal_uuid[:8]}] at {t0.strftime('%H:%M:%S')}")

        # ── Build features ─────────────────────────────────────────
        feature_vector = self.feature_builder.build_all(
            positional_signal=self._gate4_cache["positional"]["signal"]
                              if self._gate4_cache else "FLAT",
            swing_signal     =self._gate4_cache["swing"]["signal"]
                              if self._gate4_cache else "FLAT",
            intraday_signal  =self._gate4_cache["intraday"]["signal"]
                              if self._gate4_cache else "FLAT",
            positional_conf  =self._gate4_cache["positional"]["confidence"]
                              if self._gate4_cache else 0.5,
            swing_conf       =self._gate4_cache["swing"]["confidence"]
                              if self._gate4_cache else 0.5,
            intraday_conf    =self._gate4_cache["intraday"]["confidence"]
                              if self._gate4_cache else 0.5,
            signal_uuid      =signal_uuid,
        )

        raw           = feature_vector.get("raw_features", {})
        risk_context  = feature_vector.get("risk_context",  {})
        regime_result = feature_vector.get("regime",        {})
        sr_levels     = feature_vector.get("sr_levels",     {})
        current_price = feature_vector.get("current_price", 0.0)
        atr           = feature_vector.get("atr",           20.0)
        capital_mode  = risk_context.get("capital_mode",    "FULL")

        # ── Pre-Gate-1: price freshness & ATR sanity ───────────────
        # Refuse to generate a signal if we don't have a live price or
        # the ATR is non-positive. Both would silently produce bad
        # entry/stop/target levels downstream and could send a paper
        # trade with shares=0 or zero-distance stops.
        if not current_price or current_price <= 0:
            gate_results["pre_check"] = {
                "passed": False, "reason": "Stale/missing price feed",
            }
            return self._flat("Stale or missing price feed — no live entry possible",
                              gate_results, t0, signal_uuid)
        if not atr or atr <= 0:
            gate_results["pre_check"] = {
                "passed": False, "reason": f"Invalid ATR: {atr}",
            }
            return self._flat(f"Invalid ATR ({atr}) — cannot size stops",
                              gate_results, t0, signal_uuid)
        gate_results["pre_check"] = {
            "passed": True, "price": current_price, "atr": atr,
        }

        # ── Gate 1: Regime (global — direction-agnostic) ──────────
        # Validate only regime+stability here. Per-category direction is
        # re-checked after Gate 4 so each model (LONG or SHORT) gets a fair
        # shot regardless of the regime's "expected" direction.
        g1_pass, g1_ctx = self.gate1.check(
            regime_result,
            self.regime_detector.get_recent_regime_trend(),
            signal_direction=None,
        )
        gate_results["gate1"] = g1_ctx
        if not g1_pass:
            return self._flat(g1_ctx["reason"], gate_results, t0, signal_uuid)

        reasons.append(f"Regime: {g1_ctx['regime']} ✓")

        # ── Gate 2: Rule filter ────────────────────────────────────
        g2_pass, g2_ctx = self.gate2.check(
            raw,
            {
                "regime":            regime_result.get("regime", "BULL_TRENDING"),
                "fundamental_grade": raw.get("fundamental_grade", "GOOD"),
                "anomaly_severity":  risk_context.get("anomaly_severity", "LOW"),
                "anomaly_type":      risk_context.get("anomaly_type", "NONE"),
            },
        )
        gate_results["gate2"] = g2_ctx
        if not g2_pass:
            return self._flat(
                f"Rule filter: {g2_ctx['n_passed']}/{g2_ctx.get('n_total',11)} | {g2_ctx['fail_reasons'][:2]}",
                gate_results, t0, signal_uuid,
            )

        reasons.append(f"Rules: {g2_ctx['n_passed']}/{g2_ctx.get('n_total',11)} ✓")

        # ── Gate 3: Universe ranking ───────────────────────────────
        # Regime is passed so Gate 3 can apply BEAR-aware scoring:
        # in BEAR, rank-1 = weakest bank (best short) not strongest (best long).
        g3_pass, g3_ctx = self.gate3.check(
            raw,
            ticker=self.ticker,
            regime=regime_result.get("regime", "BULL_TRENDING"),
        )
        gate_results["gate3"] = g3_ctx

        # Falling knife: score too far below peers — skip this cycle
        if g3_ctx.get("disqualified"):
            return self._flat(g3_ctx["reason"], gate_results, t0, signal_uuid)

        if g3_ctx.get("should_switch"):
            reasons.append(
                f"⚡ Better stock: {g3_ctx['best_name']} "
                f"(HDFC rank #{g3_ctx['hdfc_rank']})"
            )
        else:
            reasons.append(
                f"Universe rank: #{g3_ctx.get('ticker_rank', g3_ctx.get('hdfc_rank', 1))} "
                f"size={g3_ctx.get('size_mult', 1.0):.2f} ✓"
            )

        # ── Feature quality gate (HARD for POOR, +5pp boost for DEGRADED) ─
        # Policy (locked 2026-05-26):
        #   POOR     → return FLAT immediately (cannot trust the feature vector).
        #   DEGRADED → proceed, but Gate 6 lifts the per-category threshold by
        #              5pp to demand stronger conviction when data is sparse.
        #   GOOD     → no adjustment.
        # The threshold boost is applied below (see `dq_threshold_boost`); a
        # DEGRADED signal must clear (cat_threshold + 0.05) at Gate 6.
        dq            = feature_vector.get("data_quality", {})
        dq_quality    = dq.get("quality", "GOOD")
        zero_n        = dq.get("zero_features", "?")
        total         = dq.get("total_features", "?")

        if dq_quality == "POOR":
            gate_results["data_quality"] = {
                "passed": False, "quality": "POOR",
                "zero_features": zero_n, "total_features": total,
            }
            return self._flat(
                f"Data quality POOR — {zero_n}/{total} zero features. "
                f"Refusing to trade with unreliable feature vector.",
                gate_results, t0, signal_uuid,
            )

        dq_threshold_boost = 0.05 if dq_quality == "DEGRADED" else 0.0
        gate_results["data_quality"] = {
            "passed":         True,
            "quality":        dq_quality,
            "zero_features":  zero_n,
            "total_features": total,
            "threshold_boost": dq_threshold_boost,
        }
        if dq_quality == "DEGRADED":
            logger.warning(
                f"Feature quality DEGRADED — {zero_n}/{total} zero features. "
                f"Gate 6 threshold raised by +5pp."
            )

        # ── Gate 4: ML models ──────────────────────────────────────
        # Pass regime direction so counter-regime model votes (noise) don't drag
        # an aligned signal to Grade F (EDGE-2).
        g4_pass, g4_ctx = self.gate4.check(
            feature_vector,
            regime_trade_long=bool(regime_result.get("trade_long",  True)),
            regime_trade_short=bool(regime_result.get("trade_short", True)),
        )
        gate_results["gate4"] = g4_ctx
        self._gate4_cache = g4_ctx if g4_pass else self._gate4_cache

        if not g4_pass:
            return self._flat(
                g4_ctx.get("reason", "ML models returned no signal"),
                gate_results, t0, signal_uuid,
            )

        alignment = g4_ctx["alignment"]
        self._save_predictions(g4_ctx, alignment)

        # Boosted confidences per category (Gate 4 has already added alignment boost)
        cat_conf = {
            "swing":      float(g4_ctx.get("swing_conf_boosted",      g4_ctx.get("swing", {}).get("confidence",      0))),
            "positional": float(g4_ctx.get("positional_conf_boosted", g4_ctx.get("positional", {}).get("confidence", 0))),
            "intraday":   float(g4_ctx.get("intra_conf_boosted",      g4_ctx.get("intraday", {}).get("confidence",   0))),
        }
        cat_dir = {
            "swing":      g4_ctx.get("swing",      {}).get("signal", "FLAT"),
            "positional": g4_ctx.get("positional", {}).get("signal", "FLAT"),
            "intraday":   g4_ctx.get("intraday",   {}).get("signal", "FLAT"),
        }

        # ── Signal commit layer (anti-flicker, STAB-2) ───────────────
        # Dead-band marginal votes to FLAT and require confirmation to commit a
        # NEW direction (fast to FLAT). Mirrors the regime anti-whipsaw. Raw
        # model telemetry in g4_ctx is left untouched — only the per-category
        # trading DECISION (cat_dir/cat_conf) is stabilised.
        stab_reason: Dict[str, str] = {}
        for _cat in ("swing", "positional", "intraday"):
            _st = self.stabilizer.commit(_cat, cat_dir[_cat], g4_ctx.get(_cat, {}))
            if _st["intervened"]:
                stab_reason[_cat] = _st["reason"]
                logger.info(
                    f"[{self.ticker}] {_cat} signal stabilised: "
                    f"raw={_st['raw']} → {_st['committed']} ({_st['reason']})"
                )
                cat_dir[_cat] = _st["committed"]
                if _st["committed"] == "FLAT":
                    cat_conf[_cat] = round(float(g4_ctx.get(_cat, {}).get("prob_flat", 0.0)), 4)

        reasons.append(
            f"ML: swing={cat_dir['swing']}({cat_conf['swing']:.0%}) "
            f"pos={cat_dir['positional']}({cat_conf['positional']:.0%}) "
            f"intra={cat_dir['intraday']}({cat_conf['intraday']:.0%}) "
            f"| align={alignment} ✓"
        )

        # ── Per-category pipeline (Gate 5 → Gate 6, regime-aware) ──
        # Regime is evaluated per-category BEFORE Gate 5/6 so threshold
        # adjustments take effect — Gate 6 is no longer regime-blind.
        #
        # Regime handling matrix (20-yr trader rules):
        # ┌──────────────────────────┬──────────────┬──────────────────────────────────────────┐
        # │ Regime / condition       │ Direction    │ Action                                   │
        # ├──────────────────────────┼──────────────┼──────────────────────────────────────────┤
        # │ BULL/BEAR                │ Aligned      │ Normal threshold, full size              │
        # │ BULL/BEAR                │ Counter      │ +7pp/+10pp threshold, 0.5× size          │
        # │ HIGH_VOLATILITY          │ Any swing/   │ +5pp threshold, 0.5× size                │
        # │                          │ intraday     │                                          │
        # │ CHOPPY_SIDEWAYS          │ Any          │ BLOCKED at Gate 1                        │
        # ├──────────────────────────┴──────────────┴──────────────────────────────────────────┤
        # │ Positional + counter-regime      → HARD BLOCK (2–4 wk hold vs macro trend)        │
        # │ Positional + HIGH_VOLATILITY     → HARD BLOCK (max_hold=2d vs 14–28d hold)        │
        # │ Any cat + counter-regime + Gr D  → HARD BLOCK (double jeopardy)                   │
        # │ Gate 1 low_stability (25–40%)    → +5pp Gate 6 threshold (regime weakly set)      │
        # │ Gate 1 regime_changes ≥ 2/10d   → +5pp Gate 6 threshold (HMM oscillating)        │
        # └──────────────────────────────────────────────────────────────────────────────────────┘
        regime_trend = self.regime_detector.get_recent_regime_trend()
        per_category   = {}
        signals_emitted = []

        # ── Stability / instability flags from Gate 1 → Gate 6 boost ──
        # Both flags are surfaced by gate1_regime.py and represent two
        # independent symptoms of a noisy regime environment:
        #   low_stability    — only 25–40% of the last 10 bars agree on the
        #                      current regime (weakly established; gate1 already
        #                      reduced position size, but Gate 6 also needs to
        #                      demand higher conviction).
        #   regime_changes   — HMM flipped regime ≥2 times in the last 10 days;
        #                      the model is oscillating, not cleanly transitioning.
        # Each adds +5pp to Gate 6 threshold (caps at +15pp total with others).
        regime_low_stability = g1_ctx.get("low_stability", False)
        regime_changes_count = int(g1_ctx.get("regime_changes", 0))

        # Capital-mode allowed timeframes — SMALL only trades swing, GROWING
        # adds intraday, FULL unlocks positional. A category not in the
        # allow-list short-circuits to a structured FLAT with the reason
        # surfaced so the dashboard shows WHY no signal fired (not just "FLAT").
        from risk.capital_mode import CapitalMode as _CapMode
        _cap_cfg = _CapMode.MODES.get(capital_mode, _CapMode.MODES["FULL"])
        allowed_tf = set(_cap_cfg.get("allowed_tf", ["swing", "intraday", "positional"]))

        for cat in ("swing", "positional", "intraday"):
            direction = cat_dir[cat]
            conf_b    = cat_conf[cat]

            # ── Capital-mode timeframe gate ──────────────────────
            if cat not in allowed_tf:
                per_category[cat] = {
                    "category":   cat,
                    "passed":     False,
                    "direction":  direction,
                    "confidence": round(conf_b, 4),
                    "reason":     (
                        f"{capital_mode} capital mode disallows {cat} trades "
                        f"(allowed: {', '.join(sorted(allowed_tf))})"
                    ),
                    "regime_match": None,
                    "gate5":      None,
                    "gate6":      None,
                }
                continue

            # ── FLAT category — no direction to validate ─────────
            # If the stabiliser held/de-risked this category to FLAT, surface
            # WHY (confirming / de-risk) instead of the generic "model FLAT".
            if direction == "FLAT":
                per_category[cat] = {
                    "category":   cat,
                    "passed":     False,
                    "direction":  "FLAT",
                    "confidence": round(conf_b, 4),
                    "reason":     stab_reason.get(cat, "Model predicts FLAT"),
                    "regime_match": None,
                    "gate5":      None,
                    "gate6":      None,
                }
                continue

            # ── Regime match — computed EARLY (drives threshold boost) ─
            # Must run before Gate 5/6 so the counter-regime boost is baked
            # into Gate 6's threshold rather than only affecting position size.
            # Previously this ran AFTER Gate 6 — Gate 6 was regime-blind.
            trade_long  = bool(regime_result.get("trade_long",  False))
            trade_short = bool(regime_result.get("trade_short", False))
            regime_name = regime_result.get("regime", "")
            regime_match = (
                (direction == "LONG"  and trade_long) or
                (direction == "SHORT" and trade_short)
            )

            # ── Hard block: positional counter-regime ─────────────
            # A positional trade holds 2–4 weeks. Holding LONG for weeks
            # in a confirmed BEAR (or SHORT in BULL) is fighting the macro
            # trend across an entire monthly cycle. No ML confidence justifies
            # this — the regime itself is the dominant signal at that horizon.
            # Swing and intraday counter-regime are allowed with a higher
            # confidence threshold (bounces / intraday mean-reversion exist).
            if cat == "positional" and not regime_match:
                per_category[cat] = {
                    "category":     cat,
                    "passed":       False,
                    "direction":    direction,
                    "confidence":   round(conf_b, 4),
                    "regime_match": regime_match,
                    "gate5":        None,
                    "gate6":        None,
                    "reason":       (
                        f"Positional counter-regime blocked — "
                        f"{direction} in {regime_name} "
                        f"(2–4 week hold against macro trend)"
                    ),
                }
                logger.info(
                    f"[{self.ticker}] POSITIONAL BLOCKED: counter-regime "
                    f"({direction} in {regime_name})"
                )
                continue

            # ── Hard block: positional in HIGH_VOLATILITY ─────────
            # HIGH_VOLATILITY regime has max_hold_days=2 (violent intraday
            # gaps, no persistent trend). A positional trade needs 14–28 days
            # to work — the stop gets hit in the noise before the thesis plays
            # out. This is a structural incompatibility, not a confidence issue;
            # no ML score can compensate for holding a 4-week thesis in a
            # 2-day maximum-hold environment. Swing and intraday are fine here
            # with the +5pp threshold boost already applied below.
            if cat == "positional" and regime_name == "HIGH_VOLATILITY":
                per_category[cat] = {
                    "category":     cat,
                    "passed":       False,
                    "direction":    direction,
                    "confidence":   round(conf_b, 4),
                    "regime_match": regime_match,
                    "gate5":        None,
                    "gate6":        None,
                    "reason":       (
                        "Positional blocked in HIGH_VOLATILITY — "
                        "max_hold_days=2 incompatible with 14–28 day positional horizon"
                    ),
                }
                logger.info(
                    f"[{self.ticker}] POSITIONAL BLOCKED: HIGH_VOLATILITY regime "
                    f"(max_hold_days=2 vs 14–28 day positional hold)"
                )
                continue

            # ── Gate 5: S/R per category direction ───────────────
            # per_category_mode=True drops the alignment requirement in
            # the Grade-D override — a single-model signal at high
            # confidence can pass even when alignment="C".
            g5_pass, g5_ctx = self.gate5.check(
                sr_levels,
                signal_direction=direction,
                ml_confidence=conf_b,
                alignment=alignment,
                per_category_mode=True,
                category=cat,
            )

            # ── Counter-regime + Grade D hard block ──────────────
            # Both penalties compound — S/R geometry unfavourable (Grade D)
            # AND macro regime adverse. No confidence compensates for both.
            # (Gate 6 runs next; skip it entirely for this combination.)
            if g5_pass and not regime_match and g5_ctx.get("entry_quality") == "D":
                per_category[cat] = {
                    "category":     cat,
                    "passed":       False,
                    "direction":    direction,
                    "confidence":   round(conf_b, 4),
                    "regime_match": regime_match,
                    "gate5":        g5_ctx,
                    "gate6":        None,
                    "reason":       (
                        f"Counter-regime {direction} in {regime_name} "
                        f"+ Grade D entry — double jeopardy block "
                        f"(regime_match=False + entry_quality=D)"
                    ),
                }
                logger.info(
                    f"[{self.ticker}] {cat.upper()} BLOCKED: counter-regime + Grade D "
                    f"({direction} in {regime_name}, conf={conf_b:.0%})"
                )
                continue

            # ── Gate 6: confidence with regime-aware threshold ────
            # Threshold lifters stack additively (capped at +0.15 total):
            #   • DEGRADED feature vector   → +5pp (every category)
            #   • Fundamentals DEFAULTED    → +5pp (positional — long-hold risk)
            #   • Macro data stale (>30h)   → +5pp (positional — macro matters most)
            #   • Counter-regime intraday   → +10pp (fighting trend + intraday momentum)
            #   • Counter-regime swing      → +7pp  (bounces possible but need conviction)
            #   • HIGH_VOLATILITY (aligned) → +5pp  (noise amplified; need stronger signal)
            #   • Gate 1 low_stability      → +5pp  (25–40% stability: regime weakly set)
            #   • Gate 1 regime_changes ≥2  → +5pp  (HMM oscillating in last 10 days)
            # Positional counter-regime is BLOCKED above — never reaches here.
            # Positional HIGH_VOLATILITY is BLOCKED above — never reaches here.
            cat_boost = dq_threshold_boost
            if cat == "positional" and g2_ctx.get("fundamentals_stale"):
                cat_boost = min(0.15, cat_boost + 0.05)
            if cat == "positional" and int(raw.get("macro_stale", 0)) == 1:
                cat_boost = min(0.15, cat_boost + 0.05)

            if not regime_match:
                # Counter-regime: demand extra conviction relative to the timeframe.
                # Intraday faces the steepest penalty because it fights both the
                # multi-day trend AND intraday momentum in a single session.
                counter_boost = 0.10 if cat == "intraday" else 0.07
                cat_boost = min(0.15, cat_boost + counter_boost)
                logger.debug(
                    f"[{self.ticker}] {cat.upper()} counter-regime boost "
                    f"+{counter_boost:.0%} → threshold boost = {cat_boost:.0%}"
                )
            elif regime_name == "HIGH_VOLATILITY":
                # Even aligned trades in HIGH_VOL need more conviction — signal
                # quality degrades when volatility is elevated (more false positives).
                cat_boost = min(0.15, cat_boost + 0.05)
                logger.debug(
                    f"[{self.ticker}] {cat.upper()} HIGH_VOL boost "
                    f"+5pp → threshold boost = {cat_boost:.0%}"
                )

            # Gate 1 instability signals → additional Gate 6 threshold lifts.
            # These stack on top of the regime-direction boosts above. Both are
            # independent symptoms of a degraded regime signal — even if the
            # direction is aligned and regime is tradeable, we demand more
            # conviction when the regime itself is weakly established or
            # oscillating. Each adds +5pp (total cap remains +15pp).
            if regime_low_stability:
                cat_boost = min(0.15, cat_boost + 0.05)
                logger.debug(
                    f"[{self.ticker}] {cat.upper()} low-stability boost "
                    f"+5pp → threshold boost = {cat_boost:.0%}"
                )
            if regime_changes_count >= 2:
                cat_boost = min(0.15, cat_boost + 0.05)
                logger.debug(
                    f"[{self.ticker}] {cat.upper()} regime-instability boost "
                    f"+5pp ({regime_changes_count} flips/10d) → threshold boost = {cat_boost:.0%}"
                )

            g6_pass, g6_ctx = self.gate6.check(
                primary_conf=conf_b,
                alignment=alignment,
                capital_mode=capital_mode,
                risk_context=risk_context,
                model_type=cat,
                skip_alignment=True,
                data_quality_boost=cat_boost,
                # EDGE-1: feed Gate 5 geometry so a clean A/B setup at >=2:1 R:R
                # can pass on positive expectancy even below the P(win) threshold.
                entry_quality=g5_ctx.get("entry_quality", "C"),
                reward_risk=float(g5_ctx.get("reward_risk", 0.0)),
            )

            passed_all = bool(g5_pass and g6_pass)
            per_category[cat] = {
                "category":     cat,
                "passed":       passed_all,
                "direction":    direction,
                "confidence":   round(conf_b, 4),
                "regime_match": regime_match,
                "gate5":        g5_ctx,
                "gate6":        g6_ctx,
                "reason":       None if passed_all else (
                    g5_ctx.get("reason") if not g5_pass else g6_ctx.get("reason")
                ),
            }

            if not passed_all:
                continue

            # ── Position sizing for this category ──────────────────
            # Size multiplier chain — each gate contributes a fraction:
            #   Gate 1: regime base  (BULL=1.2×, BEAR=1.0×, HIGH_VOL=0.5×)
            #   Gate 3: universe rank (rank-1 bank gets highest mult)
            #   Gate 4: ML alignment boost
            #   Gate 5: entry quality (A=1.0, B=0.85, C=0.65, D=0.40)
            #   Gate 6: circuit breaker / DD mult
            #   Regime: aligned=1.0×, counter-regime=0.5× (positional blocked above)
            #
            # Counter-regime swing/intraday: Gate 6 threshold already raised
            # (+7pp/+10pp) above. The 0.5× size on top makes this a small probe —
            # you need 67-75% confidence AND accept half-size to bet against trend.
            size_mult_cat = (
                g1_ctx.get("position_mult",  1.0) *
                g3_ctx.get("size_mult",       1.0) *
                g4_ctx.get("size_mult",       1.0) *
                g5_ctx.get("size_mult",       1.0) *
                g6_ctx.get("final_size_mult", 1.0) *
                (1.0 if regime_match else 0.50)
            )
            size_mult_cat = round(min(1.5, max(0.0, size_mult_cat)), 3)

            position = self.position_sizer.calculate(
                signal=direction,
                capital=self.capital,
                current_price=current_price,
                atr=atr,
                size_multiplier=size_mult_cat,
                capital_mode=capital_mode,
                category=cat,
            )

            # Sanity guard: refuse to emit a signal whose sizing collapsed
            # to zero shares or zero stop/target. This catches any future
            # multiplier-chain regression (e.g., DD halt × 0, F-align × 0)
            # that would otherwise produce an alert with no actionable levels.
            if (
                position.get("shares", 0) <= 0
                or position.get("stop_loss", 0) <= 0
                or position.get("target", 0) <= 0
            ):
                per_category[cat]["passed"] = False
                per_category[cat]["reason"] = (
                    f"Sizing collapsed to zero "
                    f"(size_mult={size_mult_cat:.3f} shares={position.get('shares',0)})"
                )
                continue

            signals_emitted.append({
                "category":       cat,
                "signal":         direction,
                "direction":      direction,
                "confidence":     round(conf_b, 4),
                "alignment":      alignment,
                "entry_price":    position["entry"],
                "stop_loss":      position["stop_loss"],
                "target_price":   position["target"],
                "shares":         position["shares"],
                "risk_amount":    position["risk_amount"],
                "position_value": position["position_value"],
                "reward_risk":    position["reward_risk"],
                "size_mult":      size_mult_cat,
                "regime_match":   regime_match,
                "entry_quality":  g5_ctx.get("entry_quality", "B"),
                "trade_type":     cat,
                "regime":         regime_result.get("regime", "UNKNOWN"),
                "capital_mode":   capital_mode,
                "ticker":         self.ticker,
                "current_price":  current_price,
                "atr":            atr,
                # Regime-aware holding horizon (P1-1) — persisted at open so the
                # exit engine enforces the regime's max_hold_days (BULL 21 / BEAR
                # 4 / HIGH_VOL 2) instead of a category-blind hardcoded default.
                "max_hold_days":  int(g1_ctx.get("max_hold_days", 0) or 0),
            })
            reasons.append(
                f"{cat.upper()} {direction} {conf_b:.0%} | "
                f"S/R={g5_ctx.get('entry_quality','?')} R:R={g5_ctx.get('reward_risk',0):.1f} | "
                f"thr={g6_ctx.get('threshold',0):.0%}"
                f"{' (counter-regime · 0.5×)' if not regime_match else ''} ✓"
            )

        # Stash per-category gate breakdown into gate_results for dashboard.
        # The "gate5" / "gate6" top-level keys remain populated from the best
        # passing category so legacy readers (CSV log, AlertFormatter context)
        # continue to see a single set of values.
        gate_results["per_category"]    = per_category
        gate_results["categories_passed"] = [c for c, v in per_category.items() if v.get("passed")]

        if not signals_emitted:
            # No category cleared its pipeline — return a structured FLAT that
            # still surfaces per-category reasons for the dashboard.
            blocked_reasons = [
                f"{c}: {v.get('reason','—')}"
                for c, v in per_category.items()
                if v.get("direction") not in (None, "FLAT")
            ]
            reason_txt = (
                "All categories blocked: " + " | ".join(blocked_reasons)
                if blocked_reasons
                else "All models predict FLAT"
            )
            return self._flat(reason_txt, gate_results, t0, signal_uuid)

        # ── Pick best signal as primary (backward-compat top-level) ─
        best = max(signals_emitted, key=lambda s: s["confidence"])
        gate_results["gate5"] = per_category[best["category"]]["gate5"]
        gate_results["gate6"] = per_category[best["category"]]["gate6"]

        elapsed = (datetime.now() - t0).total_seconds()
        signal_out = {
            # Core signal (highest-confidence passing category)
            "signal_uuid":    signal_uuid,
            "signal":         best["signal"],
            "confidence":     best["confidence"],
            "alignment":      alignment,
            "category":       best["category"],

            # Trade parameters (from best category)
            "entry_price":    best["entry_price"],
            "stop_loss":      best["stop_loss"],
            "target_price":   best["target_price"],
            "shares":         best["shares"],
            "risk_amount":    best["risk_amount"],
            "position_value": best["position_value"],
            "reward_risk":    best["reward_risk"],
            "size_mult":      best["size_mult"],
            "trade_type":     best["category"],

            # Context
            "regime":         regime_result.get("regime", "UNKNOWN"),
            "capital_mode":   capital_mode,
            "current_price":  current_price,
            "atr":            atr,

            # Per-category breakdown (NEW — full list of passing signals)
            "signals":        signals_emitted,
            "categories_passed": [s["category"] for s in signals_emitted],

            # Model predictions
            "positional":     g4_ctx.get("positional", {}),
            "swing":          g4_ctx.get("swing",      {}),
            "intraday":       g4_ctx.get("intraday",   {}),

            # S/R levels
            "nearest_support":    sr_levels.get("nearest_support", 0),
            "nearest_resistance": sr_levels.get("nearest_resistance", 0),
            "entry_quality":      best["entry_quality"],

            # Metadata
            "ticker":         self.ticker,
            "reasons":        reasons,
            "gate_results":   gate_results,
            "status":         "SIGNAL",
            "generated_at":   str(datetime.now()),
            "elapsed_sec":    round(elapsed, 2),
        }

        self._last_signal = best["signal"]
        self._last_conf   = best["confidence"]
        self._log_signal(signal_out)
        self._save_gate_results(gate_results, regime_result.get("regime", ""))

        logger.info(
            f"SIGNALS: {len(signals_emitted)} category(s) passed — "
            + ", ".join(
                f"{s['category'].upper()}:{s['signal']}@{s['confidence']:.0%}"
                for s in signals_emitted
            )
            + f" | entry=₹{current_price:.2f} | atr=₹{atr:.2f}"
        )
        return signal_out

    def run_universe(self) -> Dict:
        """
        Multi-asset scan — evaluate top-N ranked stocks and return the best signal.

        Flow:
        1. Gate 3 ranks the 5-bank universe; take top-N tickers.
        2. For each ticker, run the full 6-gate pipeline via a per-ticker engine.
        3. Among tickers that produce a non-FLAT signal, pick the highest confidence.
        4. Apply exposure check (per-name 40%, total 80%) before accepting.
        5. Return best signal, or FLAT if none qualify.

        At most one open trade per cycle (demo phase constraint).
        """
        # Pre-screen tickers using neutral regime — each per-ticker engine below
        # runs the full pipeline with its own regime-aware Gate 3 check, so the
        # actual size_mult and ranking for each bank's signal is correctly computed.
        top_tickers = self.gate3.get_top_tickers(n=TradingConfig.UNIVERSE_TOP_N)
        logger.info(f"Universe scan: evaluating {top_tickers}")

        candidates = []

        for ticker in top_tickers:
            try:
                engine = SignalEngine(
                    ticker=ticker,
                    capital=self.capital,
                    db_path=self.db_path,
                )
                result = engine.run()
                if result.get("signal") not in ("FLAT", None):
                    result["ticker"] = ticker
                    candidates.append(result)
                    logger.info(
                        f"  {ticker}: {result['signal']} "
                        f"conf={result.get('confidence', 0):.0%}"
                    )
            except Exception as e:
                logger.warning(f"Universe scan failed for {ticker}: {e}")

        if not candidates:
            t0 = datetime.now()
            return {
                "signal_uuid":  str(uuid.uuid4()),
                "signal":       "FLAT",
                "ticker":       top_tickers[0] if top_tickers else self.ticker,
                "confidence":   0.0,
                "reason":       "No qualifying signal from any universe ticker",
                "status":       "FLAT",
                "generated_at": str(datetime.now()),
            }

        # Pick highest-confidence signal
        best = max(candidates, key=lambda x: float(x.get("confidence", 0)))

        # Exposure check
        exposure = self.position_sizer.check_exposure(
            ticker=best["ticker"],
            proposed_value=float(best.get("position_value", 0)),
            capital=self.capital,
            db_path=self.db_path,
        )
        if not exposure["allowed"]:
            logger.warning(
                f"Universe best signal {best['ticker']} blocked: {exposure['reason']}"
            )
            t0 = datetime.now()
            return {
                "signal_uuid":  str(uuid.uuid4()),
                "signal":       "FLAT",
                "ticker":       best["ticker"],
                "confidence":   0.0,
                "reason":       f"Exposure limit: {exposure['reason']}",
                "status":       "FLAT",
                "generated_at": str(datetime.now()),
            }

        logger.info(
            f"Universe best: {best['ticker']} {best['signal']} "
            f"conf={best.get('confidence', 0):.0%} | "
            f"portfolio_exposure=₹{exposure['current_exposure']:,.0f}"
        )
        return best

    def check_exits(self) -> List[Dict]:
        """
        Check open positions for THIS ticker for stop/target/trail breaches.
        Returns a list of exit recommendation dicts; does NOT close in the DB —
        the caller (orchestrator._run_exit_checks) must do that.

        Ticker filter is CRITICAL: without it, HDFCBANK's ₹785 price would be
        applied to ICICIBANK positions at ₹1400, causing wrong stop/target hits.
        """
        price = self.feature_builder.price_fetcher.get_current_price()
        if not price or price <= 0:
            logger.warning(f"[{self.ticker}] check_exits: invalid price {price}, skipping")
            return []
        # Today's session open — for gap-fill exits on overnight positions (P1-3).
        # Cached daily fetch (PERF-1) so this adds no network cost.
        open_price = 0.0
        try:
            _df = self.feature_builder.price_fetcher.get_latest_daily(days=5)
            if _df is not None and not _df.empty and "Open" in _df.columns:
                open_price = float(_df["Open"].iloc[-1])
        except Exception:
            open_price = 0.0
        return self.exit_engine.check_all_positions(
            current_price=price,
            open_price=open_price,
            ticker=self.ticker,
        )

    def check_model_reversals(self, signal: Dict) -> List[Dict]:
        """
        Advisory check: for each open position, has the supporting model
        reversed direction or has the regime turned adverse?

        Returns a list of reversal dicts — one per affected open position.
        These are informational only; the human decides whether to close.

        Reversal types:
          MODEL_REVERSAL — model for that category now says the opposite direction
          REGIME_CHANGE  — market regime no longer supports the open trade's direction
        """
        open_positions = self.exit_engine._get_open_positions()
        if not open_positions:
            return []

        ticker      = signal.get("ticker", self.ticker)
        regime      = signal.get("regime", "")
        per_cat     = signal.get("gate_results", {}).get("per_category", {})
        alerts: List[Dict] = []

        for pos in open_positions:
            if pos.get("ticker", ticker) != ticker:
                continue

            cat       = pos.get("trade_type", "swing")
            direction = pos.get("signal", "LONG")
            cat_info  = per_cat.get(cat, {})
            model_dir = cat_info.get("direction", "FLAT")
            entry     = float(pos.get("entry_price", 0))
            stop      = float(pos.get("stop_price", 0))

            if (direction == "LONG" and model_dir == "SHORT") or \
               (direction == "SHORT" and model_dir == "LONG"):
                alerts.append({
                    "type":         "MODEL_REVERSAL",
                    "category":     cat,
                    "open_signal":  direction,
                    "model_signal": model_dir,
                    "regime":       regime,
                    "ticker":       ticker,
                    "entry_price":  entry,
                    "stop_price":   stop,
                    "position_id":  pos.get("id"),
                })
                continue

            # Regime adverse: open LONG in BEAR, or open SHORT in BULL
            regime_adverse = (
                (direction == "LONG"  and "BEAR" in regime) or
                (direction == "SHORT" and "BULL" in regime)
            )
            # ── Only flag a GENUINE regime flip, never a deliberate probe ──
            # A counter-regime swing/intraday entry is allowed by design (0.5×
            # size + raised Gate 6 threshold). It is adverse FROM BIRTH, so the
            # raw `regime_adverse` test fires the instant we re-observe it and
            # `REGIME_FLIP_EXIT` churns it out at a ~0.2% cost loop (entry==exit).
            # Real P1-2 case: a position opened ALIGNED whose regime has since
            # FLIPPED against it. Gate on both: opened aligned (regime_match=1)
            # AND the regime name actually changed from entry. Counter-regime
            # probes (regime_match=0) are left to their own stop/target/time-exit.
            opened_aligned  = bool(int(pos.get("regime_match", 0) or 0))
            regime_at_entry = str(pos.get("regime_at_entry") or "")
            regime_flipped  = bool(regime_at_entry) and (regime_at_entry != regime)
            if regime_adverse and opened_aligned and regime_flipped:
                alerts.append({
                    "type":            "REGIME_CHANGE",
                    "category":        cat,
                    "open_signal":     direction,
                    "model_signal":    model_dir,
                    "regime":          regime,
                    "regime_at_entry": regime_at_entry,
                    "ticker":          ticker,
                    "entry_price":     entry,
                    "stop_price":      stop,
                    "position_id":     pos.get("id"),
                })

        return alerts

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _flat(
        self,
        reason:        str,
        gate_results:  Dict,
        t0:            datetime,
        signal_uuid:   str = "",
    ) -> Dict:
        """Return standardised FLAT signal with reason."""
        elapsed = (datetime.now() - t0).total_seconds()
        # Always include regime so dashboard can display accurate Gate 1 status
        # even for FLAT signals (gate_results["gate1"]["regime"] if available).
        regime_from_gate = (gate_results.get("gate1") or {}).get("regime", "")
        result  = {
            "signal_uuid":  signal_uuid or str(uuid.uuid4()),
            "signal":       "FLAT",
            "ticker":       self.ticker,
            "confidence":   0.0,
            "reason":       reason,
            "regime":       regime_from_gate,
            "gate_results": gate_results,
            "status":       "FLAT",
            "generated_at": str(datetime.now()),
            "elapsed_sec":  round(elapsed, 2),
        }
        self._log_signal(result)
        self._save_gate_results(gate_results, regime_from_gate)
        logger.info(f"FLAT — {reason}")
        return result

    def _save_predictions(self, g4_ctx: Dict, alignment: str) -> None:
        """Write per-model predictions to logs/last_predictions.json for dashboard."""
        try:
            os.makedirs("logs", exist_ok=True)
            payload = {
                "positional":  g4_ctx.get("positional",  {}),
                "swing":       g4_ctx.get("swing",        {}),
                "intraday":    g4_ctx.get("intraday",     {}),
                "alignment":   alignment,
                "written_at":  str(datetime.now()),
            }
            ticker_safe = getattr(self, "ticker", "HDFCBANK").replace(".NS", "")
            with open(f"logs/last_predictions_{ticker_safe}.json", "w") as f:
                json.dump(payload, f, default=str)
        except Exception as e:
            logger.debug(f"last_predictions write failed: {e}")

    def _save_gate_results(self, gate_results: Dict, regime: str) -> None:
        """UPSERT latest gate_results into the DB so the dashboard reads one
        source of truth. Also writes the legacy JSON file as a transitional
        fallback — to be removed once the DB read is verified on the VPS.
        """
        written_at = str(datetime.now())
        try:
            payload_json = json.dumps(gate_results, default=str)
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO gate_results (ticker, gate_results, regime, written_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    gate_results = excluded.gate_results,
                    regime       = excluded.regime,
                    written_at   = excluded.written_at
            """, (self.ticker, payload_json, regime, written_at))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"gate_results DB upsert failed: {e}")

        # Legacy JSON fallback — dashboard checks DB first, this file second.
        try:
            os.makedirs("logs", exist_ok=True)
            ticker_safe = self.ticker.replace(".NS", "")
            payload = {
                "gate_results": gate_results,
                "regime":       regime,
                "written_at":   written_at,
            }
            with open(f"logs/last_gate_results_{ticker_safe}.json", "w") as f:
                json.dump(payload, f, default=str)
        except Exception as e:
            logger.debug(f"last_gate_results JSON write failed: {e}")

    def _log_signal(self, signal: Dict) -> None:
        """
        Append signal(s) to CSV log. Every signal logged — even FLAT.
        For multi-category signal cycles, writes ONE ROW PER CATEGORY so
        the log preserves the full per-category breakdown for audit and
        backtest. FLAT cycles still write a single summary row.
        """
        try:
            os.makedirs(os.path.dirname(SIGNAL_LOG), exist_ok=True)
            write_header = not os.path.exists(SIGNAL_LOG)
            with open(SIGNAL_LOG, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "timestamp", "signal_uuid", "ticker", "category",
                        "signal", "confidence", "alignment", "regime",
                        "regime_match", "capital_mode", "entry", "stop", "target",
                        "shares", "risk_amount", "reward_risk", "size_mult",
                        "entry_quality", "status", "reason",
                    ])

                rows = signal.get("signals") or []
                if rows:
                    # One CSV row per passing category.
                    for sig in rows:
                        writer.writerow([
                            signal.get("generated_at", ""),
                            signal.get("signal_uuid", ""),
                            sig.get("ticker") or signal.get("ticker", ""),
                            sig.get("category", ""),
                            sig.get("signal", "FLAT"),
                            sig.get("confidence", 0),
                            sig.get("alignment") or signal.get("alignment", ""),
                            sig.get("regime") or signal.get("regime", ""),
                            sig.get("regime_match"),
                            sig.get("capital_mode") or signal.get("capital_mode", ""),
                            sig.get("entry_price", 0),
                            sig.get("stop_loss", 0),
                            sig.get("target_price", 0),
                            sig.get("shares", 0),
                            sig.get("risk_amount", 0),
                            sig.get("reward_risk", 0),
                            sig.get("size_mult"),
                            sig.get("entry_quality", ""),
                            signal.get("status", "SIGNAL"),
                            str(signal.get("reason", ""))[:200],
                        ])
                else:
                    # FLAT or single-row cycle — preserve old behaviour.
                    writer.writerow([
                        signal.get("generated_at", ""),
                        signal.get("signal_uuid", ""),
                        signal.get("ticker", ""),
                        signal.get("category") or signal.get("trade_type", ""),
                        signal.get("signal", "FLAT"),
                        signal.get("confidence", 0),
                        signal.get("alignment", ""),
                        signal.get("regime", ""),
                        None,
                        signal.get("capital_mode", ""),
                        signal.get("entry_price", 0),
                        signal.get("stop_loss", 0),
                        signal.get("target_price", 0),
                        signal.get("shares", 0),
                        signal.get("risk_amount", 0),
                        signal.get("reward_risk", 0),
                        signal.get("size_mult"),
                        signal.get("entry_quality", ""),
                        signal.get("status", ""),
                        str(signal.get("reason", ""))[:200],
                    ])
        except Exception as e:
            logger.warning(f"Signal log write failed: {e}")

    def _setup_log(self) -> None:
        """Ensure log directory exists."""
        os.makedirs(os.path.dirname(SIGNAL_LOG), exist_ok=True)

    def has_signal_changed(self, new_signal: Dict) -> bool:
        """True if signal direction or confidence changed significantly."""
        new_dir  = new_signal.get("signal", "FLAT")
        new_conf = float(new_signal.get("confidence", 0))
        return (
            new_dir != self._last_signal or
            abs(new_conf - self._last_conf) >= 0.10
        )