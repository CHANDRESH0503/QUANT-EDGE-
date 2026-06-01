# backtest/pipeline_backtest.py
# P0-3 — Offline replay of the LIVE 6-gate pipeline over historical bars.
#
# Why this exists
# ───────────────
# backtest/engine.py runs only the bare ML model + a flat 0.55 threshold — NOT
# the gates that actually decide a live trade. CV accuracy is not strategy edge.
# Before scaling real capital we must validate the *strategy* (gates + costs)
# out-of-sample, per category and per regime.
#
# What is replayed FAITHFULLY (same code as live — no divergence)
#   Gate 1  Regime          — classify_regime_row() (ADX+EMA, shared with live)
#                             + REGIME_RULES (CHOPPY block, counter-regime, HIGH_VOL).
#   Gate 4  ML model         — per-fold trained technical model (no lookahead).
#   Gate 5  S/R validator    — Gate5SRValidator() on SupportResistanceEngine levels.
#   Gate 6  Confidence       — Gate6Confidence() with the SAME category thresholds
#                             and regime-aware boosts (counter-regime +7/+10pp,
#                             HIGH_VOL +5pp) used in signals/signal_engine.py.
#   Sizing/holds            — regime position_mult × Gate-5 grade × counter-regime
#                             0.5×; regime-aware max_hold_days (BULL 21/BEAR 4/HV 2).
#   Costs                   — 0.1% brokerage+STT, 0.05%/side slippage (≈0.2% RT).
#
# What is NEUTRALISED offline (documented limitation — cannot be reconstructed
# from OHLCV history)
#   Gate 2  Rule filter      — needs point-in-time VIX/macro/earnings/fundamentals
#                             snapshots that are not stored historically. Treated as
#                             PASS (neutral). Live Gate 2 only ever BLOCKS, so this
#                             makes the offline result a slight UPPER bound on trade
#                             count — never lets through a trade live would reject on
#                             price grounds.
#   Gate 3  Universe rank    — needs all 5 banks each bar; single-ticker offline.
#                             Size multiplier defaults to 1.0×.
#   VIX boost (Gate 6)       — no historical VIX → 0pp VIX adjustment offline.
#
# Connected to: backtest/metrics.py, backtest/walk_forward.py, the real gate
# classes in signals/, processing/support_resistance.py, features/feature_builder.

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from backtest.metrics             import BacktestMetrics
from processing.technical         import TechnicalProcessor
from features.technical_features  import TechnicalFeatures
from features.feature_builder     import classify_regime_row
from processing.support_resistance import SupportResistanceEngine
from processing.regime_detector   import RegimeDetector
from signals.gate5_sr_validator   import Gate5SRValidator
from signals.gate6_confidence     import Gate6Confidence

# One-hot regime → canonical regime name (must match REGIME_RULES keys).
_ONEHOT_TO_NAME = {
    "regime_bull":     "BULL_TRENDING",
    "regime_bear":     "BEAR_TRENDING",
    "regime_high_vol": "HIGH_VOLATILITY",
    "regime_choppy":   "CHOPPY_SIDEWAYS",
}


class PipelineBacktest:
    """
    Event-driven backtest that runs the live 6-gate decision logic over history.

    One category at a time (swing / intraday / positional) so it slots directly
    into WalkForwardValidator's per-model_type structure. Run all three to get
    the per-category breakdown; trades are tagged with regime_at_entry so the
    metrics layer can also slice by regime.
    """

    TRANSACTION_COST = 0.001
    SLIPPAGE         = 0.0005
    TOTAL_COST       = TRANSACTION_COST + 2 * SLIPPAGE

    # Category fallback hold caps (used when regime supplies no max_hold_days).
    MAX_HOLD = {"swing": 7, "intraday": 1, "positional": 21}

    # Offline backtest always validates against FULL-mode thresholds — paper
    # trading runs FULL (FORCE_CAPITAL_MODE=FULL) and the go-live bar is set on
    # the full timeframe set. Gate 6 owns the actual threshold values.
    CAPITAL_MODE = "FULL"

    # Risk per trade for offline sizing (FULL mode = 2%). Matches CapitalMode FULL.
    RISK_PCT     = 0.020
    STOP_ATR_MULT   = 2.0
    TARGET_ATR_MULT = 5.0

    def __init__(self, starting_capital: float = 100_000.0):
        self.starting_capital = starting_capital
        self.tech_processor   = TechnicalProcessor()
        self.tech_features    = TechnicalFeatures()
        self.sr_engine        = SupportResistanceEngine()
        self.gate5            = Gate5SRValidator()
        self.gate6            = Gate6Confidence()
        self.metrics_calc     = BacktestMetrics()

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def run(
        self,
        df:            pd.DataFrame,
        model,
        feature_names: List[str],
        category:      str   = "swing",
        forward_days:  int   = 5,
        verbose:       bool  = False,
    ) -> Dict:
        """
        Replay the gates over `df` for a single category.

        Args:
            df:            OHLCV DataFrame (daily). Needs 100+ bars.
            model:         Trained model exposing predict_proba (per-fold model).
            feature_names: Feature columns the model expects.
            category:      "swing" | "intraday" | "positional".
            forward_days:  Label/scan horizon — last `forward_days` bars are not
                           opened (need room for the trade to resolve).
        Returns:
            Dict with trades (each tagged regime_at_entry/category), aggregate
            metrics, and per-regime metrics.
        """
        if len(df) < 100:
            logger.error("PipelineBacktest needs 100+ bars")
            return {}

        capital     = self.starting_capital
        trades:     List[Dict] = []
        equity_log: List[Dict] = []
        in_trade    = False
        entry:      Dict = {}
        min_history = 60

        for i in range(min_history, len(df) - forward_days):
            bar    = df.iloc[i]
            price  = float(bar["Close"])
            date   = str(bar.name)[:10]

            # ── Manage open trade ──────────────────────────────────
            if in_trade:
                closed, capital = self._manage_open(
                    entry, bar, i, price, date, capital, trades
                )
                if closed:
                    in_trade = False
                equity_log.append({"date": date, "equity": round(capital, 2)})
                continue

            # ── Build technical features for this bar ──────────────
            df_slice = df.iloc[: i + 1]
            tech_df  = self.tech_processor.build_features(df_slice.copy())
            feat_row = self.tech_features.extract(tech_df)
            feat_df  = pd.DataFrame([{
                f: feat_row.get(f, 0.0) for f in feature_names
            }]).fillna(0)

            # ── Gate 4: ML prediction ──────────────────────────────
            try:
                probs = model.predict_proba(feat_df)[0]
                pred  = int(np.argmax(probs))
                conf  = float(probs[pred])
            except Exception as e:
                logger.debug(f"predict error bar {i}: {e}")
                equity_log.append({"date": date, "equity": round(capital, 2)})
                continue
            direction = {0: "SHORT", 1: "FLAT", 2: "LONG"}[pred]
            if direction == "FLAT":
                equity_log.append({"date": date, "equity": round(capital, 2)})
                continue

            # ── Gate 1: regime (shared classifier + REGIME_RULES) ──
            regime_name, rules = self._regime(feat_row)
            if regime_name == "CHOPPY_SIDEWAYS":
                equity_log.append({"date": date, "equity": round(capital, 2)})
                continue   # Gate 1 hard-blocks CHOPPY for all directions.

            regime_match = (
                (direction == "LONG"  and rules["trade_long"]) or
                (direction == "SHORT" and rules["trade_short"])
            )

            # Positional hard blocks (signal_engine rules 24 / G1-5).
            if category == "positional" and not regime_match:
                equity_log.append({"date": date, "equity": round(capital, 2)})
                continue
            if category == "positional" and regime_name == "HIGH_VOLATILITY":
                equity_log.append({"date": date, "equity": round(capital, 2)})
                continue

            # ── Gate 5: S/R entry quality (real gate code) ─────────
            sr_levels      = self.sr_engine.get_sr_features(df_slice)
            g5_pass, g5_ctx = self.gate5.check(
                sr_levels,
                signal_direction=direction,
                ml_confidence=conf,
                alignment="C",            # alignment informational in per-category mode
                per_category_mode=True,
                category=category,
            )
            if not g5_pass:
                equity_log.append({"date": date, "equity": round(capital, 2)})
                continue
            # Counter-regime + Grade D double jeopardy (rule 21).
            if not regime_match and g5_ctx.get("entry_quality") == "D":
                equity_log.append({"date": date, "equity": round(capital, 2)})
                continue

            # ── Gate 6: confidence with regime-aware boost (real code) ─
            cat_boost = 0.0
            if not regime_match:
                cat_boost = 0.10 if category == "intraday" else 0.07
            elif regime_name == "HIGH_VOLATILITY":
                cat_boost = 0.05
            g6_pass, g6_ctx = self.gate6.check(
                primary_conf=conf,
                alignment="C",
                capital_mode=self.CAPITAL_MODE,
                risk_context={
                    "trading_allowed":       True,
                    "final_size_mult":       1.0,
                    "india_vix":             0.0,       # no historical VIX offline
                    "circuit_breaker_level": "OK",
                },
                model_type=category,
                skip_alignment=True,
                data_quality_boost=cat_boost,
            )
            if not g6_pass:
                equity_log.append({"date": date, "equity": round(capital, 2)})
                continue

            # ── Open position ──────────────────────────────────────
            size_mult = (
                rules.get("position_mult", 1.0)
                * float(g5_ctx.get("size_mult", 1.0))
                * (1.0 if regime_match else 0.50)
            )
            entry = self._open(
                df, i, price, date, direction, capital,
                size_mult, regime_name, regime_match,
                g5_ctx.get("entry_quality", "C"), conf, category, rules,
            )
            if entry is None:
                equity_log.append({"date": date, "equity": round(capital, 2)})
                continue
            in_trade = True
            if verbose:
                logger.info(
                    f"  OPEN {category} {direction} @ ₹{entry['entry_price']:.2f} "
                    f"| {regime_name} {'aligned' if regime_match else 'COUNTER'} "
                    f"| conf={conf:.0%} sz×{size_mult:.2f}"
                )
            equity_log.append({"date": date, "equity": round(capital, 2)})

        # Close any residual position at the final usable bar.
        if in_trade:
            capital = self._force_close(entry, df, forward_days, capital, trades)

        agg        = self.metrics_calc.calculate(trades, self.starting_capital)
        per_regime = self.calculate_by_regime(trades)

        return {
            "category":          category,
            "total_bars":        len(df),
            "backtest_period":   f"{str(df.index[0])[:10]} to {str(df.index[-1])[:10]}",
            "starting_capital":  self.starting_capital,
            "ending_capital":    round(capital, 2),
            "trades":            trades,
            "equity_log":        equity_log,
            "metrics":           agg,
            "per_regime_metrics":per_regime,
            "run_at":            str(datetime.now()),
        }

    def calculate_by_regime(self, trades: List[Dict]) -> Dict[str, Dict]:
        """Slice metrics by the regime in force at entry."""
        out: Dict[str, Dict] = {}
        regimes = sorted({t.get("regime_at_entry", "UNKNOWN") for t in trades})
        for r in regimes:
            subset = [t for t in trades if t.get("regime_at_entry") == r]
            if len(subset) >= 5:
                out[r] = self.metrics_calc.calculate(subset, self.starting_capital)
            else:
                out[r] = {"total_trades": len(subset), "note": "too few for metrics"}
        return out

    # ─────────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────────

    def _regime(self, feat_row: Dict):
        """Classify regime from a feature row using the SHARED classifier."""
        onehot = classify_regime_row(
            adx=float(feat_row.get("adx", 0.0)),
            ema_spread=float(feat_row.get("ema_spread", 0.0)),
            returns_5d=float(feat_row.get("returns_5d", 0.0)),
        )
        name = next(
            (_ONEHOT_TO_NAME[k] for k, v in onehot.items() if v),
            "CHOPPY_SIDEWAYS",
        )
        rules = RegimeDetector.REGIME_RULES.get(name, RegimeDetector.REGIME_RULES["CHOPPY_SIDEWAYS"])
        return name, rules

    def _max_hold(self, category: str, rules: Dict) -> int:
        """Regime-aware hold cap (P1-1) with a category fallback."""
        regime_hold = int(rules.get("max_hold_days", 0) or 0)
        cat_hold    = self.MAX_HOLD.get(category, 7)
        return regime_hold if regime_hold > 0 else cat_hold

    def _open(self, df, i, price, date, direction, capital,
              size_mult, regime_name, regime_match, entry_quality,
              conf, category, rules) -> Optional[Dict]:
        atr_series = (df["High"] - df["Low"]).rolling(14).mean()
        atr        = float(atr_series.iloc[i]) if not atr_series.empty else price * 0.015
        if atr <= 0:
            return None

        stop_dist   = atr * self.STOP_ATR_MULT
        target_dist = atr * self.TARGET_ATR_MULT
        max_risk    = capital * self.RISK_PCT * max(0.0, size_mult)
        shares      = max(0, int(max_risk / stop_dist)) if stop_dist > 0 else 0
        if shares <= 0:
            return None
        # 40% per-name cap.
        if shares * price > capital * 0.40:
            shares = max(0, int(capital * 0.40 / price))
        if shares <= 0:
            return None

        entry_price = price * ((1 + self.SLIPPAGE) if direction == "LONG" else (1 - self.SLIPPAGE))
        stop_price  = entry_price - stop_dist if direction == "LONG" else entry_price + stop_dist
        target_price= entry_price + target_dist if direction == "LONG" else entry_price - target_dist

        return {
            "bar_index":      i,
            "entry_date":     date,
            "signal":         direction,
            "entry_price":    round(entry_price, 2),
            "stop_price":     round(stop_price, 2),
            "target_price":   round(target_price, 2),
            "shares":         shares,
            "position_value": round(shares * entry_price, 2),
            "confidence":     conf,
            "regime_at_entry":regime_name,
            "regime_match":   regime_match,
            "entry_quality":  entry_quality,
            "category":       category,
            "max_hold":       self._max_hold(category, rules),
        }

    def _manage_open(self, entry, bar, i, price, date, capital, trades):
        """Returns (closed: bool, capital). Mirrors BacktestEngine exit logic."""
        signal      = entry["signal"]
        entry_price = entry["entry_price"]
        stop        = entry["stop_price"]
        target      = entry["target_price"]
        shares      = entry["shares"]
        days_held   = i - entry["bar_index"]

        # Gap-fill: today's open already past the stop (P1-3 equivalent).
        bar_open = float(bar.get("Open", price))
        gap_fill = (
            (signal == "LONG"  and bar_open < stop) or
            (signal == "SHORT" and bar_open > stop)
        )
        if gap_fill:
            self._record(entry, trades, bar_open, date, "GAP_FILL", days_held, capital)
            return True, capital + trades[-1]["pnl_amount"]

        should_exit, reason = False, ""
        if signal == "LONG":
            if   price <= stop:   should_exit, reason = True, "STOP_LOSS"
            elif price >= target: should_exit, reason = True, "TARGET"
        else:
            if   price >= stop:   should_exit, reason = True, "STOP_LOSS"
            elif price <= target: should_exit, reason = True, "TARGET"
        if not should_exit and days_held >= entry["max_hold"]:
            should_exit, reason = True, "TIME_EXIT"

        if should_exit:
            exit_price = price * ((1 - self.SLIPPAGE) if signal == "LONG" else (1 + self.SLIPPAGE))
            self._record(entry, trades, exit_price, date, reason, days_held, capital)
            return True, capital + trades[-1]["pnl_amount"]
        return False, capital

    def _record(self, entry, trades, exit_price, date, reason, days_held, capital):
        signal   = entry["signal"]
        shares   = entry["shares"]
        gross    = (exit_price - entry["entry_price"]) * shares if signal == "LONG" \
                   else (entry["entry_price"] - exit_price) * shares
        net      = gross - (entry["position_value"] * self.TOTAL_COST)
        trades.append({
            "entry_date":      entry["entry_date"],
            "exit_date":       date,
            "signal":          signal,
            "category":        entry["category"],
            "regime_at_entry": entry["regime_at_entry"],
            "regime_match":    entry["regime_match"],
            "entry_quality":   entry["entry_quality"],
            "entry_price":     entry["entry_price"],
            "exit_price":      round(exit_price, 2),
            "shares":          shares,
            "gross_pnl":       round(gross, 2),
            "pnl_amount":      round(net, 2),
            "pnl_pct":         round(net / max(capital, 1), 6),
            "exit_reason":     reason,
            "days_held":       days_held,
        })

    def _force_close(self, entry, df, forward_days, capital, trades):
        exit_price = float(df["Close"].iloc[-forward_days])
        self._record(
            entry, trades, exit_price, str(df.index[-1])[:10],
            "END_OF_DATA", len(df) - entry["bar_index"], capital,
        )
        return capital + trades[-1]["pnl_amount"]
