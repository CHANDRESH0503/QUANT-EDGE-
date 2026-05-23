# backtest/engine.py
# Event-driven backtesting engine for all three model types
# Uses historical data only — strict no-lookahead enforcement
# Connected to: features/feature_builder.py, models/, backtest/metrics.py

import numpy as np
import pandas as pd
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from backtest.metrics  import BacktestMetrics
from processing.technical import TechnicalProcessor
from features.technical_features import TechnicalFeatures
from risk.position_sizer import PositionSizer
from risk.capital_mode   import CapitalMode

RESULTS_PATH = "models/evaluation/backtest_results.json"


class BacktestEngine:
    """
    Event-driven backtesting engine.

    20yr trader rule on backtesting:
    A backtest is a compass, not a GPS.
    It tells you approximate direction — not the exact path.
    The three cardinal sins of backtesting:
    1. Lookahead bias    — using future data to make past decisions
    2. Overfitting       — optimising so hard the system only works on history
    3. Ignoring costs    — slippage + brokerage destroys thin edges

    This engine:
    - Walks forward in time strictly — no peeking at future bars
    - Applies 0.1% transaction cost (realistic for HDFC Bank NSE)
    - Applies 0.05% slippage on entry and exit
    - Enforces max hold days per model type
    - Tracks equity curve for drawdown analysis
    """

    # Realistic Indian equity trading costs
    TRANSACTION_COST = 0.001   # 0.1% brokerage + STT + exchange charges
    SLIPPAGE         = 0.0005  # 0.05% per side entry/exit slippage
    TOTAL_COST       = TRANSACTION_COST + 2 * SLIPPAGE  # ~0.2% round trip

    MAX_HOLD = {"swing": 7, "intraday": 1, "positional": 21}

    def __init__(
        self,
        starting_capital: float = 100_000.0,
        capital_mode:     str   = "FULL",
    ):
        self.starting_capital = starting_capital
        self.capital_mode     = capital_mode
        self.tech_processor   = TechnicalProcessor()
        self.tech_features    = TechnicalFeatures()
        self.position_sizer   = PositionSizer()
        self.metrics_calc     = BacktestMetrics()

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def run(
        self,
        df:             pd.DataFrame,
        model,                          # trained XGBoost model
        feature_names:  List[str],
        model_type:     str   = "swing",
        forward_days:   int   = 5,
        threshold:      float = 0.025,
        verbose:        bool  = False,
    ) -> Dict:
        """
        Run full backtest on historical OHLCV data.

        Args:
            df:            OHLCV DataFrame (daily/weekly/5-min)
            model:         Trained XGBoost model from models/
            feature_names: Feature list used by the model
            model_type:    "swing" | "intraday" | "positional"
            forward_days:  Prediction horizon in bars
            threshold:     Minimum price move to generate signal
            verbose:       Log every trade

        Returns:
            Full backtest results dict with metrics and trade log
        """
        if len(df) < 100:
            logger.error("Need 100+ bars for meaningful backtest")
            return {}

        logger.info(
            f"Backtest starting: {model_type} | "
            f"{len(df)} bars | "
            f"capital=₹{self.starting_capital:,.0f}"
        )

        capital     = self.starting_capital
        trades      = []
        equity_log  = []
        in_trade    = False
        trade_entry = {}
        max_hold    = self.MAX_HOLD.get(model_type, 7)
        min_history = 60

        for i in range(min_history, len(df) - forward_days):
            current_bar  = df.iloc[i]
            current_price= float(current_bar["Close"])
            current_date = str(current_bar.name)[:10]

            # ── Manage open trade ─────────────────────────────────
            if in_trade:
                days_held   = i - trade_entry["bar_index"]
                entry_price = trade_entry["entry_price"]
                signal      = trade_entry["signal"]
                stop        = trade_entry["stop_price"]
                target      = trade_entry["target_price"]

                # Gap-fill: use today's open if it's already past the stop
                bar_open    = float(current_bar.get("Open", current_price))
                gap_fill    = (
                    (signal == "LONG"  and bar_open < stop) or
                    (signal == "SHORT" and bar_open > stop)
                )
                if gap_fill:
                    # Actual fill at open — may be worse than the stop
                    actual_exit_price = bar_open
                    gross_pnl = (actual_exit_price - entry_price) * trade_entry["shares"] \
                                if signal == "LONG" \
                                else (entry_price - actual_exit_price) * trade_entry["shares"]
                    net_pnl = gross_pnl - (trade_entry["position_value"] * self.TOTAL_COST)
                    capital += net_pnl
                    in_trade = False
                    trades.append({
                        "entry_date":  trade_entry["entry_date"],
                        "exit_date":   current_date,
                        "signal":      signal,
                        "entry_price": entry_price,
                        "exit_price":  round(actual_exit_price, 2),
                        "shares":      trade_entry["shares"],
                        "pnl_amount":  round(net_pnl, 2),
                        "pnl_pct":     round(net_pnl / max(capital, 1), 6),
                        "exit_reason": "GAP_FILL",
                        "days_held":   days_held,
                    })
                    equity_log.append({"date": current_date, "equity": round(capital, 2)})
                    continue

                # Exit conditions
                should_exit = False
                exit_reason = ""

                if signal == "LONG":
                    pnl_pct = (current_price - entry_price) / entry_price
                    if current_price <= stop:
                        should_exit = True; exit_reason = "STOP_LOSS"
                    elif current_price >= target:
                        should_exit = True; exit_reason = "TARGET"
                else:  # SHORT
                    pnl_pct = (entry_price - current_price) / entry_price
                    if current_price >= stop:
                        should_exit = True; exit_reason = "STOP_LOSS"
                    elif current_price <= target:
                        should_exit = True; exit_reason = "TARGET"

                if days_held >= max_hold and not should_exit:
                    should_exit = True; exit_reason = "TIME_EXIT"

                if should_exit:
                    # Apply transaction cost + slippage
                    exit_price  = current_price * (
                        (1 - self.SLIPPAGE) if signal == "LONG"
                        else (1 + self.SLIPPAGE)
                    )
                    shares      = trade_entry["shares"]
                    gross_pnl   = (exit_price - trade_entry["entry_price"]) * shares \
                                  if signal == "LONG" \
                                  else (trade_entry["entry_price"] - exit_price) * shares
                    # Subtract round-trip transaction cost
                    net_pnl     = gross_pnl - (trade_entry["position_value"] * self.TOTAL_COST)
                    net_pnl_pct = net_pnl / capital

                    capital    += net_pnl
                    in_trade    = False

                    trade_record = {
                        "entry_date":  trade_entry["entry_date"],
                        "exit_date":   current_date,
                        "signal":      signal,
                        "entry_price": trade_entry["entry_price"],
                        "exit_price":  round(exit_price, 2),
                        "shares":      shares,
                        "pnl_amount":  round(net_pnl, 2),
                        "pnl_pct":     round(net_pnl_pct, 6),
                        "exit_reason": exit_reason,
                        "days_held":   days_held,
                    }
                    trades.append(trade_record)

                    if verbose:
                        logger.info(
                            f"  EXIT {signal} @ ₹{exit_price:.2f} | "
                            f"pnl={net_pnl_pct:+.2%} | "
                            f"reason={exit_reason} | "
                            f"equity=₹{capital:,.0f}"
                        )

                equity_log.append({"date": current_date, "equity": round(capital, 2)})
                continue

            # ── Generate signal for next period ───────────────────
            df_slice     = df.iloc[: i + 1].copy()
            tech_df      = self.tech_processor.build_features(df_slice)
            feat_row     = self.tech_features.extract(tech_df)
            feat_df      = pd.DataFrame([{
                f: feat_row.get(f, 0.0) for f in feature_names
            }]).fillna(0)

            try:
                probs    = model.predict_proba(feat_df)[0]
                pred     = int(np.argmax(probs))
                conf     = float(probs[pred])
            except Exception as e:
                logger.debug(f"Prediction error at bar {i}: {e}")
                equity_log.append({"date": current_date, "equity": round(capital, 2)})
                continue

            label_map = {0: "SHORT", 1: "FLAT", 2: "LONG"}
            signal    = label_map[pred]

            # Skip flat or low confidence
            if signal == "FLAT" or conf < 0.55:
                equity_log.append({"date": current_date, "equity": round(capital, 2)})
                continue

            # ── Open new position ─────────────────────────────────
            atr_series   = (df["High"] - df["Low"]).rolling(14).mean()
            atr          = float(atr_series.iloc[i]) if not atr_series.empty else current_price * 0.015
            cfg          = CapitalMode.MODES.get(self.capital_mode, CapitalMode.MODES["FULL"])
            stop_mult    = cfg["stop_atr_mult"]
            target_mult  = cfg["target_atr_mult"]
            max_risk     = capital * cfg["max_risk_pct"]

            stop_dist    = atr * stop_mult
            target_dist  = atr * target_mult
            shares       = max(1, int(max_risk / stop_dist))

            # Cap at 40% capital
            if shares * current_price > capital * 0.40:
                shares = max(1, int(capital * 0.40 / current_price))

            # Apply entry slippage
            entry_price  = current_price * (
                (1 + self.SLIPPAGE) if signal == "LONG"
                else (1 - self.SLIPPAGE)
            )
            stop_price   = entry_price - stop_dist if signal == "LONG" else entry_price + stop_dist
            target_price = entry_price + target_dist if signal == "LONG" else entry_price - target_dist

            in_trade     = True
            trade_entry  = {
                "bar_index":     i,
                "entry_date":    current_date,
                "signal":        signal,
                "entry_price":   round(entry_price, 2),
                "stop_price":    round(stop_price, 2),
                "target_price":  round(target_price, 2),
                "shares":        shares,
                "position_value":round(shares * entry_price, 2),
                "confidence":    conf,
            }

            equity_log.append({"date": current_date, "equity": round(capital, 2)})

        # ── Close any open position at last bar ───────────────────
        if in_trade and len(df) > 0:
            exit_price  = float(df["Close"].iloc[-forward_days])
            signal      = trade_entry["signal"]
            shares      = trade_entry["shares"]
            gross_pnl   = (exit_price - trade_entry["entry_price"]) * shares \
                          if signal == "LONG" \
                          else (trade_entry["entry_price"] - exit_price) * shares
            net_pnl     = gross_pnl - (trade_entry["position_value"] * self.TOTAL_COST)
            trades.append({
                "entry_date":  trade_entry["entry_date"],
                "exit_date":   str(df.index[-1])[:10],
                "signal":      signal,
                "entry_price": trade_entry["entry_price"],
                "exit_price":  round(exit_price, 2),
                "shares":      shares,
                "pnl_amount":  round(net_pnl, 2),
                "pnl_pct":     round(net_pnl / capital, 6),
                "exit_reason": "END_OF_DATA",
                "days_held":   len(df) - trade_entry["bar_index"],
            })

        # ── Calculate metrics ─────────────────────────────────────
        metrics  = self.metrics_calc.calculate(trades, self.starting_capital)
        logger.info(
            f"Backtest complete: {model_type} | "
            f"{self.metrics_calc.summary_line(metrics)}"
        )

        result = {
            "model_type":       model_type,
            "total_bars":       len(df),
            "backtest_period":  f"{str(df.index[0])[:10]} to {str(df.index[-1])[:10]}",
            "starting_capital": self.starting_capital,
            "ending_capital":   round(capital, 2),
            "trades":           trades,
            "equity_log":       equity_log,
            "metrics":          metrics,
            "run_at":           str(datetime.now()),
        }

        self._save_results(result, model_type)
        return result

    # ─────────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────────

    def _save_results(self, result: Dict, model_type: str) -> None:
        """Persist backtest results for dashboard display."""
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        existing = {}
        if os.path.exists(RESULTS_PATH):
            try:
                with open(RESULTS_PATH) as f:
                    existing = json.load(f)
            except Exception:
                pass
        # Store without full trade list (too large) — metrics only
        existing[model_type] = {
            "metrics":         result["metrics"],
            "backtest_period": result["backtest_period"],
            "total_trades":    len(result["trades"]),
            "run_at":          result["run_at"],
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        logger.info(f"Backtest results saved: {RESULTS_PATH}")
