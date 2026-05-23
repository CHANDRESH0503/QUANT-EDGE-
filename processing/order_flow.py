# processing/order_flow.py
# Analyses buying vs selling pressure from OHLCV data
# Cumulative delta, VWAP, absorption detection, institutional bias
# Connected to: price_fetcher.py (intraday), technical.py (ATR)

import numpy as np
import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class OrderFlowAnalyzer:
    """
    Infers institutional order flow from OHLCV data.

    20yr trader truth:
    You can not see the order book at retail level.
    But you CAN infer it from volume patterns.
    High volume + small move = someone absorbing everything.
    Price up + falling delta = distribution.
    These patterns lead price by 1-2 bars consistently.

    Features generated (for intraday model primarily):
    - cum_delta           : net buying minus selling pressure
    - delta_divergence    : when delta disagrees with price direction
    - vwap_distance       : above/below institutional reference
    - absorption_detected : high vol + tiny price = big player present
    - inst_bias           : institutional buying or selling inferred
    - order_flow_score    : composite -1 to +1

    Connected to:
    - price_fetcher.py: uses 5-min intraday candles
    - technical.py: uses ATR for absorption threshold
    - signal_engine.py: intraday gate — order flow must confirm
    """

    def get_order_flow_features(self, df: pd.DataFrame,
                                 atr: float = 0.0) -> Dict:
        """
        Master method — compute all order flow features.
        Works best with 5-min intraday data.
        Falls back gracefully with daily data.

        Args:
            df:  OHLCV DataFrame (5-min preferred, daily acceptable)
            atr: Average True Range from technical.py for thresholds
        """
        if len(df) < 5:
            return self._empty_features()

        delta         = self._calculate_delta(df)
        vwap          = self._calculate_vwap(df)
        absorption    = self._detect_absorption(df, atr)
        inst_bias     = self._institutional_bias(df, delta)
        divergence    = self._delta_divergence(df, delta)
        composite     = self._composite_score(delta, vwap, absorption,
                                               inst_bias, divergence, df)

        return {
            # Delta analysis
            "cum_delta":            float(delta["cum"].iloc[-1]),
            "cum_delta_norm":       float(delta["cum_norm"].iloc[-1]),
            "delta_trend":          delta["trend"],
            "delta_divergence":     divergence["type"],
            "delta_divergence_flag":int(divergence["flagged"]),

            # VWAP
            "vwap_distance":        float(vwap["distance"].iloc[-1]),
            "price_above_vwap":     int(float(vwap["distance"].iloc[-1]) > 0),
            "vwap_signal":          vwap["signal"],

            # Absorption
            "absorption_detected":  int(absorption["detected"]),
            "absorption_strength":  absorption["strength"],

            # Institutional inference
            "inst_bias":            inst_bias["direction"],
            "inst_conviction":      inst_bias["conviction"],
            "large_candle_ratio":   inst_bias["large_candle_ratio"],

            # Composite
            "order_flow_score":     composite,
            "order_flow_signal":    self._score_to_signal(composite),
        }

    # ─────────────────────────────────────────────────────────────
    # DELTA ANALYSIS
    # ─────────────────────────────────────────────────────────────

    def _calculate_delta(self, df: pd.DataFrame) -> Dict:
        """
        Estimate buy vs sell volume per candle.
        Bullish candle (C > O) = mostly buying.
        Bearish candle (C < O) = mostly selling.
        Proportional split based on close position in candle range.
        """
        close    = df["Close"]
        open_    = df["Open"]
        high     = df["High"]
        low      = df["Low"]
        volume   = df["Volume"]

        # Close position in candle range (0=at low, 1=at high)
        rng          = (high - low).replace(0, np.nan)
        close_pos    = (close - low) / rng

        buy_vol      = close_pos.fillna(0.5) * volume
        sell_vol     = (1 - close_pos.fillna(0.5)) * volume
        delta        = buy_vol - sell_vol
        cum_delta    = delta.cumsum()

        # Normalise cumulative delta by average volume for comparability
        avg_vol      = volume.rolling(20).mean()
        cum_norm     = cum_delta / (avg_vol.cumsum() + 1)

        # Trend: is cumulative delta rising or falling?
        if len(cum_delta) >= 5:
            recent = float(cum_delta.iloc[-1])
            older  = float(cum_delta.iloc[-5])
            trend  = "RISING" if recent > older else "FALLING"
        else:
            trend  = "FLAT"

        return {
            "buy_vol":   buy_vol,
            "sell_vol":  sell_vol,
            "delta":     delta,
            "cum":       cum_delta,
            "cum_norm":  cum_norm,
            "trend":     trend,
        }

    def _delta_divergence(self, df: pd.DataFrame, delta: Dict) -> Dict:
        """
        Price direction disagrees with delta direction.
        Price up + delta falling = buyers exhausted = bearish.
        Price down + delta rising = sellers exhausted = bullish.
        """
        if len(df) < 5:
            return {"type": "NONE", "flagged": False}

        price_up  = float(df["Close"].iloc[-1]) > float(df["Close"].iloc[-3])
        delta_up  = float(delta["cum"].iloc[-1]) > float(delta["cum"].iloc[-3])

        if price_up and not delta_up:
            return {"type": "BEARISH_DIVERGENCE", "flagged": True}
        elif not price_up and delta_up:
            return {"type": "BULLISH_DIVERGENCE", "flagged": True}
        return {"type": "CONFIRMED", "flagged": False}

    # ─────────────────────────────────────────────────────────────
    # VWAP
    # ─────────────────────────────────────────────────────────────

    def _calculate_vwap(self, df: pd.DataFrame) -> Dict:
        """
        Volume Weighted Average Price — institutional reference level.
        Institutional desks benchmark all executions to VWAP.
        """
        tp   = (df["High"] + df["Low"] + df["Close"]) / 3
        cum_tp_vol = (tp * df["Volume"]).cumsum()
        cum_vol    = df["Volume"].cumsum()
        vwap       = cum_tp_vol / (cum_vol + 1e-8)
        distance   = (df["Close"] - vwap) / (vwap + 1e-8)

        latest_dist = float(distance.iloc[-1])
        signal = (
            "ABOVE_VWAP"   if latest_dist >  0.003 else
            "BELOW_VWAP"   if latest_dist < -0.003 else
            "AT_VWAP"
        )
        return {"vwap": vwap, "distance": distance, "signal": signal}

    # ─────────────────────────────────────────────────────────────
    # ABSORPTION
    # ─────────────────────────────────────────────────────────────

    def _detect_absorption(self, df: pd.DataFrame, atr: float) -> Dict:
        """
        High volume + small price range = large player absorbing orders.
        This is the fingerprint of institutional accumulation/distribution.
        Always precedes a major move — either direction.
        Position: size down, wait for resolution.
        """
        volume   = df["Volume"]
        avg_vol  = float(volume.rolling(20).mean().iloc[-1])
        today_vol= float(volume.iloc[-1])
        vol_ratio= today_vol / max(avg_vol, 1)

        candle_range = float(df["High"].iloc[-1] - df["Low"].iloc[-1])
        atr_threshold= atr if atr > 0 else float(
            (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
        )

        # Absorption: volume 2.5x normal but range < 30% of ATR
        detected = vol_ratio > 2.5 and candle_range < atr_threshold * 0.3

        strength = (
            "STRONG" if vol_ratio > 5 and detected else
            "MODERATE" if detected else
            "NONE"
        )

        return {"detected": detected, "strength": strength,
                "vol_ratio": round(vol_ratio, 2)}

    # ─────────────────────────────────────────────────────────────
    # INSTITUTIONAL BIAS
    # ─────────────────────────────────────────────────────────────

    def _institutional_bias(self, df: pd.DataFrame, delta: Dict) -> Dict:
        """
        Infer institutional activity from large candles and volume patterns.
        Candles 2x+ average size = institutional participation.
        """
        volume    = df["Volume"]
        avg_vol   = volume.rolling(20).mean()
        large     = volume > avg_vol * 2

        bull_large = ((df["Close"] > df["Open"]) & large).sum()
        bear_large = ((df["Close"] < df["Open"]) & large).sum()
        total_large= int(large.sum())

        large_candle_ratio = total_large / max(len(df), 1)

        if bull_large > bear_large * 1.5:
            direction  = "INSTITUTIONAL_BUYING"
            conviction = "HIGH" if bull_large > bear_large * 2 else "MEDIUM"
        elif bear_large > bull_large * 1.5:
            direction  = "INSTITUTIONAL_SELLING"
            conviction = "HIGH" if bear_large > bull_large * 2 else "MEDIUM"
        else:
            direction  = "NEUTRAL"
            conviction = "LOW"

        return {
            "direction":          direction,
            "conviction":         conviction,
            "large_candle_ratio": round(large_candle_ratio, 3),
            "bull_large":         int(bull_large),
            "bear_large":         int(bear_large),
        }

    # ─────────────────────────────────────────────────────────────
    # COMPOSITE
    # ─────────────────────────────────────────────────────────────

    def _composite_score(self, delta: Dict, vwap: Dict,
                          absorption: Dict, inst: Dict,
                          divergence: Dict, df: pd.DataFrame) -> float:
        """
        Composite order flow score from -1.0 to +1.0.
        Positive = buying pressure dominating.
        """
        score = 0.0

        # Delta trend
        if delta["trend"] == "RISING":   score += 0.25
        elif delta["trend"] == "FALLING": score -= 0.25

        # VWAP
        if vwap["signal"] == "ABOVE_VWAP":  score += 0.20
        elif vwap["signal"] == "BELOW_VWAP": score -= 0.20

        # Delta divergence (flip interpretation)
        if divergence["type"] == "BEARISH_DIVERGENCE": score -= 0.25
        elif divergence["type"] == "BULLISH_DIVERGENCE": score += 0.25

        # Institutional bias
        if inst["direction"] == "INSTITUTIONAL_BUYING":
            score += 0.20 if inst["conviction"] == "HIGH" else 0.10
        elif inst["direction"] == "INSTITUTIONAL_SELLING":
            score -= 0.20 if inst["conviction"] == "HIGH" else 0.10

        # Absorption reduces confidence in either direction
        if absorption["detected"]:
            score *= 0.5  # uncertainty — big player = unpredictable

        return round(max(-1.0, min(1.0, score)), 3)

    def _score_to_signal(self, score: float) -> str:
        if score >  0.4: return "STRONG_BUYING_PRESSURE"
        if score >  0.15: return "MILD_BUYING_PRESSURE"
        if score < -0.4: return "STRONG_SELLING_PRESSURE"
        if score < -0.15: return "MILD_SELLING_PRESSURE"
        return "BALANCED_FLOW"

    def _empty_features(self) -> Dict:
        return {
            "cum_delta": 0, "cum_delta_norm": 0, "delta_trend": "FLAT",
            "delta_divergence": "NONE", "delta_divergence_flag": 0,
            "vwap_distance": 0, "price_above_vwap": 0, "vwap_signal": "AT_VWAP",
            "absorption_detected": 0, "absorption_strength": "NONE",
            "inst_bias": "NEUTRAL", "inst_conviction": "LOW",
            "large_candle_ratio": 0, "order_flow_score": 0.0,
            "order_flow_signal": "BALANCED_FLOW",
        }