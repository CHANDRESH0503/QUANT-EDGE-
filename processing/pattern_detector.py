# processing/pattern_detector.py
# Detects candlestick patterns and classic chart patterns
# Individual patterns weak — composite score is powerful
# Connects to: technical.py features, signal_engine.py gates

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not installed. pip install TA-Lib. Using manual patterns.")


class PatternDetector:
    """
    Detects candlestick + chart patterns and returns composite scores.

    20yr trader insight:
    No single candlestick pattern has >55% reliability alone.
    But when 3+ patterns align at a key S/R level with volume — 
    that combination has real edge. Score everything, threshold the composite.

    Pattern score: -100 (strong bear) to +100 (strong bull)
    Threshold for signal: abs(score) > 40 at minimum
    """

    # Weights by pattern reliability (backtested over 20yr career)
    PATTERN_WEIGHTS = {
        # High reliability
        "engulfing":       3.0,
        "morning_star":    3.0,
        "evening_star":    3.0,
        "three_soldiers":  2.5,
        "three_crows":     2.5,
        # Medium reliability
        "hammer":          2.0,
        "shooting_star":   2.0,
        "doji":            1.5,
        "harami":          1.5,
        "marubozu":        2.0,
        # Lower reliability
        "spinning_top":    1.0,
    }

    def get_pattern_features(self, df: pd.DataFrame) -> Dict:
        """
        Master method — detect all patterns and return composite score.
        Returns dict of individual pattern flags + composite score.
        """
        if len(df) < 5:
            return self._empty_patterns()

        o = df["Open"].values.astype(float)
        h = df["High"].values.astype(float)
        l = df["Low"].values.astype(float)
        c = df["Close"].values.astype(float)

        patterns = {}

        if TALIB_AVAILABLE:
            patterns = self._detect_talib(o, h, l, c)
        else:
            patterns = self._detect_manual(o, h, l, c)

        # Composite score
        score = self._composite_score(patterns)

        # Chart patterns (longer-term)
        chart  = self._detect_chart_patterns(df)

        return {
            **patterns,
            **chart,
            "pattern_score":        round(score, 1),
            "pattern_signal":       self._score_to_signal(score),
            "pattern_bull_count":   sum(1 for v in patterns.values() if v > 0),
            "pattern_bear_count":   sum(1 for v in patterns.values() if v < 0),
        }

    # ─────────────────────────────────────────────────────────────
    # TALIB DETECTION (preferred)
    # ─────────────────────────────────────────────────────────────

    def _detect_talib(self, o, h, l, c) -> Dict:
        """Use TA-Lib for accurate pattern detection."""
        patterns = {}
        talib_funcs = {
            "engulfing":       (talib.CDLENGULFING,    True),
            "hammer":          (talib.CDLHAMMER,       True),
            "shooting_star":   (talib.CDLSHOOTINGSTAR, False),
            "doji":            (talib.CDLDOJI,         None),
            "morning_star":    (talib.CDLMORNINGSTAR,  True),
            "evening_star":    (talib.CDLEVENINGSTAR,  False),
            "three_soldiers":  (talib.CDL3WHITESOLDIERS, True),
            "three_crows":     (talib.CDL3BLACKCROWS,  False),
            "harami":          (talib.CDLHARAMI,       None),
            "marubozu":        (talib.CDLMARUBOZU,     None),
            "spinning_top":    (talib.CDLSPINNINGTOP,  None),
        }
        for name, (func, _) in talib_funcs.items():
            try:
                result = func(o, h, l, c)
                val    = int(result[-1])
                # TA-Lib returns +100 (bullish), -100 (bearish), 0 (none)
                patterns[f"pat_{name}"] = val / 100.0
            except Exception:
                patterns[f"pat_{name}"] = 0.0
        return patterns

    # ─────────────────────────────────────────────────────────────
    # MANUAL DETECTION (fallback when TA-Lib not available)
    # ─────────────────────────────────────────────────────────────

    def _detect_manual(self, o, h, l, c) -> Dict:
        """Manual pattern detection — less precise but no TA-Lib dependency."""
        patterns = {}
        patterns["pat_engulfing"]    = self._engulfing(o, h, l, c)
        patterns["pat_hammer"]       = self._hammer(o, h, l, c)
        patterns["pat_shooting_star"]= self._shooting_star(o, h, l, c)
        patterns["pat_doji"]         = self._doji(o, h, l, c)
        patterns["pat_morning_star"] = self._morning_star(o, h, l, c)
        patterns["pat_evening_star"] = self._evening_star(o, h, l, c)
        patterns["pat_marubozu"]     = self._marubozu(o, h, l, c)
        patterns["pat_three_soldiers"] = 0.0
        patterns["pat_three_crows"]    = 0.0
        patterns["pat_harami"]         = self._harami(o, h, l, c)
        patterns["pat_spinning_top"]   = self._spinning_top(o, h, l, c)
        return patterns

    def _body(self, o, c, i):
        return abs(c[i] - o[i])

    def _range(self, h, l, i):
        return h[i] - l[i]

    def _is_bull(self, o, c, i):
        return c[i] > o[i]

    def _engulfing(self, o, h, l, c) -> float:
        if len(c) < 2:
            return 0.0
        i = len(c) - 1
        prev_body = self._body(o, c, i-1)
        curr_body = self._body(o, c, i)
        if curr_body <= prev_body:
            return 0.0
        # Bullish engulfing
        if not self._is_bull(o, c, i-1) and self._is_bull(o, c, i):
            if o[i] <= c[i-1] and c[i] >= o[i-1]:
                return 1.0
        # Bearish engulfing
        if self._is_bull(o, c, i-1) and not self._is_bull(o, c, i):
            if o[i] >= c[i-1] and c[i] <= o[i-1]:
                return -1.0
        return 0.0

    def _hammer(self, o, h, l, c) -> float:
        i = len(c) - 1
        body  = self._body(o, c, i)
        rng   = self._range(h, l, i)
        if rng == 0:
            return 0.0
        lower_wick = min(o[i], c[i]) - l[i]
        upper_wick = h[i] - max(o[i], c[i])
        if lower_wick > 2 * body and upper_wick < body * 0.3 and body > 0:
            return 1.0
        return 0.0

    def _shooting_star(self, o, h, l, c) -> float:
        i = len(c) - 1
        body  = self._body(o, c, i)
        rng   = self._range(h, l, i)
        if rng == 0:
            return 0.0
        upper_wick = h[i] - max(o[i], c[i])
        lower_wick = min(o[i], c[i]) - l[i]
        if upper_wick > 2 * body and lower_wick < body * 0.3 and body > 0:
            return -1.0
        return 0.0

    def _doji(self, o, h, l, c) -> float:
        i    = len(c) - 1
        body = self._body(o, c, i)
        rng  = self._range(h, l, i)
        if rng == 0:
            return 0.0
        return 0.5 if body / rng < 0.1 else 0.0  # neutral doji

    def _morning_star(self, o, h, l, c) -> float:
        if len(c) < 3:
            return 0.0
        i = len(c) - 1
        if (not self._is_bull(o, c, i-2) and
                self._body(o, c, i-1) < self._body(o, c, i-2) * 0.3 and
                self._is_bull(o, c, i) and
                c[i] > (o[i-2] + c[i-2]) / 2):
            return 1.0
        return 0.0

    def _evening_star(self, o, h, l, c) -> float:
        if len(c) < 3:
            return 0.0
        i = len(c) - 1
        if (self._is_bull(o, c, i-2) and
                self._body(o, c, i-1) < self._body(o, c, i-2) * 0.3 and
                not self._is_bull(o, c, i) and
                c[i] < (o[i-2] + c[i-2]) / 2):
            return -1.0
        return 0.0

    def _marubozu(self, o, h, l, c) -> float:
        i    = len(c) - 1
        body = self._body(o, c, i)
        rng  = self._range(h, l, i)
        if rng == 0:
            return 0.0
        if body / rng > 0.92:
            return 1.0 if self._is_bull(o, c, i) else -1.0
        return 0.0

    def _harami(self, o, h, l, c) -> float:
        if len(c) < 2:
            return 0.0
        i = len(c) - 1
        prev_high = max(o[i-1], c[i-1])
        prev_low  = min(o[i-1], c[i-1])
        curr_high = max(o[i], c[i])
        curr_low  = min(o[i], c[i])
        if curr_high < prev_high and curr_low > prev_low:
            return 0.5 if self._is_bull(o, c, i) else -0.5
        return 0.0

    def _spinning_top(self, o, h, l, c) -> float:
        i    = len(c) - 1
        body = self._body(o, c, i)
        rng  = self._range(h, l, i)
        if rng == 0:
            return 0.0
        upper = h[i] - max(o[i], c[i])
        lower = min(o[i], c[i]) - l[i]
        if body / rng < 0.3 and upper > body and lower > body:
            return 0.3  # indecision
        return 0.0

    # ─────────────────────────────────────────────────────────────
    # CHART PATTERNS (longer-term)
    # ─────────────────────────────────────────────────────────────

    def _detect_chart_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect multi-bar chart patterns: squeeze, breakout, golden/death cross."""
        result = {}
        close  = df["Close"]
        volume = df["Volume"]

        # Golden cross (EMA20 crosses above EMA50)
        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()
        result["chart_golden_cross"] = int(
            ema20.iloc[-2] < ema50.iloc[-2] and ema20.iloc[-1] > ema50.iloc[-1]
        )
        result["chart_death_cross"] = int(
            ema20.iloc[-2] > ema50.iloc[-2] and ema20.iloc[-1] < ema50.iloc[-1]
        )

        # Volume breakout: price new 20d high + volume spike
        high_20d = df["High"].rolling(20).max().iloc[-2]
        vol_avg  = volume.rolling(10).mean().iloc[-1]
        result["chart_volume_breakout"] = int(
            close.iloc[-1] > high_20d and
            volume.iloc[-1] > vol_avg * 1.5
        )

        # Bollinger squeeze breakout
        mid   = close.rolling(20).mean()
        sigma = close.rolling(20).std()
        bb_w  = (4 * sigma / mid)
        was_squeezed = float(bb_w.iloc[-5]) < float(bb_w.rolling(60).quantile(0.15).iloc[-5])
        now_expanding= float(bb_w.iloc[-1]) > float(bb_w.iloc[-3])
        result["chart_bb_breakout"] = int(was_squeezed and now_expanding)

        # Inside bar (consolidation before move)
        result["chart_inside_bar"] = int(
            df["High"].iloc[-1] < df["High"].iloc[-2] and
            df["Low"].iloc[-1]  > df["Low"].iloc[-2]
        )

        return result

    # ─────────────────────────────────────────────────────────────
    # SCORING
    # ─────────────────────────────────────────────────────────────

    def _composite_score(self, patterns: Dict) -> float:
        """
        Weight each pattern by reliability and sum.
        Returns -100 to +100 composite score.
        """
        score = 0.0
        for key, val in patterns.items():
            name = key.replace("pat_", "")
            w    = self.PATTERN_WEIGHTS.get(name, 1.0)
            score += val * w * 20  # scale to ±100

        return max(-100.0, min(100.0, score))

    def _score_to_signal(self, score: float) -> str:
        if score >  60: return "STRONG_BULL_PATTERN"
        if score >  30: return "BULL_PATTERN"
        if score < -60: return "STRONG_BEAR_PATTERN"
        if score < -30: return "BEAR_PATTERN"
        return "NO_PATTERN"

    def _empty_patterns(self) -> Dict:
        keys = [f"pat_{n}" for n in self.PATTERN_WEIGHTS]
        return {**{k: 0.0 for k in keys},
                "pattern_score": 0.0, "pattern_signal": "NO_PATTERN",
                "pattern_bull_count": 0, "pattern_bear_count": 0,
                "chart_golden_cross": 0, "chart_death_cross": 0,
                "chart_volume_breakout": 0, "chart_bb_breakout": 0,
                "chart_inside_bar": 0}