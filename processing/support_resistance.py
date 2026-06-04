# processing/support_resistance.py
# Finds key support and resistance levels using multiple methods
# S/R proximity is a trade quality metric — not just a signal
# Connected to: price_fetcher.py, options_fetcher.py (max OI strikes)

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)


class SupportResistanceEngine:
    """
    Identifies high-quality support and resistance levels.

    20yr trader truth:
    Entry near strong support = tight stop, great R:R.
    Entry 5% above support = wide stop, poor R:R.
    The same signal has completely different quality depending
    on where price is relative to key levels.

    Three methods — stronger when multiple methods agree:
    1. Swing highs/lows     — price memory, most reliable
    2. Round numbers        — psychological magnets (₹1800, ₹1850)
    3. Volume nodes (POC)   — where most trading happened = defended

    Connected to:
    - options_fetcher.py: max OI strikes added as S/R levels
    - signal_engine.py gate 5: entry quality validation
    - risk/exit_engine.py: sets profit targets at resistance
    """

    # Round number intervals for HDFC Bank price range ~₹1500-2500
    ROUND_INTERVALS = [50, 100]

    # ── ATR-relative entry geometry (GATE5-1) ────────────────────────
    # The execution layer trades a fixed ATR geometry: stop = 2×ATR,
    # target = 5×ATR → a real 2.5:1 reward:risk on EVERY trade. The old
    # reward_risk_sr (nearest_res/nearest_sup) was a ratio of two sub-ATR
    # noise distances — disconnected from the actual trade — and it
    # hard-blocked ~37% of genuine 2.5:1 trades at random.
    #
    # Fix (Rule #20/#26/#27 superseded): reward_risk_sr = the REAL trade
    # R:R, so the hard floor never spuriously fires; entry_quality becomes
    # an ATR-relative *runway* grade used ONLY to modulate SIZE (clear
    # runway → A/full size, entering into a strong wall → D/probe size).
    # S/R sizes the trade; it does not veto it.
    STOP_ATR_MULT    = 2.0   # mirrors PipelineBacktest / exit engine
    TARGET_ATR_MULT  = 5.0
    REAL_TRADE_RR    = TARGET_ATR_MULT / STOP_ATR_MULT          # 2.5:1
    WALL_STRENGTH    = 6     # only high-confluence levels cap the runway
    ATR_PERIOD       = 14

    def get_sr_features(self, df: pd.DataFrame,
                         max_call_oi_strike: float = 0,
                         max_put_oi_strike:  float = 0) -> Dict:
        """
        Master method — find all S/R levels, compute distances.

        Args:
            df:                 OHLCV DataFrame (252 days minimum)
            max_call_oi_strike: from options_fetcher (key resistance)
            max_put_oi_strike:  from options_fetcher (key support)
        """
        if len(df) < 30:
            return self._empty_sr()

        current = float(df["Close"].iloc[-1])
        levels  = self._find_all_levels(df, max_call_oi_strike,
                                         max_put_oi_strike)

        supports     = sorted(
            [l for l in levels if l["price"] < current],
            key=lambda x: x["price"], reverse=True
        )
        resistances  = sorted(
            [l for l in levels if l["price"] > current],
            key=lambda x: x["price"]
        )

        # Nearest levels
        nearest_sup  = supports[0]    if supports    else None
        nearest_res  = resistances[0] if resistances else None

        # Second nearest (next target after first is broken)
        second_sup   = supports[1]    if len(supports)    > 1 else None
        second_res   = resistances[1] if len(resistances) > 1 else None

        # Distance calculations
        sup_dist  = self._dist(current, nearest_sup["price"])  if nearest_sup  else 0.05
        res_dist  = self._dist(nearest_res["price"], current)  if nearest_res  else 0.05
        sup_str   = nearest_sup["strength"]  if nearest_sup  else 1
        res_str   = nearest_res["strength"]  if nearest_res  else 1

        # ── ATR-relative runway grades (GATE5-1) ──────────────────
        # entry_quality measures how much CLEAR RUNWAY the trade has to its
        # ATR target before hitting a strong wall — a size signal, not a veto.
        # Computed for both directions (SR is direction-agnostic); Gate 5
        # picks the one matching the signal. reward_risk_sr is the REAL
        # trade R:R (2.5:1) so the hard floor never fires on S/R noise.
        atr_pct      = self._atr_pct(df, current)
        eq_long      = self._runway_grade("LONG",  current, levels, atr_pct)
        eq_short     = self._runway_grade("SHORT", current, levels, atr_pct)

        return {
            # Nearest support
            "nearest_support":       nearest_sup["price"]    if nearest_sup  else current * 0.95,
            "support_distance_pct":  round(sup_dist * 100, 3),
            "support_strength":      sup_str,
            "near_support":          int(sup_dist < 0.012),  # within 1.2% = near

            # Nearest resistance
            "nearest_resistance":    nearest_res["price"]   if nearest_res  else current * 1.05,
            "resistance_distance_pct": round(res_dist * 100, 3),
            "resistance_strength":   res_str,
            "near_resistance":       int(res_dist < 0.012),

            # Second levels (for targets)
            "second_support":        second_sup["price"]  if second_sup  else current * 0.90,
            "second_resistance":     second_res["price"]  if second_res  else current * 1.10,

            # Trade quality — ATR-relative runway grade (GATE5-1).
            # entry_quality is the LONG grade (kept as the default field for
            # backward compatibility); entry_quality_short is the SHORT grade.
            # Gate 5 selects by signal direction. Both are SIZE signals only.
            "entry_quality":         eq_long,           # A / B / C / D (LONG)
            "entry_quality_short":   eq_short,          # A / B / C / D (SHORT)
            "entry_quality_score":   self._quality_to_score(eq_long),
            "atr_pct":               round(atr_pct, 4),
            # Real trade R:R (target 5×ATR / stop 2×ATR). The trade does not
            # use S/R for execution, so the honest R:R is this constant — not
            # the old nearest_res/nearest_sup noise ratio.
            "reward_risk_sr":        round(self.REAL_TRADE_RR, 2),

            # Breakout detection
            "near_breakout":         int(res_dist < 0.005 and
                                         float(df["Volume"].iloc[-1]) >
                                         float(df["Volume"].rolling(10).mean().iloc[-1]) * 1.5),
        }

    # ─────────────────────────────────────────────────────────────
    # LEVEL DETECTION
    # ─────────────────────────────────────────────────────────────

    def _find_all_levels(self, df: pd.DataFrame,
                          max_call_strike: float,
                          max_put_strike:  float) -> List[Dict]:
        """Combine all three methods into unified level list."""
        levels = []
        levels += self._swing_levels(df)
        levels += self._round_number_levels(df)
        levels += self._volume_node_levels(df)

        # Add options-derived levels (strongest institutional reference)
        if max_call_strike > 0:
            levels.append({"price": max_call_strike, "type": "RESISTANCE",
                           "method": "options_oi", "strength": 4})
        if max_put_strike > 0:
            levels.append({"price": max_put_strike, "type": "SUPPORT",
                           "method": "options_oi", "strength": 4})

        # Cluster levels within 0.5% of each other
        return self._cluster_levels(levels)

    def _swing_levels(self, df: pd.DataFrame, order: int = 8) -> List[Dict]:
        """
        Swing highs = resistance memory.
        Swing lows  = support memory.
        Multiple touches = stronger level.
        """
        levels = []
        highs  = argrelextrema(df["High"].values,  np.greater, order=order)[0]
        lows   = argrelextrema(df["Low"].values,   np.less,    order=order)[0]

        for idx in highs:
            price = float(df["High"].iloc[idx])
            # Count touches — how many times has this level been tested?
            touches = self._count_touches(df, price, is_resistance=True)
            levels.append({
                "price":    price,
                "type":     "RESISTANCE",
                "method":   "swing_high",
                "strength": min(touches, 5),
            })

        for idx in lows:
            price = float(df["Low"].iloc[idx])
            touches = self._count_touches(df, price, is_resistance=False)
            levels.append({
                "price":    price,
                "type":     "SUPPORT",
                "method":   "swing_low",
                "strength": min(touches, 5),
            })

        return levels

    def _round_number_levels(self, df: pd.DataFrame) -> List[Dict]:
        """
        Psychological levels: ₹1800, ₹1850, ₹1900 etc.
        Self-fulfilling — enough traders watch them to make them real.
        """
        levels   = []
        current  = float(df["Close"].iloc[-1])

        for interval in self.ROUND_INTERVALS:
            base = round(current / interval) * interval
            for mult in range(-8, 9):
                level = base + mult * interval
                if level <= 0:
                    continue
                dist  = abs(level - current) / current
                if dist > 0.15:  # only look ±15% from current
                    continue
                levels.append({
                    "price":    float(level),
                    "type":     "RESISTANCE" if level > current else "SUPPORT",
                    "method":   "round_number",
                    "strength": 3 if interval == 100 else 2,
                })
        return levels

    def _volume_node_levels(self, df: pd.DataFrame, bins: int = 25) -> List[Dict]:
        """
        High Volume Nodes — prices where most volume traded.
        These are the most institutionally defended levels.
        Millions of buyers have their cost at these prices — they defend them.
        """
        levels   = []
        current  = float(df["Close"].iloc[-1])

        if len(df) < 20:
            return levels

        price_min = float(df["Low"].min())
        price_max = float(df["High"].max())
        if price_max <= price_min:
            return levels

        bucket_size = (price_max - price_min) / bins
        vol_by_bucket: Dict[int, float] = {}

        for i in range(len(df)):
            bucket = int((float(df["Close"].iloc[i]) - price_min) / (bucket_size + 1e-8))
            bucket = max(0, min(bucket, bins - 1))
            vol_by_bucket[bucket] = (
                vol_by_bucket.get(bucket, 0) + float(df["Volume"].iloc[i])
            )

        if not vol_by_bucket:
            return levels

        # Top 20% by volume = high-volume nodes
        vol_values  = list(vol_by_bucket.values())
        threshold   = np.percentile(vol_values, 80)

        for bucket, vol in vol_by_bucket.items():
            if vol >= threshold:
                mid_price = price_min + (bucket + 0.5) * bucket_size
                levels.append({
                    "price":    round(mid_price, 2),
                    "type":     "RESISTANCE" if mid_price > current else "SUPPORT",
                    "method":   "volume_node",
                    "strength": 4,  # volume nodes = strongest S/R
                })
        return levels

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _count_touches(self, df: pd.DataFrame, price: float,
                        is_resistance: bool, tolerance: float = 0.005) -> int:
        """How many times did price come within tolerance of this level?"""
        if is_resistance:
            touches = ((df["High"] - price).abs() / price < tolerance).sum()
        else:
            touches = ((df["Low"]  - price).abs() / price < tolerance).sum()
        return max(1, int(touches))

    def _cluster_levels(self, levels: List[Dict],
                         tolerance: float = 0.005) -> List[Dict]:
        """Merge levels within tolerance% of each other. Average price, sum strength."""
        if not levels:
            return []

        sorted_levels = sorted(levels, key=lambda x: x["price"])
        clustered     = []
        current_group = [sorted_levels[0]]

        for lvl in sorted_levels[1:]:
            prev_price = current_group[-1]["price"]
            if abs(lvl["price"] - prev_price) / max(prev_price, 1) <= tolerance:
                current_group.append(lvl)
            else:
                clustered.append(self._merge_group(current_group))
                current_group = [lvl]

        clustered.append(self._merge_group(current_group))
        return clustered

    def _merge_group(self, group: List[Dict]) -> Dict:
        avg_price = float(np.mean([g["price"] for g in group]))
        total_str = min(sum(g["strength"] for g in group), 8)
        types     = [g["type"] for g in group]
        dominant  = max(set(types), key=types.count)
        return {
            "price":    round(avg_price, 2),
            "type":     dominant,
            "method":   "+".join(set(g["method"] for g in group)),
            "strength": total_str,
            "touches":  len(group),
        }

    def _dist(self, a: float, b: float) -> float:
        return abs(a - b) / max(b, 1)

    def _atr_pct(self, df: pd.DataFrame, current: float) -> float:
        """ATR(14) as a fraction of price. 0.0 if not computable."""
        if len(df) < self.ATR_PERIOD + 1 or current <= 0:
            return 0.0
        h, l, c = df["High"], df["Low"], df["Close"]
        prev_c  = c.shift()
        tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()],
                       axis=1).max(axis=1)
        atr = float(tr.rolling(self.ATR_PERIOD).mean().iloc[-1])
        if not np.isfinite(atr) or atr <= 0:
            return 0.0
        return atr / current

    def _runway_grade(self, direction: str, current: float,
                      levels: List[Dict], atr_pct: float) -> str:
        """
        ATR-relative entry-quality grade — a SIZE signal, never a veto.

        The trade targets TARGET_ATR_MULT×ATR with a STOP_ATR_MULT×ATR stop.
        A strong wall (strength ≥ WALL_STRENGTH) between entry and that target
        caps the realistic reward — so the grade measures how much CLEAR
        RUNWAY the trade has:
            A = full runway to the ATR target (R:R ≥ 2.0)
            B = mild cap                      (R:R ≥ 1.5)
            C = capped but still positive     (R:R ≥ 1.0)
            D = entering into a near wall      (R:R < 1.0) → probe size
        Minor levels do NOT cap a momentum move (it breaks through them);
        only high-confluence walls count.
        """
        if atr_pct <= 0:
            return "C"   # neutral grade when ATR unknown → mid size
        risk = self.STOP_ATR_MULT * atr_pct
        if direction == "LONG":
            walls = [(lvl["price"] - current) / current for lvl in levels
                     if lvl["price"] > current and lvl["strength"] >= self.WALL_STRENGTH]
        else:
            walls = [(current - lvl["price"]) / current for lvl in levels
                     if lvl["price"] < current and lvl["strength"] >= self.WALL_STRENGTH]
        runway = min([w for w in walls if w > 0], default=99.0)
        reward = min(self.TARGET_ATR_MULT * atr_pct, runway)
        rr = reward / max(risk, 1e-9)
        if   rr >= 2.0: return "A"
        elif rr >= 1.5: return "B"
        elif rr >= 1.0: return "C"
        return "D"

    def _quality_to_score(self, quality: str) -> float:
        return {"A": 1.0, "B": 0.6, "C": 0.2, "D": -0.2}.get(quality, 0.0)

    def _empty_sr(self) -> Dict:
        # <30 bars = no S/R info → Grade D (probe size, 0.40×). Under GATE5-1
        # S/R never VETOES, so reward_risk_sr is the real trade R:R (2.5) and
        # the trade passes at reduced size rather than being blocked on no data.
        return {
            "nearest_support": 0, "support_distance_pct": 5.0,
            "support_strength": 1, "near_support": 0,
            "nearest_resistance": 0, "resistance_distance_pct": 5.0,
            "resistance_strength": 1, "near_resistance": 0,
            "second_support": 0, "second_resistance": 0,
            "entry_quality": "D", "entry_quality_short": "D",
            "entry_quality_score": -0.2, "atr_pct": 0.0,
            "reward_risk_sr": round(self.REAL_TRADE_RR, 2), "near_breakout": 0,
        }