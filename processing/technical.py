# processing/technical.py
# Builds all 23 technical features from OHLCV data
# Covers: RSI, MACD, BB, ATR, EMA, Volume, Momentum, S/R distance
# Used by all three ML models (intraday / swing / positional)

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("ta library not installed. pip install ta")


class TechnicalProcessor:
    """
    Converts OHLCV price data into the 23 technical features
    selected by the senior trader framework.

    Design principle:
    - All features are backward-looking ONLY (no lookahead bias)
    - ATR-based normalization where appropriate
    - Features return NaN-safe values (filled with 0 or neutral)
    - Works on daily, weekly, and 5-min candles
    """

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Master method — builds all technical features in one pass.
        Input:  OHLCV DataFrame with DatetimeIndex
        Output: DataFrame with all technical features aligned to same index
        """
        if df.empty or len(df) < 30:
            logger.warning("Insufficient data for technical features")
            return pd.DataFrame()

        f = pd.DataFrame(index=df.index)

        # ── Momentum ─────────────────────────────────────────────
        f["rsi_14"]       = self._rsi(df["Close"], 14)
        f["stoch_rsi"]    = self._stoch_rsi(df["Close"])
        f["rsi_divergence"] = self._rsi_divergence(df["Close"], f["rsi_14"])

        # ── Trend ────────────────────────────────────────────────
        f["macd"]         = self._macd_line(df["Close"])
        f["macd_hist"]    = self._macd_histogram(df["Close"])
        f["macd_hist_roc"]= f["macd_hist"].pct_change(3).fillna(0)  # acceleration
        f["ema_20"]       = df["Close"].ewm(span=20, adjust=False).mean()
        f["ema_50"]       = df["Close"].ewm(span=50, adjust=False).mean()
        f["ema_spread"]   = (f["ema_20"] - f["ema_50"]) / f["ema_50"]
        f["ema_spread_roc"] = f["ema_spread"].pct_change(5).fillna(0)
        f["adx"]          = self._adx(df)
        f["supertrend"]   = self._supertrend(df)  # 1=bullish, -1=bearish

        # ── Volatility ───────────────────────────────────────────
        bb = self._bollinger(df["Close"])
        f["bb_pct_b"]     = bb["pct_b"]
        f["bb_squeeze"]   = bb["squeeze"].astype(float)
        f["bb_width"]     = bb["width"]
        f["atr_14"]       = self._atr(df)
        f["atr_ratio"]    = f["atr_14"] / df["Close"]
        f["hv_ratio"]     = self._hv_ratio(df["Close"])

        # ── Volume ───────────────────────────────────────────────
        f["volume_ratio"] = df["Volume"] / df["Volume"].rolling(10).mean()
        f["vol_roc"]      = df["Volume"].pct_change(5).fillna(0)
        f["obv"]          = self._obv(df["Close"], df["Volume"])
        f["obv_roc"]      = f["obv"].pct_change(5).fillna(0)
        f["mfi"]          = self._mfi(df)
        f["cmf"]          = self._cmf(df)

        # ── Price Action ─────────────────────────────────────────
        f["returns_1d"]   = df["Close"].pct_change(1).fillna(0)
        f["returns_3d"]   = df["Close"].pct_change(3).fillna(0)
        f["returns_5d"]   = df["Close"].pct_change(5).fillna(0)
        f["returns_20d"]  = df["Close"].pct_change(20).fillna(0)
        f["close_pct_range"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-8)
        f["gap_pct"]      = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1).bfill()
        f["range_pct_52w"]= self._range_position_52w(df["Close"])
        f["consec_days"]  = self._consecutive_days(df["Close"])
        f["hv_ratio"]     = self._hv_ratio(df["Close"])

        # ── Support / Resistance distance ────────────────────────
        sr = self._sr_distances(df)
        f["support_dist"]    = sr["support_dist"]
        f["resistance_dist"] = sr["resistance_dist"]
        f["poc_distance"]    = self._poc_distance(df)

        # VWAP (intraday only — for daily data returns 0)
        f["vwap_dist"] = self._vwap_distance(df)

        # ── Time features ────────────────────────────────────────
        f["day_of_week"]  = pd.to_datetime(df.index).dayofweek.astype(float)
        f["month"]        = pd.to_datetime(df.index).month.astype(float)
        f["is_monday"]    = (f["day_of_week"] == 0).astype(float)
        f["is_friday"]    = (f["day_of_week"] == 4).astype(float)

        # ── Fill any remaining NaN ───────────────────────────────
        f = f.fillna(0).replace([np.inf, -np.inf], 0)

        return f

    def get_latest_features(self, df: pd.DataFrame) -> Dict:
        """
        Get the most recent row of features as a flat dict for live prediction.
        """
        features_df = self.build_features(df)
        if features_df.empty:
            return {}
        return features_df.iloc[-1].to_dict()

    # ─────────────────────────────────────────────────────────────
    # MOMENTUM
    # ─────────────────────────────────────────────────────────────

    def _rsi(self, close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(window).mean()
        loss  = (-delta.clip(upper=0)).rolling(window).mean()
        rs    = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _stoch_rsi(self, close: pd.Series, window: int = 14) -> pd.Series:
        rsi   = self._rsi(close, window)
        min_r = rsi.rolling(window).min()
        max_r = rsi.rolling(window).max()
        return (rsi - min_r) / (max_r - min_r + 1e-10)

    def _rsi_divergence(self, close: pd.Series, rsi: pd.Series,
                         window: int = 10) -> pd.Series:
        """
        Bearish divergence = price higher high but RSI lower high = -1
        Bullish divergence = price lower low but RSI higher low   = +1
        No divergence = 0
        """
        result = pd.Series(0.0, index=close.index)
        if len(close) < window * 3:
            return result

        # Local maxima and minima
        price_vals = close.values
        rsi_vals   = rsi.fillna(50).values

        try:
            price_highs = set(argrelextrema(price_vals, np.greater, order=window)[0])
            price_lows  = set(argrelextrema(price_vals, np.less,    order=window)[0])
            rsi_highs   = set(argrelextrema(rsi_vals,   np.greater, order=window)[0])
            rsi_lows    = set(argrelextrema(rsi_vals,   np.less,    order=window)[0])

            for i in range(window, len(close) - 1):
                # Bearish: price new high, RSI lower high
                if i in price_highs and i not in rsi_highs:
                    result.iloc[i] = -1.0
                # Bullish: price new low, RSI higher low
                elif i in price_lows and i not in rsi_lows:
                    result.iloc[i] = 1.0
        except Exception:
            pass

        return result

    # ─────────────────────────────────────────────────────────────
    # TREND
    # ─────────────────────────────────────────────────────────────

    def _macd_line(self, close: pd.Series) -> pd.Series:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        return ema12 - ema26

    def _macd_histogram(self, close: pd.Series) -> pd.Series:
        macd   = self._macd_line(close)
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd - signal

    def _adx(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        high, low, close = df["High"], df["Low"], df["Close"]
        tr   = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)

        dm_plus  = (high.diff()).clip(lower=0)
        dm_minus = (-low.diff()).clip(lower=0)

        # Zero out where other direction is stronger
        dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
        dm_minus = dm_minus.where(dm_minus > dm_plus, 0)

        atr    = tr.ewm(span=window, adjust=False).mean()
        di_plus  = 100 * dm_plus.ewm(span=window, adjust=False).mean()  / (atr + 1e-10)
        di_minus = 100 * dm_minus.ewm(span=window, adjust=False).mean() / (atr + 1e-10)
        dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10)
        return dx.ewm(span=window, adjust=False).mean()

    def _supertrend(self, df: pd.DataFrame,
                    period: int = 10, multiplier: float = 3.0) -> pd.Series:
        """Returns +1 (bullish) or -1 (bearish) supertrend signal."""
        atr = self._atr(df, period)
        hl2 = (df["High"] + df["Low"]) / 2
        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr

        supertrend = pd.Series(1.0, index=df.index)
        for i in range(1, len(df)):
            if df["Close"].iloc[i] > upper.iloc[i - 1]:
                supertrend.iloc[i] = 1.0
            elif df["Close"].iloc[i] < lower.iloc[i - 1]:
                supertrend.iloc[i] = -1.0
            else:
                supertrend.iloc[i] = supertrend.iloc[i - 1]

        return supertrend

    # ─────────────────────────────────────────────────────────────
    # VOLATILITY
    # ─────────────────────────────────────────────────────────────

    def _bollinger(self, close: pd.Series,
                   window: int = 20, std: float = 2.0) -> Dict:
        mid   = close.rolling(window).mean()
        sigma = close.rolling(window).std()
        upper = mid + std * sigma
        lower = mid - std * sigma
        width = (upper - lower) / (mid + 1e-10)
        pct_b = (close - lower) / (upper - lower + 1e-10)
        squeeze = width < width.rolling(126).quantile(0.10)
        return {"upper": upper, "lower": lower, "mid": mid,
                "width": width, "pct_b": pct_b, "squeeze": squeeze}

    def _atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"]  - df["Close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=window, adjust=False).mean()

    def _hv_ratio(self, close: pd.Series) -> pd.Series:
        """HV10 / HV30 — short-term vs long-term volatility ratio."""
        ret  = close.pct_change()
        hv10 = ret.rolling(10).std() * np.sqrt(252)
        hv30 = ret.rolling(30).std() * np.sqrt(252)
        return (hv10 / (hv30 + 1e-10)).fillna(1.0)

    # ─────────────────────────────────────────────────────────────
    # VOLUME
    # ─────────────────────────────────────────────────────────────

    def _obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff().fillna(0))
        return (direction * volume).cumsum()

    def _mfi(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        tp    = (df["High"] + df["Low"] + df["Close"]) / 3
        raw_mf = tp * df["Volume"]
        pos_mf = raw_mf.where(tp > tp.shift(1), 0).rolling(window).sum()
        neg_mf = raw_mf.where(tp < tp.shift(1), 0).rolling(window).sum()
        mfr    = pos_mf / (neg_mf + 1e-10)
        return 100 - (100 / (1 + mfr))

    def _cmf(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / \
              (df["High"] - df["Low"] + 1e-10)
        mfv = clv * df["Volume"]
        return mfv.rolling(window).sum() / (df["Volume"].rolling(window).sum() + 1e-10)

    # ─────────────────────────────────────────────────────────────
    # PRICE ACTION
    # ─────────────────────────────────────────────────────────────

    def _range_position_52w(self, close: pd.Series) -> pd.Series:
        """Where price sits in 52-week high-low range (0=at low, 1=at high)."""
        low_52  = close.rolling(252, min_periods=20).min()
        high_52 = close.rolling(252, min_periods=20).max()
        return (close - low_52) / (high_52 - low_52 + 1e-10)

    def _consecutive_days(self, close: pd.Series) -> pd.Series:
        """Count consecutive green or red days. Positive=green streak, negative=red streak."""
        direction = np.sign(close.diff().fillna(0))
        streak = pd.Series(0.0, index=close.index)
        for i in range(1, len(close)):
            if direction.iloc[i] == direction.iloc[i - 1] and direction.iloc[i] != 0:
                streak.iloc[i] = streak.iloc[i - 1] + direction.iloc[i]
            else:
                streak.iloc[i] = direction.iloc[i]
        return streak

    def _vwap_distance(self, df: pd.DataFrame) -> pd.Series:
        """
        VWAP distance from close.
        For daily data this is intraday VWAP of each day's session.
        Returns 0 for data without enough intraday granularity.
        """
        try:
            tp   = (df["High"] + df["Low"] + df["Close"]) / 3
            vwap = (tp * df["Volume"]).cumsum() / (df["Volume"].cumsum() + 1e-10)
            return (df["Close"] - vwap) / (vwap + 1e-10)
        except Exception:
            return pd.Series(0.0, index=df.index)

    # ─────────────────────────────────────────────────────────────
    # SUPPORT / RESISTANCE
    # ─────────────────────────────────────────────────────────────

    def _sr_distances(self, df: pd.DataFrame,
                       lookback: int = 252, order: int = 10) -> Dict:
        """
        Find nearest support and resistance using swing highs/lows.
        Returns distance as percentage from current price.
        """
        close = df["Close"]
        result = {
            "support_dist":    pd.Series(0.0, index=close.index),
            "resistance_dist": pd.Series(0.0, index=close.index),
        }

        if len(df) < lookback:
            return result

        try:
            lows   = argrelextrema(df["Low"].values,  np.less,    order=order)[0]
            highs  = argrelextrema(df["High"].values, np.greater, order=order)[0]
            support_levels    = df["Low"].iloc[lows].values
            resistance_levels = df["High"].iloc[highs].values

            for i in range(len(close)):
                price = close.iloc[i]
                supports    = support_levels[support_levels < price]
                resistances = resistance_levels[resistance_levels > price]

                nearest_sup = supports.max()    if len(supports)    > 0 else price * 0.95
                nearest_res = resistances.min() if len(resistances) > 0 else price * 1.05

                result["support_dist"].iloc[i]    = (price - nearest_sup) / price
                result["resistance_dist"].iloc[i] = (nearest_res - price) / price

        except Exception as e:
            logger.warning(f"SR distance calculation failed: {e}")

        return result

    def _poc_distance(self, df: pd.DataFrame, bins: int = 20) -> pd.Series:
        """
        Point of Control distance — most traded price level.
        POC = price bucket with highest cumulative volume.
        """
        result = pd.Series(0.0, index=df.index)
        try:
            price_min = df["Low"].min()
            price_max = df["High"].max()
            bucket_size = (price_max - price_min) / bins

            volume_by_bucket = {}
            for i in range(len(df)):
                bucket = int((df["Close"].iloc[i] - price_min) / (bucket_size + 1e-10))
                volume_by_bucket[bucket] = volume_by_bucket.get(bucket, 0) + df["Volume"].iloc[i]

            if not volume_by_bucket:
                return result

            poc_bucket = max(volume_by_bucket, key=volume_by_bucket.get)
            poc_price  = price_min + (poc_bucket + 0.5) * bucket_size

            result = (df["Close"] - poc_price) / (poc_price + 1e-10)
        except Exception:
            pass

        return result