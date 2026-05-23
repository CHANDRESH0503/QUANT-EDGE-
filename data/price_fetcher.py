# data/price_fetcher.py
# Fetches OHLCV price data for all three timeframes
# Sources: yfinance (free, auto-adjusted)
# Covers: 5-min (intraday), daily (swing), weekly (positional)

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

_DROP_COLS = {"Dividends", "Stock Splits", "Capital Gains"}


def yf_safe_download(
    ticker: str,
    period: str = "30d",
    interval: str = "1d",
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """
    Reliable replacement for yf.download() using Ticker.history().
    Uses Ticker.history() which is the v1.x preferred API with its own
    session/cookie handling.
    """
    try:
        t = yf.Ticker(ticker)
        kw: dict = dict(interval=interval, auto_adjust=True, prepost=False)
        if start:
            kw["start"] = start
            if end:
                kw["end"] = end
        else:
            kw["period"] = period
        df = t.history(**kw)
        if df.empty:
            return df
        drop = [c for c in df.columns if c in _DROP_COLS]
        if drop:
            df = df.drop(columns=drop)
        return df
    except Exception as e:
        logger.debug(f"yf_safe_download({ticker}): {e}")
        return pd.DataFrame()


class PriceFetcher:
    """
    Fetches and cleans price data for all three timeframes.

    Timeframes:
        - 5-min  : intraday model (last 60 days max from yfinance)
        - daily  : swing model    (25 years of history)
        - weekly : positional     (25 years of history)
    """

    def __init__(self, ticker: str = "HDFCBANK.NS"):
        self.ticker = ticker

    # ─────────────────────────────────────────────────────────────
    # PUBLIC METHODS
    # ─────────────────────────────────────────────────────────────

    def get_daily(self, start: str = "2000-01-01") -> pd.DataFrame:
        """
        Daily OHLCV — 25 years of history for swing model training.
        auto_adjust=True handles splits and dividends automatically.
        """
        try:
            df = yf_safe_download(self.ticker, start=start, interval="1d")
            df = self._clean(df)
            logger.info(f"Daily data fetched: {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch daily data: {e}")
            return pd.DataFrame()

    def get_weekly(self, start: str = "2000-01-01") -> pd.DataFrame:
        """
        Weekly OHLCV — for positional model training.
        """
        try:
            df = yf_safe_download(self.ticker, start=start, interval="1wk")
            df = self._clean(df)
            logger.info(f"Weekly data fetched: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch weekly data: {e}")
            return pd.DataFrame()

    def get_intraday(self, interval: str = "5m") -> pd.DataFrame:
        """
        Intraday OHLCV — 5-min candles, last 60 days max from yfinance.
        For more history use Zerodha Kite API.
        """
        try:
            df = yf_safe_download(self.ticker, period="60d", interval=interval)
            df = self._clean(df)
            # Keep only market hours 9:15 to 15:30 IST
            df = self._filter_market_hours(df)
            logger.info(f"Intraday ({interval}) data fetched: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch intraday data: {e}")
            return pd.DataFrame()

    def get_latest_daily(self, days: int = 60) -> pd.DataFrame:
        """
        Last N days of daily data — used for live predictions.
        """
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return self.get_daily(start=start)

    def get_latest_intraday(self) -> pd.DataFrame:
        """
        Today's 5-min candles — used for live intraday signals.
        Primary: Angel One SmartAPI (real-time).
        Fallback: yfinance (15-min delayed).
        """
        # Category 2: Angel One real-time intraday candles
        try:
            from data.angel_fetcher import get_candles, is_available
            if is_available():
                now      = datetime.now()
                from_dt  = now.replace(hour=9, minute=15, second=0, microsecond=0)
                # If before market open, use yesterday
                if now.hour < 9 or (now.hour == 9 and now.minute < 15):
                    from_dt -= timedelta(days=1)
                df = get_candles(
                    self.ticker,
                    interval="5m",
                    from_date=from_dt.strftime("%Y-%m-%d %H:%M"),
                    to_date=now.strftime("%Y-%m-%d %H:%M"),
                )
                if not df.empty:
                    df = self._clean(df)
                    logger.debug(f"Intraday (5m) from Angel One: {len(df)} rows")
                    return df
        except Exception as e:
            logger.debug(f"Angel One intraday failed for {self.ticker}, using yfinance: {e}")

        # Fallback: yfinance (15-min delayed for NSE)
        try:
            df = yf_safe_download(self.ticker, period="1d", interval="5m")
            df = self._clean(df)
            df = self._filter_market_hours(df)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch today's intraday data: {e}")
            return pd.DataFrame()

    def get_current_price(self) -> float:
        """
        Real-time current price.
        Primary: Angel One SmartAPI LTP (true real-time).
        Fallback: yfinance fast_info (15-min delayed for NSE).
        """
        # Category 1: Angel One real-time LTP
        try:
            from data.angel_fetcher import get_ltp, is_available
            if is_available():
                price = get_ltp(self.ticker)
                if price > 0:
                    logger.debug(f"Current price from Angel One: {self.ticker} = {price}")
                    return price
        except Exception as e:
            logger.debug(f"Angel One LTP failed for {self.ticker}, using yfinance: {e}")

        # Fallback: yfinance fast_info
        try:
            t    = yf.Ticker(self.ticker)
            data = t.fast_info
            return float(data.last_price)
        except Exception as e:
            logger.error(f"Failed to fetch current price for {self.ticker}: {e}")
            return 0.0

    def get_price_metadata(self) -> dict:
        """
        Day high, low, volume, 52-week high/low for dashboard.
        """
        try:
            df_daily = self.get_latest_daily(days=252)
            df_today = self.get_latest_intraday()

            current = float(df_daily["Close"].iloc[-1])
            prev_close = float(df_daily["Close"].iloc[-2])
            change = current - prev_close
            change_pct = (change / prev_close) * 100

            day_high = float(df_today["High"].max()) if not df_today.empty else current
            day_low = float(df_today["Low"].min()) if not df_today.empty else current
            day_volume = int(df_today["Volume"].sum()) if not df_today.empty else 0
            avg_volume = int(df_daily["Volume"].rolling(10).mean().iloc[-1])
            volume_ratio = day_volume / avg_volume if avg_volume > 0 else 1.0

            week52_high = float(df_daily["High"].rolling(252).max().iloc[-1])
            week52_low = float(df_daily["Low"].rolling(252).min().iloc[-1])

            return {
                "ticker":       self.ticker,
                "current":      current,
                "prev_close":   prev_close,
                "change":       change,
                "change_pct":   change_pct,
                "day_high":     day_high,
                "day_low":      day_low,
                "day_volume":   day_volume,
                "volume_ratio": round(volume_ratio, 2),
                "week52_high":  week52_high,
                "week52_low":   week52_low,
                "fetched_at":   str(datetime.now()),
            }
        except Exception as e:
            logger.error(f"Failed to fetch price metadata: {e}")
            return {}

    # ─────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standard cleaning:
        1. Forward-fill missing values (holidays etc.)
        2. Drop remaining NaNs
        3. Remove obvious data errors (>25% single-day move)
        4. Ensure float types
        """
        if df.empty:
            return df

        # Flatten multi-level columns if present (yfinance quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Forward fill
        df = df.ffill()
        df = df.dropna()

        # Remove impossible single-day moves (data errors, not circuit breakers)
        daily_returns = df["Close"].pct_change().abs()
        df = df[daily_returns < 0.26]

        # Ensure float
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = df[col].astype(float)

        return df

    def _filter_market_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only NSE market hours: 9:15 to 15:30 IST.
        yfinance returns UTC — convert to IST first.
        """
        if df.empty:
            return df

        try:
            # Convert to IST (UTC+5:30)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert("Asia/Kolkata")

            # Filter 9:15 to 15:30
            df = df.between_time("09:15", "15:30")
            df.index = df.index.tz_localize(None)  # remove tz for cleaner handling
        except Exception as e:
            logger.warning(f"Could not filter market hours: {e}")

        return df
