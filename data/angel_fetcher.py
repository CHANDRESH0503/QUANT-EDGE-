# data/angel_fetcher.py
# Angel One SmartAPI client — real-time price data for all 5 banks
#
# Replaces yfinance (15-min delayed NSE feed) for:
#   Category 1 — get_current_price()    : exit monitor, signal engine, Telegram
#   Category 2 — get_latest_intraday()  : 5-min OHLCV for ML feature builder
#   Category 3 — dashboard_api.yf_quote(): live prices shown on dashboard
#
# Falls back to yfinance automatically on any failure.
#
# REQUIRED .env keys:
#   ANGEL_API_KEY      — from Angel One developer portal
#   ANGEL_CLIENT_ID    — your trading login ID (e.g. AACH929531)
#   ANGEL_MPIN         — 4-digit MPIN
#   ANGEL_TOTP_SECRET  — Base32 TOTP key from Angel One app
#                        (Settings → My Profile → Security → TOTP Setup)
#                        Looks like: JBSWY3DPEHPK3PXP  (NOT a UUID)
#
# TOTP secret must be a Base32 string (no dashes) from Angel One app
# → My Profile → Security → Authenticator App → "Can't scan?" → copy the key.

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# NSE EQ tokens (from Angel One OpenAPIScripMaster, exch_seg=NSE, symbol ends -EQ)
ANGEL_TOKENS: Dict[str, str] = {
    "HDFCBANK.NS":   "1333",
    "ICICIBANK.NS":  "4963",
    "KOTAKBANK.NS":  "1922",
    "AXISBANK.NS":   "5900",
    "INDUSINDBK.NS": "5258",
}

# Angel One tradingsymbol (what ltpData / getCandleData expect)
ANGEL_SYMBOLS: Dict[str, str] = {
    "HDFCBANK.NS":   "HDFCBANK-EQ",
    "ICICIBANK.NS":  "ICICIBANK-EQ",
    "KOTAKBANK.NS":  "KOTAKBANK-EQ",
    "AXISBANK.NS":   "AXISBANK-EQ",
    "INDUSINDBK.NS": "INDUSINDBK-EQ",
}

# yfinance interval → Angel One interval
INTERVAL_MAP: Dict[str, str] = {
    "1m":  "ONE_MINUTE",
    "3m":  "THREE_MINUTE",
    "5m":  "FIVE_MINUTE",
    "10m": "TEN_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "1h":  "ONE_HOUR",
    "1d":  "ONE_DAY",
}

_session: Optional[object] = None
_session_ts: Optional[datetime] = None
_SESSION_TTL_SEC = 7 * 3600  # 7 hours; Angel One tokens expire ~8h


# ─────────────────────────────────────────────────────────────────────────────
# Session management
# ─────────────────────────────────────────────────────────────────────────────

def _is_valid() -> bool:
    if _session is None or _session_ts is None:
        return False
    return (datetime.now() - _session_ts).total_seconds() < _SESSION_TTL_SEC


def _create_session() -> object:
    """Authenticate and cache an Angel One SmartConnect session."""
    import pyotp
    from SmartApi import SmartConnect
    global _session, _session_ts

    api_key     = os.getenv("ANGEL_API_KEY",    "").strip()
    client_id   = os.getenv("ANGEL_CLIENT_ID",  "").strip()
    mpin        = os.getenv("ANGEL_MPIN",       "").strip()
    totp_secret = os.getenv("ANGEL_TOTP_SECRET","").strip()

    if not all([api_key, client_id, mpin]):
        raise ValueError("ANGEL_API_KEY / ANGEL_CLIENT_ID / ANGEL_MPIN missing in .env")

    if totp_secret and "-" in totp_secret:
        raise ValueError(
            "ANGEL_TOTP_SECRET looks like a UUID — use the Base32 key from "
            "Angel One app → My Profile → Security → TOTP Setup → 'Can't scan?'"
        )

    totp = pyotp.TOTP(totp_secret).now() if totp_secret else ""
    obj  = SmartConnect(api_key=api_key)
    resp = obj.generateSession(client_id, mpin, totp)

    if not resp.get("status"):
        raise RuntimeError(
            f"Angel One auth failed: {resp.get('message')} "
            f"[{resp.get('errorcode')}]"
        )

    _session    = obj
    _session_ts = datetime.now()
    logger.info("Angel One: session created OK")
    return obj


def get_session() -> object:
    """Return active session, refreshing if TTL expired."""
    if not _is_valid():
        return _create_session()
    return _session


def invalidate_session() -> None:
    global _session, _session_ts
    _session    = None
    _session_ts = None


def is_available() -> bool:
    """
    True when all env vars are present, TOTP secret is valid Base32,
    and the required packages (pyotp, SmartApi) are installed.
    """
    try:
        import pyotp  # noqa
        from SmartApi import SmartConnect  # noqa
    except ImportError:
        return False
    key    = os.getenv("ANGEL_API_KEY",    "").strip()
    cid    = os.getenv("ANGEL_CLIENT_ID",  "").strip()
    mpin   = os.getenv("ANGEL_MPIN",       "").strip()
    secret = os.getenv("ANGEL_TOTP_SECRET","").strip()
    return bool(key and cid and mpin and secret and "-" not in secret)


# ─────────────────────────────────────────────────────────────────────────────
# Category 1 — Real-time LTP
# ─────────────────────────────────────────────────────────────────────────────

def get_ltp(ticker: str) -> float:
    """
    Real-time Last Traded Price for a single ticker.
    Raises RuntimeError on failure so caller can fall back to yfinance.
    """
    token  = ANGEL_TOKENS.get(ticker)
    symbol = ANGEL_SYMBOLS.get(ticker)
    if not token:
        raise ValueError(f"No Angel One token configured for {ticker}")

    def _call(obj) -> float:
        resp = obj.ltpData("NSE", symbol, token)
        if resp.get("status") and resp.get("data"):
            price = float(resp["data"]["ltp"])
            if price > 0:
                return price
        raise RuntimeError(f"ltpData bad response for {ticker}: {resp}")

    try:
        return _call(get_session())
    except RuntimeError:
        raise
    except Exception:
        invalidate_session()
        return _call(get_session())  # single retry with fresh session


def get_all_ltps() -> Dict[str, float]:
    """
    Batch real-time LTP for all 5 banks in one API call.
    Returns {ticker: price}. Any missing ticker is absent from result.
    """
    try:
        obj  = get_session()
        resp = obj.getMarketData("LTP", {"NSE": list(ANGEL_TOKENS.values())})
        if resp.get("status") and resp.get("data"):
            tok_to_ticker = {v: k for k, v in ANGEL_TOKENS.items()}
            result: Dict[str, float] = {}
            for item in resp["data"].get("fetched", []):
                tok = item.get("symbolToken")
                ltp = item.get("ltp")
                t   = tok_to_ticker.get(tok)
                if t and ltp:
                    result[t] = float(ltp)
            if result:
                return result
        logger.warning("get_all_ltps: batch response empty, falling back to individual calls")
    except Exception as e:
        logger.warning(f"get_all_ltps batch failed: {e}, falling back to individual calls")

    result = {}
    for ticker in ANGEL_TOKENS:
        try:
            result[ticker] = get_ltp(ticker)
        except Exception as ex:
            logger.warning(f"get_ltp({ticker}) failed: {ex}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Category 2 — Intraday OHLCV candles
# ─────────────────────────────────────────────────────────────────────────────

def get_candles(
    ticker: str,
    interval: str = "5m",
    from_date: Optional[str] = None,
    to_date:   Optional[str] = None,
) -> pd.DataFrame:
    """
    Intraday OHLCV candles from Angel One.
    Returns DataFrame with columns [Open, High, Low, Close, Volume]
    and a timezone-naive DatetimeIndex (IST), matching PriceFetcher output.

    interval: yfinance style ('1m', '5m', '15m', '1h', '1d').
    from_date / to_date: 'YYYY-MM-DD HH:MM' strings. Defaults to today's session.
    """
    token = ANGEL_TOKENS.get(ticker)
    if not token:
        raise ValueError(f"No Angel One token configured for {ticker}")

    angel_ivl = INTERVAL_MAP.get(interval, "FIVE_MINUTE")

    now = datetime.now()
    if from_date is None:
        from_dt   = now.replace(hour=9, minute=15, second=0, microsecond=0)
        from_date = from_dt.strftime("%Y-%m-%d %H:%M")
    if to_date is None:
        to_date = now.strftime("%Y-%m-%d %H:%M")

    obj  = get_session()
    resp = obj.getCandleData({
        "exchange":    "NSE",
        "symboltoken": token,
        "interval":    angel_ivl,
        "fromdate":    from_date,
        "todate":      to_date,
    })

    if not resp.get("status") or not resp.get("data"):
        raise RuntimeError(f"getCandleData failed for {ticker}: {resp}")

    # Angel One returns [[ts, o, h, l, c, v], ...]
    rows = resp["data"]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    if df["Datetime"].dt.tz is not None:
        df["Datetime"] = df["Datetime"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.set_index("Datetime")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)
    return df
