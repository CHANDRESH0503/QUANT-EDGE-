# dashboard_api.py
# FastAPI backend that serves live data to the trading dashboard
# Run: uvicorn dashboard_api:app --host 0.0.0.0 --port 8000 --reload
# Dashboard fetches from: http://localhost:8000/api/live?ticker=HDFCBANK.NS

# ── Prevent loky/multiprocessing semaphore leak on Python 3.14 / macOS ───────
# Must be set BEFORE any transformers / tokenizers import.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import sqlite3
import logging
import json
import math
import os
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import pandas as pd
    import yfinance as _yf
    from data.price_fetcher import yf_safe_download
    YF_OK = True
except ImportError:
    YF_OK = False

logger = logging.getLogger(__name__)

# NSE market schedule — single shared instance (thread-safe reads)
from scheduler.market_schedule import MarketSchedule as _MarketSchedule
_market_schedule = _MarketSchedule()

# In-memory cache: last known good quote per ticker
_price_cache: Dict[str, dict] = {}

# ── Zero-fill sentinel returned when all data sources fail ────────────────────
_ZERO_QUOTE: Dict[str, Any] = {
    "price": 0, "prev": 0, "change": 0, "change_pct": 0,
    "high": 0, "low": 0, "volume": 0, "vol_ratio": 1,
}

# ── Batch quote cache — avoids serial yf calls on every /api/live hit ────────
_quotes_cache: Dict[str, dict] = {}
_quotes_cache_ts: float = 0.0
_QUOTES_TTL        = 15    # seconds when market is open
_QUOTES_TTL_CLOSED = 300   # 5 min between refreshes when market is closed

_week52_cache: Dict[str, dict] = {}  # {ticker: {"high": float, "ts": float}}
_WEEK52_TTL = 3600  # 1 hour

_UNIVERSE_SYMS = ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS"]
_STRIP_SYMS    = ["^NSEBANK", "^NSEI", "^INDIAVIX", "USDINR=X", "CL=F", "GC=F", "^GSPC"]
_ALL_BATCH_SYMS = _UNIVERSE_SYMS + _STRIP_SYMS


def _fetch_one_yf(ticker: str) -> tuple:
    """Fetch 5-day daily history for one ticker via yfinance. Thread-safe."""
    try:
        import yfinance as _yf_t
        df = _yf_t.Ticker(ticker).history(
            period="5d", interval="1d", auto_adjust=True, prepost=False
        )
        df = df.dropna(subset=["Close"])
        if len(df) >= 2:
            curr = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            chg  = curr - prev
            chgp = (chg / prev * 100) if prev else 0
            vol  = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0
            avgv = max(float(df["Volume"].mean()) if "Volume" in df.columns else 1, 1)
            return ticker, {
                "price": round(curr, 2), "prev": round(prev, 2),
                "change": round(chg, 2), "change_pct": round(chgp, 2),
                "high":  round(float(df["High"].iloc[-1]), 2),
                "low":   round(float(df["Low"].iloc[-1]),  2),
                "volume": int(vol), "vol_ratio": round(vol / avgv, 2),
            }
    except Exception:
        pass
    return ticker, dict(_ZERO_QUOTE)


def _fetch_quotes_batch() -> Dict[str, dict]:
    """
    Parallel-fetch quotes for all 12 tickers in one shot.
    During market hours: refreshes every 15s + overlays Angel One real-time LTP.
    Outside market hours: refreshes every 5 min (last traded price from yfinance).
    Angel One LTP is never called outside market hours.
    """
    global _quotes_cache, _quotes_cache_ts
    now_ts   = time.time()
    is_open  = _market_schedule.is_market_open()
    ttl      = _QUOTES_TTL if is_open else _QUOTES_TTL_CLOSED

    if now_ts - _quotes_cache_ts < ttl and _quotes_cache:
        return _quotes_cache

    result: Dict[str, dict] = {t: dict(_ZERO_QUOTE) for t in _ALL_BATCH_SYMS}

    # Step 1 — parallel yfinance (prev_close + OHLCV for all tickers)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_fetch_one_yf, t): t for t in _ALL_BATCH_SYMS}
        for fut in as_completed(futs, timeout=12):
            try:
                ticker, q = fut.result(timeout=0)
                if q["price"] > 0:
                    result[ticker] = q
            except Exception:
                pass

    # Step 2 — Angel One real-time LTP only during market hours
    if is_open:
        try:
            from data.angel_fetcher import get_all_ltps, is_available
            if is_available():
                ltps = get_all_ltps()
                for t, ltp in ltps.items():
                    if t in result and ltp > 0:
                        prev = result[t].get("prev") or ltp
                        chg  = round(ltp - prev, 2)
                        chgp = round((chg / prev * 100) if prev else 0, 2)
                        result[t].update({"price": round(ltp, 2), "change": chg, "change_pct": chgp})
        except Exception as e:
            logger.debug(f"Angel One batch LTP override failed: {e}")

    _quotes_cache = result
    _quotes_cache_ts = now_ts
    return result


def _get_week52_high(ticker: str) -> float:
    """52-week high, cached per ticker for 1 hour."""
    now_ts = time.time()
    cached = _week52_cache.get(ticker, {})
    if cached and now_ts - cached.get("ts", 0) < _WEEK52_TTL:
        return cached["high"]
    high = 0.0
    try:
        df = yf_safe_download(ticker, period="1y")
        if not df.empty:
            high = round(float(df["High"].max()), 2)
    except Exception:
        pass
    _week52_cache[ticker] = {"high": high, "ts": now_ts}
    return high


_news_fetch_lock = threading.Lock()
_news_fetch_active = False

def _trigger_news_fetch_bg(db_path: str = "database/trading.db") -> None:
    """Fetch + score news in a background thread.

    Two independent triggers:
      1. Stale news (>15 min since last insert) → fetch new articles
      2. Any rows with processed=0 → score them through FinBERT
    Either trigger alone is sufficient — we used to skip scoring on the
    "still fresh" branch, which left freshly-inserted articles stuck at
    sentiment_score=0.0 indefinitely.

    Non-blocking — returns immediately. At most one task runs at a time.
    """
    global _news_fetch_active
    if _news_fetch_active:
        return

    needs_fetch = True
    needs_score = False
    try:
        conn = sqlite3.connect(db_path)
        row  = conn.execute("SELECT MAX(created_at) FROM news").fetchone()
        unscored = conn.execute(
            "SELECT COUNT(*) FROM news WHERE processed = 0"
        ).fetchone()
        conn.close()
        if row and row[0]:
            last = datetime.fromisoformat(str(row[0])[:19])
            if (datetime.now() - last).total_seconds() < 900:
                needs_fetch = False
        needs_score = bool(unscored and unscored[0])
    except Exception:
        return
    if not needs_fetch and not needs_score:
        return

    def _run():
        global _news_fetch_active
        with _news_fetch_lock:
            _news_fetch_active = True
            try:
                if needs_fetch:
                    from data.news_fetcher import NewsFetcher
                    NewsFetcher.fetch_all_banks(db_path=db_path)
                # Always score after fetching, and also score on a pure
                # "scoring is behind" trigger when fetch was skipped.
                from processing.finbert_sentiment import FinBERTSentiment
                FinBERTSentiment(db_path=db_path).process_unscored(batch_size=100)
                logger.info("Background news fetch + score complete")
            except Exception as e:
                logger.warning(f"Background news fetch failed: {e}")
            finally:
                _news_fetch_active = False

    threading.Thread(target=_run, daemon=True).start()


app = FastAPI(title="QUANT EDGE API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_TICKERS = [
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "INDUSINDBK.NS",
]

# Tickers that have Angel One tokens configured
_ANGEL_TICKERS = set(VALID_TICKERS)


@app.get("/")
async def serve_dashboard():
    """Serve the dashboard HTML so fetch() works same-origin (no file:// CORS block)."""
    return FileResponse("tradingDashboard.html", media_type="text/html")

DB_PATH = "database/trading.db"

TICKERS = {
    "HDFCBANK":  "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "AXISBANK":  "AXISBANK.NS",
    "SBIN":      "SBIN.NS",
    "INDUSINDBK":"INDUSINDBK.NS",
    "BANKNIFTY": "^NSEBANK",
    "NIFTY50":   "^NSEI",
    "INDIAVIX":  "^INDIAVIX",
    "USDINR":    "USDINR=X",
    "CRUDE":     "CL=F",
    "GOLD":      "GC=F",
    "SP500":     "^GSPC",
}


def db_query(sql: str, params=()) -> list:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"DB query failed: {e}")
        return []


def db_one(sql: str, params=()) -> dict:
    rows = db_query(sql, params)
    return rows[0] if rows else {}


def safe_float(v, default=0.0) -> float:
    try:
        f = float(v)
        return f if (f == f and abs(f) < 1e15) else default
    except (TypeError, ValueError):
        return default


def _clean_json(obj):
    """
    Recursively walk the response tree and sanitize values that would make
    json.dumps() raise ``ValueError: Out of range float values are not JSON
    compliant``.  Specifically:
      - float nan / inf / -inf  → 0.0
      - numpy scalar types      → native Python int / float (then re-checked)
      - everything else         → unchanged
    Applied to the full response dict before JSONResponse() so a single NaN
    anywhere in a nested calculation cannot bring down the entire endpoint.
    """
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(v) for v in obj]
    # numpy scalars — convert first, then fall through to float/int check
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        obj = obj.item()          # → Python float / int / bool
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    if isinstance(obj, int):
        # guard against Python 2 compat (irrelevant here, but be safe)
        return obj
    return obj


def _build_quote(price: float, prev: float, high: float, low_: float,
                  vol: float, avg_vol: float) -> dict:
    change  = price - prev
    chg_pct = (change / prev * 100) if prev else 0
    return {
        "price":     round(price, 2),
        "prev":      round(prev, 2),
        "change":    round(change, 2),
        "change_pct":round(chg_pct, 2),
        "high":      round(high, 2),
        "low":       round(low_, 2),
        "volume":    int(vol),
        "vol_ratio": round(vol / max(avg_vol, 1), 2),
    }


def yf_quote(ticker: str, period: str = "2d") -> dict:
    """
    Category 3 — Live price for dashboard display.
    Priority: Angel One LTP → yfinance history → yfinance fast_info → cache.
    Angel One is real-time; yfinance is 15-min delayed for NSE.
    """
    # ── Priority 1: Angel One real-time LTP ──────────────────────
    # For bank tickers use batch cache first (populated by _fetch_quotes_batch)
    if ticker in _ANGEL_TICKERS and _quotes_cache.get(ticker, {}).get("price", 0) > 0:
        return dict(_quotes_cache[ticker])

    try:
        from data.angel_fetcher import get_ltp, is_available
        if is_available() and ticker in _ANGEL_TICKERS:
            ltp = get_ltp(ticker)
            if ltp > 0:
                cached = _price_cache.get(ticker, {})
                prev_c = cached.get("prev", ltp)
                high_  = cached.get("high", ltp)
                low__  = cached.get("low",  ltp)
                vol    = cached.get("volume", 0)
                avg_v  = max(vol, 1)
                result = _build_quote(ltp, prev_c, high_, low__, vol, avg_v)
                _price_cache[ticker] = result
                return result
    except Exception as e:
        logger.debug(f"Angel One quote failed for {ticker}: {e}")

    if not YF_OK:
        return _price_cache.get(ticker, {})

    # ── Priority 2: yfinance Ticker.history() ────────────────────
    try:
        df = yf_safe_download(ticker, period=period)
        if not df.empty and len(df) >= 1:
            close   = df["Close"]
            price   = float(close.iloc[-1])
            prev    = float(close.iloc[-2]) if len(close) > 1 else price
            high    = float(df["High"].iloc[-1])
            low_    = float(df["Low"].iloc[-1])
            vol     = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0
            avg_vol = float(df["Volume"].mean())   if "Volume" in df.columns else 1
            if price > 0:
                result = _build_quote(price, prev, high, low_, vol, avg_vol)
                _price_cache[ticker] = result
                return result
    except Exception as e:
        logger.warning(f"yf_quote history failed {ticker}: {e}")

    # ── Priority 3: yfinance fast_info ───────────────────────────
    try:
        t   = _yf.Ticker(ticker)
        fi  = t.fast_info
        last = float(fi.last_price or 0)
        if last > 0:
            prev_c = float(fi.previous_close or last)
            high_  = float(fi.day_high  or last)
            low__  = float(fi.day_low   or last)
            result = _build_quote(last, prev_c, high_, low__, 0, 1)
            _price_cache[ticker] = result
            return result
    except Exception as e:
        logger.warning(f"yf_quote fast_info failed {ticker}: {e}")

    # ── Priority 4: stale cache, then zero dict ───────────────────
    cached = _price_cache.get(ticker, {})
    if cached:
        logger.info(f"yf_quote returning stale cache for {ticker}")
        return cached
    return dict(_ZERO_QUOTE)


def yf_history(ticker: str, period: str = "1d",
               interval: str = "5m") -> list:
    """
    OHLC candles for chart rendering.
    For intraday (period=1d/5d, interval=5m/15m) on bank tickers:
      Primary: Angel One candles (real-time).
      Fallback: yfinance (15-min delayed).
    For longer periods (1mo daily): yfinance only.
    """
    _is_intraday = interval in ("1m", "5m", "15m", "30m")

    # ── Angel One: intraday candles for bank tickers ──────────────
    if _is_intraday and ticker in _ANGEL_TICKERS:
        try:
            from data.angel_fetcher import get_candles, is_available
            if is_available():
                now = datetime.now()
                if period in ("1d",):
                    from_dt = now.replace(hour=9, minute=15, second=0, microsecond=0)
                else:
                    days = {"5d": 5, "1wk": 5}.get(period, 5)
                    from_dt = (now - timedelta(days=days)).replace(
                        hour=9, minute=15, second=0, microsecond=0)
                df = get_candles(
                    ticker,
                    interval=interval,
                    from_date=from_dt.strftime("%Y-%m-%d %H:%M"),
                    to_date=now.strftime("%Y-%m-%d %H:%M"),
                )
                if not df.empty:
                    return [
                        {
                            "t": str(idx)[:16],
                            "o": round(float(r["Open"]),  2),
                            "h": round(float(r["High"]),  2),
                            "l": round(float(r["Low"]),   2),
                            "c": round(float(r["Close"]), 2),
                            "v": int(r.get("Volume", 0)),
                        }
                        for idx, r in df.iterrows()
                    ]
        except Exception as e:
            logger.debug(f"Angel One chart candles failed for {ticker}: {e}")

    # ── Fallback: yfinance ────────────────────────────────────────
    if not YF_OK:
        return []
    try:
        df = yf_safe_download(ticker, period=period, interval=interval)
        if df.empty:
            return []
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("Asia/Kolkata")
        if _is_intraday:
            df = df.between_time("09:15", "15:30")
        return [
            {
                "t": str(idx)[:16],
                "o": round(float(r["Open"]),  2),
                "h": round(float(r["High"]),  2),
                "l": round(float(r["Low"]),   2),
                "c": round(float(r["Close"]), 2),
                "v": int(r.get("Volume", 0)),
            }
            for idx, r in df.iterrows()
        ]
    except Exception as e:
        logger.warning(f"yfinance history error {ticker}: {e}")
        return []


# ══════════════════════════════════════════════════════════════
# MASTER LIVE ENDPOINT
# ══════════════════════════════════════════════════════════════

@app.get("/api/live")
async def live_data(
    ticker: str = Query(default="HDFCBANK.NS"),
) -> JSONResponse:
    """
    Single endpoint that returns ALL dashboard data.
    Dashboard polls this every 15 seconds.
    Pass ?ticker=ICICIBANK.NS to switch banks.
    """
    if ticker not in VALID_TICKERS:
        ticker = "HDFCBANK.NS"
    ticker_safe = ticker.replace(".NS", "")

    # `now` stays naive (server local time) for all timedelta / DB comparisons.
    # `_now_ist` is IST-aware, used ONLY for the ts/ist display fields so the
    # dashboard always shows Kolkata time regardless of VPS timezone (UTC).
    import pytz as _pytz
    now = datetime.now()
    _now_ist = datetime.now(_pytz.timezone("Asia/Kolkata"))

    # ── 1. Batch-fetch all 12 market quotes in parallel (cached 15s) ──
    # Also kicks off the Angel One batch LTP call for all 5 banks.
    all_quotes = _fetch_quotes_batch()
    bank_quote = all_quotes.get(ticker, dict(_ZERO_QUOTE))
    bank_quote["week52_high"] = _get_week52_high(ticker)

    # ── 2. Latest signal from DB (ticker-scoped) ───────────────
    sig_row = db_one("""
        SELECT signal, confidence, alignment, regime,
               created_at
        FROM   signal_outcomes
        WHERE  ticker = ?
        ORDER  BY created_at DESC LIMIT 1
    """, (ticker,))
    if not sig_row:
        sig_row = db_one("""
            SELECT signal, confidence, alignment, regime,
                   created_at
            FROM   signal_outcomes
            ORDER  BY created_at DESC LIMIT 1
        """)
    # Fallback to signal_log CSV
    if not sig_row:
        try:
            import csv
            with open("logs/signal_log.csv") as f:
                rows = list(csv.DictReader(f))
                sig_row = rows[-1] if rows else {}
        except Exception:
            pass

    # ── 3. Open trade (ticker-scoped) ──────────────────────────
    open_trade = db_one(
        "SELECT * FROM open_trades WHERE status='OPEN' AND ticker=? ORDER BY opened_at DESC LIMIT 1",
        (ticker,)
    )
    if not open_trade:
        open_trade = db_one(
            "SELECT * FROM open_trades WHERE status='OPEN' ORDER BY opened_at DESC LIMIT 1"
        )

    # ── 4. Regime from DB ──────────────────────────────────────
    regime_row = db_one("""
        SELECT regime, stability, bull_prob, bear_prob,
               high_vol_prob, choppy_prob
        FROM   regime_snapshots
        ORDER  BY detected_at DESC LIMIT 1
    """)

    # ── 5. Sentiment from DB (ticker-scoped) ──────────────────
    since_24h = str(now - timedelta(hours=24))
    sent_row  = db_one("""
        SELECT COUNT(*) as cnt,
               AVG(sentiment_score) as avg_score,
               SUM(CASE WHEN sentiment_score > 0.2 THEN 1 ELSE 0 END) as pos,
               SUM(CASE WHEN sentiment_score < -0.2 THEN 1 ELSE 0 END) as neg
        FROM   news
        WHERE  processed=1 AND created_at > ? AND ticker = ?
    """, (since_24h, ticker))

    # ── 6. Latest news headlines (ticker-scoped, last 7 days) ───
    # Order by *publication* date when available, falling back to insert
    # timestamp for legacy rows. The 7-day window matches the fetcher's
    # MAX_ARTICLE_AGE_DAYS so users see every article we ingested, but
    # months-old articles indexed today still get filtered out.
    news_rows = db_query("""
        SELECT title, source, sentiment_score, processed, link,
               COALESCE(published_iso, created_at) AS ts
        FROM   news
        WHERE  ticker = ?
          AND  COALESCE(published_iso, created_at) > datetime('now', '-7 days')
        ORDER  BY COALESCE(published_iso, created_at) DESC
        LIMIT  20
    """, (ticker,))

    # ── 7. FII/DII flow ────────────────────────────────────────
    # Prefer daily rows (those with DII data). Monthly aggregates land on
    # month-end dates and typically have dii_net_cr=NULL. Pull the 5 most
    # recent trading days; if fewer daily rows exist, fall back to all rows.
    fii_rows  = db_query("""
        SELECT trade_date, fii_net_cr, dii_net_cr
        FROM   fii_data
        WHERE  trade_date <= date('now')
        ORDER  BY (dii_net_cr IS NULL) ASC,   -- rows with DII first
                  trade_date DESC
        LIMIT  5
    """)
    # Strictly chronological for sparkline — separate query
    fii_history = db_query("""
        SELECT trade_date, fii_net_cr, dii_net_cr
        FROM   fii_data
        WHERE  trade_date <= date('now')
        ORDER  BY trade_date DESC
        LIMIT  5
    """)

    # ── 8. Options ─────────────────────────────────────────────
    opts_row  = db_one("""
        SELECT pcr, max_pain, iv_percentile,
               implied_move_pct, max_call_oi_strike,
               max_put_oi_strike, total_call_oi,
               total_put_oi, fetched_at
        FROM   options_snapshots
        ORDER  BY fetched_at DESC LIMIT 1
    """)

    # ── 9. Global markets ──────────────────────────────────────
    global_row = db_one("""
        SELECT sp500_1d_pct, india_vix_level, crude_level,
               crude_5d_pct, dxy_level, dxy_5d_pct,
               usdinr_level, usdinr_5d_pct,
               us_10y_level, banknifty_5d_pct,
               jpmorgan_5d_pct, verdict
        FROM   global_snapshots
        ORDER  BY fetched_at DESC LIMIT 1
    """)

    # ── 10. Performance stats (ticker-scoped) ──────────────────
    perf = db_one("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN pnl_pct>0 THEN 1 ELSE 0 END) as wins,
               AVG(CASE WHEN pnl_pct>0 THEN pnl_pct END) as avg_win,
               AVG(CASE WHEN pnl_pct<=0 THEN pnl_pct END) as avg_loss,
               SUM(pnl_amount) as total_pnl
        FROM   closed_trades WHERE status='CLOSED' AND ticker=?
    """, (ticker,))

    month_pnl = db_one("""
        SELECT COALESCE(SUM(pnl_amount),0) as mpnl
        FROM   closed_trades
        WHERE  status='CLOSED' AND ticker=?
        AND    close_date >= date('now','start of month')
    """, (ticker,))

    consec = db_query("""
        SELECT pnl_pct FROM closed_trades
        WHERE status='CLOSED' AND ticker=?
        ORDER BY close_date DESC LIMIT 10
    """, (ticker,))
    consec_losses = 0
    for r in consec:
        if safe_float(r.get("pnl_pct")) <= 0:
            consec_losses += 1
        else:
            break

    # ── 11a. Cross-ticker open exposure ────────────────────────
    all_open = db_query(
        "SELECT shares, entry_price FROM open_trades WHERE status='OPEN'"
    )

    # ── 11b. Sharpe ratio (annualised) from all closed trades ──
    all_pnl = db_query(
        "SELECT pnl_pct FROM closed_trades WHERE status='CLOSED' ORDER BY close_date"
    )
    pnl_arr = [safe_float(r.get("pnl_pct", 0)) * 100 for r in all_pnl]
    if len(pnl_arr) >= 3:
        _mean = sum(pnl_arr) / len(pnl_arr)
        _std  = (sum((x - _mean) ** 2 for x in pnl_arr) / len(pnl_arr)) ** 0.5
        sharpe = round(_mean / _std * (252 ** 0.5), 2) if _std > 0 else 0.0
    else:
        sharpe = 0.0

    # ── 11c. Capital mode + active signals from signal_log ─────
    # signal_log is always written (even when scheduler isn't running).
    # Used for: capital_mode, and as a fallback for open exposure when
    # open_trades is empty (signals fired via --mode=signal, not via scheduler).
    capital_mode  = "SMALL"
    _log_open: list = []
    try:
        import csv as _csv
        _last_per_key: dict = {}
        with open("logs/signal_log.csv") as _f:
            for _r in _csv.DictReader(_f):
                # Track capital mode from any row that has it
                if _r.get("capital_mode"):
                    capital_mode = _r["capital_mode"]
                # Last row per (ticker, category) wins — captures flips to FLAT
                _key = (_r.get("ticker", ""), _r.get("category", ""))
                _last_per_key[_key] = _r
        # Active signal = last row for that (ticker, cat) has status=SIGNAL
        # with valid sizing (shares > 0, stop > 0).
        for _v in _last_per_key.values():
            if (_v.get("status") == "SIGNAL"
                    and float(_v.get("shares", 0) or 0) > 0
                    and float(_v.get("stop",   0) or 0) > 0):
                _log_open.append({
                    "shares":      int(float(_v.get("shares", 0))),
                    "entry_price": float(_v.get("entry", 0) or 0),
                    "ticker":      _v.get("ticker", ""),
                    "trade_type":  _v.get("category", "swing"),
                    "signal":      _v.get("signal", ""),
                    "confidence":  float(_v.get("confidence", 0) or 0),
                })
    except Exception:
        pass

    # Use DB trades when available; fall back to signal_log-derived positions
    _effective_open   = all_open if all_open else _log_open
    open_exposure     = sum(int(r.get("shares", 0)) * safe_float(r.get("entry_price", 0)) for r in _effective_open)
    open_trades_count = len(_effective_open)

    # ── 11. Universe ranking (from batch cache — no extra network calls) ──
    universe = []
    for sym, name in [
        ("HDFCBANK.NS",  "HDFCBANK"),  ("ICICIBANK.NS",  "ICICIBANK"),
        ("KOTAKBANK.NS", "KOTAKBANK"), ("AXISBANK.NS",   "AXISBANK"),
        ("INDUSINDBK.NS","INDUSINDBK"),
    ]:
        q = all_quotes.get(sym, dict(_ZERO_QUOTE))
        universe.append({
            "name":       name,
            "price":      q["price"],
            "change_pct": q["change_pct"],
        })
    universe.sort(key=lambda x: x["change_pct"], reverse=True)
    for i, u in enumerate(universe):
        u["rank"]  = i + 1
        u["score"] = round(u["change_pct"] * 2.5, 1)

    # ── 12. Ticker strip (from batch cache) ───────────────────
    bn  = all_quotes.get("^NSEBANK",    dict(_ZERO_QUOTE))
    n50 = all_quotes.get("^NSEI",       dict(_ZERO_QUOTE))
    vix = all_quotes.get("^INDIAVIX",   dict(_ZERO_QUOTE))
    inr = all_quotes.get("USDINR=X",    dict(_ZERO_QUOTE))
    cru = all_quotes.get("CL=F",        dict(_ZERO_QUOTE))
    gld = all_quotes.get("GC=F",        dict(_ZERO_QUOTE))
    sp5 = all_quotes.get("^GSPC",       dict(_ZERO_QUOTE))
    ici = all_quotes.get("ICICIBANK.NS",dict(_ZERO_QUOTE))
    kot = all_quotes.get("KOTAKBANK.NS",dict(_ZERO_QUOTE))
    axs = all_quotes.get("AXISBANK.NS", dict(_ZERO_QUOTE))
    ind = all_quotes.get("INDUSINDBK.NS",dict(_ZERO_QUOTE))

    # ── 13. Chart data (parallel) ─────────────────────────────
    with ThreadPoolExecutor(max_workers=2) as _cp:
        _f1d = _cp.submit(yf_history, ticker, "6mo", "1d")
        _f1w = _cp.submit(yf_history, ticker, "2y",  "1wk")
        try:
            chart_1d = _f1d.result(timeout=12)
        except Exception:
            chart_1d = []
        try:
            chart_1w = _f1w.result(timeout=12)
        except Exception:
            chart_1w = []

    # ── 14. Feature importance (ticker-scoped metrics file) ────
    feat_imp = []
    metrics_path = f"models/evaluation/{ticker_safe}_model_metrics.json"
    # Fallback to legacy path for HDFCBANK
    if not os.path.exists(metrics_path):
        metrics_path = "models/evaluation/feature_importance.json"
    try:
        with open(metrics_path) as f:
            data = json.load(f)
            raw  = data.get("swing", {})
            top  = sorted(raw.items(), key=lambda x: x[1], reverse=True)[:8]
            total= sum(v for _, v in top) or 1
            feat_imp = [
                {"name": k.replace("_", " ").upper(),
                 "pct":  round(v / total * 100, 1)}
                for k, v in top
            ]
    except Exception:
        feat_imp = []

    # ── 15. LLM scores ─────────────────────────────────────────
    llm_earn = db_one("""
        SELECT earnings_data, analyzed_at
        FROM   llm_earnings_cache
        ORDER  BY analyzed_at DESC LIMIT 1
    """)
    earn_score = 0.0
    earn_text  = ""
    if llm_earn:
        try:
            ed = json.loads(llm_earn.get("earnings_data", "{}"))
            earn_score = safe_float(ed.get("overall_score", 0))
            earn_text  = ed.get("reasoning", "")
        except Exception:
            pass

    llm_ann = db_one("""
        SELECT impact_score, reason, analyzed_at
        FROM   llm_announcement_scores
        ORDER  BY analyzed_at DESC LIMIT 1
    """)

    # ── 16. Macro ──────────────────────────────────────────────
    macro_row = db_one("""
        SELECT rbi_rate, credit_growth, cpi_yoy, fetched_at
        FROM   macro_data
        ORDER  BY fetched_at DESC LIMIT 1
    """)

    # Days to next earnings (ticker-scoped, future dates only)
    earn_cal = db_one("""
        SELECT result_date
        FROM   earnings_calendar
        WHERE  ticker = ? AND result_date >= date('now')
        ORDER  BY result_date ASC LIMIT 1
    """, (ticker,))
    if not earn_cal:
        earn_cal = db_one("""
            SELECT result_date
            FROM   earnings_calendar
            WHERE  result_date >= date('now')
            ORDER  BY result_date ASC LIMIT 1
        """)

    # ── 16b. Order Flow from recent daily candles ─────────────
    # Uses the same chart_1d already fetched — no extra network call.
    # OrderFlowAnalyzer infers institutional buy/sell pressure from OHLCV.
    order_flow: Dict[str, Any] = {}
    try:
        from processing.order_flow import OrderFlowAnalyzer
        import pandas as pd
        if chart_1d and len(chart_1d) >= 5:
            df_of = pd.DataFrame(chart_1d).rename(columns={
                "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"
            })
            for col in ("Open", "High", "Low", "Close", "Volume"):
                df_of[col] = pd.to_numeric(df_of[col], errors="coerce").fillna(0)
            price_now = safe_float(bank_quote.get("price", 0))
            atr_est   = price_now * 0.02 if price_now > 0 else 10.0
            of = OrderFlowAnalyzer().get_order_flow_features(df_of, atr=atr_est)

            # Per-bar delta for chart (last 20 bars)
            recent = df_of.tail(20)
            bar_deltas = []
            for _, row in recent.iterrows():
                rng = max(float(row["High"]) - float(row["Low"]), 0.001)
                close_pos = (float(row["Close"]) - float(row["Low"])) / rng
                bvol = int(close_pos * float(row["Volume"]))
                svol = int((1 - close_pos) * float(row["Volume"]))
                bar_deltas.append({
                    "buy":  bvol,
                    "sell": svol,
                    "bull": bool(float(row["Close"]) >= float(row["Open"])),
                })

            order_flow = {
                "cum_delta":          round(float(of.get("cum_delta", 0))),
                "delta_trend":        of.get("delta_trend", "FLAT"),
                "vwap_signal":        of.get("vwap_signal", "AT_VWAP"),
                "absorption":         of.get("absorption_strength", "NONE"),
                "absorption_detected":bool(of.get("absorption_detected", 0)),
                "inst_bias":          of.get("inst_bias", "NEUTRAL"),
                "inst_conviction":    of.get("inst_conviction", "LOW"),
                "order_flow_score":   float(of.get("order_flow_score", 0.0)),
                "order_flow_signal":  of.get("order_flow_signal", "BALANCED_FLOW"),
                "delta_divergence":   of.get("delta_divergence", "NONE"),
                "bar_deltas":         bar_deltas,
            }
    except Exception as _e:
        logger.debug(f"Order flow compute: {_e}")

    # ── 17. Alternative data ───────────────────────────────────
    alt_rows = db_query("""
        SELECT keyword, slope, latest_value, fetched_at
        FROM   alt_trends
        WHERE  fetched_at > ?
        ORDER  BY fetched_at DESC
    """, (str(now - timedelta(days=8)),))

    # ── 18. Psychology ─────────────────────────────────────────
    psych = {"discipline": 80, "adherence": 88,
             "early_exit": 20, "overtrading": 15, "loss_aversion": 30}

    # ── 19. Exit engine status ─────────────────────────────────
    unrealized_pnl = 0.0
    trail_stop     = 0.0
    if open_trade:
        entry  = safe_float(open_trade.get("entry_price", 0))
        price_ = bank_quote.get("price", entry)
        sig    = open_trade.get("signal", "LONG")
        if sig == "LONG" and entry > 0:
            unrealized_pnl = (price_ - entry) / entry * 100
        elif sig == "SHORT" and entry > 0:
            unrealized_pnl = (entry - price_) / entry * 100
        high_  = safe_float(open_trade.get("highest_price", entry))
        trail_stop = round(high_ * 0.985, 2)

    # ── 20. Fundamentals (ticker-scoped) ──────────────────────
    fund_row = db_one("""
        SELECT nim, npa_gross, npa_net, casa_ratio,
               loan_growth, roe, pb_ratio, fetched_at
        FROM   fundamentals
        WHERE  ticker = ?
        ORDER  BY fetched_at DESC LIMIT 1
    """, (ticker,))
    if not fund_row:
        fund_row = db_one("""
            SELECT nim, npa_gross, npa_net, casa_ratio,
                   loan_growth, roe, pb_ratio, fetched_at
            FROM   fundamentals
            ORDER  BY fetched_at DESC LIMIT 1
        """)

    # ── days_held for open trade ──────────────────────────────
    days_held = 0
    if open_trade:
        try:
            opened = datetime.fromisoformat(str(open_trade.get("opened_at", "")).split(".")[0])
            days_held = (now - opened).days
        except Exception:
            days_held = 0

    # ── ASSEMBLE RESPONSE ──────────────────────────────────────
    total_tr = safe_float(perf.get("total"), 0)
    wins_tr  = safe_float(perf.get("wins"),  0)
    win_rate = round(wins_tr / total_tr * 100, 1) if total_tr > 0 else 0
    avg_win  = safe_float(perf.get("avg_win"), 0) * 100
    avg_loss = safe_float(perf.get("avg_loss"),0) * 100
    pf       = round(abs(avg_win * wins_tr) / abs(avg_loss * (total_tr - wins_tr)), 2) \
               if (avg_loss and total_tr > wins_tr) else 0

    fii_net  = safe_float(fii_rows[0].get("fii_net_cr")) if fii_rows else 0
    dii_net  = safe_float(fii_rows[0].get("dii_net_cr")) if fii_rows else 0
    fii_5d   = sum(safe_float(r.get("fii_net_cr")) for r in fii_history)
    dii_5d   = sum(safe_float(r.get("dii_net_cr")) for r in fii_history)
    risk_badge = (
        "DANGER"  if consec_losses >= 4 or safe_float(month_pnl.get("mpnl")) < -3000 else
        "WARNING" if consec_losses >= 2 or safe_float(month_pnl.get("mpnl")) < -1000 else
        "SAFE"
    )

    pcr          = safe_float(opts_row.get("pcr", 1.0))
    max_pain     = safe_float(opts_row.get("max_pain", 0))
    impl_move    = safe_float(opts_row.get("implied_move_pct", 0))
    call_oi      = safe_float(opts_row.get("total_call_oi", 0))
    put_oi       = safe_float(opts_row.get("total_put_oi",  0))
    iv_pct       = safe_float(opts_row.get("iv_percentile", 50))

    # ── Market open/closed — delegate entirely to MarketSchedule ─────
    # MarketSchedule handles weekends, NSE holidays, and IST conversion
    # correctly via pytz. No duplicated logic here.
    _is_open   = _market_schedule.is_market_open()
    _is_pre    = _market_schedule.is_pre_market()
    _mkt_status = (
        "MARKET OPEN"  if _is_open else
        "PRE-MARKET"   if _is_pre  else
        "MARKET CLOSED"
    )

    # Kick off a background news fetch if news is stale — non-blocking
    _trigger_news_fetch_bg()

    return JSONResponse(_clean_json({
        # Display fields use _now_ist (Kolkata time) so VPS in UTC still shows IST.
        "ts":            _now_ist.strftime("%H:%M:%S"),
        "date":          _now_ist.strftime("%a %d %b %Y"),
        "ist":           _now_ist.strftime("%d %b %Y %H:%M IST"),
        "selected_ticker": ticker,
        "market_open":   _is_open,
        "market_status": _mkt_status,

        # ── Selected bank price ─────────────────────────────────
        "selected_bank": {
            "price":       bank_quote.get("price",       0),
            "change":      bank_quote.get("change",      0),
            "change_pct":  bank_quote.get("change_pct",  0),
            "high":        bank_quote.get("high",         0),
            "low":         bank_quote.get("low",          0),
            "volume":      bank_quote.get("volume",       0),
            "vol_ratio":   bank_quote.get("vol_ratio",    1),
            "prev_close":  bank_quote.get("prev",         0),
            "week52_high": bank_quote.get("week52_high",  0),
        },

        # ── Signal ──────────────────────────────────────────────
        "signal": {
            "direction":   sig_row.get("signal",     "FLAT"),
            "confidence":  safe_float(sig_row.get("confidence", 0)) * 100,
            "alignment":   sig_row.get("alignment",  ""),
            "regime":      sig_row.get("regime",     regime_row.get("regime", "UNKNOWN")),
            "ticker":      sig_row.get("ticker",     open_trade.get("ticker", ticker) if open_trade else ticker),
            "gate_results": _load_last_gate_results(ticker_safe),
        },

        # ── Open trade ──────────────────────────────────────────
        "trade": {
            "active":         bool(open_trade),
            "direction":      open_trade.get("signal",       "") if open_trade else "",
            "entry":          safe_float(open_trade.get("entry_price"))  if open_trade else 0,
            "stop":           safe_float(open_trade.get("stop_price"))   if open_trade else 0,
            "target":         safe_float(open_trade.get("target_price")) if open_trade else 0,
            "shares":         int(open_trade.get("shares", 0))           if open_trade else 0,
            "risk_amount":    safe_float(open_trade.get("risk_amount"))  if open_trade else 0,
            "trade_type":     open_trade.get("trade_type", "swing")      if open_trade else "",
            "unrealized_pnl": round(unrealized_pnl, 2),
            "trail_stop":     trail_stop,
        },

        # ── Regime ──────────────────────────────────────────────
        "regime": {
            "name":       regime_row.get("regime",     "UNKNOWN"),
            "stability":  safe_float(regime_row.get("stability", 0)) * 100,
            "bull_prob":  safe_float(regime_row.get("bull_prob",  0)) * 100,
            "bear_prob":  safe_float(regime_row.get("bear_prob",  0)) * 100,
            "hv_prob":    safe_float(regime_row.get("high_vol_prob", 0)) * 100,
            "choppy_prob":safe_float(regime_row.get("choppy_prob",   0)) * 100,
        },

        # ── Sentiment ───────────────────────────────────────────
        "sentiment": {
            "score":     round(safe_float(sent_row.get("avg_score")), 3),
            "count":     int(safe_float(sent_row.get("cnt"))),
            "positive":  int(safe_float(sent_row.get("pos"))),
            "negative":  int(safe_float(sent_row.get("neg"))),
        },

        # ── News ────────────────────────────────────────────────
        "news": [
            {
                "title":    r.get("title", ""),
                "source":   r.get("source", ""),
                "score":    round(safe_float(r.get("sentiment_score")), 3),
                "scored":   bool(r.get("processed")),
                "time":     str(r.get("ts", ""))[:16],
                "link":     r.get("link", "") or "",
            }
            for r in news_rows
        ],

        # ── FII/DII ─────────────────────────────────────────────
        "fii": {
            "net_today":  round(fii_net, 0),
            "dii_today":  round(dii_net, 0),
            "fii_5d":     round(fii_5d,  0),
            "dii_5d":     round(dii_5d,  0),
            "trend":      "BUYING" if fii_5d > 500  else ("SELLING" if fii_5d < -500  else "NEUTRAL"),
            "dii_trend":  "BUYING" if dii_5d > 500  else ("SELLING" if dii_5d < -500  else "NEUTRAL"),
            "confluence": (
                "BULLISH" if fii_5d > 500  and dii_5d > 500  else
                "BEARISH" if fii_5d < -500 and dii_5d < -500 else "MIXED"
            ),
            "history":    [
                {
                    "date":    r.get("trade_date", ""),
                    "fii_net": round(safe_float(r.get("fii_net_cr")), 0),
                    "dii_net": round(safe_float(r.get("dii_net_cr")), 0),
                }
                for r in fii_history
            ],
        },

        # ── Options ─────────────────────────────────────────────
        "options": {
            "pcr":         round(pcr, 3),
            "pcr_signal":  "BULLISH" if pcr > 1.3 else ("BEARISH" if pcr < 0.7 else "NEUTRAL"),
            "max_pain":    round(max_pain, 0),
            "iv_pct":      round(iv_pct, 1),
            "implied_move":round(impl_move, 2),
            "call_oi":     round(call_oi / 1e6, 2),
            "put_oi":      round(put_oi  / 1e6, 2),
            "call_wall":   safe_float(opts_row.get("max_call_oi_strike")),
            "put_wall":    safe_float(opts_row.get("max_put_oi_strike")),
        },

        # ── Global markets ──────────────────────────────────────
        "global": {
            "sp500_1d":      safe_float(global_row.get("sp500_1d_pct")),
            "vix":           safe_float(global_row.get("india_vix_level")),
            "crude":         safe_float(global_row.get("crude_level")),
            "crude_5d":      safe_float(global_row.get("crude_5d_pct")),
            "dxy":           safe_float(global_row.get("dxy_level")),
            "dxy_5d":        safe_float(global_row.get("dxy_5d_pct")),
            "usdinr":        safe_float(global_row.get("usdinr_level")),
            "usdinr_5d":     safe_float(global_row.get("usdinr_5d_pct")),
            "us10y":         safe_float(global_row.get("us_10y_level")),
            "banknifty_5d":  safe_float(global_row.get("banknifty_5d_pct")),
            "jpmorgan_5d":   safe_float(global_row.get("jpmorgan_5d_pct")),
            "verdict":       global_row.get("verdict", "NEUTRAL"),
            # Live ticker quotes
            "banknifty":     bn,
            "nifty50":       n50,
            "indiavix":      vix,
            "usdinr_live":   inr,
            "crude_live":    cru,
            "gold_live":     gld,
            "sp500_live":    sp5,
            "icicibank":     ici,
            "kotakbank":     kot,
            "axisbank":      axs,
            "indusindbk":    ind,
        },

        # ── Macro ───────────────────────────────────────────────
        "macro": {
            # Use `is not None` so a legitimate 0.0 rate isn't suppressed
            "rbi_rate":      safe_float(macro_row.get("rbi_rate"))      if macro_row.get("rbi_rate")      is not None else None,
            "credit_growth": safe_float(macro_row.get("credit_growth")) if macro_row.get("credit_growth") is not None else None,
            "cpi":           safe_float(macro_row.get("cpi_yoy"))       if macro_row.get("cpi_yoy")       is not None else None,
            "days_to_earnings": (
                (datetime.strptime(earn_cal["result_date"], "%Y-%m-%d") - now).days
                if earn_cal and earn_cal.get("result_date") else None
            ),
        },

        # ── Fundamentals ────────────────────────────────────────
        "fundamentals": {
            "nim":         safe_float(fund_row.get("nim",         3.8)),
            "npa_gross":   safe_float(fund_row.get("npa_gross",   1.3)),
            "npa_net":     safe_float(fund_row.get("npa_net",     0.4)),
            "casa_ratio":  safe_float(fund_row.get("casa_ratio",  44.0)),
            "loan_growth": safe_float(fund_row.get("loan_growth", 16.0)),
            "roe":         safe_float(fund_row.get("roe",         16.0)),
            "pb_ratio":    safe_float(fund_row.get("pb_ratio",    2.8)),
        },

        # ── LLM ─────────────────────────────────────────────────
        "llm": {
            "earnings_score":      round(earn_score, 2),
            "earnings_text":       earn_text[:200],
            "announcement_score":  round(safe_float(llm_ann.get("impact_score")), 2),
            "announcement_reason": llm_ann.get("reason", "")[:120],
            "combined":            round((earn_score * 0.7 +
                                    safe_float(llm_ann.get("impact_score")) * 0.3), 2),
        },

        # ── Alternative data ─────────────────────────────────────
        "alt_data": [
            {
                "keyword": r.get("keyword", "").replace("HDFC ",""),
                "slope":   round(safe_float(r.get("slope")), 4),
                "value":   round(safe_float(r.get("latest_value")), 1),
            }
            for r in alt_rows
        ],

        # ── Performance ─────────────────────────────────────────
        "performance": {
            "total_trades":     int(total_tr),
            "wins":             int(wins_tr),
            "losses":           int(total_tr - wins_tr),
            "win_rate":         win_rate,
            "avg_win_pct":      round(avg_win,  2),
            "avg_loss_pct":     round(avg_loss, 2),
            "profit_factor":    pf,
            "total_pnl":        round(safe_float(perf.get("total_pnl")), 2),
            "monthly_pnl":      round(safe_float(month_pnl.get("mpnl")), 2),
            "consec_losses":    consec_losses,
            "capital_mode":     capital_mode,
            "open_exposure":    round(open_exposure, 0),
            "open_trades_count":open_trades_count,
            "sharpe":           sharpe,
            "risk_badge":       risk_badge,
        },

        # ── Universe ranking ─────────────────────────────────────
        "universe": universe,

        # ── Feature importance ───────────────────────────────────
        "feature_importance": feat_imp,

        # ── Exit engine ──────────────────────────────────────────
        "exit_engine": {
            "active":        bool(open_trade),
            "unrealized_pnl":round(unrealized_pnl, 2),
            "trail_stop":    trail_stop,
            "days_held":     days_held,
        },

        # ── Chart data ───────────────────────────────────────────
        "chart_1d": chart_1d[-80:] if chart_1d else [],
        "chart_1w": chart_1w[-80:] if chart_1w else [],

        # ── Psychology ───────────────────────────────────────────
        "psychology": psych,

        # ── Per-model predictions (written by signal engine) ─────
        "models": _load_last_predictions(ticker_safe),

        # ── Order flow (computed from chart candles) ─────────────
        "order_flow": order_flow,
    }))


def _load_last_gate_results(ticker_safe: str = "HDFCBANK") -> dict:
    """Load last gate_results written by signal_engine for dashboard display."""
    path = f"logs/last_gate_results_{ticker_safe}.json"
    try:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return data.get("gate_results", {})
    except Exception:
        pass
    return {}


def _load_last_predictions(ticker_safe: str = "HDFCBANK") -> dict:
    """Read last per-model predictions written by signal_engine. Falls back to empty."""
    _default = lambda t: {"signal": None, "confidence": 0,
                          "prob_long": 0, "prob_flat": 0, "prob_short": 0,
                          "model_type": t}
    path = f"logs/last_predictions_{ticker_safe}.json"
    # Backward compat: fall back to shared file for HDFCBANK
    if not os.path.exists(path) and ticker_safe == "HDFCBANK":
        path = "logs/last_predictions.json"
    try:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return {
                "positional": data.get("positional", _default("positional")),
                "swing":      data.get("swing",      _default("swing")),
                "intraday":   data.get("intraday",   _default("intraday")),
                "alignment":  data.get("alignment",  ""),
                "written_at": data.get("written_at", ""),
            }
    except Exception as e:
        logger.debug(f"last_predictions load failed: {e}")
    return {
        "positional": _default("positional"),
        "swing":      _default("swing"),
        "intraday":   _default("intraday"),
        "alignment":  "",
        "written_at": "",
    }


@app.get("/api/trades")
async def trades_data() -> JSONResponse:
    """
    Paper-trading ledger endpoint — returns open positions + closed trade history.
    Called by the Portfolio modal when the heat-map card is clicked.
    """
    import pytz as _pytz
    _IST = _pytz.timezone("Asia/Kolkata")
    now  = datetime.now(_IST)

    # ── Open positions ────────────────────────────────────────────
    open_rows = db_query("""
        SELECT id, ticker, signal, trade_type, entry_price,
               stop_price, target_price, shares, risk_amount,
               highest_price, lowest_price, opened_at, signal_uuid
        FROM   open_trades
        WHERE  status = 'OPEN'
        ORDER  BY opened_at DESC
    """)

    # ── Signal-log fallback (mirrors /api/live logic exactly) ─────
    # When open_trades DB is empty (e.g. VPS not yet updated, or signals
    # fired via --mode=signal without the orchestrator opening positions),
    # fall back to signal_log.csv so the modal shows the same count as
    # the Portfolio Heat Map card.  Without this fallback the heat map
    # says "3 open trades" but the modal shows 0 — a confusing mismatch.
    if not open_rows:
        try:
            import csv as _csv
            _last_per_key: dict = {}
            with open("logs/signal_log.csv") as _f:
                for _r in _csv.DictReader(_f):
                    _key = (_r.get("ticker", ""), _r.get("category", ""))
                    _last_per_key[_key] = _r
            for _v in _last_per_key.values():
                if (_v.get("status") == "SIGNAL"
                        and float(_v.get("shares", 0) or 0) > 0
                        and float(_v.get("stop",   0) or 0) > 0):
                    _ep = float(_v.get("entry", 0) or 0)
                    open_rows.append({
                        "id":           None,           # no DB id — log-derived
                        "ticker":       _v.get("ticker", "HDFCBANK.NS"),
                        "signal":       _v.get("signal", "LONG"),
                        "trade_type":   _v.get("category", "swing"),
                        "entry_price":  _ep,
                        "stop_price":   float(_v.get("stop",   0) or 0),
                        "target_price": float(_v.get("target", 0) or 0),
                        "shares":       int(float(_v.get("shares", 0) or 0)),
                        "risk_amount":  float(_v.get("risk_amount", 0) or 0),
                        "highest_price":_ep,
                        "lowest_price": _ep,
                        "opened_at":    _v.get("timestamp", ""),
                        "signal_uuid":  _v.get("signal_uuid", ""),
                        "_from_log":    True,   # flag for UI labelling
                    })
        except Exception:
            pass

    # Enrich with current price from batch cache (already populated by /api/live)
    _sym_map = {t.replace(".NS","").upper(): t for t in _UNIVERSE_SYMS}
    open_positions = []
    for r in open_rows:
        ticker     = r.get("ticker", "HDFCBANK.NS")
        sym        = ticker if ticker.endswith(".NS") else ticker + ".NS"
        q          = _quotes_cache.get(sym, {})
        cur_price  = safe_float(q.get("price", 0)) or safe_float(r.get("entry_price", 0))
        entry      = safe_float(r.get("entry_price", 0))
        stop       = safe_float(r.get("stop_price",  0))
        target     = safe_float(r.get("target_price",0))
        signal     = r.get("signal", "LONG")

        # Unrealized P&L
        if entry > 0:
            unreal_pct = ((cur_price - entry) / entry) if signal == "LONG" else ((entry - cur_price) / entry)
            unreal_amt = round(unreal_pct * entry * int(r.get("shares", 0) or 0), 2)
        else:
            unreal_pct = unreal_amt = 0.0

        # Distance to stop / target in %
        stop_dist   = round(abs(cur_price - stop)   / cur_price * 100, 2) if cur_price > 0 else 0
        target_dist = round(abs(target - cur_price) / cur_price * 100, 2) if cur_price > 0 else 0

        # Risk:Reward remaining
        risk_rem   = abs(cur_price - stop)
        reward_rem = abs(target - cur_price)
        rr_rem     = round(reward_rem / risk_rem, 2) if risk_rem > 0 else 0

        # Days held
        try:
            opened_dt = datetime.fromisoformat(str(r.get("opened_at",""))[:19])
            days_held = (now.replace(tzinfo=None) - opened_dt).days
        except Exception:
            days_held = 0

        # Progress toward target (0–100%)
        total_move  = abs(target - entry)
        done_move   = abs(cur_price - entry)
        progress    = min(100, round(done_move / total_move * 100)) if total_move > 0 else 0

        open_positions.append({
            "id":          r.get("id"),
            "ticker":      ticker.replace(".NS",""),
            "signal":      signal,
            "category":    r.get("trade_type", "swing"),
            "entry":       round(entry, 2),
            "current":     round(cur_price, 2),
            "stop":        round(stop, 2),
            "target":      round(target, 2),
            "shares":      int(r.get("shares", 0) or 0),
            "risk_amount": round(safe_float(r.get("risk_amount", 0)), 2),
            "unreal_pct":  round(unreal_pct * 100, 2),
            "unreal_amt":  unreal_amt,
            "stop_dist":   stop_dist,
            "target_dist": target_dist,
            "rr_rem":      rr_rem,
            "days_held":   days_held,
            "progress":    progress,
            "opened_at":   str(r.get("opened_at",""))[:16],
            "from_log":    bool(r.get("_from_log")),   # True = signal-log derived, no DB row
        })

    # ── Closed trades (last 50) ───────────────────────────────────
    closed_rows = db_query("""
        SELECT id, ticker, signal, trade_type, entry_price, exit_price,
               shares, pnl_amount, pnl_pct, exit_reason,
               close_date, signal_uuid
        FROM   closed_trades
        WHERE  status = 'CLOSED'
        ORDER  BY close_date DESC
        LIMIT  50
    """)

    reason_map = {
        "TARGET_HIT":      "🎯 Target Hit",
        "STOP_LOSS":       "🛑 Stop Loss",
        "TRAILING_STOP":   "📉 Trail Stop",
        "TIME_EXIT":       "⏰ Time Exit",
        "INTRADAY_CLOSE":  "🔔 Intraday Close",
        "GAP_FILL":        "⚡ Gap Fill",
        "TARGET_1R":       "🎯 1R Target",
        "TARGET_2R":       "🎯 2R Target",
        "CHANDELIER_TRAIL":"📉 Chandelier",
        # Partial suffixes
        "TARGET_HIT_PARTIAL":      "🎯 Target (Partial)",
        "STOP_LOSS_PARTIAL":       "🛑 Stop (Partial)",
        "TRAILING_STOP_PARTIAL":   "📉 Trail (Partial)",
        "TARGET_1R_PARTIAL":       "🎯 1R (Partial)",
        "TARGET_2R_PARTIAL":       "🎯 2R (Partial)",
        "CHANDELIER_TRAIL_PARTIAL":"📉 Chandelier (Partial)",
    }

    closed_trades = []
    for r in closed_rows:
        pnl_pct = safe_float(r.get("pnl_pct", 0))
        pnl_amt = safe_float(r.get("pnl_amount", 0))
        entry   = safe_float(r.get("entry_price", 0))
        exit_p  = safe_float(r.get("exit_price",  0))
        reason  = r.get("exit_reason", "") or ""
        # trade_type is now a proper DB column; fall back to uuid hint for legacy rows
        uuid    = r.get("signal_uuid","") or ""
        cat_db  = r.get("trade_type") or ""
        if not cat_db:
            for _c in ("intraday", "positional", "swing"):
                if _c in uuid.lower():
                    cat_db = _c
                    break
            cat_db = cat_db or "swing"

        reason_label = reason_map.get(reason, reason.replace("_"," ").title())

        closed_trades.append({
            "id":         r.get("id"),
            "ticker":     (r.get("ticker") or "HDFCBANK").replace(".NS",""),
            "signal":     r.get("signal","LONG"),
            "category":   cat_db,
            "entry":      round(entry, 2),
            "exit":       round(exit_p, 2),
            "shares":     int(r.get("shares", 0) or 0),
            "pnl_pct":    round(pnl_pct * 100, 2),
            "pnl_amt":    round(pnl_amt, 2),
            "reason":     reason_label,
            "raw_reason": reason,
            "close_date": str(r.get("close_date",""))[:16],
        })

    # ── Summary stats ─────────────────────────────────────────────
    total  = len(closed_trades)
    wins   = sum(1 for t in closed_trades if t["pnl_pct"] > 0)
    losses = total - wins
    gross_profit = sum(t["pnl_amt"] for t in closed_trades if t["pnl_amt"] > 0)
    gross_loss   = abs(sum(t["pnl_amt"] for t in closed_trades if t["pnl_amt"] < 0))
    net_pnl      = sum(t["pnl_amt"] for t in closed_trades)

    return JSONResponse(_clean_json({
        "open":   open_positions,
        "closed": closed_trades,
        "summary": {
            "total":         total,
            "wins":          wins,
            "losses":        losses,
            "win_rate":      round(wins / total * 100, 1) if total > 0 else 0,
            "net_pnl":       round(net_pnl, 2),
            "gross_profit":  round(gross_profit, 2),
            "gross_loss":    round(gross_loss, 2),
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
            "open_count":    len(open_positions),
        },
    }))


@app.get("/api/health")
async def health():
    return {"status": "ok", "ts": str(datetime.now())}


@app.get("/api/banks")
async def list_banks():
    """Return all configured bank tickers with display names."""
    from config import BANK_CONFIG
    return {
        "banks": [
            {"ticker": t, "name": v["display_name"]}
            for t, v in BANK_CONFIG.items()
        ]
    }


@app.get("/api/chart/{period}")
async def chart_data(
    period: str = "1d",
    ticker: str = Query(default="HDFCBANK.NS"),
):
    if ticker not in VALID_TICKERS:
        ticker = "HDFCBANK.NS"
    interval_map = {
        "5m":  ("1d",  "5m"),
        "15m": ("5d",  "15m"),
        "1d":  ("6mo", "1d"),
        "1w":  ("2y",  "1wk"),
        "1m":  ("5y",  "1mo"),
    }
    per, itv = interval_map.get(period, ("6mo", "1d"))
    return {"data": yf_history(ticker, period=per, interval=itv)}
