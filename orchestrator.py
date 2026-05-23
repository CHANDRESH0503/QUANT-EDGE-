# orchestrator.py
# Single-process runner for all 5 banks.
#
# Why a separate entry point instead of N copies of main.py?
#   - FinBERT is ~1.6 GB resident. Five separate processes = ~8 GB just for
#     the model. With this orchestrator FinBERT is loaded once and shared via
#     FinBERTSentiment._shared_pipeline.
#   - Global data (FII, DII, options, world markets, news headlines) is the
#     same regardless of which bank we're scoring. Fetched once per cycle
#     instead of five times.
#   - One interpreter, one log file, one systemd unit.
#
# Run:
#   python orchestrator.py
#   python orchestrator.py --interval 900 --tickers HDFCBANK.NS,ICICIBANK.NS
#   python orchestrator.py --once       # single cycle, useful for cron/cron-debug

import argparse
import logging
import os
import signal as _signal
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional

# ── Logging — same setup as main.py ──────────────────────────────────────────
from config import LogConfig, PathConfig, APIConfig, TradingConfig

PathConfig.ensure_dirs()
_handlers = [logging.StreamHandler(sys.stdout)]
if LogConfig.LOG_TO_FILE:
    _handlers.append(logging.FileHandler(LogConfig.LOG_FILE, encoding="utf-8"))
logging.basicConfig(
    level   = getattr(logging, LogConfig.LEVEL, logging.INFO),
    format  = LogConfig.FORMAT,
    datefmt = LogConfig.DATE_FORMAT,
    handlers= _handlers,
)
for _lib in ("httpx", "httpcore", "huggingface_hub", "filelock",
             "urllib3", "transformers.modeling_utils", "yfinance"):
    logging.getLogger(_lib).setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

logger = logging.getLogger("orchestrator")


# ── Defaults ─────────────────────────────────────────────────────────────────
ALL_BANKS: List[str] = [
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "INDUSINDBK.NS",
]

CYCLE_INTERVAL_SEC = 900          # 15 min — matches task_runner
NEWS_INTERVAL_SEC  = 900
OPTIONS_INTERVAL_SEC = 1800
FII_INTERVAL_SEC   = 7200
GLOBAL_INTERVAL_SEC= 3600


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class MultiBankOrchestrator:
    """
    Runs the signal pipeline for every ticker in `tickers` on a single
    process. Heavy resources (FinBERT pipeline, Telegram bot, scheduler,
    market-wide data fetchers) are instantiated once and reused.

    Per-bank state is kept in `self.engines` (one SignalEngine per ticker).
    Dedup state for alerts is kept in `self._last_signal_by_cat[ticker]`.

    The main loop is intentionally simple — one cycle per
    CYCLE_INTERVAL_SEC. Inside a cycle we:
        1. Run global / market-wide fetchers if their interval is due
        2. For each ticker, run the signal pipeline and send alerts
    """

    def __init__(
        self,
        tickers:  List[str],
        capital:  float = TradingConfig.STARTING_CAPITAL,
        db_path:  str   = PathConfig.DB_PATH,
        interval: int   = CYCLE_INTERVAL_SEC,
    ):
        self.tickers   = tickers
        self.capital   = capital
        self.db_path   = db_path
        self.interval  = interval
        self._running  = True

        # ── Shared, market-wide components ──────────────────────────
        from scheduler.market_schedule    import MarketSchedule
        from alerts.telegram_bot          import TelegramBot
        from data.fii_fetcher             import FIIFetcher
        from data.global_fetcher          import GlobalFetcher
        from data.options_fetcher         import OptionsFetcher
        from data.news_fetcher            import NewsFetcher
        from data.social_fetcher          import SocialFetcher
        from data.bse_fetcher             import BSEFetcher
        from processing.finbert_sentiment import FinBERTSentiment
        from risk.exit_engine             import ExitEngine
        from risk.circuit_breaker         import CircuitBreaker

        self.schedule        = MarketSchedule()
        self.telegram        = TelegramBot()
        self.fii_fetcher     = FIIFetcher(db_path)
        self.global_fetcher  = GlobalFetcher(db_path)
        self.options_fetcher = OptionsFetcher(db_path)
        self.social_fetcher  = SocialFetcher(db_path)
        self.exit_engine     = ExitEngine(db_path)
        self.circuit_breaker = CircuitBreaker(db_path)

        # FinBERT — the heavy one. FinBERTSentiment._shared_pipeline caches
        # the model across instances, so we just construct one and let any
        # other FinBERTSentiment(ticker=...) reuse the cached pipeline.
        self.finbert = FinBERTSentiment(db_path=db_path)

        # ── Per-ticker components ───────────────────────────────────
        from signals.signal_engine import SignalEngine

        self.engines: Dict[str, SignalEngine] = {}
        self.news_fetchers: Dict[str, NewsFetcher] = {}
        self.bse_fetchers:  Dict[str, BSEFetcher]  = {}
        for t in tickers:
            self.engines[t]        = SignalEngine(t, capital, db_path)
            self.news_fetchers[t]  = NewsFetcher(db_path=db_path, ticker=t)
            self.bse_fetchers[t]   = BSEFetcher(db_path=db_path, ticker=t)

        # Per-ticker alert dedup: { ticker: { category: (direction, conf) } }
        self._last_signal_by_cat: Dict[str, Dict[str, tuple]] = {
            t: {} for t in tickers
        }

        # Global "last ran" timestamps for the data tasks
        self._last_news_ts    = 0.0
        self._last_options_ts = 0.0
        self._last_fii_ts     = 0.0
        self._last_global_ts  = 0.0
        self._task_errors: Dict[str, int] = {}

        # Graceful shutdown
        _signal.signal(_signal.SIGINT,  self._stop)
        _signal.signal(_signal.SIGTERM, self._stop)

    # ─────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────────────────────────

    def start(self) -> None:
        logger.info("=" * 60)
        logger.info("QUANT EDGE — Orchestrator starting")
        logger.info(f"Banks    : {', '.join(self.tickers)}")
        logger.info(f"Capital  : ₹{self.capital:,.0f}")
        logger.info(f"Interval : {self.interval}s ({self.interval // 60} min)")
        logger.info(f"DB       : {self.db_path}")
        logger.info("=" * 60)

        # Ensure schema is up to date — adds the multi-bank `ticker` column to
        # open_trades, closed_trades, signal_outcomes, feature_snapshots,
        # options_snapshots, fundamentals, earnings_calendar if missing.
        # Safe to run repeatedly (idempotent).
        self._ensure_schema()

        # Load all per-ticker models once
        from models.train_all import ModelTrainer
        for t in self.tickers:
            try:
                ModelTrainer(t, self.capital, self.db_path).load_all()
            except Exception as e:
                logger.warning(f"[{t}] model load: {e}")

        # Warm FinBERT once at boot so the first cycle doesn't pay the cost
        try:
            self.finbert._get_pipeline()
        except Exception:
            pass

        self.telegram.test_connection()
        if APIConfig.TELEGRAM_TOKEN:
            try:
                self.telegram.send_raw(
                    f"🟢 QUANT EDGE orchestrator up · "
                    f"{len(self.tickers)} banks · "
                    f"{self.interval//60}-min cycle"
                )
            except Exception:
                pass

        while self._running:
            t0 = time.time()
            try:
                self._cycle()
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"cycle error: {e}\n{traceback.format_exc()}")
                self._safe_telegram(f"Orchestrator cycle error: {e}", "cycle")

            elapsed = time.time() - t0
            sleep_for = max(5.0, self.interval - elapsed)
            logger.info(f"cycle done in {elapsed:.1f}s — sleeping {sleep_for:.0f}s")
            self._sleep(sleep_for)

        logger.info("Orchestrator stopped.")
        if APIConfig.TELEGRAM_TOKEN:
            try:
                self.telegram.send_raw("⛔ QUANT EDGE orchestrator stopped")
            except Exception:
                pass

    def run_once(self) -> None:
        """Single cycle then exit. Useful for testing / cron-debug."""
        self._ensure_schema()
        from models.train_all import ModelTrainer
        for t in self.tickers:
            try:
                ModelTrainer(t, self.capital, self.db_path).load_all()
            except Exception as e:
                logger.warning(f"[{t}] model load: {e}")
        self._cycle()

    def _ensure_schema(self) -> None:
        """Create any missing tables and apply pending ALTER TABLE migrations
        (notably the multi-bank `ticker` columns). Mirrors main.py's startup.

        Run defensively at every boot — DatabaseSetup.setup_all() is built to
        be idempotent (`CREATE TABLE IF NOT EXISTS`, ALTER only if column
        missing). Catching a fresh deploy that copied an older trading.db
        without the ticker columns is the whole reason we run this here."""
        try:
            from database.db_setup import DatabaseSetup
            DatabaseSetup(self.db_path).setup_all()
        except Exception as e:
            logger.error(f"_ensure_schema failed: {e}")
            logger.error(traceback.format_exc())
            # Don't crash — let the cycle attempt run; many tables will work
            # even if migrations couldn't complete.

    # ─────────────────────────────────────────────────────────────────
    # ONE CYCLE
    # ─────────────────────────────────────────────────────────────────

    def _cycle(self) -> None:
        now = time.time()
        is_trading = self.schedule.is_trading_day()
        is_open    = self.schedule.is_market_open()

        # ── 1. Global / market-wide data (once per cycle) ───────────
        if is_trading and (now - self._last_news_ts) >= NEWS_INTERVAL_SEC:
            self._safe("news_all_banks",
                       lambda: self._fetch_news_and_score())
            self._last_news_ts = now

        if is_open and (now - self._last_options_ts) >= OPTIONS_INTERVAL_SEC:
            self._safe("options",
                       lambda: self.options_fetcher.fetch_and_calculate())
            self._last_options_ts = now

        if is_trading and (now - self._last_fii_ts) >= FII_INTERVAL_SEC:
            self._safe("fii", lambda: self.fii_fetcher.fetch_daily())
            self._last_fii_ts = now

        if (now - self._last_global_ts) >= GLOBAL_INTERVAL_SEC:
            self._safe("global", lambda: self.global_fetcher.fetch_all())
            self._last_global_ts = now

        # ── 2. Per-ticker signal pipeline ───────────────────────────
        for ticker in self.tickers:
            try:
                self._run_ticker(ticker)
            except Exception as e:
                logger.error(f"[{ticker}] pipeline error: {e}")
                logger.error(traceback.format_exc())
                self._safe_telegram(
                    f"[{ticker}] pipeline failed: {e}", f"pipeline_{ticker}"
                )

        # ── 3. Exit checks (open trades across all banks) ───────────
        if is_open:
            self._safe("exits", self.exit_engine.check_all_open_trades)

    def _run_ticker(self, ticker: str) -> None:
        engine = self.engines[ticker]
        logger.info(f"[{ticker}] running signal pipeline")
        signal = engine.run()
        self._dispatch_alerts(ticker, signal)

    def _fetch_news_and_score(self) -> None:
        """Fetch RSS for every bank then run FinBERT once on the combined
        unscored pool. Cheaper than per-ticker scoring loops."""
        for t in self.tickers:
            try:
                self.news_fetchers[t].fetch_all()
            except Exception as e:
                logger.warning(f"[{t}] news fetch: {e}")
        # Scoring is global — one batch through FinBERT covers every bank
        try:
            n = self.finbert.process_unscored(batch_size=100)
            if n:
                logger.info(f"FinBERT scored {n} articles")
        except Exception as e:
            logger.warning(f"finbert.process_unscored: {e}")

    # ─────────────────────────────────────────────────────────────────
    # ALERT DISPATCH
    # ─────────────────────────────────────────────────────────────────

    def _dispatch_alerts(self, ticker: str, signal: Dict) -> None:
        """Emit a Telegram alert for any category that flips direction or
        crosses a meaningful confidence delta since we last alerted on it."""
        per_cat = (signal or {}).get("gate_results", {}).get("per_category", {})
        if not per_cat:
            return

        dedup = self._last_signal_by_cat[ticker]
        for category, cat_signal in per_cat.items():
            direction = cat_signal.get("direction") or cat_signal.get("signal", "FLAT")
            if direction in (None, "FLAT"):
                continue
            confidence = float(cat_signal.get("confidence", 0) or 0)

            prev = dedup.get(category)
            same_dir = prev and prev[0] == direction
            small_delta = same_dir and abs(prev[1] - confidence) < 0.05
            if same_dir and small_delta:
                continue  # nothing new worth pinging the user about

            payload = {
                **signal,                  # preserve regime/gate/etc context
                "ticker":     ticker,
                "category":   category,
                "trade_type": category,
                "signal":     direction,
                "confidence": confidence,
            }
            # Pull per-category entry/stop/target if the signal engine attached them
            for k in ("entry_price", "stop_loss", "target_price", "shares"):
                if k in cat_signal:
                    payload[k] = cat_signal[k]

            try:
                self.telegram.send_signal(payload)
                logger.info(
                    f"[{ticker}] ALERT {category} {direction} "
                    f"@ conf={confidence:.0%}"
                )
                dedup[category] = (direction, confidence)
            except Exception as e:
                logger.warning(f"[{ticker}] telegram send: {e}")

    # ─────────────────────────────────────────────────────────────────
    # UTIL
    # ─────────────────────────────────────────────────────────────────

    def _safe(self, name: str, fn) -> None:
        try:
            fn()
            self._task_errors[name] = 0
        except Exception as e:
            self._task_errors[name] = self._task_errors.get(name, 0) + 1
            logger.warning(f"task {name} failed: {e}")
            if self._task_errors[name] == 3:
                self._safe_telegram(
                    f"task '{name}' failed 3x in a row: {e}", f"task_{name}"
                )

    def _safe_telegram(self, msg: str, dedup_key: str) -> None:
        if not APIConfig.TELEGRAM_TOKEN:
            return
        try:
            self.telegram.send_raw(f"⚠️ {msg}")
        except Exception:
            pass

    def _sleep(self, seconds: float) -> None:
        """Interruptible sleep — wakes immediately on SIGINT/SIGTERM."""
        end = time.time() + seconds
        while self._running and time.time() < end:
            time.sleep(min(1.0, end - time.time()))

    def _stop(self, *_args) -> None:
        logger.info("shutdown requested")
        self._running = False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QUANT EDGE — multi-bank orchestrator (single process)",
    )
    p.add_argument(
        "--tickers",
        default=",".join(ALL_BANKS),
        help=f"Comma-separated NSE symbols (default: all 5 banks)",
    )
    p.add_argument(
        "--interval",
        type=int,
        default=CYCLE_INTERVAL_SEC,
        help="Seconds between cycles (default: 900)",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Run a single cycle and exit (useful for cron / debugging)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    for t in tickers:
        if t not in ALL_BANKS:
            logger.error(f"Unknown ticker: {t}. Allowed: {ALL_BANKS}")
            sys.exit(2)

    orch = MultiBankOrchestrator(
        tickers  = tickers,
        capital  = TradingConfig.STARTING_CAPITAL,
        db_path  = PathConfig.DB_PATH,
        interval = args.interval,
    )

    if args.once:
        orch.run_once()
    else:
        orch.start()


if __name__ == "__main__":
    main()
