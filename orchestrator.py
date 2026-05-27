# orchestrator.py
# ═══════════════════════════════════════════════════════════════════════════════
# QUANT EDGE — Complete Multi-Bank Autonomous Trading System (Single Process)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This file IS the entire system. It replaces both orchestrator.py + task_runner.py.
# One process, one FinBERT load (~1.6 GB), one scheduler for all 5 banks.
#
# Cycle architecture (30s main loop tick):
#   ├── 5-min  [market hours 9:15–15:30]   → intraday signal for all 5 banks
#   ├── 15-min [pre-market 6:00+ & hours]  → full pipeline (swing+positional+intraday)
#   ├──  1-min [market hours]              → exit monitor (stop/target breach)
#   ├── 15-min [trading days]              → news fetch + FinBERT scoring
#   ├── 30-min [market hours]              → options chain refresh
#   ├── 30-min [trading days]              → BSE announcements + LLM
#   ├──  2-hr  [trading days]              → FII/DII data + social sentiment
#   ├──  1-hr  [always]                    → global macro (S&P, DXY, VIX, crude)
#   ├── daily  06:00 IST                   → pre-market global fetch + signal
#   ├── daily  09:15 IST                   → market open alert
#   ├── daily  15:15 IST                   → intraday close warning
#   ├── daily  15:45 IST                   → EOD summary + heatmap
#   ├── daily  18:00 IST                   → evening performance report
#   ├── weekly Sunday 00:00 IST            → retrain all 5 bank models
#   └── weekly Sunday 06:00 IST            → fundamentals + alt data refresh
#
# Why a single process?
#   - FinBERT is ~1.6 GB resident. 5 processes = ~8 GB. Here it loads once.
#   - Global data (FII, options, news, macro) is fetched once per cycle, not 5×.
#   - One log file, one systemd unit, one Telegram connection.
#
# Run:
#   python orchestrator.py                              # live 24×7 for all 5 banks
#   python orchestrator.py --tickers HDFCBANK.NS        # single bank
#   python orchestrator.py --once                       # one full cycle, then exit

import argparse
import logging
import os
import sqlite3

# Prevent loky/multiprocessing semaphore leak on Python 3.14 / macOS.
# Must be set before any transformers / tokenizers import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import signal as _signal
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional

# ── Logging — configure before any other import ───────────────────────────────
from config import LogConfig, PathConfig, APIConfig, TradingConfig

PathConfig.ensure_dirs()
_handlers: list = [logging.StreamHandler(sys.stdout)]
if LogConfig.LOG_TO_FILE:
    _handlers.append(logging.FileHandler(LogConfig.LOG_FILE, encoding="utf-8"))
logging.basicConfig(
    level   = getattr(logging, LogConfig.LEVEL, logging.INFO),
    format  = LogConfig.FORMAT,
    datefmt = LogConfig.DATE_FORMAT,
    handlers= _handlers,
)
for _noisy in (
    "httpx", "httpcore", "huggingface_hub", "filelock",
    "urllib3", "transformers.modeling_utils", "yfinance",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

logger = logging.getLogger("orchestrator")


# ── Interval constants (seconds) ──────────────────────────────────────────────
ALL_BANKS: List[str] = [
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "INDUSINDBK.NS",
]

INTRADAY_INTERVAL = 300    # 5-min   intraday-only pipeline (market hours)
FULL_INTERVAL     = 900    # 15-min  full 3-category pipeline
NEWS_INTERVAL     = 900    # 15-min  news fetch + FinBERT
OPTIONS_INTERVAL  = 1800   # 30-min  options chain
BSE_INTERVAL      = 1800   # 30-min  BSE announcements
FII_INTERVAL      = 7200   # 2-hr    FII/DII
SOCIAL_INTERVAL   = 7200   # 2-hr    social sentiment
GLOBAL_INTERVAL   = 3600   # 1-hr    global macro
EXIT_INTERVAL     = 60     # 1-min   stop/target exit monitor
TICK_SLEEP        = 30     # main loop resolution (s)
REVERSAL_COOLDOWN_SEC = 1800  # 30-min throttle on per-(ticker, cat, type) reversal alerts


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class MultiBankOrchestrator:
    """
    Complete autonomous trading system for 5 Indian bank stocks.

    Heavy resources are instantiated ONCE and shared across all banks:
      FinBERT, Telegram, FII/Options/Global/Social fetchers, ExitEngine.

    Per-bank resources (one instance each):
      SignalEngine  — holds Gate-4 ML model cache and feature state
      NewsFetcher   — RSS feeds are bank-specific
      BSEFetcher    — BSE code is bank-specific

    Two-tier signal cycle:
      15-min (pre-market + market hours) → full pipeline: swing + positional + intraday
       5-min (market hours only)         → intraday pipeline only
      → Swing/positional: 4 checks/hour   Intraday: 12 checks/hour
    """

    def __init__(
        self,
        tickers: List[str],
        capital: float = TradingConfig.STARTING_CAPITAL,
        db_path: str   = PathConfig.DB_PATH,
    ):
        self.tickers  = tickers
        self.capital  = capital
        self.db_path  = db_path
        self._running = True

        # ── Shared / market-wide components ──────────────────────────────────
        from scheduler.market_schedule    import MarketSchedule
        from alerts.telegram_bot          import TelegramBot
        from data.fii_fetcher             import FIIFetcher
        from data.global_fetcher          import GlobalFetcher
        from data.options_fetcher         import OptionsFetcher
        from data.social_fetcher          import SocialFetcher
        from processing.finbert_sentiment import FinBERTSentiment
        from risk.exit_engine             import ExitEngine
        from risk.circuit_breaker         import CircuitBreaker
        from risk.portfolio_tracker       import PortfolioTracker

        self.schedule        = MarketSchedule()
        self.telegram        = TelegramBot()
        self.fii_fetcher     = FIIFetcher(db_path)
        self.global_fetcher  = GlobalFetcher(db_path)
        self.social_fetcher  = SocialFetcher(db_path)
        self.exit_engine     = ExitEngine(db_path)
        self.circuit_breaker = CircuitBreaker(db_path)
        self.portfolio       = PortfolioTracker(capital, db_path)

        # FinBERT — the heavy singleton. FinBERTSentiment caches the pipeline
        # across instances so every bank reuses the same loaded model.
        self.finbert = FinBERTSentiment(db_path=db_path)

        # Optional components — fail gracefully if unavailable
        self.llm             = self._try_import(
            "processing.llm_analyzer", "LLMAnalyzer", db_path)
        self.insider_fetcher = self._try_import(
            "data.insider_fetcher", "InsiderFetcher", db_path)
        self.alt_fetcher     = self._try_import(
            "data.alternative_fetcher", "AlternativeFetcher", db_path)

        # ── Per-ticker components ─────────────────────────────────────────────
        from signals.signal_engine import SignalEngine
        from data.news_fetcher     import NewsFetcher
        from data.bse_fetcher      import BSEFetcher

        self.engines:         Dict[str, SignalEngine]   = {}
        self.news_fetchers:   Dict[str, NewsFetcher]    = {}
        self.bse_fetchers:    Dict[str, BSEFetcher]     = {}
        self.options_fetchers: Dict[str, OptionsFetcher] = {}

        for t in tickers:
            self.engines[t]         = SignalEngine(t, capital, db_path)
            self.news_fetchers[t]   = NewsFetcher(db_path=db_path, ticker=t)
            self.bse_fetchers[t]    = BSEFetcher(db_path=db_path, ticker=t)
            self.options_fetchers[t]= OptionsFetcher(db_path=db_path, ticker=t)

        # ── Alert dedup state ─────────────────────────────────────────────────
        # Per-ticker, per-category: { ticker: { category: (direction, confidence) } }
        self._last_signal_by_cat: Dict[str, Dict[str, tuple]] = {
            t: {} for t in tickers
        }
        # Reversal alerts: { (ticker, category, reversal_type) → last_sent_ts }
        # Compound tuple key (NOT position_id) so dedup survives position
        # close→reopen for the same ticker+category. Also throttled by
        # REVERSAL_COOLDOWN_SEC — without a time floor, fast model oscillation
        # produced 5–15 Telegram alerts per 15-min cycle (audit CR-3).
        # Entries are cleared when the position for that ticker actually closes.
        self._sent_reversals: Dict[tuple, float] = {}
        # Last pipeline output per ticker (for market-open alert)
        self._last_signal_per_ticker: Dict[str, Dict] = {}

        # ── Interval timestamps ───────────────────────────────────────────────
        self._last_intraday_ts = 0.0   # 5-min intraday pipeline
        self._last_full_ts     = 0.0   # 15-min full pipeline
        self._last_news_ts     = 0.0
        self._last_options_ts  = 0.0
        self._last_bse_ts      = 0.0
        self._last_fii_ts      = 0.0
        self._last_social_ts   = 0.0
        self._last_global_ts   = 0.0
        self._last_exit_ts     = 0.0
        self._task_errors: Dict[str, int] = {}

        # ── Daily / weekly one-shot flags (reset at midnight) ─────────────────
        self._premarket_done        = False
        # Per-ticker 09:15 alert flag. Previously a single bool — once any
        # bank's alert fired, the other 4 banks were silently suppressed for
        # the rest of the day. Audit CR-2.
        self._opened_today: Dict[str, bool] = {t: False for t in tickers}
        self._eod_sent              = False
        self._evening_sent          = False
        self._intraday_close_sent   = False
        self._retrain_done          = False
        self._sunday_refresh_done   = False

        # Graceful shutdown on SIGINT / SIGTERM
        _signal.signal(_signal.SIGINT,  self._stop)
        _signal.signal(_signal.SIGTERM, self._stop)

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINTS
    # ─────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Boot and run forever. Call from __main__."""
        self._boot()

        while self._running:
            try:
                self._tick()
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"tick error: {e}\n{traceback.format_exc()}")
                self._safe_telegram(f"Orchestrator tick error: {e}", "tick_error")

            # Interruptible sleep — wakes immediately on SIGINT/SIGTERM
            deadline = time.time() + TICK_SLEEP
            while self._running and time.time() < deadline:
                time.sleep(1.0)

        logger.info("Orchestrator stopped.")
        if APIConfig.TELEGRAM_TOKEN:
            try:
                self.telegram.send_raw("⛔ QUANT EDGE orchestrator stopped")
            except Exception:
                pass

    def run_once(self) -> None:
        """Single full cycle then exit — useful for --once / cron debug."""
        self._ensure_schema()
        self._load_models()
        self._run_pipeline_cycle(categories=None)   # all 3 categories

    # ─────────────────────────────────────────────────────────────────────────
    # BOOT SEQUENCE
    # ─────────────────────────────────────────────────────────────────────────

    def _boot(self) -> None:
        logger.info("=" * 65)
        logger.info("QUANT EDGE — Complete Orchestrator starting")
        logger.info(f"  Banks    : {', '.join(self.tickers)}")
        logger.info(f"  Capital  : ₹{self.capital:,.0f}")
        logger.info(f"  DB       : {self.db_path}")
        logger.info(f"  Cycles   : intraday={INTRADAY_INTERVAL}s  full={FULL_INTERVAL}s  tick={TICK_SLEEP}s")
        logger.info("=" * 65)

        self._ensure_schema()
        self._load_models()

        # Warm FinBERT now — first cycle pays zero model-load cost
        try:
            self.finbert._get_pipeline()
            logger.info("FinBERT warmed ✓")
        except Exception as e:
            logger.warning(f"FinBERT warm-up skipped: {e}")

        # Seed static reference data (idempotent — safe to run on every boot)
        try:
            from data.rbi_fetcher import RBIFetcher
            RBIFetcher(self.db_path).seed_historical_rates()
        except Exception as e:
            logger.warning(f"RBI seed: {e}")

        # Bootstrap global macro if the table is sparse (<7 rows needed for
        # rolling features to be meaningful). Synthesises history from the
        # 30-day price series we already fetch — lookahead-safe.
        try:
            if self.global_fetcher.get_row_count() < 7:
                logger.info("global_snapshots <7 rows — running bootstrap_history")
                self.global_fetcher.bootstrap_history(days=30)
        except Exception as e:
            logger.warning(f"global bootstrap: {e}")

        # Bootstrap fundamentals on first boot. The Sunday 06:00 refresh
        # populates these, but a fresh deployment would wait up to a week,
        # and every positional signal eats a +5pp Gate 6 threshold via the
        # H6 fundamentals_stale bubble until rows exist.
        try:
            conn = sqlite3.connect(self.db_path)
            n_funds = conn.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0]
            conn.close()
            if n_funds < len(self.tickers):
                logger.info(
                    f"fundamentals has {n_funds} rows (<{len(self.tickers)}) — "
                    "running first-boot fetch for all banks"
                )
                from processing.fundamental import FundamentalProcessor
                for t in self.tickers:
                    self._safe(
                        f"fundamentals_bootstrap_{t}",
                        FundamentalProcessor(self.db_path, t).fetch_and_update,
                    )
        except Exception as e:
            logger.warning(f"fundamentals bootstrap: {e}")

        for t in self.tickers:
            try:
                self.bse_fetchers[t].sync_earnings_calendar()
            except Exception as e:
                logger.warning(f"[{t}] earnings calendar sync: {e}")

        self.telegram.test_connection()
        self.schedule.log_schedule()

        if APIConfig.TELEGRAM_TOKEN:
            try:
                self.telegram.send_raw(
                    f"🟢 QUANT EDGE up · {len(self.tickers)} banks\n"
                    f"intraday={INTRADAY_INTERVAL//60}m · full={FULL_INTERVAL//60}m · tick={TICK_SLEEP}s"
                )
            except Exception:
                pass

    def _load_models(self) -> None:
        """Load all saved model artefacts for every ticker into memory."""
        from models.train_all import ModelTrainer
        for t in self.tickers:
            try:
                ModelTrainer(t, self.capital, self.db_path).load_all()
                logger.info(f"[{t}] models loaded ✓")
            except Exception as e:
                logger.warning(f"[{t}] model load: {e}")

    def _ensure_schema(self) -> None:
        """Idempotent DB schema setup — safe to run on every boot."""
        try:
            from database.db_setup import DatabaseSetup
            DatabaseSetup(self.db_path).setup_all()
        except Exception as e:
            logger.error(f"schema setup: {e}\n{traceback.format_exc()}")

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN TICK  (runs every 30 seconds)
    # ─────────────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        now        = time.time()
        now_dt     = self.schedule.now_ist()
        is_trading = self.schedule.is_trading_day()
        is_open    = self.schedule.is_market_open()
        is_pre     = self.schedule.is_pre_market_analysis_time()  # 6:00–9:15

        # ── Midnight: reset daily flags ───────────────────────────────────────
        if now_dt.hour == 0 and now_dt.minute == 0:
            self._reset_daily_flags(now_dt)

        # ── Sunday 00:00–00:30 — Weekly model retrain ─────────────────────────
        if self.schedule.is_sunday_midnight() and not self._retrain_done:
            self._safe("weekly_retrain", self._run_weekly_retrain)
            self._retrain_done = True

        # ── Sunday 06:00–06:30 — Weekly data refresh ──────────────────────────
        if self.schedule.is_sunday_morning() and not self._sunday_refresh_done:
            self._safe("weekly_data_refresh", self._run_weekly_data_refresh)
            self._sunday_refresh_done = True

        # ── 06:00 — Pre-market analysis (fires once per trading day) ──────────
        if (is_trading and is_pre
                and now_dt.hour == 6 and now_dt.minute < 15
                and not self._premarket_done):
            self._safe("premarket", self._run_pre_market)
            self._premarket_done = True

        # ── Data fetchers (each gated by its own interval) ────────────────────
        self._tick_data_tasks(now, is_trading, is_open)

        # ── Exit monitor — every 60s during market hours ──────────────────────
        if is_open and (now - self._last_exit_ts) >= EXIT_INTERVAL:
            self._run_exit_checks()
            self._last_exit_ts = now

        # ── Circuit breaker — every tick on trading days ──────────────────────
        if is_trading:
            self._safe("circuit_breaker", self._run_circuit_breaker_check)

        # ── Signal pipelines ──────────────────────────────────────────────────
        if is_open:
            full_due     = (now - self._last_full_ts)     >= FULL_INTERVAL
            intraday_due = (now - self._last_intraday_ts) >= INTRADAY_INTERVAL

            if full_due:
                # 15-min boundary: full pipeline — swing + positional + intraday
                self._run_pipeline_cycle(categories=None)
                self._last_full_ts     = now
                self._last_intraday_ts = now          # reset so 5-min doesn't double-fire

            elif intraday_due:
                # Between 15-min marks: intraday only
                self._run_pipeline_cycle(categories=["intraday"])
                self._last_intraday_ts = now

        elif is_trading and is_pre:
            # Pre-market 6:00–9:15: swing + positional only (no intraday in pre-market)
            if (now - self._last_full_ts) >= FULL_INTERVAL:
                self._run_pipeline_cycle(categories=["swing", "positional"])
                self._last_full_ts     = now
                self._last_intraday_ts = now

        # ── 09:15 — Market open alert (one-shot per ticker) ───────────────────
        if (is_open and now_dt.hour == 9 and now_dt.minute < 20
                and not all(self._opened_today.values())):
            self._send_market_open_alerts()  # internally honors per-ticker flag

        # ── 15:15–15:30 — Intraday close warning ──────────────────────────────
        if self.schedule.is_intraday_close_window():
            self._run_exit_checks()
            if not self._intraday_close_sent and now_dt.minute >= 15:
                self._safe_telegram(
                    self.telegram.send_intraday_close_warning,
                    "intraday_close_warning",
                )
                self._intraday_close_sent = True

        # ── 15:45 — EOD summary ───────────────────────────────────────────────
        if self.schedule.is_end_of_day() and not self._eod_sent:
            self._safe("eod_summary", self._run_eod_summary)
            self._eod_sent = True

        # ── 18:00 — Evening performance report ───────────────────────────────
        if self.schedule.is_evening_report_time() and not self._evening_sent:
            self._safe("evening_report", self._run_evening_report)
            self._evening_sent = True

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL PIPELINE
    # ─────────────────────────────────────────────────────────────────────────

    def _run_pipeline_cycle(self, categories: Optional[List[str]]) -> None:
        """
        Run the 6-gate signal pipeline for every ticker.

        categories=None              → full cycle: swing + positional + intraday
        categories=["intraday"]      → 5-min cycle: intraday alerts only
        categories=["swing","positional"] → pre-market: no intraday trades
        """
        label = "FULL" if categories is None else "+".join(c.upper() for c in categories)
        logger.info(f"── Pipeline [{label}] {'─' * (50 - len(label))}")

        for ticker in self.tickers:
            try:
                signal = self.engines[ticker].run()
                if signal:
                    self._last_signal_per_ticker[ticker] = signal
                    self._dispatch_alerts(ticker, signal, categories)
                    self._dispatch_reversal_alerts(ticker, signal, categories)
                    self._open_paper_trades(ticker, signal, categories)
            except Exception as e:
                logger.error(f"[{ticker}] pipeline error: {e}\n{traceback.format_exc()}")
                self._safe_telegram(
                    f"[{ticker}] pipeline failed: {e}", f"pipeline_{ticker}"
                )

    def _open_paper_trades(
        self,
        ticker:     str,
        signal:     Dict,
        categories: Optional[List[str]],
    ) -> None:
        """
        Auto-open one paper position per passing category.
        Skips categories not in scope for this cycle (e.g. swing on 5-min run).

        Risk guards (in order, all per ticker × category):
          1. Capital-mode allowed_tf — SMALL only opens swing, etc.
          2. Duplicate (ticker, category) already open → skip.
          3. max_positions cap from CapitalMode (2/3/5).
          4. PositionSizer.check_exposure() — 40% per-name, 80% total caps.
        Exposure/position state is recomputed every iteration so the second
        signal in a batch sees the first one's open position.
        """
        from risk.capital_mode  import CapitalMode
        from risk.position_sizer import PositionSizer

        emitted = signal.get("signals") or []
        if not emitted:
            return

        # Belt-and-suspenders: if the circuit breaker is currently PAUSE/HALT
        # in the DB, refuse to open new positions regardless of what
        # Gate 6 emitted earlier in this cycle.
        cb_state = self.circuit_breaker.get_current_state()
        if cb_state.get("level") in ("PAUSE", "HALT"):
            logger.info(
                f"[{ticker}] open suppressed — circuit breaker "
                f"{cb_state['level']}: {cb_state.get('reason','')}"
            )
            return

        cap_cfg     = CapitalMode.get_config(self.capital)
        allowed_tf  = set(cap_cfg.get("allowed_tf", ["swing", "intraday", "positional"]))
        max_pos     = int(cap_cfg.get("max_positions", 5))
        cap_mode    = cap_cfg.get("mode", "FULL")
        sizer       = PositionSizer()

        for sig in emitted:
            cat = sig.get("category", sig.get("trade_type", "swing"))

            if categories and cat not in categories:
                continue

            # Re-read state inside the loop — each successful open updates DB.
            open_positions = self.exit_engine._get_open_positions()

            # Guard 1 — capital-mode timeframe allow-list.
            if cat not in allowed_tf:
                logger.info(
                    f"[{ticker}] skipped {cat}: {cap_mode} mode disallows "
                    f"(allowed={sorted(allowed_tf)})"
                )
                continue

            # Guard 2 — duplicate (ticker, category) already open.
            already_open = any(
                p.get("ticker") == ticker and p.get("trade_type") == cat
                for p in open_positions
            )
            if already_open:
                continue

            # Guard 3 — max_positions cap (portfolio-wide).
            if len(open_positions) >= max_pos:
                logger.info(
                    f"[{ticker}] skipped {cat}: {cap_mode} max_positions={max_pos} "
                    f"reached (currently {len(open_positions)} open)"
                )
                continue

            # Guard 4 — per-name (40%) + total portfolio (80%) exposure caps.
            entry  = float(sig.get("entry_price", 0))
            shares = int(sig.get("shares", 0))
            proposed_value = entry * max(shares, 0)
            exp = sizer.check_exposure(
                ticker         = ticker,
                proposed_value = proposed_value,
                capital        = self.capital,
                db_path        = self.db_path,
            )
            if not exp.get("allowed", True):
                logger.info(
                    f"[{ticker}] skipped {cat}: exposure block — {exp.get('reason','?')}"
                )
                continue

            try:
                self.exit_engine.open_position(
                    signal       = sig["signal"],
                    entry_price  = entry,
                    stop_price   = float(sig.get("stop_loss",    0)),
                    target_price = float(sig.get("target_price", 0)),
                    shares       = shares,
                    risk_amount  = float(sig.get("risk_amount",  0)),
                    trade_type   = cat,
                    signal_uuid  = signal.get("signal_uuid", ""),
                    alignment    = signal.get("alignment",   ""),
                    regime       = signal.get("regime",      ""),
                    confidence   = float(sig.get("confidence", 0)),
                    entry_quality= sig.get("entry_quality", ""),
                    ticker       = ticker,
                )
                logger.info(
                    f"[{ticker}] opened paper trade: {cat} {sig['signal']} "
                    f"@ ₹{entry:.2f} · stop=₹{sig.get('stop_loss', 0):.2f} · "
                    f"target=₹{sig.get('target_price', 0):.2f} "
                    f"(value=₹{proposed_value:,.0f}, "
                    f"name_exp_after=₹{exp.get('ticker_exposure',0)+proposed_value:,.0f})"
                )
            except Exception as e:
                logger.warning(f"[{ticker}] open_position({cat}): {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # ALERT DISPATCH
    # ─────────────────────────────────────────────────────────────────────────

    def _dispatch_alerts(
        self,
        ticker:     str,
        signal:     Dict,
        categories: Optional[List[str]],
    ) -> None:
        """
        Emit a Telegram alert for each category whose direction or confidence
        changed materially since the last alert for that category.

        categories=None → check all categories.
        categories=[…]  → only check the listed categories; others' dedup state
                          is left untouched so they fire correctly on next full cycle.

        Dedup rules (per ticker × category):
          • New non-FLAT signal for a category that was FLAT → send
          • Direction flipped LONG↔SHORT                    → send
          • Same direction but confidence shifted ≥5%       → send
          • Category went FLAT after being active           → send "went flat"
          • Same direction, small delta                     → suppress
        """
        per_cat = (signal or {}).get("gate_results", {}).get("per_category", {})
        if not per_cat:
            return

        # Build lookup: category → emitted signal dict (has entry/stop/target)
        emitted_by_cat: Dict[str, Dict] = {
            s.get("category"): s
            for s in (signal.get("signals") or [])
        }

        dedup = self._last_signal_by_cat[ticker]

        for category, cat_info in per_cat.items():
            if categories and category not in categories:
                continue

            direction  = cat_info.get("direction", "FLAT")
            confidence = float(cat_info.get("confidence", 0) or 0)
            passed     = cat_info.get("passed", False)

            # ── Category is FLAT or failed gates ──────────────────────────────
            if not passed or direction in (None, "FLAT"):
                prev = dedup.get(category)
                if prev and prev[0] not in (None, "FLAT"):
                    # Was active last cycle — notify once that it went flat
                    try:
                        self.telegram.send_raw(
                            f"⬜ [{ticker}] {category.upper()} → FLAT "
                            f"(was {prev[0]} @ {prev[1]:.0%})"
                        )
                    except Exception as e:
                        logger.warning(f"[{ticker}] flat notify ({category}): {e}")
                    dedup.pop(category, None)
                continue

            # ── Dedup check ───────────────────────────────────────────────────
            prev       = dedup.get(category)
            same_dir   = prev and prev[0] == direction
            small_move = same_dir and abs(prev[1] - confidence) < 0.05
            if same_dir and small_move:
                continue

            # ── Build alert payload ───────────────────────────────────────────
            emitted = emitted_by_cat.get(category, {})
            payload  = {
                **signal,
                "ticker":     ticker,
                "category":   category,
                "trade_type": category,
                "signal":     direction,
                "confidence": confidence,
            }
            # Overlay per-category trade levels (entry/stop/target/shares)
            for key in ("entry_price", "stop_loss", "target_price",
                        "shares", "reward_risk", "size_mult"):
                if key in emitted:
                    payload[key] = emitted[key]

            try:
                self.telegram.send_signal(payload)
                logger.info(
                    f"[{ticker}] ALERT {category.upper()} {direction} "
                    f"@ conf={confidence:.0%}"
                )
                dedup[category] = (direction, confidence)
            except Exception as e:
                logger.warning(f"[{ticker}] send_signal ({category}): {e}")

    def _dispatch_reversal_alerts(
        self,
        ticker:     str,
        signal:     Dict,
        categories: Optional[List[str]] = None,
    ) -> None:
        """
        Advisory alert when an open position's model direction reverses or the
        regime turns adverse. One alert per reversal event; suppressed until the
        position for that ticker+category closes.

        Dedup key: (ticker, category, reversal_type) tuple — NOT position_id.

        Why NOT position_id:
          When a paper position closes and immediately re-opens in the same cycle
          (e.g. AXIS POSITIONAL), the new position gets a fresh DB id, bypassing
          any dedup keyed by the old id and re-firing the same alert every 5 min.
          A compound key survives position ID churn: once (AXISBANK.NS, positional,
          REGIME_CHANGE) is recorded, it won't fire again until that ticker's
          position actually closes — at which point _run_exit_checks clears all
          keys for that ticker.

        Category scoping:
          During 5-min intraday-only cycles (categories=["intraday"]), we skip
          POSITIONAL and SWING reversal checks. Those categories are evaluated on
          the next 15-min full cycle, preventing false alerts for long-duration
          trades during short cycles.
        """
        try:
            reversals = self.engines[ticker].check_model_reversals(signal)
        except Exception as e:
            logger.warning(f"[{ticker}] check_model_reversals: {e}")
            return

        for rev in reversals:
            cat   = rev.get("category", "")
            rtype = rev.get("type", "")

            # Skip categories not in scope for this cycle
            if categories and cat not in categories:
                continue

            key = (ticker, cat, rtype)
            last_sent = self._sent_reversals.get(key, 0.0)
            if (time.time() - last_sent) < REVERSAL_COOLDOWN_SEC:
                continue    # throttled — same alert within cooldown window

            self._sent_reversals[key] = time.time()
            logger.warning(
                f"[{ticker}] reversal: {rtype} cat={cat} "
                f"{rev.get('open_signal')} → {rev.get('model_signal')}"
            )
            try:
                self.telegram.send_raw(
                    self.telegram.formatter.format_model_reversal(rev)
                )
            except Exception as e:
                logger.warning(f"[{ticker}] reversal telegram: {e}")

    def _run_exit_checks(self) -> None:
        """
        Check all open positions across every ticker for stop/target/trail breaches.

        Architecture:
          Each SignalEngine.check_exits() fetches ITS OWN ticker's current price
          and filters DB to positions for THAT ticker only (ticker-scoped).
          This prevents HDFCBANK's ₹785 price from being applied to ICICIBANK
          positions at ₹1400, which was the root cause of wrong stop/target hits.

          After detecting an exit, we CLOSE the position in the DB immediately
          before moving to the next engine. This ensures:
            • The same position is never processed by two different engines.
            • Positions don't accumulate and spam alerts every 60s.

          _closed_this_cycle is a per-call safeguard against the edge case where
          two engines both see the same position in the same snapshot.
        """
        _closed_this_cycle: set = set()

        for ticker in self.tickers:
            try:
                exits = self.engines[ticker].check_exits()
                for exit_info in exits:
                    pos_id = exit_info.get("position_id")
                    if not pos_id or pos_id in _closed_this_cycle:
                        continue  # already handled by a previous engine this cycle
                    _closed_this_cycle.add(pos_id)
                    try:
                        # Close position in DB FIRST — prevents repeat alerts
                        if exit_info.get("partial"):
                            closed = self.exit_engine.close_position_partial(
                                pos_id,
                                exit_info["exit_price"],
                                exit_info["reason"],
                                exit_info.get("book_pct", 0.50),
                            )
                        else:
                            closed = self.exit_engine.close_position(
                                pos_id,
                                exit_info["exit_price"],
                                exit_info["reason"],
                            )
                        # Merge closed info (pnl_amount, hold_days) for Telegram
                        merged = {**exit_info, **(closed or {})}
                        self.telegram.send_exit(merged)
                        logger.info(
                            f"[{ticker}] EXIT: reason={exit_info.get('reason','?')} "
                            f"pos={pos_id} price=₹{exit_info.get('exit_price',0):.2f} "
                            f"pnl={exit_info.get('pnl_pct', 0):+.2%}"
                        )
                        # Clear ALL reversal-dedup entries for this ticker.
                        # Using tuple key (ticker, cat, rtype) so we can cleanly
                        # purge by ticker without needing the old position_id.
                        for _key in [k for k in self._sent_reversals if k[0] == ticker]:
                            del self._sent_reversals[_key]
                    except Exception as e:
                        logger.warning(f"[{ticker}] close+exit pos={pos_id}: {e}")
            except Exception as e:
                logger.warning(f"[{ticker}] exit check: {e}")

    def _send_market_open_alerts(self) -> None:
        """09:15 — send market-open confirmation for each bank that has a live
        non-FLAT signal. One alert per ticker per day (per-ticker dedup); the
        prior implementation broadcast a single universe-wide alert.
        """
        for ticker in self.tickers:
            if self._opened_today.get(ticker, False):
                continue
            sig = self._last_signal_per_ticker.get(ticker)
            if sig and sig.get("signal") not in ("FLAT", None):
                try:
                    self.telegram.send_market_open(sig)
                    self._opened_today[ticker] = True
                except Exception as e:
                    logger.warning(f"[{ticker}] market open alert: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # DATA TASKS
    # ─────────────────────────────────────────────────────────────────────────

    def _tick_data_tasks(self, now: float, is_trading: bool, is_open: bool) -> None:
        """Check and run each data fetcher if its interval has elapsed.

        Weekend gating rationale:
          News / social  — financial news and social posts happen 24/7; fetch always.
          Options chain  — meaningful only when NSE is open; gated on is_open.
          BSE / FII/DII  — NSE publishes only on trading days; gated on is_trading.
          Global macro   — always (S&P, VIX, DXY don't take weekends off).
        """

        # News + FinBERT — every 15 min, 24/7 (news publishes on weekends too)
        # NewsFetcher.fetch_all() has its own 900s internal guard so this is safe.
        if (now - self._last_news_ts) >= NEWS_INTERVAL:
            self._safe("news", self._fetch_news_and_score)
            self._last_news_ts = now

        # Options chain — every 30 min during market hours only, per bank.
        # Each bank's options chain is fetched from NSE separately so PCR /
        # max-pain / IV percentile features stay isolated.
        if is_open and (now - self._last_options_ts) >= OPTIONS_INTERVAL:
            for t, fetcher in self.options_fetchers.items():
                self._safe(f"options[{t}]", fetcher.fetch_and_calculate)
            self._last_options_ts = now

        # BSE announcements — every 30 min on trading days (NSE only publishes weekdays)
        if is_trading and (now - self._last_bse_ts) >= BSE_INTERVAL:
            self._safe("bse", self._run_bse_pipeline)
            self._last_bse_ts = now

        # FII/DII — every 2 hr on trading days (NSE API only has weekday data)
        if is_trading and (now - self._last_fii_ts) >= FII_INTERVAL:
            self._safe("fii",      self.fii_fetcher.fetch_fii_dii_data)
            self._safe("fii_bulk", self.fii_fetcher.fetch_bulk_deals)
            self._last_fii_ts = now

        # Social sentiment — every 2 hr, 24/7 (Reddit/Twitter post on weekends)
        if (now - self._last_social_ts) >= SOCIAL_INTERVAL:
            self._safe("social", self.social_fetcher.fetch_all)
            self._last_social_ts = now

        # Global macro — every 1 hr always (S&P/VIX/DXY needed for Gate 2)
        if (now - self._last_global_ts) >= GLOBAL_INTERVAL:
            self._safe("global", self.global_fetcher.fetch_overnight_signals)
            self._last_global_ts = now

    def _fetch_news_and_score(self) -> None:
        """
        Fetch RSS for all 5 banks then run FinBERT once on the combined
        unscored pool. Cheaper than 5 separate scoring loops.
        Aggregates per-source counts into one cycle summary line so we can
        spot a feed source going dark from a single log grep.
        """
        from data.news_fetcher import NewsFetcher
        agg: Dict[str, int] = {}
        for t in self.tickers:
            try:
                for source, n in self.news_fetchers[t].fetch_all().items():
                    agg[source] = agg.get(source, 0) + n
            except Exception as e:
                logger.warning(f"[{t}] news fetch: {e}")

        total = sum(agg.values())
        summary = " ".join(f"{k}={v}" for k, v in sorted(agg.items()))
        logger.info(f"news cycle: {summary} [total={total}]")

        if total == 0:
            NewsFetcher._consecutive_empty_cycles += 1
            if (NewsFetcher._consecutive_empty_cycles >= 2
                    and not NewsFetcher._empty_alarm_sent):
                msg = (
                    f"⚠️ News pipeline empty for "
                    f"{NewsFetcher._consecutive_empty_cycles} consecutive cycles — "
                    f"every feed source returned 0 articles."
                )
                logger.warning(msg)
                try:
                    self.telegram.send_raw(msg)
                    NewsFetcher._empty_alarm_sent = True
                except Exception as e:
                    logger.warning(f"news empty-cycle alarm telegram failed: {e}")
        else:
            NewsFetcher._consecutive_empty_cycles = 0
            NewsFetcher._empty_alarm_sent = False

        try:
            n = self.finbert.process_unscored(batch_size=100)
            if n:
                logger.info(f"FinBERT scored {n} articles")
        except Exception as e:
            logger.warning(f"FinBERT process_unscored: {e}")

    def _run_bse_pipeline(self) -> None:
        """
        Fetch BSE announcements for all banks.
        Run LLM on up to 2 high-priority announcements (cost control).
        """
        llm_calls = 0
        for t in self.tickers:
            try:
                self.bse_fetchers[t].fetch_bse_announcements()
            except Exception as e:
                logger.warning(f"[{t}] BSE fetch: {e}")
                continue

            if self.llm and llm_calls < 2:
                try:
                    high = self.bse_fetchers[t].get_high_priority_unanalyzed() or []
                    for ann in high[:1]:    # max 1 LLM call per bank per BSE cycle
                        result = self.llm.analyze_bse_announcement(
                            ann["title"], ann["description"]
                        )
                        if result:
                            score = float(result.get("impact_score", 0))
                            self.bse_fetchers[t].mark_llm_analyzed(ann["id"], score)
                            if abs(score) > 0.4:
                                self._safe_telegram(
                                    f"📋 [{t}] {ann['title'][:100]}\n"
                                    f"Impact: {score:+.2f} | {result.get('reason', '')[:80]}",
                                    f"bse_{t}",
                                )
                            llm_calls += 1
                except Exception as e:
                    logger.warning(f"[{t}] BSE LLM: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # DAILY TASKS
    # ─────────────────────────────────────────────────────────────────────────

    def _run_pre_market(self) -> None:
        """
        06:00 IST — fetch overnight global data, run swing+positional pipeline,
        and send pre-market alert. Updates _last_full_ts so the regular 15-min
        pipeline doesn't immediately double-fire.
        """
        logger.info("── Pre-market analysis (06:00) ──────────────────────────")
        self._safe("global_overnight", self.global_fetcher.fetch_overnight_signals)
        self._run_pipeline_cycle(categories=["swing", "positional"])
        # Update timestamps so we don't repeat for another FULL_INTERVAL seconds
        now = time.time()
        self._last_full_ts     = now
        self._last_intraday_ts = now
        logger.info("── Pre-market complete ──────────────────────────────────")

    def _run_circuit_breaker_check(self) -> None:
        """
        Check portfolio drawdown / consecutive losses.
        `CircuitBreaker.check()` persists state transitions to DB, so future
        signal-engine cycles read the live level via RiskFeatures regardless
        of which tick discovered the breach.
        """
        try:
            from risk.capital_mode import CapitalMode
            cap_mode = CapitalMode.detect(self.capital)

            stats         = self.portfolio.get_performance_stats() or {}
            monthly_dd    = float(stats.get("monthly_dd_pct",      0))
            consec_losses = int(stats.get("consecutive_losses",     0))
            _, cb_status  = self.circuit_breaker.check(
                capital_mode       = cap_mode,
                monthly_dd_pct     = monthly_dd,
                consecutive_losses = consec_losses,
            )
            if (cb_status or {}).get("level") in ("PAUSE", "HALT"):
                self._safe_telegram(
                    lambda s=cb_status: self.telegram.send_circuit_breaker(s),
                    "circuit_breaker",
                )
        except Exception as e:
            logger.warning(f"circuit breaker check: {e}")

    def _run_eod_summary(self) -> None:
        """15:45 IST — final exit check + portfolio heatmap via Telegram."""
        logger.info("── EOD Summary ──────────────────────────────────────────")
        self._run_exit_checks()
        try:
            from dashboard.portfolio_heatmap import PortfolioHeatmap
            hm   = PortfolioHeatmap(self.capital, self.db_path)
            heat = hm.generate()
            self._safe_telegram(hm.format_telegram(heat), "eod_heatmap")
        except Exception as e:
            logger.warning(f"EOD heatmap: {e}")

    def _run_evening_report(self) -> None:
        """18:00 IST — performance report. Psychology check added on Fridays."""
        logger.info("── Evening Report ───────────────────────────────────────")
        try:
            from dashboard.performance import PerformanceDashboard
            perf   = PerformanceDashboard(self.capital, self.db_path)
            report = perf.daily_report()
            self._safe_telegram(perf.format_telegram(report), "evening_perf")
        except Exception as e:
            logger.warning(f"evening performance report: {e}")

        if datetime.now().weekday() == 4:   # Friday → psychology check
            try:
                from dashboard.psychology_tracker import PsychologyTracker
                psych = PsychologyTracker(self.db_path)
                self._safe_telegram(
                    psych.format_telegram(psych.analyse()), "psychology"
                )
            except Exception as e:
                logger.warning(f"psychology report: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # WEEKLY TASKS
    # ─────────────────────────────────────────────────────────────────────────

    def _run_weekly_retrain(self) -> None:
        """
        Sunday 00:00 IST — retrain all 5 bank models.
        Runs sequentially (one bank at a time) to stay within RAM budget.
        Reloads fresh model weights into SignalEngines after training.
        """
        logger.info("=" * 60)
        logger.info("WEEKLY RETRAIN — Sunday 00:00 IST")
        logger.info("=" * 60)

        try:
            self.telegram.send_raw(
                f"🔄 Weekly retrain starting ({len(self.tickers)} banks)…\n"
                "Expected: 20–40 min."
            )
        except Exception:
            pass

        from models.train_all import ModelTrainer
        results: Dict[str, Dict] = {}

        for t in self.tickers:
            logger.info(f"Retraining [{t}] …")
            try:
                summary  = ModelTrainer(t, self.capital, self.db_path).train_all()
                results[t] = summary or {}
                logger.info(f"[{t}] retrain complete")
            except Exception as e:
                logger.error(f"[{t}] retrain failed: {e}\n{traceback.format_exc()}")
                results[t] = {"error": str(e)}

        ok = sum(1 for v in results.values() if "error" not in v)
        status_lines = "\n".join(
            f"  {'✓' if 'error' not in v else '✗'} {t}"
            + (f": {v['error'][:60]}" if "error" in v else "")
            for t, v in results.items()
        )

        try:
            self.telegram.send_raw(
                f"{'✅' if ok == len(self.tickers) else '⚠️'} "
                f"Weekly retrain: {ok}/{len(self.tickers)} banks\n{status_lines}"
            )
            # Detailed report for the primary bank
            primary = results.get(TradingConfig.TICKER, {})
            if primary and "error" not in primary:
                self.telegram.send_retrain_report(primary)
        except Exception:
            pass

        # Reload fresh weights so next cycle uses the new models
        self._load_models()

    def _run_weekly_data_refresh(self) -> None:
        """
        Sunday 06:00 IST — refresh slow-moving data:
        insider trades, alternative data, RBI rates, fundamentals.
        """
        logger.info("── Weekly data refresh (Sunday 06:00) ──────────────────")

        if self.insider_fetcher:
            self._safe("insider_refresh", self.insider_fetcher.fetch_all)

        if self.alt_fetcher:
            self._safe("alt_data_refresh", self.alt_fetcher.fetch_all)

        try:
            from data.rbi_fetcher import RBIFetcher
            self._safe("rbi_seed", RBIFetcher(self.db_path).seed_historical_rates)
        except Exception as e:
            logger.warning(f"RBI refresh: {e}")

        # Fundamentals for all 5 banks (Screener.in, quarterly data)
        try:
            from processing.fundamental import FundamentalProcessor
            for t in self.tickers:
                self._safe(
                    f"fundamentals_{t}",
                    FundamentalProcessor(self.db_path, t).fetch_and_update,
                )
        except Exception as e:
            logger.warning(f"fundamentals refresh: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_daily_flags(self, now_dt: datetime) -> None:
        """Reset one-shot daily flags at midnight. Weekly flags reset on Monday."""
        self._premarket_done      = False
        for _t in self._opened_today:
            self._opened_today[_t] = False
        self._eod_sent            = False
        self._evening_sent        = False
        self._intraday_close_sent = False
        if now_dt.weekday() == 0:   # Monday midnight → reset weekly flags too
            self._retrain_done        = False
            self._sunday_refresh_done = False
        logger.info(f"Daily flags reset ({now_dt.strftime('%A')})")

    def _safe(self, name: str, fn) -> None:
        """
        Run fn(), catch all exceptions, log them.
        After 3 consecutive failures of the same task, escalate to Telegram.
        """
        try:
            fn()
            self._task_errors[name] = 0
        except Exception as e:
            count = self._task_errors.get(name, 0) + 1
            self._task_errors[name] = count
            logger.warning(f"task '{name}' failed (×{count}): {e}")
            if count == 3:
                self._safe_telegram(
                    f"task '{name}' failed 3× in a row: {e}", f"task_{name}"
                )

    def _safe_telegram(self, content, dedup_key: str = "") -> None:
        """Send a Telegram message or call a lambda safely. Never raises."""
        if not APIConfig.TELEGRAM_TOKEN:
            return
        try:
            if callable(content):
                content()
            else:
                self.telegram.send_raw(f"⚠️ {content}")
        except Exception as e:
            logger.warning(f"telegram send ({dedup_key}): {e}")

    def _stop(self, *_args) -> None:
        logger.info("Shutdown signal received — stopping after current tick")
        self._running = False

    @staticmethod
    def _try_import(module: str, cls: str, *args):
        """Attempt to import and instantiate an optional component."""
        try:
            import importlib
            mod = importlib.import_module(module)
            return getattr(mod, cls)(*args)
        except Exception as e:
            logger.warning(f"{cls} unavailable ({module}): {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QUANT EDGE — complete multi-bank trading system (single process)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orchestrator.py                               # all 5 banks, live 24×7
  python orchestrator.py --tickers HDFCBANK.NS         # single bank
  python orchestrator.py --tickers HDFCBANK.NS,ICICIBANK.NS
  python orchestrator.py --once                        # one full cycle then exit
""",
    )
    p.add_argument(
        "--tickers",
        default=",".join(ALL_BANKS),
        help="Comma-separated NSE symbols (default: all 5 banks)",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Run one full cycle then exit (useful for cron / debugging)",
    )
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    invalid = [t for t in tickers if t not in ALL_BANKS]
    if invalid:
        logger.error(f"Unknown ticker(s): {invalid}. Allowed: {ALL_BANKS}")
        sys.exit(2)

    orch = MultiBankOrchestrator(
        tickers = tickers,
        capital = TradingConfig.STARTING_CAPITAL,
        db_path = PathConfig.DB_PATH,
    )

    if args.once:
        orch.run_once()
    else:
        orch.start()


if __name__ == "__main__":
    main()
