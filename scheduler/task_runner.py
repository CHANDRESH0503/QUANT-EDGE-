# scheduler/task_runner.py
# Master task runner — orchestrates all scheduled jobs
# Entry point for the running system: python main.py → TaskRunner.start()
# Connected to: every module in the system

import time
import logging
import traceback
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

from scheduler.market_schedule   import MarketSchedule
from signals.signal_engine       import SignalEngine
from alerts.telegram_bot         import TelegramBot
from dashboard.performance       import PerformanceDashboard
from dashboard.psychology_tracker import PsychologyTracker
from dashboard.portfolio_heatmap  import PortfolioHeatmap
from risk.portfolio_tracker      import PortfolioTracker
from risk.circuit_breaker        import CircuitBreaker
from risk.exit_engine            import ExitEngine
from models.train_all            import ModelTrainer
from data.news_fetcher           import NewsFetcher
from data.bse_fetcher            import BSEFetcher
from data.fii_fetcher            import FIIFetcher
from data.global_fetcher         import GlobalFetcher
from data.options_fetcher        import OptionsFetcher
from data.social_fetcher         import SocialFetcher
from data.insider_fetcher        import InsiderFetcher
from data.alternative_fetcher    import AlternativeFetcher
from processing.finbert_sentiment import FinBERTSentiment
from processing.llm_analyzer     import LLMAnalyzer


class TaskRunner:
    """
    Master task orchestrator — runs the entire system on a schedule.

    20yr trader principle:
    The system must run itself. If it requires manual intervention
    every day, it will fail on the one day you are not watching.
    Build for unattended 24x7 operation from day one.

    Task hierarchy:
    ─────────────────────────────────────────────────────────────
    CRITICAL (run every 15 min, market hours):
      signal pipeline → exits check → circuit breaker check

    HIGH (run daily):
      06:00 global fetch → pre-market signal
      09:15 market open alert
      15:15 intraday close warning
      15:45 EOD summary
      18:00 performance report

    MEDIUM (run on trigger):
      every 15 min: news fetch → FinBERT scoring
      every 30 min: BSE announcements → LLM if high priority
      every 30 min: options chain refresh
      every 2 hours: FII data + social sentiment

    LOW (run weekly):
      Sunday 00:00: retrain ALL models
      Sunday 06:00: fundamentals + alt data refresh
      Sunday 18:00: psychology + walk-forward report
    ─────────────────────────────────────────────────────────────

    Error policy:
    Any single task failure → log + Telegram error alert → continue.
    Three consecutive failures of the SAME task → escalate alert.
    System NEVER stops running due to one component failure.
    """

    SIGNAL_INTERVAL_SEC   = 900   # 15 minutes
    NEWS_INTERVAL_SEC     = 900   # 15 minutes
    OPTIONS_INTERVAL_SEC  = 1800  # 30 minutes
    BSE_INTERVAL_SEC      = 1800  # 30 minutes
    FII_INTERVAL_SEC      = 7200  # 2 hours
    SOCIAL_INTERVAL_SEC   = 7200  # 2 hours

    def __init__(
        self,
        ticker:  str   = "HDFCBANK.NS",
        capital: float = 100_000.0,
        db_path: str   = "database/trading.db",
    ):
        self.ticker  = ticker
        self.capital = capital
        self.db_path = db_path

        # ── Core components ────────────────────────────────────────
        self.schedule        = MarketSchedule()
        self.signal_engine   = SignalEngine(ticker, capital, db_path)
        self.telegram        = TelegramBot(ticker=ticker)
        self.performance     = PerformanceDashboard(capital, db_path)
        self.psychology      = PsychologyTracker(db_path)
        self.heatmap         = PortfolioHeatmap(capital, db_path)
        self.portfolio       = PortfolioTracker(capital, db_path)
        self.circuit_breaker = CircuitBreaker(db_path)
        self.exit_engine     = ExitEngine(db_path)
        self.trainer         = ModelTrainer(ticker, capital, db_path)

        # ── Data fetchers ──────────────────────────────────────────
        self.news_fetcher   = NewsFetcher(db_path, ticker=ticker)
        self.bse_fetcher    = BSEFetcher(db_path, ticker=ticker)
        self.fii_fetcher    = FIIFetcher(db_path)
        self.global_fetcher = GlobalFetcher(db_path)
        self.options_fetcher= OptionsFetcher(db_path)
        self.social_fetcher = SocialFetcher(db_path)
        self.insider_fetcher= InsiderFetcher(db_path)
        self.alt_fetcher    = AlternativeFetcher(db_path)

        # ── Processors ────────────────────────────────────────────
        self.finbert         = FinBERTSentiment(db_path)
        self.llm             = LLMAnalyzer(db_path)

        # ── State tracking ────────────────────────────────────────
        self._last_signal     : Optional[dict] = None
        # Per-category alert dedup: {category: {direction, confidence}}.
        # Allows swing/positional/intraday to flap independently without
        # one category's change suppressing another's fresh alert.
        self._last_signal_by_cat : dict        = {}
        # Reversal alert dedup: {position_id: reversal_type}.
        # Prevents spamming the same MODEL_REVERSAL / REGIME_CHANGE alert
        # every 15 minutes for the same open trade. Cleared when the trade closes
        # (position_id disappears from open_trades).
        self._sent_reversals  : dict           = {}
        self._last_signal_ts  : float          = 0.0
        self._last_news_ts    : float          = 0.0
        self._last_options_ts : float          = 0.0
        self._last_bse_ts     : float          = 0.0
        self._last_fii_ts     : float          = 0.0
        self._last_social_ts  : float          = 0.0
        self._task_errors     : dict           = {}   # task → consecutive error count
        self._opened_today    : bool           = False
        self._eod_sent        : bool           = False
        self._evening_sent    : bool           = False
        self._retrain_done    : bool           = False
        self._sunday_refresh  : bool           = False
        self._last_exit_ts    : float          = 0.0   # last stop/target check
        EXIT_INTERVAL_SEC     = 60             # check every 60s during market hours
        self.EXIT_INTERVAL_SEC= EXIT_INTERVAL_SEC

        logger.info(
            f"TaskRunner initialised | "
            f"ticker={ticker} capital=₹{capital:,.0f}"
        )

    # ─────────────────────────────────────────────────────────────
    # PRIMARY ENTRY POINT
    # ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Start the task runner loop.
        Runs forever — call from main.py.
        Handles all errors gracefully — never stops.
        """
        logger.info("=" * 60)
        logger.info("QUANT EDGE — Task Runner starting")
        logger.info("=" * 60)

        # Load all models from disk on startup
        loaded = self.trainer.load_all()
        logger.info(f"Models loaded: {loaded}")

        # Seed RBI rates and earnings calendar on startup
        try:
            from data.rbi_fetcher import RBIFetcher
            RBIFetcher(self.db_path).seed_historical_rates()
            self.bse_fetcher.sync_earnings_calendar()
        except Exception as e:
            logger.warning(f"Startup seeding: {e}")

        # Test Telegram connection
        self.telegram.test_connection()
        self.schedule.log_schedule()

        # Reset daily flags at startup
        self._reset_daily_flags()

        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                logger.info("TaskRunner stopped by user")
                self.telegram.send_raw("⛔ QUANT EDGE stopped by user")
                break
            except Exception as e:
                logger.error(f"TaskRunner main loop error: {e}")
                logger.error(traceback.format_exc())
                self._safe_telegram(f"TaskRunner error: {e}", "main_loop")
                time.sleep(60)   # brief pause then continue

    def run_once(self) -> Optional[dict]:
        """
        Run one signal cycle manually — useful for testing.
        Returns the signal result.
        """
        return self._run_signal_pipeline()

    # ─────────────────────────────────────────────────────────────
    # MAIN TICK
    # ─────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        """
        One iteration of the main loop.
        Checks all scheduled tasks and runs what is due.
        Sleeps for the remainder of the 15-minute window.
        """
        now     = time.time()
        now_dt  = datetime.now()
        schedule= self.schedule

        # ── Daily flag reset at midnight ──────────────────────────
        if now_dt.hour == 0 and now_dt.minute < 2:
            self._reset_daily_flags()

        # ── Sunday 00:00 — weekly retrain ─────────────────────────
        if schedule.is_sunday_midnight() and not self._retrain_done:
            self._run_weekly_retrain()
            self._retrain_done = True

        # ── Sunday 06:00 — data refresh ───────────────────────────
        if schedule.is_sunday_morning() and not self._sunday_refresh:
            self._run_weekly_data_refresh()
            self._sunday_refresh = True

        # ── 06:00 — global pre-market fetch ───────────────────────
        if (now_dt.hour == 6 and now_dt.minute < 5
                and schedule.is_trading_day()):
            self._run_pre_market()

        # ── NEWS — every 15 minutes (trading days only) ──────────
        # Weekend/holiday news can be fetched but FinBERT scoring deferred
        # to avoid unnecessary GPU use on non-trading days.
        if (now - self._last_news_ts >= self.NEWS_INTERVAL_SEC
                and schedule.is_trading_day()):
            self._run_news_pipeline()
            self._last_news_ts = now

        # ── OPTIONS — every 30 minutes during market hours ────────
        if (now - self._last_options_ts >= self.OPTIONS_INTERVAL_SEC
                and schedule.is_market_open()):
            self._safe_run(
                self.options_fetcher.fetch_and_calculate,
                "options_fetch",
            )
            self._last_options_ts = now

        # ── BSE announcements — trading days only ─────────────────
        if (now - self._last_bse_ts >= self.BSE_INTERVAL_SEC
                and schedule.is_trading_day()):
            self._run_bse_pipeline()
            self._last_bse_ts = now

        # ── FII / DII — trading days only (NSE publishes EOD) ─────
        if (now - self._last_fii_ts >= self.FII_INTERVAL_SEC
                and schedule.is_trading_day()):
            self._safe_run(self.fii_fetcher.fetch_fii_dii_data, "fii_fetch")
            self._safe_run(self.fii_fetcher.fetch_bulk_deals,   "bulk_deals")
            self._last_fii_ts = now

        # ── Social — trading days only ────────────────────────────
        if (now - self._last_social_ts >= self.SOCIAL_INTERVAL_SEC
                and schedule.is_trading_day()):
            self._safe_run(self.social_fetcher.fetch_all, "social_fetch")
            self._last_social_ts = now

        # ── Circuit breaker check ─────────────────────────────────
        if schedule.is_trading_day():
            self._run_circuit_breaker_check()

        # ── EXIT MONITOR — every 60s during market hours ──────────
        # Checks stop-loss and target against live price on every tick.
        # Fires a Telegram alert immediately if stop/target is breached —
        # does NOT wait for the 15-min signal cycle.
        if (schedule.is_market_open()
                and now - self._last_exit_ts >= self.EXIT_INTERVAL_SEC):
            self._last_exit_ts = now
            exits = self._safe_run(self.signal_engine.check_exits, "exit_check")
            if exits:
                for exit_info in exits:
                    self._safe_telegram(
                        lambda e=exit_info: self.telegram.send_exit(e),
                        "exit_alert",
                    )

        # ── SIGNAL PIPELINE — every 15 min during active hours ────
        if (schedule.should_run_signal_pipeline()
                and now - self._last_signal_ts >= self.SIGNAL_INTERVAL_SEC):
            signal = self._run_signal_pipeline()
            self._last_signal_ts = now

            # Per-category alerting: each passing category (swing /
            # positional / intraday) gets its OWN Telegram alert if its
            # direction or confidence changed materially. A single FLAT
            # cycle that follows an active per-category signal also sends
            # a "category went flat" notification.
            if signal:
                self._dispatch_category_alerts(signal)
                self._last_signal = signal

        # ── 09:15 — market open ────────────────────────────────────
        if (schedule.is_market_open()
                and not self._opened_today
                and schedule.now_ist().hour == 9
                and schedule.now_ist().minute < 20):
            if self._last_signal:
                self._safe_telegram(
                    lambda: self.telegram.send_market_open(self._last_signal),
                    "market_open_alert",
                )
            self._opened_today = True

        # ── 15:15 — intraday close warning ────────────────────────
        if schedule.is_intraday_close_window():
            exits = self._safe_run(
                self.signal_engine.check_exits, "exit_check"
            )
            if exits:
                for exit_info in exits:
                    self._safe_telegram(
                        lambda e=exit_info: self.telegram.send_exit(e),
                        "exit_alert",
                    )
            if (schedule.now_ist().minute >= 15
                    and schedule.now_ist().minute < 20):
                self._safe_telegram(
                    self.telegram.send_intraday_close_warning,
                    "intraday_close_warning",
                )

        # ── 15:45 — EOD summary ────────────────────────────────────
        if schedule.is_end_of_day() and not self._eod_sent:
            self._run_eod_summary()
            self._eod_sent = True

        # ── 18:00 — evening performance report ────────────────────
        if schedule.is_evening_report_time() and not self._evening_sent:
            self._run_evening_report()
            self._evening_sent = True

        # ── Sleep until next 15-min boundary ──────────────────────
        sleep_sec = min(
            self.schedule.get_next_run_seconds(),
            60,   # max 60s sleep so we catch daily events promptly
        )
        time.sleep(sleep_sec)

    # ─────────────────────────────────────────────────────────────
    # TASK IMPLEMENTATIONS
    # ─────────────────────────────────────────────────────────────

    def _run_pre_market(self) -> None:
        """
        6 AM pre-market routine:
        1. Fetch global overnight data (S&P500, DXY, crude, VIX)
        2. Run full signal pipeline
        3. Send pre-market alert to Telegram
        """
        logger.info("─── Pre-market analysis (06:00) ───")
        self._safe_run(self.global_fetcher.fetch_overnight_signals, "global_fetch")
        signal = self._run_signal_pipeline()
        if signal:
            self._safe_telegram_signal(signal)
            self._last_signal = signal
        logger.info("─── Pre-market analysis complete ───")

    def _run_news_pipeline(self) -> None:
        """Fetch news for all 5 banks → score unscored articles (keyword or FinBERT)."""
        from data.news_fetcher import NewsFetcher as _NF
        self._safe_run(
            lambda: _NF.fetch_all_banks(self.db_path), "news_fetch"
        )
        # Always run scoring — clears backlog even when no new articles fetched
        self._safe_run(self.finbert.process_unscored, "finbert_score")

    def _run_bse_pipeline(self) -> None:
        """Fetch BSE announcements → LLM analysis for high-priority ones."""
        self._safe_run(self.bse_fetcher.fetch_bse_announcements, "bse_fetch")

        # LLM analysis for high-priority announcements
        high_priority = self._safe_run(
            self.bse_fetcher.get_high_priority_unanalyzed, "bse_priority"
        ) or []

        for ann in high_priority[:2]:   # max 2 LLM calls per cycle (cost control)
            result = self._safe_run(
                lambda a=ann: self.llm.analyze_bse_announcement(
                    a["title"], a["description"]
                ),
                "llm_announcement",
            )
            if result and abs(float(result.get("impact_score", 0))) > 0.4:
                self._safe_telegram(
                    lambda r=result, a=ann: self.telegram.send_raw(
                        f"📋 *BSE Announcement Alert*\n"
                        f"{a['title'][:100]}\n"
                        f"Impact: {r['impact_score']:+.2f} | "
                        f"{r.get('reason', '')}"
                    ),
                    "bse_alert",
                )
            if result:
                self.bse_fetcher.mark_llm_analyzed(
                    ann["id"],
                    float(result.get("impact_score", 0)),
                )

    def _run_signal_pipeline(self) -> Optional[dict]:
        """Core 15-minute signal generation cycle."""
        try:
            signal = self.signal_engine.run()

            # Auto-open one paper trade per passing category. Each
            # category (swing/positional/intraday) is a distinct time-horizon
            # trade and is opened independently — so a swing LONG and a
            # positional SHORT can both run at the same time without
            # collision in the exit engine.
            if signal and signal.get("signal") not in ("FLAT", None):
                per_cat = signal.get("signals") or [{
                    # Backward-compat shim — older engine versions returned
                    # only a single primary signal at top-level.
                    "category":     signal.get("trade_type", "swing"),
                    "signal":       signal["signal"],
                    "entry_price":  signal.get("entry_price", 0),
                    "stop_loss":    signal.get("stop_loss", 0),
                    "target_price": signal.get("target_price", 0),
                    "shares":       signal.get("shares", 0),
                    "risk_amount":  signal.get("risk_amount", 0),
                    "confidence":   signal.get("confidence", 0),
                    "entry_quality":signal.get("entry_quality", ""),
                }]
                open_positions = self.exit_engine._get_open_positions()
                ticker = signal.get("ticker", self.ticker)
                for sig in per_cat:
                    cat = sig.get("category", sig.get("trade_type", "swing"))
                    already_open = any(
                        p.get("ticker") == ticker
                        and p.get("trade_type") == cat
                        for p in open_positions
                    )
                    if already_open:
                        continue
                    self.exit_engine.open_position(
                        signal       = sig["signal"],
                        entry_price  = float(sig.get("entry_price", 0)),
                        stop_price   = float(sig.get("stop_loss", 0)),
                        target_price = float(sig.get("target_price", 0)),
                        shares       = int(sig.get("shares", 0)),
                        risk_amount  = float(sig.get("risk_amount", 0)),
                        trade_type   = cat,
                        signal_uuid  = signal.get("signal_uuid", ""),
                        alignment    = signal.get("alignment", ""),
                        regime       = signal.get("regime", ""),
                        confidence   = float(sig.get("confidence", 0)),
                        entry_quality= sig.get("entry_quality", ""),
                        ticker       = ticker,
                    )

            # Advisory: check if any open position's model or regime has reversed.
            # Send one Telegram alert per new reversal event; suppress repeats until
            # the position closes (its ID disappears from open_trades).
            if signal:
                self._dispatch_reversal_alerts(signal)

            return signal
        except Exception as e:
            logger.error(f"Signal pipeline error: {e}")
            self._record_error("signal_pipeline")
            if self._task_errors.get("signal_pipeline", 0) >= 3:
                self._safe_telegram(
                    f"Signal pipeline failed 3× in a row: {e}",
                    "signal_pipeline",
                )
            return None

    def _run_circuit_breaker_check(self) -> None:
        """Check circuit breaker and alert if triggered."""
        try:
            stats     = self.portfolio.get_performance_stats()
            mode      = stats.get("capital_mode", "FULL") if hasattr(stats, "get") else "FULL"
            monthly_dd= float(stats.get("monthly_dd_pct", 0))
            c_losses  = int(stats.get("consecutive_losses", 0))

            allowed, cb_status = self.circuit_breaker.check(
                capital_mode="FULL",
                monthly_dd_pct=monthly_dd,
                consecutive_losses=c_losses,
            )

            level = cb_status.get("level", "OK")
            if level in ("PAUSE", "HALT"):
                self._safe_telegram(
                    lambda s=cb_status: self.telegram.send_circuit_breaker(s),
                    "circuit_breaker",
                )
        except Exception as e:
            logger.warning(f"Circuit breaker check error: {e}")

    def _run_eod_summary(self) -> None:
        """3:45 PM end-of-day summary."""
        logger.info("─── EOD Summary ───")
        try:
            # Check remaining exits
            exits = self.signal_engine.check_exits()
            for e in exits:
                self.telegram.send_exit(e)

            # Heatmap
            heatmap = self.heatmap.generate()
            self.telegram.send_raw(self.heatmap.format_telegram(heatmap))
        except Exception as e:
            logger.error(f"EOD summary error: {e}")

    def _run_evening_report(self) -> None:
        """6 PM evening performance and psychology report."""
        logger.info("─── Evening Report ───")
        try:
            report    = self.performance.daily_report()
            msg       = self.performance.format_telegram(report)
            self.telegram.send_raw(msg)

            # Psychology check (weekly on Fridays)
            if datetime.now().weekday() == 4:  # Friday
                psych  = self.psychology.analyse()
                pmsg   = self.psychology.format_telegram(psych)
                self.telegram.send_raw(pmsg)
        except Exception as e:
            logger.error(f"Evening report error: {e}")

    def _run_weekly_retrain(self) -> None:
        """Sunday midnight — retrain all models for all 5 banks."""
        logger.info("=" * 50)
        logger.info("WEEKLY RETRAIN — Sunday midnight")
        logger.info("=" * 50)
        try:
            from config import TradingConfig
            from models.train_all import ModelTrainer
            self.telegram.send_raw(
                "🔄 Weekly model retrain starting (5 banks)...\n"
                "Expected duration: 20–40 minutes."
            )
            all_results = {}
            for ticker in TradingConfig.UNIVERSE:
                logger.info(f"Retraining {ticker}...")
                try:
                    trainer = ModelTrainer(ticker, self.capital, self.db_path)
                    summary = trainer.train_all()
                    all_results[ticker] = summary
                    logger.info(f"  {ticker}: done")
                except Exception as e:
                    logger.error(f"  {ticker}: retrain failed — {e}")
                    all_results[ticker] = {"error": str(e)}
            # Send report for primary ticker
            primary_summary = all_results.get(TradingConfig.TICKER, {})
            self.telegram.send_retrain_report(primary_summary)
            succeeded = sum(1 for v in all_results.values() if "error" not in v)
            self.telegram.send_raw(
                f"✅ Weekly retrain complete: {succeeded}/{len(TradingConfig.UNIVERSE)} banks retrained"
            )
        except Exception as e:
            logger.error(f"Weekly retrain failed: {e}")
            self.telegram.send_error("Weekly Retrain", str(e))

    def _run_weekly_data_refresh(self) -> None:
        """Sunday 6 AM — refresh slow-moving data sources."""
        logger.info("─── Weekly data refresh ───")
        self._safe_run(self.insider_fetcher.fetch_all,         "insider_refresh")
        self._safe_run(self.alt_fetcher.fetch_all,             "alt_data_refresh")
        # Re-seed RBI rates in case new rates were added to RBI_RATE_HISTORY
        try:
            from data.rbi_fetcher import RBIFetcher
            self._safe_run(RBIFetcher(self.db_path).seed_historical_rates, "rbi_seed")
        except Exception as e:
            logger.warning(f"RBI seed in weekly refresh: {e}")
        # Refresh fundamentals for all 5 banks (quarterly data via Screener.in)
        try:
            from processing.fundamental import FundamentalProcessor
            from config import TradingConfig
            for ticker in TradingConfig.UNIVERSE:
                self._safe_run(
                    FundamentalProcessor(self.db_path, ticker).fetch_and_update,
                    f"fundamentals_{ticker}",
                )
        except Exception as e:
            logger.warning(f"Fundamentals refresh error: {e}")

    # ─────────────────────────────────────────────────────────────
    # ALERT LOGIC
    # ─────────────────────────────────────────────────────────────

    def _dispatch_category_alerts(self, signal: dict) -> None:
        """
        Route per-category signal events to Telegram with independent dedup.

        For every category (swing / positional / intraday), compare against
        the previous cycle's state for that same category:
          - new non-FLAT signal           → send
          - direction flipped LONG↔SHORT  → send
          - confidence shifted ≥10%       → send
          - category went FLAT after non-FLAT → send "no longer active"

        Each category is independent — a swing flip doesn't suppress a
        positional signal that hasn't changed.
        """
        per_cat = signal.get("signals") or []
        active_by_cat = {s.get("category"): s for s in per_cat if s.get("category")}

        # If the engine returned a structured FLAT (no signals[]) but the
        # primary slot is non-FLAT (backward-compat path), promote it.
        if not per_cat and signal.get("signal") not in ("FLAT", None):
            active_by_cat = {signal.get("trade_type", "swing"): signal}

        seen_cats = set(active_by_cat) | set(self._last_signal_by_cat)
        any_sent  = False

        for cat in seen_cats:
            new   = active_by_cat.get(cat)
            old   = self._last_signal_by_cat.get(cat)
            send  = False

            if new and not old:
                send = True   # first time this category fired
            elif new and old:
                if new["signal"] != old.get("signal"):
                    send = True
                elif abs(float(new.get("confidence", 0)) -
                         float(old.get("confidence", 0))) >= 0.10:
                    send = True
            elif old and not new:
                # Category was active last cycle, now silent — notify once.
                send = True

            if send:
                payload = new if new else {
                    "signal":   "FLAT",
                    "category": cat,
                    "ticker":   signal.get("ticker", self.ticker),
                    "reason":   f"{cat.upper()} no longer active",
                    "regime":   signal.get("regime", ""),
                }
                self._safe_telegram_signal(payload)
                any_sent = True

            # Update state cache
            if new:
                self._last_signal_by_cat[cat] = new
            elif cat in self._last_signal_by_cat:
                self._last_signal_by_cat.pop(cat, None)

        # If nothing was sent and there's no active or prior state, this
        # was just a steady FLAT — silent, no alert.
        if not any_sent and not active_by_cat and not self._last_signal_by_cat:
            return

    def _dispatch_reversal_alerts(self, signal: dict) -> None:
        """
        Check open positions for model/regime reversals and send one advisory
        Telegram alert per new reversal event.

        Dedup key: position_id → reversal_type. A reversal alert for position 42
        (MODEL_REVERSAL) is sent once. If the model goes back to the original
        direction the dedup key is cleared, so a future re-reversal fires again.
        Closed positions vanish from open_trades, so their dedup entries are pruned
        here to keep _sent_reversals tidy.
        """
        try:
            reversals = self.signal_engine.check_model_reversals(signal)
        except Exception as e:
            logger.warning(f"check_model_reversals error: {e}")
            return

        # Prune stale entries — positions that have since been closed
        active_ids = {r["position_id"] for r in reversals if r.get("position_id")}
        stale = [k for k in self._sent_reversals if k not in active_ids]
        for k in stale:
            del self._sent_reversals[k]

        for rev in reversals:
            pos_id = rev.get("position_id")
            rtype  = rev.get("type")
            if self._sent_reversals.get(pos_id) == rtype:
                continue   # already alerted for this reversal event

            self._sent_reversals[pos_id] = rtype
            logger.warning(
                f"Reversal alert: pos={pos_id} type={rtype} "
                f"cat={rev.get('category')} "
                f"open={rev.get('open_signal')} model={rev.get('model_signal')} "
                f"regime={rev.get('regime')}"
            )
            self._safe_telegram(
                lambda r=rev: self.telegram.send_raw(
                    self.telegram.formatter.format_model_reversal(r)
                ),
                "reversal_alert",
            )

    def _safe_telegram_signal(self, signal: dict) -> None:
        """Send signal alert, catching all errors."""
        try:
            self.telegram.send_signal(signal)
        except Exception as e:
            logger.error(f"Telegram signal send failed: {e}")

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _safe_run(self, fn, task_name: str):
        """
        Run a task function, catch all exceptions, log errors.
        Three consecutive failures → escalate Telegram alert.
        """
        try:
            result = fn()
            self._task_errors[task_name] = 0   # reset on success
            return result
        except Exception as e:
            count = self._task_errors.get(task_name, 0) + 1
            self._task_errors[task_name] = count
            logger.warning(f"Task '{task_name}' failed (attempt {count}): {e}")
            if count >= 3:
                self._safe_telegram(
                    f"Task '{task_name}' failed {count}× consecutively: {str(e)[:100]}",
                    f"error_{task_name}",
                )
            return None

    def _safe_telegram(self, content, task_name: str = "") -> None:
        """Send Telegram message or execute lambda safely."""
        try:
            if callable(content):
                content()
            else:
                self.telegram.send_error(task_name, str(content))
        except Exception as e:
            logger.error(f"Telegram send failed ({task_name}): {e}")

    def _record_error(self, task: str) -> None:
        self._task_errors[task] = self._task_errors.get(task, 0) + 1

    def _reset_daily_flags(self) -> None:
        """Reset all daily one-shot flags at midnight."""
        self._opened_today = False
        self._eod_sent     = False
        self._evening_sent = False
        self._last_exit_ts = 0.0
        # Weekly flags reset on Monday midnight
        if datetime.now().weekday() == 0:
            self._retrain_done    = False
            self._sunday_refresh  = False
        logger.info("Daily flags reset")