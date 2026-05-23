# main.py
# Entry point for the QUANT EDGE trading system
# Run: python main.py
# Run with mode: python main.py --mode=signal | train | backtest | setup

import os
import sys
import logging
import argparse
import traceback
from datetime import datetime

# ── Load configuration first ──────────────────────────────────
from config import (
    LogConfig, PathConfig, APIConfig,
    TradingConfig, print_config_summary, validate_config,
)

# ── Configure logging ─────────────────────────────────────────
def setup_logging() -> None:
    PathConfig.ensure_dirs()
    handlers = [logging.StreamHandler(sys.stdout)]
    if LogConfig.LOG_TO_FILE:
        handlers.append(
            logging.FileHandler(LogConfig.LOG_FILE, encoding="utf-8")
        )
    logging.basicConfig(
        level   = getattr(logging, LogConfig.LEVEL, logging.INFO),
        format  = LogConfig.FORMAT,
        datefmt = LogConfig.DATE_FORMAT,
        handlers= handlers,
    )
    # Silence chatty third-party libraries that log at INFO by default
    for _lib in ("httpx", "httpcore", "huggingface_hub", "filelock",
                 "urllib3", "transformers.modeling_utils", "yfinance"):
        logging.getLogger(_lib).setLevel(logging.WARNING)
    # huggingface_hub.utils._http warns about missing HF_TOKEN on every model
    # load — harmless for public models (ProsusAI/finbert needs no auth).
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

setup_logging()
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# MAIN MODES
# ════════════════════════════════════════════════════════════════

def mode_setup() -> None:
    """
    First-time setup:
    1. Create all directories
    2. Initialise database
    3. Validate configuration
    4. Test Telegram connection
    5. Run initial model training
    """
    print_config_summary()

    logger.info("Running first-time setup...")

    # Create directories
    PathConfig.ensure_dirs()
    logger.info("✅ Directories created")

    # Initialise database
    from database.db_setup import DatabaseSetup
    db = DatabaseSetup(PathConfig.DB_PATH)
    db.setup_all()
    counts = db.verify()
    logger.info(f"✅ Database initialised: {len(counts)} tables")

    # Validate config
    is_valid, issues = validate_config()
    if issues:
        for issue in issues:
            logger.warning(f"Config issue: {issue}")
    else:
        logger.info("✅ Configuration valid")

    # Test Telegram
    if APIConfig.TELEGRAM_TOKEN and APIConfig.TELEGRAM_CHAT_ID:
        from alerts.telegram_bot import TelegramBot
        bot = TelegramBot()
        if bot.test_connection():
            logger.info("✅ Telegram connected")
        else:
            logger.warning("⚠️  Telegram connection failed — check credentials")
    else:
        logger.warning("⚠️  Telegram not configured — set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in .env")

    # Initial model training
    logger.info("Starting initial model training (this may take 15–30 minutes)...")
    mode_train(force=True, ticker=TradingConfig.TICKER)

    logger.info("=" * 55)
    logger.info("Setup complete. Run: python main.py to start the system.")
    logger.info("=" * 55)


def mode_backfill_fii() -> None:
    """Load historical FII daily data from CSV files into the DB."""
    from data.fii_fetcher import FIIFetcher
    fetcher = FIIFetcher(PathConfig.DB_PATH)
    inserted, skipped = fetcher.load_from_csv("fii_cash_daily_data")
    logger.info("Done — run --mode=train to retrain with the new data")


def mode_train(force: bool = False, model: str = "all",
               ticker: str = None) -> None:
    """
    Train models. model='all' | 'swing' | 'regime'
    """
    effective_ticker = ticker or TradingConfig.TICKER
    from models.train_all import ModelTrainer
    trainer = ModelTrainer(
        ticker  = effective_ticker,
        capital = TradingConfig.STARTING_CAPITAL,
        db_path = PathConfig.DB_PATH,
    )

    if model == "swing":
        logger.info("Training swing model only...")
        metrics = trainer.train_swing_only(force=force)
        if metrics:
            cv = metrics.get("cv_accuracy_mean", 0)
            n  = metrics.get("n_samples", 0)
            logger.info(f"✅ Swing model trained: CV={cv:.3f} ({n:,} samples)")
            summary = {"success": True, "models": {"swing": metrics}}
        else:
            logger.error("❌ Swing model failed to train")
            summary = {"success": False, "models": {}}
    elif model == "regime":
        logger.info("Training regime model only...")
        metrics = trainer.train_regime_only()
        if metrics:
            logger.info("✅ Regime model trained")
            summary = {"success": True, "models": {"regime": metrics}}
        else:
            logger.error("❌ Regime model failed to train")
            summary = {"success": False, "models": {}}
    else:
        logger.info("Training all models...")
        summary = trainer.train_all(force=force)
        if summary.get("success"):
            logger.info("✅ All models trained successfully")
            for model_name, m in summary.get("models", {}).items():
                if m:
                    cv = m.get("cv_accuracy_mean", 0)
                    n  = m.get("n_samples", 0)
                    logger.info(f"  {model_name:12}: CV={cv:.3f} ({n:,} samples)")
        else:
            logger.error("❌ One or more models failed to train")

    # Send retrain report
    if APIConfig.TELEGRAM_TOKEN:
        try:
            from alerts.telegram_bot import TelegramBot
            TelegramBot().send_retrain_report(summary)
        except Exception as e:
            logger.warning(f"Could not send retrain report: {e}")


def mode_signal(ticker: str = None) -> None:
    """
    Run one signal cycle manually.
    Useful for testing the pipeline end-to-end.
    """
    effective_ticker = ticker or TradingConfig.TICKER
    from signals.signal_engine import SignalEngine
    engine = SignalEngine(
        ticker  = effective_ticker,
        capital = TradingConfig.STARTING_CAPITAL,
        db_path = PathConfig.DB_PATH,
    )

    logger.info("Running signal pipeline...")
    signal = engine.run()
    direction = signal.get("signal", "FLAT")
    confidence= signal.get("confidence", 0)
    alignment = signal.get("alignment", "")
    regime    = signal.get("regime",    "")

    if direction != "FLAT":
        logger.info(
            f"SIGNAL: {direction} | "
            f"conf={confidence:.0%} | "
            f"align={alignment} | "
            f"regime={regime} | "
            f"entry=₹{signal.get('entry_price', 0):.2f} | "
            f"stop=₹{signal.get('stop_loss', 0):.2f} | "
            f"target=₹{signal.get('target_price', 0):.2f}"
        )
        # Send to Telegram
        if APIConfig.TELEGRAM_TOKEN:
            from alerts.telegram_bot import TelegramBot
            TelegramBot().send_signal(signal)
    else:
        logger.info(f"FLAT — {signal.get('reason', 'No signal')}")


def mode_backtest() -> None:
    """
    Run full backtest + walk-forward validation on the swing model.
    Produces a deployment readiness report.
    """
    from data.price_fetcher    import PriceFetcher
    from models.train_swing    import SwingModelTrainer
    from features.feature_builder import FeatureBuilder
    from backtest.engine       import BacktestEngine
    from backtest.walk_forward import WalkForwardValidator

    logger.info("Starting backtest analysis...")

    fetcher = PriceFetcher(TradingConfig.TICKER)
    df      = fetcher.get_daily(start="2000-01-01")
    logger.info(f"Loaded {len(df)} daily bars")

    fb      = FeatureBuilder(
        ticker  = TradingConfig.TICKER,
        capital = TradingConfig.STARTING_CAPITAL,
        db_path = PathConfig.DB_PATH,
    )
    X, y = fb.build_training_dataset(
        df, model_type="swing",
        forward_days=5, threshold=0.025,
    )

    trainer = SwingModelTrainer()
    trainer.train(X, y, list(X.columns), save=False)

    engine = BacktestEngine(
        starting_capital = TradingConfig.STARTING_CAPITAL,
        capital_mode     = "FULL",
    )
    result = engine.run(
        df            = df,
        model         = trainer.model,
        feature_names = trainer.features,
        model_type    = "swing",
    )
    m = result.get("metrics", {})

    logger.info("=" * 55)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 55)
    logger.info(f"  Trades:       {m.get('total_trades', 0)}")
    logger.info(f"  Win rate:     {m.get('win_rate', 0):.1%}")
    logger.info(f"  Profit factor:{m.get('profit_factor', 0):.2f}")
    logger.info(f"  Sharpe:       {m.get('sharpe_ratio', 0):.2f}")
    logger.info(f"  Max DD:       {m.get('max_drawdown_pct', 0):.1%}")
    logger.info(f"  Return:       {m.get('total_return_pct', 0):.1%}")
    logger.info(f"  Grade:        {m.get('grade', 'D')}")
    logger.info(f"  Deployable:   {'YES' if m.get('is_deployable') else 'NO'}")

    # Walk-forward
    logger.info("\nRunning walk-forward validation...")
    wf = WalkForwardValidator(
        train_years      = 2,
        test_months      = 6,
        starting_capital = TradingConfig.STARTING_CAPITAL,
    )
    wf_result = wf.validate(df, model_type="swing")
    if wf_result:
        assess = wf_result.get("assessment", {})
        logger.info(f"  Walk-forward: {assess.get('recommendation', 'No result')}")


def mode_dashboard() -> None:
    """Print current system status to console."""
    from database.queries       import Queries
    from dashboard.performance  import PerformanceDashboard
    from dashboard.portfolio_heatmap import PortfolioHeatmap

    q    = Queries(PathConfig.DB_PATH)
    perf = PerformanceDashboard(TradingConfig.STARTING_CAPITAL, PathConfig.DB_PATH)
    hm   = PortfolioHeatmap(TradingConfig.STARTING_CAPITAL, PathConfig.DB_PATH)

    report  = perf.daily_report()
    heatmap = hm.generate()

    print(perf.format_telegram(report))
    print()
    print(hm.format_telegram(heatmap))

    # Data freshness
    fresh = q.get_data_freshness()
    print("\nData source freshness:")
    for src, info in fresh.items():
        if src == "_summary":
            continue
        hours = info.get("hours_ago")
        status = "✅" if not info.get("stale") else "⚠️ STALE"
        age_str = f"{hours:.1f}h ago" if hours is not None else "never"
        print(f"  {src:15}: {age_str:12} {status}")


def mode_run(ticker: str = None) -> None:
    """
    Start the live task runner — runs forever.
    This is the main production mode.
    """
    effective_ticker = ticker or TradingConfig.TICKER
    print_config_summary()

    # Validate credentials
    _, issues = validate_config()
    critical  = [i for i in issues if "TELEGRAM" in i]
    if critical:
        logger.error("Critical credentials missing:")
        for issue in critical:
            logger.error(f"  {issue}")
        logger.error("Set credentials in .env and restart.")
        sys.exit(1)

    # Ensure database exists
    from database.db_setup import DatabaseSetup
    DatabaseSetup(PathConfig.DB_PATH).setup_all()

    # Load models (train if none exist)
    from models.train_all import ModelTrainer
    trainer = ModelTrainer(
        ticker  = effective_ticker,
        capital = TradingConfig.STARTING_CAPITAL,
        db_path = PathConfig.DB_PATH,
    )
    loaded = trainer.load_all()
    not_loaded = [k for k, v in loaded.items() if not v]

    if not_loaded:
        logger.warning(f"Models not found: {not_loaded}")
        logger.info(f"Run: python main.py --mode=train --ticker={effective_ticker}  to train models first")
        logger.info("Or:  python main.py --mode=setup  for full first-time setup")

        response = input("\nTrain models now? This takes 15–30 min [y/N]: ").strip().lower()
        if response == "y":
            mode_train(force=True, ticker=effective_ticker)
        else:
            logger.info("Starting with no models — signals will be FLAT until models are trained")

    # Start the task runner
    from scheduler.task_runner import TaskRunner
    runner = TaskRunner(
        ticker  = effective_ticker,
        capital = TradingConfig.STARTING_CAPITAL,
        db_path = PathConfig.DB_PATH,
    )

    logger.info("=" * 55)
    logger.info("QUANT EDGE LIVE — Running")
    logger.info(f"Ticker:  {effective_ticker}")
    logger.info(f"Capital: ₹{TradingConfig.STARTING_CAPITAL:,.0f}")
    logger.info(f"Time:    {datetime.now().strftime('%d %b %Y %H:%M IST')}")
    logger.info("=" * 55)
    logger.info("Press Ctrl+C to stop")

    runner.start()


# ════════════════════════════════════════════════════════════════
# CLI ARGUMENT PARSING
# ════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="QUANT EDGE — Autonomous Trading Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  run        Start live trading system (default)
  setup      First-time setup: DB, dirs, initial training
  train      Retrain all 4 ML models
  signal     Run one signal cycle and print result
  backtest   Full backtest + walk-forward validation
  dashboard  Print current portfolio status

Examples:
  python main.py                    # Start live system
  python main.py --mode=setup       # First-time setup
  python main.py --mode=signal      # Single signal test
  python main.py --mode=train       # Retrain models
  python main.py --mode=backtest    # Backtest analysis
  python main.py --mode=dashboard   # Portfolio status
        """,
    )
    parser.add_argument(
        "--mode",
        default="run",
        choices=["run", "setup", "train", "signal", "backtest", "dashboard", "backfill-fii"],
        help="Operating mode (default: run)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force model replacement even if accuracy drops (use with --mode=train)",
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "swing", "regime"],
        help="Which model to train (default: all, use with --mode=train)",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        choices=["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS",
                 "AXISBANK.NS", "INDUSINDBK.NS"],
        help="Bank ticker to trade (default: HDFCBANK.NS from config)",
    )
    return parser.parse_args()


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    effective_ticker = args.ticker or TradingConfig.TICKER

    try:
        if args.mode == "setup":
            mode_setup()
        elif args.mode == "backfill-fii":
            mode_backfill_fii()
        elif args.mode == "train":
            mode_train(force=args.force, model=args.model, ticker=effective_ticker)
        elif args.mode == "signal":
            mode_signal(ticker=effective_ticker)
        elif args.mode == "backtest":
            mode_backtest()
        elif args.mode == "dashboard":
            mode_dashboard()
        else:
            mode_run(ticker=effective_ticker)

    except KeyboardInterrupt:
        logger.info("\nQuant Edge stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error in mode={args.mode}: {e}")
        logger.error(traceback.format_exc())

        # Try to send error alert
        if APIConfig.TELEGRAM_TOKEN:
            try:
                from alerts.telegram_bot import TelegramBot
                TelegramBot().send_error("main.py", str(e)[:200])
            except Exception:
                pass

        sys.exit(1)


if __name__ == "__main__":
    main()