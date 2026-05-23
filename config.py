
# config.py
# Single source of truth for all system configuration
# NEVER hardcode credentials anywhere else — always read from here
# .env file overrides these defaults in production

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict

# Load .env file at import time
load_dotenv(Path(__file__).parent / ".env")


# ══════════════════════════════════════════════════════════════
# TRADING CONFIGURATION
# ══════════════════════════════════════════════════════════════

class TradingConfig:
    """Core trading parameters — the rules the system trades by."""

    # Primary ticker (kept for single-ticker backward compatibility)
    TICKER          = os.getenv("TICKER", "HDFCBANK.NS")
    EXCHANGE        = "NSE"

    # Multi-asset universe — top-5 private banks ranked by liquidity + quality
    # SBI excluded: PSU bank with different fundamental dynamics
    UNIVERSE = [
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "KOTAKBANK.NS",
        "AXISBANK.NS",
        "INDUSINDBK.NS",
    ]

    # Universe scanning: evaluate top-N from gate3 ranking each cycle
    UNIVERSE_TOP_N = int(os.getenv("UNIVERSE_TOP_N", "5"))

    # Position concentration limits
    MAX_PER_NAME_PCT      = float(os.getenv("MAX_PER_NAME_PCT",  "0.40"))  # 40%
    MAX_TOTAL_EXPOSURE_PCT= float(os.getenv("MAX_EXPOSURE_PCT",  "0.80"))  # 80%

    # Starting capital (₹) — update this to your actual account size
    STARTING_CAPITAL = float(os.getenv("STARTING_CAPITAL", "100000"))

    # Capital mode auto-detected but can be forced:
    # "SMALL" = < ₹50K  | "GROWING" = ₹50K–₹2L  | "FULL" = > ₹2L
    FORCE_CAPITAL_MODE = os.getenv("FORCE_CAPITAL_MODE", "")  # "" = auto-detect

    # Signal thresholds
    MIN_CONFIDENCE_SWING      = float(os.getenv("MIN_CONF_SWING",      "0.60"))
    MIN_CONFIDENCE_INTRADAY   = float(os.getenv("MIN_CONF_INTRADAY",   "0.65"))
    MIN_CONFIDENCE_POSITIONAL = float(os.getenv("MIN_CONF_POSITIONAL", "0.55"))

    # Position sizing
    MAX_RISK_PER_TRADE_PCT    = float(os.getenv("MAX_RISK_PCT",  "0.02"))   # 2%
    MAX_POSITION_PCT_CAPITAL  = float(os.getenv("MAX_POS_PCT",   "0.40"))   # 40%
    STOP_ATR_MULTIPLIER       = float(os.getenv("STOP_ATR_MULT", "2.0"))
    TARGET_ATR_MULTIPLIER     = float(os.getenv("TGT_ATR_MULT",  "5.0"))
    MIN_REWARD_RISK           = float(os.getenv("MIN_RR",         "2.0"))

    # Maximum hold days per timeframe
    MAX_HOLD_SWING            = int(os.getenv("MAX_HOLD_SWING",       "7"))
    MAX_HOLD_INTRADAY         = int(os.getenv("MAX_HOLD_INTRADAY",    "1"))
    MAX_HOLD_POSITIONAL       = int(os.getenv("MAX_HOLD_POSITIONAL",  "21"))

    # Trailing stop — activate after this % profit
    TRAIL_TRIGGER_PCT         = float(os.getenv("TRAIL_TRIGGER", "0.02"))   # 2%
    TRAIL_DISTANCE_PCT        = float(os.getenv("TRAIL_DIST",    "0.015"))  # 1.5%


# ══════════════════════════════════════════════════════════════
# CIRCUIT BREAKER THRESHOLDS
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# PER-BANK CONFIGURATION
# ══════════════════════════════════════════════════════════════

BANK_CONFIG: Dict[str, Dict] = {
    "HDFCBANK.NS": {
        "bse_code":     "500180",
        "nse_symbol":   "HDFCBANK",
        "screener_id":  "HDFCBANK",
        "display_name": "HDFC Bank",
        "earnings_dates": [
            "2026-01-20", "2026-04-20", "2026-07-19", "2026-10-18",
            "2027-01-19", "2027-04-19", "2027-07-18", "2027-10-17",
        ],
        "peers": ["ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
    },
    "ICICIBANK.NS": {
        "bse_code":     "532174",
        "nse_symbol":   "ICICIBANK",
        "screener_id":  "ICICIBANK",
        "display_name": "ICICI Bank",
        "earnings_dates": [
            "2026-01-25", "2026-04-26", "2026-07-26", "2026-10-25",
            "2027-01-24", "2027-04-25", "2027-07-25", "2027-10-24",
        ],
        "peers": ["HDFCBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
    },
    "KOTAKBANK.NS": {
        "bse_code":     "500247",
        "nse_symbol":   "KOTAKBANK",
        "screener_id":  "KOTAKBANK",
        "display_name": "Kotak Bank",
        "earnings_dates": [
            "2026-01-18", "2026-04-18", "2026-07-18", "2026-10-17",
            "2027-01-17", "2027-04-17", "2027-07-17", "2027-10-16",
        ],
        "peers": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS"],
    },
    "AXISBANK.NS": {
        "bse_code":     "532215",
        "nse_symbol":   "AXISBANK",
        "screener_id":  "AXISBANK",
        "display_name": "Axis Bank",
        "earnings_dates": [
            "2026-01-16", "2026-04-25", "2026-07-25", "2026-10-24",
            "2027-01-15", "2027-04-24", "2027-07-24", "2027-10-23",
        ],
        "peers": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS"],
    },
    "INDUSINDBK.NS": {
        "bse_code":     "532187",
        "nse_symbol":   "INDUSINDBK",
        "screener_id":  "INDUSINDBK",
        "display_name": "IndusInd Bank",
        "earnings_dates": [
            "2026-01-14", "2026-04-13", "2026-07-14", "2026-10-13",
            "2027-01-13", "2027-04-12", "2027-07-13", "2027-10-12",
        ],
        "peers": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS"],
    },
}


def get_bank_config(ticker: str) -> Dict:
    """Return per-bank config dict. Raises KeyError for unknown tickers."""
    cfg = BANK_CONFIG.get(ticker)
    if cfg is None:
        raise KeyError(f"Unknown ticker '{ticker}'. Valid: {list(BANK_CONFIG.keys())}")
    return cfg


class RiskConfig:
    """Circuit breaker and risk management thresholds."""

    # Monthly drawdown limits by capital mode
    HALT_DD_SMALL   = float(os.getenv("HALT_DD_SMALL",   "-0.08"))  # -8%
    HALT_DD_GROWING = float(os.getenv("HALT_DD_GROWING", "-0.12"))  # -12%
    HALT_DD_FULL    = float(os.getenv("HALT_DD_FULL",    "-0.15"))  # -15%

    PAUSE_DD_SMALL   = float(os.getenv("PAUSE_DD_SMALL",   "-0.05"))
    PAUSE_DD_GROWING = float(os.getenv("PAUSE_DD_GROWING", "-0.08"))
    PAUSE_DD_FULL    = float(os.getenv("PAUSE_DD_FULL",    "-0.10"))

    # Consecutive loss limits
    CONSEC_LOSS_PAUSE = int(os.getenv("CONSEC_PAUSE", "3"))
    CONSEC_LOSS_HALT  = int(os.getenv("CONSEC_HALT",  "5"))

    # Model accuracy minimum (rolling 10 signals)
    MIN_MODEL_ACCURACY = float(os.getenv("MIN_MODEL_ACC", "0.40"))

    # Portfolio heat (total risk deployed)
    MAX_PORTFOLIO_HEAT = float(os.getenv("MAX_HEAT", "0.10"))   # 10% of equity


# ══════════════════════════════════════════════════════════════
# EXTERNAL API CREDENTIALS
# ══════════════════════════════════════════════════════════════

class APIConfig:
    """All external API credentials — loaded from .env only."""

    # Telegram (required — primary delivery channel)
    TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN",   "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

    # Anthropic Claude (required — LLM earnings analysis)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    # Twitter/X (optional — social sentiment)
    TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

    # Reddit (optional — social sentiment)
    REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID",     "")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
    REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT",    "QuantEdge/1.0")

    # Zerodha Kite (optional — for auto-execution in phase 2)
    KITE_API_KEY    = os.getenv("KITE_API_KEY",    "")
    KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")
    KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")

    @classmethod
    def validate(cls) -> list:
        """
        Check required credentials are set.
        Returns list of missing credentials.
        """
        missing = []
        if not cls.TELEGRAM_TOKEN:
            missing.append("TELEGRAM_TOKEN")
        if not cls.TELEGRAM_CHAT_ID:
            missing.append("TELEGRAM_CHAT_ID")
        if not cls.ANTHROPIC_API_KEY:
            missing.append("ANTHROPIC_API_KEY")
        return missing


# ══════════════════════════════════════════════════════════════
# SCHEDULER CONFIGURATION
# ══════════════════════════════════════════════════════════════

class SchedulerConfig:
    """Task runner timing configuration."""

    # How often to run the signal pipeline (seconds)
    SIGNAL_INTERVAL_SEC   = int(os.getenv("SIGNAL_INTERVAL",   "900"))   # 15 min
    NEWS_INTERVAL_SEC     = int(os.getenv("NEWS_INTERVAL",     "900"))   # 15 min
    OPTIONS_INTERVAL_SEC  = int(os.getenv("OPTIONS_INTERVAL",  "1800"))  # 30 min
    BSE_INTERVAL_SEC      = int(os.getenv("BSE_INTERVAL",      "1800"))  # 30 min
    FII_INTERVAL_SEC      = int(os.getenv("FII_INTERVAL",      "7200"))  # 2 hr
    SOCIAL_INTERVAL_SEC   = int(os.getenv("SOCIAL_INTERVAL",   "7200"))  # 2 hr

    # Weekly retrain: Sunday midnight
    RETRAIN_DAY   = int(os.getenv("RETRAIN_DAY",  "6"))   # 6 = Sunday
    RETRAIN_HOUR  = int(os.getenv("RETRAIN_HOUR", "0"))   # midnight


# ══════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ══════════════════════════════════════════════════════════════

class ModelConfig:
    """ML model training parameters."""

    # Swing model (primary)
    SWING_FORWARD_DAYS = int(os.getenv("SWING_FWD_DAYS",   "5"))
    SWING_THRESHOLD    = float(os.getenv("SWING_THRESH",   "0.025"))
    SWING_CV_SPLITS    = int(os.getenv("SWING_CV",         "10"))

    # Intraday model
    INTRADAY_FORWARD_BARS = int(os.getenv("INTRA_FWD_BARS", "6"))
    INTRADAY_THRESHOLD    = float(os.getenv("INTRA_THRESH", "0.004"))
    INTRADAY_CV_SPLITS    = int(os.getenv("INTRA_CV",       "5"))

    # Positional model
    POS_FORWARD_WEEKS  = int(os.getenv("POS_FWD_WEEKS",    "3"))
    POS_THRESHOLD      = float(os.getenv("POS_THRESH",     "0.05"))
    POS_CV_SPLITS      = int(os.getenv("POS_CV",           "5"))

    # Model replacement policy
    # New model replaces old only if accuracy >= old - this margin
    REPLACE_ACCURACY_MARGIN = float(os.getenv("REPLACE_MARGIN", "0.01"))

    # XGBoost common params
    EARLY_STOPPING_ROUNDS = int(os.getenv("EARLY_STOPPING", "30"))
    N_JOBS                = int(os.getenv("N_JOBS",          "-1"))  # all CPUs


# ══════════════════════════════════════════════════════════════
# DATABASE & FILE PATHS
# ══════════════════════════════════════════════════════════════

class PathConfig:
    """File system paths for all data, models, and logs."""

    BASE_DIR     = Path(__file__).parent

    # Database
    DB_PATH      = str(BASE_DIR / os.getenv("DB_PATH", "database/trading.db"))

    # Model artifacts
    MODELS_DIR   = str(BASE_DIR / "models/saved")
    EVAL_DIR     = str(BASE_DIR / "models/evaluation")

    # Logs
    LOGS_DIR     = str(BASE_DIR / "logs")
    SIGNAL_LOG   = str(BASE_DIR / "logs/signal_log.csv")
    TRADE_LOG    = str(BASE_DIR / "logs/trade_log.csv")
    ERROR_LOG    = str(BASE_DIR / "logs/error_log.txt")
    RETRAIN_LOG  = str(BASE_DIR / "logs/retrain_log.txt")

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create all required directories on startup."""
        dirs = [
            cls.MODELS_DIR, cls.EVAL_DIR, cls.LOGS_DIR,
            str(cls.BASE_DIR / "database"),
            str(cls.BASE_DIR / "models/saved/finbert_cache"),
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        cls._migrate_model_paths()

    @classmethod
    def _migrate_model_paths(cls) -> None:
        """One-time rename of old model files to ticker-prefixed names."""
        _renames = {
            "models/saved/swing_model.pkl":                "models/saved/HDFCBANK_technical.pkl",
            "models/saved/swing_model_calibrated.pkl":     "models/saved/HDFCBANK_technical_calibrated.pkl",
            "models/saved/swing_features.pkl":             "models/saved/HDFCBANK_technical_features.pkl",
            "models/saved/swing_model_full.pkl":           "models/saved/HDFCBANK_full.pkl",
            "models/saved/swing_model_full_calibrated.pkl":"models/saved/HDFCBANK_full_calibrated.pkl",
            "models/saved/swing_features_full.pkl":        "models/saved/HDFCBANK_full_features.pkl",
            "models/saved/intraday_model.pkl":             "models/saved/HDFCBANK_intraday_technical.pkl",
            "models/saved/intraday_model_calibrated.pkl":  "models/saved/HDFCBANK_intraday_technical_calibrated.pkl",
            "models/saved/intraday_features.pkl":          "models/saved/HDFCBANK_intraday_technical_features.pkl",
            "models/saved/intraday_model_full.pkl":        "models/saved/HDFCBANK_intraday_full.pkl",
            "models/saved/intraday_model_full_calibrated.pkl": "models/saved/HDFCBANK_intraday_full_calibrated.pkl",
            "models/saved/intraday_features_full.pkl":     "models/saved/HDFCBANK_intraday_full_features.pkl",
            "models/saved/positional_model.pkl":           "models/saved/HDFCBANK_positional_technical.pkl",
            "models/saved/positional_model_calibrated.pkl":"models/saved/HDFCBANK_positional_technical_calibrated.pkl",
            "models/saved/positional_features.pkl":        "models/saved/HDFCBANK_positional_technical_features.pkl",
            "models/saved/positional_model_full.pkl":      "models/saved/HDFCBANK_positional_full.pkl",
            "models/saved/positional_model_full_calibrated.pkl": "models/saved/HDFCBANK_positional_full_calibrated.pkl",
            "models/saved/positional_features_full.pkl":   "models/saved/HDFCBANK_positional_full_features.pkl",
        }
        for old, new in _renames.items():
            old_path = str(cls.BASE_DIR / old)
            new_path = str(cls.BASE_DIR / new)
            if os.path.exists(old_path) and not os.path.exists(new_path):
                os.rename(old_path, new_path)


# ══════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ══════════════════════════════════════════════════════════════

class LogConfig:
    """Logging level and format."""

    LEVEL  = os.getenv("LOG_LEVEL", "INFO")    # DEBUG / INFO / WARNING / ERROR
    FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    # Write to file as well as console
    LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"
    LOG_FILE    = PathConfig.ERROR_LOG


# ══════════════════════════════════════════════════════════════
# NSE CALENDAR
# ══════════════════════════════════════════════════════════════

class CalendarConfig:
    """NSE holidays and important dates."""

    # HDFC Bank quarterly result dates — reference BANK_CONFIG as single source of truth
    HDFC_EARNINGS_DATES = BANK_CONFIG["HDFCBANK.NS"]["earnings_dates"]

    # RBI MPC meeting dates 2026
    RBI_MPC_DATES = [
        "2026-02-07", "2026-04-09", "2026-06-06",
        "2026-08-08", "2026-10-08", "2026-12-05",
    ]

    # NSE trading holidays 2026
    NSE_HOLIDAYS = [
        "2026-01-26", "2026-03-25", "2026-04-02",
        "2026-04-14", "2026-04-17", "2026-05-01",
        "2026-08-15", "2026-10-02", "2026-10-20",
        "2026-11-04", "2026-12-25",
    ]

    # Trading session times (IST)
    MARKET_OPEN_TIME  = "09:15"
    MARKET_CLOSE_TIME = "15:30"
    PRE_OPEN_TIME     = "09:00"
    INTRADAY_CUTOFF   = "15:15"


# ══════════════════════════════════════════════════════════════
# ENVIRONMENT VALIDATION
# ══════════════════════════════════════════════════════════════

def validate_config() -> tuple:
    """
    Validate entire configuration on startup.
    Returns (is_valid: bool, issues: list).
    """
    issues = []

    # Missing credentials
    missing_creds = APIConfig.validate()
    for cred in missing_creds:
        issues.append(f"Missing credential: {cred}")

    # Capital sanity check
    if TradingConfig.STARTING_CAPITAL <= 0:
        issues.append("STARTING_CAPITAL must be positive")

    # Risk parameter sanity
    if TradingConfig.MIN_REWARD_RISK < 1.5:
        issues.append("MIN_REWARD_RISK below 1.5 — dangerous")

    if TradingConfig.MAX_RISK_PER_TRADE_PCT > 0.05:
        issues.append("MAX_RISK_PER_TRADE_PCT above 5% — dangerous")

    is_valid = len([i for i in issues if "Missing" in i and "ANTHROPIC" not in i and
                    "TELEGRAM" not in i]) == 0

    return is_valid, issues


def print_config_summary() -> None:
    """Print configuration summary on startup."""
    print("=" * 55)
    print("QUANT EDGE — Configuration")
    print("=" * 55)
    print(f"  Ticker:          {TradingConfig.TICKER}")
    print(f"  Universe:        {', '.join(TradingConfig.UNIVERSE)}")
    print(f"  Capital:         ₹{TradingConfig.STARTING_CAPITAL:,.0f}")
    print(f"  Max risk/trade:  {TradingConfig.MAX_RISK_PER_TRADE_PCT:.1%}")
    print(f"  Min R:R:         {TradingConfig.MIN_REWARD_RISK:.1f}:1")
    print(f"  DB path:         {PathConfig.DB_PATH}")
    print(f"  Log level:       {LogConfig.LEVEL}")
    print(f"  Telegram:        {'✅ SET' if APIConfig.TELEGRAM_TOKEN else '❌ NOT SET'}")
    print(f"  Anthropic API:   {'✅ SET' if APIConfig.ANTHROPIC_API_KEY else '❌ NOT SET'}")
    print(f"  Twitter:         {'✅ SET' if APIConfig.TWITTER_BEARER_TOKEN else '⚠️  OPTIONAL'}")
    print("=" * 55)

    _, issues = validate_config()
    if issues:
        print("  ⚠️  Configuration issues:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ✅ Configuration valid")
    print("=" * 55)