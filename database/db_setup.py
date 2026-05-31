# database/db_setup.py
# Creates ALL SQLite tables on first run
# Single source of truth for schema — every table defined here
# Run once at deployment: python -c "from database.db_setup import DatabaseSetup; DatabaseSetup().setup_all()"

import sqlite3
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = "database/trading.db"


class DatabaseSetup:
    """
    Initialises the complete SQLite database schema.

    Design decisions:
    - SQLite over PostgreSQL: zero infrastructure, runs on Railway free tier
    - One DB file: easier backup — just copy trading.db
    - UNIQUE constraints everywhere: prevents duplicate data on retry
    - Indexes on all date/timestamp columns: fast time-range queries
    - signal_uuid stitches feature_snapshots → open_trades → closed_trades → signal_outcomes
      so any closed trade can be traced back to the exact features that produced it.
      FKs are declared and enforced via PRAGMA foreign_keys=ON.

    Tables (20 total):
    Data layer     : news, bse_announcements, options_snapshots,
                     fii_data, bulk_deals, global_snapshots,
                     social_posts, shareholding_pattern,
                     insider_block_deals, alt_trends
    Processing     : llm_earnings_cache, llm_announcement_scores,
                     fundamentals, regime_snapshots, macro_data
    Trading        : open_trades, closed_trades, signal_outcomes
    System         : feature_snapshots, earnings_history,
                     earnings_calendar
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def setup_all(self) -> None:
        """Create all tables. Safe to call multiple times (IF NOT EXISTS)."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = self._connect()
        try:
            self._data_tables(conn)
            self._processing_tables(conn)
            self._trading_tables(conn)
            self._system_tables(conn)
            self._migrate_columns(conn)
            self._create_indexes(conn)
            conn.commit()
            logger.info(f"Database initialised: {self.db_path}")
        finally:
            conn.close()

    def _migrate_columns(self, conn: sqlite3.Connection) -> None:
        """
        Idempotently add columns to existing tables.

        SQLite can't ALTER TABLE with FK or DEFAULT non-constant; we add the
        column nullable. New rows are written with the column populated, old
        rows stay NULL.
        """
        migrations = [
            # Audit trail UUID (added in Tier-2 phase)
            ("feature_snapshots", "signal_uuid", "TEXT"),
            ("open_trades",       "signal_uuid", "TEXT"),
            ("closed_trades",     "signal_uuid", "TEXT"),
            ("signal_outcomes",   "signal_uuid", "TEXT"),
            # Multi-asset ticker column (added in multi-bank expansion)
            ("open_trades",       "ticker", "TEXT DEFAULT 'HDFCBANK.NS'"),
            ("closed_trades",     "ticker", "TEXT DEFAULT 'HDFCBANK.NS'"),
            ("signal_outcomes",   "ticker", "TEXT DEFAULT 'HDFCBANK.NS'"),
            ("feature_snapshots", "ticker", "TEXT DEFAULT 'HDFCBANK.NS'"),
            ("options_snapshots", "ticker", "TEXT DEFAULT 'HDFCBANK.NS'"),
            # fundamentals.ticker is now a base column + UNIQUE — see
            # _migrate_fundamentals_table for legacy rebuild.
            ("regime_snapshots",  "ticker", "TEXT DEFAULT 'HDFCBANK.NS'"),
            ("bse_announcements", "ticker", "TEXT DEFAULT 'HDFCBANK.NS'"),
            # Full feature snapshot for win/loss attribution
            ("feature_snapshots", "feature_values", "TEXT"),
            # RBI rate effective date — keyed seed for macro_data (P2)
            ("macro_data",        "effective_date", "TEXT"),
            # earnings_calendar ticker column (added in P2 phase)
            ("earnings_calendar", "ticker", "TEXT DEFAULT 'HDFCBANK.NS'"),
            # Per-category tracking in closed_trades (added 2026-05-25)
            # Allows per-category win-rate / PF analysis on historical trades.
            ("closed_trades", "trade_type", "TEXT DEFAULT 'swing'"),
            # ── Attributable outcome loop (added 2026-05-31) ──────────────
            # Entry context — captured at open, carried to close, so paper-
            # trading P&L can be sliced by the decisions that produced it.
            # Without these the learning loop records blind outcomes.
            ("open_trades",   "confidence",      "REAL"),
            ("open_trades",   "alignment",       "TEXT"),
            ("open_trades",   "regime_at_entry", "TEXT"),
            ("open_trades",   "entry_quality",   "TEXT"),
            ("open_trades",   "regime_match",    "INTEGER"),
            ("open_trades",   "size_mult",       "REAL"),
            ("open_trades",   "atr_at_entry",    "REAL"),
            ("open_trades",   "reward_risk",     "REAL"),
            ("closed_trades", "confidence",      "REAL"),
            ("closed_trades", "alignment",       "TEXT"),
            ("closed_trades", "regime_at_entry", "TEXT"),
            ("closed_trades", "entry_quality",   "TEXT"),
            ("closed_trades", "regime_match",    "INTEGER"),
            ("closed_trades", "size_mult",       "REAL"),
            ("closed_trades", "atr_at_entry",    "REAL"),
            ("closed_trades", "reward_risk",     "REAL"),
            # Exit analytics — R-multiple, hold time, and MFE/MAE (max
            # favorable / adverse excursion: reveals stop/target misplacement).
            ("closed_trades", "r_multiple",      "REAL"),
            ("closed_trades", "hold_hours",      "REAL"),
            ("closed_trades", "mfe_pct",         "REAL"),
            ("closed_trades", "mae_pct",         "REAL"),
            # signal_outcomes — rich attribution tags for calibration slicing.
            ("signal_outcomes", "category",      "TEXT"),
            ("signal_outcomes", "entry_quality", "TEXT"),
            ("signal_outcomes", "regime_match",  "INTEGER"),
            ("signal_outcomes", "r_multiple",    "REAL"),
            ("signal_outcomes", "hold_hours",    "REAL"),
            ("signal_outcomes", "exit_reason",   "TEXT"),
            ("signal_outcomes", "mfe_pct",       "REAL"),
            ("signal_outcomes", "mae_pct",       "REAL"),
        ]
        for table, col, coltype in migrations:
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
                logger.info(f"Migrated: {table}.{col} added")

    def _migrate_fundamentals_table(self, conn: sqlite3.Connection) -> None:
        """Rebuild legacy `fundamentals` (no UNIQUE(ticker, fetched_at)) if needed.

        Legacy table: no ticker column, no UNIQUE. New shape: ticker NOT NULL,
        UNIQUE(ticker, fetched_at). Backfills ticker='HDFCBANK.NS' on copy.
        """
        cols = {row[1] for row in conn.execute("PRAGMA table_info(fundamentals)")}
        if not cols:
            return  # CREATE just ran — already new shape

        has_unique = False
        for idx in conn.execute("PRAGMA index_list(fundamentals)").fetchall():
            idx_name = idx[1]
            idx_cols = [c[2] for c in conn.execute(f"PRAGMA index_info('{idx_name}')").fetchall()]
            if set(idx_cols) == {"ticker", "fetched_at"}:
                has_unique = True
                break
        if "ticker" in cols and has_unique:
            return  # already correct shape

        logger.info("Rebuilding fundamentals to add UNIQUE(ticker, fetched_at)")
        conn.execute("ALTER TABLE fundamentals RENAME TO fundamentals_old")
        conn.execute("""
            CREATE TABLE fundamentals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL DEFAULT 'HDFCBANK.NS',
                nim             REAL,  npa_gross      REAL,
                npa_net         REAL,  casa_ratio      REAL,
                loan_growth     REAL,  deposit_growth  REAL,
                car             REAL,  pcr             REAL,
                roe             REAL,  roa             REAL,
                cost_to_income  REAL,  pb_ratio        REAL,
                pe_ratio        REAL,  eps_surprise    REAL,
                fetched_at      TEXT,
                UNIQUE(ticker, fetched_at)
            )
        """)
        old_cols = {row[1] for row in conn.execute("PRAGMA table_info(fundamentals_old)")}
        ticker_expr = "COALESCE(ticker, 'HDFCBANK.NS')" if "ticker" in old_cols else "'HDFCBANK.NS'"
        conn.execute(f"""
            INSERT OR IGNORE INTO fundamentals
            (ticker, nim, npa_gross, npa_net, casa_ratio, loan_growth,
             deposit_growth, car, pcr, roe, roa, cost_to_income,
             pb_ratio, pe_ratio, eps_surprise, fetched_at)
            SELECT {ticker_expr}, nim, npa_gross, npa_net, casa_ratio, loan_growth,
                   deposit_growth, car, pcr, roe, roa, cost_to_income,
                   pb_ratio, pe_ratio, eps_surprise, fetched_at
            FROM fundamentals_old
        """)
        conn.execute("DROP TABLE fundamentals_old")
        logger.info("fundamentals migrated to UNIQUE(ticker, fetched_at)")

    def verify(self) -> dict:
        """Verify all tables exist and return row counts."""
        conn   = self._connect()
        tables = [
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        counts = {}
        for t in tables:
            try:
                row = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
                counts[t] = row[0]
            except Exception:
                counts[t] = -1
        conn.close()
        return counts

    # ─────────────────────────────────────────────────────────────
    # TABLE DEFINITIONS
    # ─────────────────────────────────────────────────────────────

    def _data_tables(self, conn: sqlite3.Connection) -> None:
        """Raw data from external sources."""

        conn.execute("""
            CREATE TABLE IF NOT EXISTS news (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                title           TEXT    UNIQUE,
                summary         TEXT,
                published       TEXT,
                source          TEXT,
                link            TEXT,
                sentiment_score REAL    DEFAULT 0.0,
                llm_score       REAL    DEFAULT 0.0,
                is_important    INTEGER DEFAULT 0,
                processed       INTEGER DEFAULT 0,
                llm_analyzed    INTEGER DEFAULT 0,
                created_at      TEXT
            )
        """)

        # bse_announcements — multi-bank scoped.
        # Old shape: UNIQUE(title) collapses two banks' filings with the same title.
        # New shape: UNIQUE(ticker, title, announcement_date). If an old single-
        # column UNIQUE table is present, migrate by copying into a new table.
        old_idx = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='bse_announcements' AND name='sqlite_autoindex_bse_announcements_1'"
        ).fetchone()
        if old_idx:
            conn.execute("ALTER TABLE bse_announcements RENAME TO bse_announcements_old")
            conn.execute("""
                CREATE TABLE bse_announcements (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker            TEXT    NOT NULL DEFAULT 'HDFCBANK.NS',
                    title             TEXT,
                    description       TEXT,
                    category          TEXT,
                    source            TEXT,
                    announcement_date TEXT,
                    is_high_priority  INTEGER DEFAULT 0,
                    llm_analyzed      INTEGER DEFAULT 0,
                    llm_impact_score  REAL    DEFAULT 0.0,
                    created_at        TEXT,
                    UNIQUE(ticker, title, announcement_date)
                )
            """)
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO bse_announcements
                    (ticker, title, description, category, source,
                     announcement_date, is_high_priority,
                     llm_analyzed, llm_impact_score, created_at)
                    SELECT 'HDFCBANK.NS', title, description, category, source,
                           announcement_date, is_high_priority,
                           llm_analyzed, llm_impact_score, created_at
                    FROM bse_announcements_old
                """)
            except Exception as e:
                logger.warning(f"bse_announcements legacy copy skipped: {e}")
            conn.execute("DROP TABLE bse_announcements_old")
            logger.info("bse_announcements migrated to UNIQUE(ticker, title, announcement_date)")
        else:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bse_announcements (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker            TEXT    NOT NULL DEFAULT 'HDFCBANK.NS',
                    title             TEXT,
                    description       TEXT,
                    category          TEXT,
                    source            TEXT,
                    announcement_date TEXT,
                    is_high_priority  INTEGER DEFAULT 0,
                    llm_analyzed      INTEGER DEFAULT 0,
                    llm_impact_score  REAL    DEFAULT 0.0,
                    created_at        TEXT,
                    UNIQUE(ticker, title, announcement_date)
                )
            """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS options_snapshots (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker             TEXT    NOT NULL DEFAULT 'HDFCBANK.NS',
                pcr                REAL,
                pcr_roc            REAL,
                max_pain           REAL,
                iv_percentile      REAL,
                atm_iv             REAL,
                max_call_oi_strike REAL,
                max_put_oi_strike  REAL,
                implied_move_pct   REAL,
                total_call_oi      INTEGER,
                total_put_oi       INTEGER,
                spot_price         REAL,
                fetched_at         TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS fii_data (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date   TEXT    UNIQUE,
                fii_buy_cr   REAL,
                fii_sell_cr  REAL,
                fii_net_cr   REAL,
                dii_buy_cr   REAL,
                dii_sell_cr  REAL,
                dii_net_cr   REAL,
                fetched_at   TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS bulk_deals (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date   TEXT,
                client_name  TEXT,
                trade_type   TEXT,
                quantity     REAL,
                price        REAL,
                value_cr     REAL,
                deal_type    TEXT    DEFAULT 'BULK',
                created_at   TEXT,
                UNIQUE(trade_date, client_name, trade_type, quantity)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS global_snapshots (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                sp500_1d_pct      REAL,  sp500_5d_pct     REAL,
                nasdaq_1d_pct     REAL,  india_vix_level   REAL,
                india_vix_1d_pct  REAL,  crude_1d_pct      REAL,
                crude_5d_pct      REAL,  crude_level       REAL,
                dxy_level         REAL,  dxy_5d_pct        REAL,
                dxy_above_20ma    INTEGER,
                usdinr_level      REAL,  usdinr_5d_pct     REAL,
                us_10y_level      REAL,  us_10y_5d_chg     REAL,
                banknifty_1d_pct  REAL,  banknifty_5d_pct  REAL,
                jpmorgan_5d_pct   REAL,  verdict           TEXT,
                fetched_at        TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS social_posts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash       TEXT    UNIQUE,
                text_snippet    TEXT,
                sentiment_label TEXT,
                score           REAL,
                source          TEXT,
                created_at      TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS shareholding_pattern (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                quarter       TEXT    UNIQUE,
                promoter_pct  REAL,
                fii_pct       REAL,
                dii_pct       REAL,
                public_pct    REAL,
                pledge_pct    REAL,
                total_shares  REAL,
                created_at    TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS insider_block_deals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date  TEXT,
                client_name TEXT,
                buy_sell    TEXT,
                quantity    REAL,
                price       REAL,
                value_cr    REAL,
                deal_type   TEXT    DEFAULT 'BLOCK',
                created_at  TEXT,
                UNIQUE(trade_date, client_name, buy_sell, quantity)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS alt_trends (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword       TEXT,
                group_name    TEXT,
                recent_avg    REAL,
                older_avg     REAL,
                slope         REAL,
                latest_value  REAL,
                fetched_at    TEXT
            )
        """)

    def _processing_tables(self, conn: sqlite3.Connection) -> None:
        """Derived data from processing layer."""

        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_earnings_cache (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                earnings_data TEXT,
                analyzed_at   TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_announcement_scores (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                title        TEXT,
                impact_score REAL,
                confidence   REAL,
                timeframe    TEXT,
                reason       TEXT,
                analyzed_at  TEXT
            )
        """)

        # fundamentals — multi-bank scoped with UNIQUE(ticker, fetched_at) so
        # Sunday refresh re-runs UPSERT cleanly. Legacy table (no ticker column,
        # no UNIQUE) is rebuilt below in _migrate_fundamentals_table.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL DEFAULT 'HDFCBANK.NS',
                nim             REAL,  npa_gross      REAL,
                npa_net         REAL,  casa_ratio      REAL,
                loan_growth     REAL,  deposit_growth  REAL,
                car             REAL,  pcr             REAL,
                roe             REAL,  roa             REAL,
                cost_to_income  REAL,  pb_ratio        REAL,
                pe_ratio        REAL,  eps_surprise    REAL,
                fetched_at      TEXT,
                UNIQUE(ticker, fetched_at)
            )
        """)
        self._migrate_fundamentals_table(conn)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_snapshots (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker        TEXT    NOT NULL DEFAULT 'HDFCBANK.NS',
                regime        TEXT,
                stability     REAL,
                bull_prob     REAL,
                bear_prob     REAL,
                high_vol_prob REAL,
                choppy_prob   REAL,
                detected_at   TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS macro_data (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                rbi_rate      REAL,
                credit_growth REAL,
                cpi_yoy       REAL,
                fetched_at    TEXT
            )
        """)

    def _trading_tables(self, conn: sqlite3.Connection) -> None:
        """Trade lifecycle — open, closed, outcomes."""

        conn.execute("""
            CREATE TABLE IF NOT EXISTS open_trades (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_uuid    TEXT,
                signal         TEXT,
                entry_price    REAL,
                stop_price     REAL,
                target_price   REAL,
                shares         INTEGER,
                risk_amount    REAL,
                trade_type     TEXT    DEFAULT 'swing',
                highest_price  REAL,
                lowest_price   REAL,
                status         TEXT    DEFAULT 'OPEN',
                opened_at      TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS closed_trades (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_uuid  TEXT,
                ticker       TEXT    DEFAULT 'HDFCBANK.NS',
                signal       TEXT,
                trade_type   TEXT    DEFAULT 'swing',
                entry_price  REAL,
                exit_price   REAL,
                shares       INTEGER,
                pnl_amount   REAL,
                pnl_pct      REAL,
                exit_reason  TEXT,
                close_date   TEXT,
                status       TEXT    DEFAULT 'CLOSED'
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_outcomes (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_uuid  TEXT,
                signal       TEXT,
                confidence   REAL,
                alignment    TEXT,
                regime       TEXT,
                outcome      TEXT,
                pnl_pct      REAL,
                pnl_amount   REAL,
                created_at   TEXT
            )
        """)

    def _system_tables(self, conn: sqlite3.Connection) -> None:
        """System monitoring and audit tables."""

        # gate_results — per-ticker latest snapshot of the 6-gate pipeline.
        # One row per ticker (UPSERT on each signal cycle); dashboard reads
        # straight from here instead of `logs/last_gate_results_*.json`.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gate_results (
                ticker        TEXT    NOT NULL PRIMARY KEY,
                gate_results  TEXT,
                regime        TEXT,
                written_at    TEXT    NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_snapshots (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker         TEXT    NOT NULL DEFAULT 'HDFCBANK.NS',
                signal_uuid    TEXT,
                built_at       TEXT,
                build_time_sec REAL,
                data_quality   TEXT,
                n_features     INTEGER,
                current_price  REAL,
                regime         TEXT,
                risk_allowed   INTEGER
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS earnings_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                result_date   TEXT    UNIQUE,
                ticker        TEXT    DEFAULT 'HDFCBANK.NS',
                actual_eps    REAL,
                estimated_eps REAL,
                surprise_pct  REAL,
                beat_estimates INTEGER,
                created_at    TEXT
            )
        """)

        # Multi-bank: composite UNIQUE(ticker, result_date) instead of UNIQUE(result_date)
        # First check if old single-column UNIQUE table exists and migrate it
        old_idx = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='earnings_calendar' AND name='sqlite_autoindex_earnings_calendar_1'"
        ).fetchone()
        if old_idx:
            conn.execute("ALTER TABLE earnings_calendar RENAME TO earnings_calendar_old")
            conn.execute("""
                CREATE TABLE earnings_calendar (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker      TEXT    NOT NULL DEFAULT 'HDFCBANK.NS',
                    result_date TEXT    NOT NULL,
                    source      TEXT    DEFAULT 'hardcoded',
                    confirmed   INTEGER DEFAULT 0,
                    created_at  TEXT,
                    UNIQUE(ticker, result_date)
                )
            """)
            conn.execute(
                "INSERT OR IGNORE INTO earnings_calendar "
                "(ticker, result_date, source, confirmed, created_at) "
                "SELECT ticker, result_date, source, confirmed, created_at "
                "FROM earnings_calendar_old"
            )
            conn.execute("DROP TABLE earnings_calendar_old")
            logger.info("earnings_calendar migrated to UNIQUE(ticker, result_date)")
        else:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS earnings_calendar (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker      TEXT    NOT NULL DEFAULT 'HDFCBANK.NS',
                    result_date TEXT    NOT NULL,
                    source      TEXT    DEFAULT 'hardcoded',
                    confirmed   INTEGER DEFAULT 0,
                    created_at  TEXT,
                    UNIQUE(ticker, result_date)
                )
            """)

    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """Performance indexes on all frequently-queried columns."""
        indexes = [
            ("idx_news_created",       "news",                  "created_at DESC"),
            ("idx_news_processed",     "news",                  "processed, created_at DESC"),
            ("idx_bse_created",        "bse_announcements",     "created_at DESC"),
            ("idx_bse_priority",       "bse_announcements",     "is_high_priority, llm_analyzed"),
            ("idx_options_fetched",    "options_snapshots",     "fetched_at DESC"),
            ("idx_fii_date",           "fii_data",              "trade_date DESC"),
            ("idx_bulk_date",          "bulk_deals",            "trade_date DESC"),
            ("idx_global_fetched",     "global_snapshots",      "fetched_at DESC"),
            ("idx_social_created",     "social_posts",          "created_at DESC"),
            ("idx_sh_quarter",         "shareholding_pattern",  "quarter DESC"),
            ("idx_bd_date",            "insider_block_deals",   "trade_date DESC"),
            ("idx_alt_fetched",        "alt_trends",            "fetched_at DESC"),
            ("idx_regime_detected",    "regime_snapshots",      "detected_at DESC"),
            ("idx_open_status",        "open_trades",           "status"),
            ("idx_open_uuid",          "open_trades",           "signal_uuid"),
            ("idx_closed_date",        "closed_trades",         "close_date DESC"),
            ("idx_closed_uuid",        "closed_trades",         "signal_uuid"),
            ("idx_outcomes_created",   "signal_outcomes",       "created_at DESC"),
            ("idx_outcomes_uuid",      "signal_outcomes",       "signal_uuid"),
            ("idx_feature_built",      "feature_snapshots",     "built_at DESC"),
            ("idx_feature_uuid",       "feature_snapshots",     "signal_uuid"),
            # Multi-asset ticker indexes
            ("idx_open_ticker",        "open_trades",           "ticker, status"),
            ("idx_closed_ticker",      "closed_trades",         "ticker, close_date DESC"),
            ("idx_outcomes_ticker",    "signal_outcomes",       "ticker, created_at DESC"),
            # Earnings calendar
            ("idx_earn_cal_date",      "earnings_calendar",     "result_date ASC"),
            ("idx_earn_cal_ticker",    "earnings_calendar",     "ticker, result_date ASC"),
            # Per-ticker hot paths (added 2026-05-26 with multi-bank scoping)
            ("idx_options_ticker_fetched", "options_snapshots", "ticker, fetched_at DESC"),
            ("idx_feature_ticker_built",   "feature_snapshots", "ticker, built_at DESC"),
            ("idx_regime_ticker_detected", "regime_snapshots",  "ticker, detected_at DESC"),
            ("idx_bse_ticker_created",     "bse_announcements", "ticker, created_at DESC"),
            ("idx_fundamentals_ticker_fetched", "fundamentals", "ticker, fetched_at DESC"),
        ]
        for idx_name, table, cols in indexes:
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {idx_name}
                ON {table} ({cols})
            """)

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        # Performance pragmas
        c.execute("PRAGMA journal_mode=WAL")    # concurrent reads during writes
        c.execute("PRAGMA synchronous=NORMAL")  # safe + faster than FULL
        c.execute("PRAGMA cache_size=10000")    # 10MB page cache
        c.execute("PRAGMA temp_store=MEMORY")
        c.execute("PRAGMA foreign_keys=ON")     # enforce FK constraints
        return c
