# data/rbi_fetcher.py
# RBI repo rate history and direction detection
# No external API calls — all data is hardcoded historical + manual update
# Run: python -m data.rbi_fetcher  (seeds DB and prints current rate + direction)

import sqlite3
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ── Known RBI repo rate changes (effective_date → rate) ──────────────────────
# Source: RBI monetary policy committee announcements
RBI_RATE_HISTORY = {
    "2020-03-27": 4.40,   # emergency cut (COVID)
    "2020-05-22": 4.00,
    "2022-05-04": 4.40,   # hiking cycle begins
    "2022-06-08": 4.90,
    "2022-08-05": 5.40,
    "2022-09-30": 5.90,
    "2022-12-07": 6.25,
    "2023-02-08": 6.50,   # peak
    "2025-02-07": 6.25,   # cutting cycle begins
    "2025-04-09": 6.00,
    "2025-06-06": 5.75,
    "2026-02-07": 5.50,
    "2026-04-09": 5.25,
}

# ── Approximate system credit growth (%YoY) — quarterly snapshots ────────────
RBI_CREDIT_GROWTH_HISTORY = {
    "2022-04-01": 11.5,
    "2023-04-01": 15.8,
    "2024-04-01": 16.3,
    "2025-04-01": 12.1,
    "2026-04-01": 11.0,
}


class RBIFetcher:
    """
    Manages RBI repo rate and credit growth data in the local SQLite DB.

    Design:
    - All data is hardcoded (RBI doesn't have a public machine-readable API).
    - seed_historical_rates() populates macro_data on first run.
    - get_current_rate() / get_rate_direction() are fast DB queries used
      by MacroAnalyzer at every signal cycle.
    - When RBI announces a new rate, add it to RBI_RATE_HISTORY and rerun
      python -m data.rbi_fetcher  to seed the new entry.
    """

    def __init__(self, db_path: str = "database/trading.db"):
        self.db_path = db_path

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def seed_historical_rates(self) -> int:
        """
        Insert all known RBI rate changes into macro_data.
        Uses INSERT OR IGNORE keyed on effective_date — safe to call multiple times.
        Skips seeding if table already has 5+ non-null effective_date rows.

        Returns count of rows inserted.
        """
        try:
            conn = self._connect()
            # Check existing count
            existing = conn.execute(
                "SELECT COUNT(*) FROM macro_data WHERE effective_date IS NOT NULL"
            ).fetchone()[0]

            if existing >= 5:
                logger.debug(f"RBI seed skipped — {existing} rows already present")
                conn.close()
                return 0

            inserted = 0
            for date_str, rate in sorted(RBI_RATE_HISTORY.items()):
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO macro_data
                        (rbi_rate, effective_date, fetched_at)
                        VALUES (?, ?, ?)
                    """, (rate, date_str, str(datetime.now())))
                    if conn.execute("SELECT changes()").fetchone()[0] > 0:
                        inserted += 1
                except Exception as e:
                    logger.warning(f"RBI rate insert failed for {date_str}: {e}")

            conn.commit()
            conn.close()
            logger.info(f"RBI historical rates seeded: {inserted} rows inserted")
            return inserted
        except Exception as e:
            logger.error(f"seed_historical_rates failed: {e}")
            return 0

    def seed_credit_growth(self) -> int:
        """
        Insert credit growth history into macro_data.
        Returns count of rows inserted.
        """
        try:
            conn = self._connect()
            inserted = 0
            for date_str, growth in sorted(RBI_CREDIT_GROWTH_HISTORY.items()):
                try:
                    # Check if a credit_growth entry for this date already exists
                    existing = conn.execute(
                        "SELECT id FROM macro_data WHERE effective_date = ? AND credit_growth IS NOT NULL",
                        (date_str,)
                    ).fetchone()
                    if existing:
                        continue

                    conn.execute("""
                        INSERT OR IGNORE INTO macro_data
                        (credit_growth, effective_date, fetched_at)
                        VALUES (?, ?, ?)
                    """, (growth, date_str, str(datetime.now())))
                    if conn.execute("SELECT changes()").fetchone()[0] > 0:
                        inserted += 1
                except Exception as e:
                    logger.warning(f"Credit growth insert failed for {date_str}: {e}")

            conn.commit()
            conn.close()
            logger.info(f"Credit growth seeded: {inserted} rows inserted")
            return inserted
        except Exception as e:
            logger.error(f"seed_credit_growth failed: {e}")
            return 0

    def get_current_rate(self) -> float:
        """
        Returns the latest known RBI repo rate from DB.
        Falls back to 6.50 (last known peak) if no data found.
        """
        try:
            conn = self._connect()
            row = conn.execute("""
                SELECT rbi_rate FROM macro_data
                WHERE rbi_rate IS NOT NULL AND rbi_rate > 0
                ORDER BY COALESCE(effective_date, fetched_at) DESC
                LIMIT 1
            """).fetchone()
            conn.close()
            if row:
                return float(row[0])
        except Exception as e:
            logger.warning(f"get_current_rate DB error: {e}")
        return 6.50

    def get_rate_direction(self) -> str:
        """
        Returns rate direction based on last 3 distinct rate values.
        CUTTING: latest < previous
        HIKING:  latest > previous
        HOLDING: latest == previous or only 1 data point
        """
        try:
            conn = self._connect()
            rows = conn.execute("""
                SELECT rbi_rate FROM macro_data
                WHERE rbi_rate IS NOT NULL AND rbi_rate > 0
                ORDER BY COALESCE(effective_date, fetched_at) DESC
                LIMIT 3
            """).fetchall()
            conn.close()

            if len(rows) < 2:
                return "HOLDING"
            rates = [float(r[0]) for r in rows]
            if rates[0] < rates[1]:
                return "CUTTING"
            if rates[0] > rates[1]:
                return "HIKING"
            return "HOLDING"
        except Exception as e:
            logger.warning(f"get_rate_direction DB error: {e}")
            return "HOLDING"

    # ─────────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Ensure DB directory exists
    os.makedirs("database", exist_ok=True)

    # Run db_setup first to ensure tables + migrations exist
    try:
        from database.db_setup import DatabaseSetup
        DatabaseSetup().setup_all()
    except Exception as e:
        logger.error(f"DB setup failed: {e}")
        sys.exit(1)

    fetcher = RBIFetcher()

    print("\n=== RBI Fetcher — Historical Seed ===")
    n_rates = fetcher.seed_historical_rates()
    n_credit = fetcher.seed_credit_growth()
    print(f"Rates seeded:         {n_rates} rows")
    print(f"Credit growth seeded: {n_credit} rows")

    rate = fetcher.get_current_rate()
    direction = fetcher.get_rate_direction()
    print(f"\nCurrent RBI repo rate: {rate:.2f}%")
    print(f"Rate direction:        {direction}")
    print("=====================================\n")
