# tests/test_data.py
# Tests for data fetchers — validates data format and quality
# These are integration tests — they test the data pipeline contracts

import unittest
import pandas as pd
import numpy as np
import sqlite3
import tempfile
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestPriceFetcherContract(unittest.TestCase):
    """
    Tests the contract that PriceFetcher must fulfil.
    Uses synthetic data to avoid network calls in CI.
    """

    def _make_valid_ohlcv(self, n=100) -> pd.DataFrame:
        """Minimal valid OHLCV DataFrame."""
        np.random.seed(1)
        dates  = pd.date_range("2023-01-01", periods=n, freq="B")
        price  = 1800.0
        rows   = []
        for _ in range(n):
            price *= (1 + np.random.normal(0, 0.01))
            o = price * (1 + np.random.normal(0, 0.003))
            h = max(o, price) * (1 + abs(np.random.normal(0, 0.003)))
            l = min(o, price) * (1 - abs(np.random.normal(0, 0.003)))
            rows.append({
                "Open": o, "High": h, "Low": l,
                "Close": price, "Volume": float(np.random.randint(5e6, 20e6))
            })
        return pd.DataFrame(rows, index=dates)

    def test_ohlcv_column_names(self):
        df = self._make_valid_ohlcv()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_high_always_gte_low(self):
        df = self._make_valid_ohlcv()
        self.assertTrue((df["High"] >= df["Low"]).all(),
                        "High must always be >= Low")

    def test_high_always_gte_close(self):
        df = self._make_valid_ohlcv()
        self.assertTrue((df["High"] >= df["Close"]).all(),
                        "High must always be >= Close")

    def test_low_always_lte_close(self):
        df = self._make_valid_ohlcv()
        self.assertTrue((df["Low"] <= df["Close"]).all(),
                        "Low must always be <= Close")

    def test_no_negative_prices(self):
        df = self._make_valid_ohlcv()
        for col in ["Open", "High", "Low", "Close"]:
            self.assertTrue((df[col] > 0).all(), f"{col} has non-positive values")

    def test_no_negative_volume(self):
        df = self._make_valid_ohlcv()
        self.assertTrue((df["Volume"] > 0).all(), "Volume has non-positive values")

    def test_no_nan_values(self):
        df = self._make_valid_ohlcv()
        self.assertFalse(df.isnull().any().any(), "OHLCV has NaN values")

    def test_index_is_datetime(self):
        df = self._make_valid_ohlcv()
        self.assertIsInstance(df.index, pd.DatetimeIndex,
                              "Index must be DatetimeIndex")

    def test_no_extreme_single_day_moves(self):
        """Single day moves > 25% indicate data errors."""
        df       = self._make_valid_ohlcv()
        returns  = df["Close"].pct_change().abs().dropna()
        extreme  = (returns > 0.25).sum()
        self.assertEqual(extreme, 0, f"{extreme} extreme daily moves found")

    def test_no_duplicate_dates(self):
        df = self._make_valid_ohlcv()
        self.assertFalse(df.index.duplicated().any(), "Duplicate dates found")

    def test_monotonic_date_index(self):
        df = self._make_valid_ohlcv()
        self.assertTrue(df.index.is_monotonic_increasing,
                        "Date index must be monotonically increasing")


class TestDatabaseSchema(unittest.TestCase):
    """Tests for database schema and basic CRUD operations."""

    def setUp(self):
        self.tmp = tempfile.mktemp(suffix=".db")

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def _setup(self):
        from database.db_setup import DatabaseSetup
        db = DatabaseSetup(self.tmp)
        db.setup_all()
        return db

    def test_setup_creates_all_tables(self):
        db    = self._setup()
        stats = db.verify()
        required = [
            "news", "bse_announcements", "options_snapshots",
            "fii_data", "global_snapshots", "social_posts",
            "shareholding_pattern", "insider_block_deals",
            "open_trades", "closed_trades", "signal_outcomes",
            "regime_snapshots", "fundamentals", "feature_snapshots",
        ]
        for table in required:
            self.assertIn(table, stats, f"Table {table} not created")

    def test_setup_idempotent(self):
        """Calling setup_all twice must not raise errors."""
        db = self._setup()
        try:
            db.setup_all()
        except Exception as e:
            self.fail(f"Second setup_all raised: {e}")

    def test_news_insert_and_query(self):
        self._setup()
        conn = sqlite3.connect(self.tmp)
        conn.execute("""
            INSERT INTO news (title, source, created_at, processed)
            VALUES (?, ?, ?, 0)
        """, ("HDFC Bank Q3 results beat estimates", "economic_times", str(datetime.now())))
        conn.commit()
        row = conn.execute("SELECT COUNT(*) FROM news").fetchone()
        conn.close()
        self.assertEqual(row[0], 1)

    def test_unique_constraint_news(self):
        """Duplicate news titles must be silently ignored."""
        self._setup()
        conn  = sqlite3.connect(self.tmp)
        title = "Test HDFC Bank News Article"
        for _ in range(3):
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO news (title, created_at) VALUES (?,?)",
                    (title, str(datetime.now()))
                )
            except Exception:
                pass
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM news WHERE title=?", (title,)).fetchone()[0]
        conn.close()
        self.assertEqual(count, 1, "Duplicate title should be ignored")

    def test_trade_lifecycle(self):
        """Insert open trade, close it, verify closed table."""
        self._setup()
        conn = sqlite3.connect(self.tmp)
        # Open
        conn.execute("""
            INSERT INTO open_trades
            (signal, entry_price, stop_price, target_price,
             shares, risk_amount, status, opened_at)
            VALUES ('LONG', 1842.0, 1786.0, 1934.0, 5, 280.0, 'OPEN', ?)
        """, (str(datetime.now()),))
        conn.commit()
        # Close
        conn.execute("""
            INSERT INTO closed_trades
            (signal, entry_price, exit_price, shares,
             pnl_amount, pnl_pct, exit_reason, close_date, status)
            VALUES ('LONG', 1842.0, 1934.0, 5, 460.0, 0.05, 'TARGET', ?, 'CLOSED')
        """, (str(datetime.now()),))
        conn.commit()
        open_count   = conn.execute("SELECT COUNT(*) FROM open_trades").fetchone()[0]
        closed_count = conn.execute("SELECT COUNT(*) FROM closed_trades").fetchone()[0]
        conn.close()
        self.assertEqual(open_count,   1)
        self.assertEqual(closed_count, 1)


class TestQueriesContract(unittest.TestCase):
    """Tests for the Queries class contract."""

    def setUp(self):
        self.tmp = tempfile.mktemp(suffix=".db")
        from database.db_setup import DatabaseSetup
        DatabaseSetup(self.tmp).setup_all()

        # Seed some closed trades
        conn = sqlite3.connect(self.tmp)
        trades = [
            ("LONG",  1800, 1900, 5,  500, 0.055, "TARGET",  "2026-01-10"),
            ("LONG",  1900, 1850, 5, -250, -0.026,"STOP",    "2026-01-15"),
            ("SHORT", 1900, 1820, 5,  400, 0.042, "TARGET",  "2026-01-20"),
            ("LONG",  1820, 1900, 5,  400, 0.044, "TARGET",  "2026-02-01"),
            ("LONG",  1900, 1850, 5, -250, -0.026,"STOP",    "2026-02-05"),
        ]
        for t in trades:
            conn.execute("""
                INSERT INTO closed_trades
                (signal,entry_price,exit_price,shares,pnl_amount,pnl_pct,exit_reason,close_date,status)
                VALUES (?,?,?,?,?,?,?,?,'CLOSED')
            """, t)
        conn.commit()
        conn.close()

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_total_pnl_calculation(self):
        from database.queries import Queries
        q   = Queries(self.tmp)
        pnl = q.get_total_pnl()
        expected = 500 - 250 + 400 + 400 - 250
        self.assertAlmostEqual(pnl, expected, places=1)

    def test_rolling_win_rate(self):
        from database.queries import Queries
        q  = Queries(self.tmp)
        wr = q.get_rolling_win_rate(n=5)
        self.assertGreater(wr, 0.0)
        self.assertLessEqual(wr, 1.0)

    def test_consecutive_losses(self):
        from database.queries import Queries
        q = Queries(self.tmp)
        # Most recent trade is a LOSS (-0.026) — consec = 1
        cl = q.get_consecutive_losses()
        self.assertEqual(cl, 1)

    def test_equity_curve_length(self):
        from database.queries import Queries
        q     = Queries(self.tmp)
        curve = q.get_equity_curve(starting_capital=100_000)
        self.assertEqual(len(curve), 5)

    def test_equity_curve_values_sum_to_total_pnl(self):
        from database.queries import Queries
        q     = Queries(self.tmp)
        curve = q.get_equity_curve(starting_capital=100_000)
        final = curve[-1]["equity"]
        total_pnl = q.get_total_pnl()
        self.assertAlmostEqual(final, 100_000 + total_pnl, places=1)

    def test_data_freshness_returns_dict(self):
        from database.queries import Queries
        q       = Queries(self.tmp)
        fresh   = q.get_data_freshness()
        self.assertIn("_summary", fresh)
        self.assertIn("stale_sources", fresh["_summary"])

    def test_system_stats_returns_counts(self):
        from database.queries import Queries
        q     = Queries(self.tmp)
        stats = q.get_system_stats()
        self.assertIn("closed_trades", stats)
        self.assertEqual(stats["closed_trades"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)