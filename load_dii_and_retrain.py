"""
load_dii_and_retrain.py
Loads MF equity net flow CSVs into fii_data.dii_net_cr, then retrains
all 5 banks × 3 models with the updated features.

Run: python3 load_dii_and_retrain.py
"""
import glob
import logging
import sqlite3
from datetime import datetime

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DB_PATH  = "database/trading.db"
DII_DIR  = "dii_cash_daily_data"
TICKERS  = ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS"]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load and parse DII CSVs
# ─────────────────────────────────────────────────────────────────────────────

def load_dii_csvs() -> pd.DataFrame:
    files = sorted(glob.glob(f"{DII_DIR}/*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {DII_DIR}/")

    dfs = [pd.read_csv(f) for f in files]
    df  = pd.concat(dfs, ignore_index=True)

    # Keep only what we need
    df = df[["Date", "MF_Equity_Net_Purchase_Sales"]].copy()
    df.columns = ["raw_date", "dii_net_cr"]

    # Drop NA / non-numeric rows
    df = df.dropna(subset=["dii_net_cr"])
    df["dii_net_cr"] = pd.to_numeric(df["dii_net_cr"], errors="coerce")
    df = df.dropna(subset=["dii_net_cr"])

    # Parse "1 Jan 2020" → "2020-01-01"
    df["trade_date"] = pd.to_datetime(df["raw_date"], format="%d %b %Y").dt.strftime("%Y-%m-%d")
    df = df[["trade_date", "dii_net_cr"]].drop_duplicates(subset="trade_date")

    log.info(f"DII CSVs: {len(files)} files → {len(df)} unique trading days")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Upsert into fii_data
# ─────────────────────────────────────────────────────────────────────────────

def upsert_dii(df: pd.DataFrame) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # Fetch all existing dates for fast lookup
    existing = {
        r[0]: r[1]
        for r in cur.execute("SELECT trade_date, dii_net_cr FROM fii_data").fetchall()
    }

    updated  = 0
    inserted = 0
    skipped  = 0   # dates that already had real DII data from the live API

    for _, row in df.iterrows():
        date     = row["trade_date"]
        dii_net  = float(row["dii_net_cr"])
        now_str  = datetime.now().isoformat()

        if date in existing:
            current_dii = existing[date]
            # Skip if already populated by the live API (non-zero, non-null)
            if current_dii is not None and abs(current_dii) > 0.01:
                skipped += 1
                continue
            cur.execute(
                "UPDATE fii_data SET dii_net_cr = ?, fetched_at = ? WHERE trade_date = ?",
                (dii_net, now_str, date),
            )
            updated += 1
        else:
            # Date missing from fii_data entirely (e.g. day only in DII CSV)
            cur.execute(
                """INSERT INTO fii_data
                   (trade_date, fii_net_cr, dii_net_cr, fetched_at)
                   VALUES (?, 0, ?, ?)""",
                (date, dii_net, now_str),
            )
            inserted += 1

    conn.commit()

    # Verify
    total_dii = conn.execute(
        "SELECT COUNT(*) FROM fii_data WHERE dii_net_cr IS NOT NULL"
    ).fetchone()[0]
    conn.close()

    log.info(
        f"DII upsert complete — updated={updated}, inserted={inserted}, "
        f"skipped(API data)={skipped} | total rows with DII={total_dii}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Retrain all 5 banks × 3 models (prod variant)
# ─────────────────────────────────────────────────────────────────────────────

def retrain_all() -> None:
    from models.train_all import ModelTrainer

    for ticker in TICKERS:
        log.info(f"{'─'*50}")
        log.info(f"Retraining {ticker} ...")
        try:
            trainer = ModelTrainer(ticker=ticker, db_path=DB_PATH)
            summary = trainer.train_all(force=True)
            for model, metrics in summary.get("models", {}).items():
                if not isinstance(metrics, dict):
                    continue
                cv = metrics.get("cv_accuracy_mean", 0)
                log.info(f"  {ticker} {model:12s} CV={cv:.3f}")
        except Exception as e:
            log.error(f"  {ticker} retrain failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("DII Backfill + Full Retrain")
    log.info("=" * 60)

    df = load_dii_csvs()
    upsert_dii(df)
    retrain_all()

    log.info("=" * 60)
    log.info("Done.")
