# data/historical/options_backfill.py
# Backfill historical options snapshots for HDFCBANK from NSE F&O bhavcopy archives.
#
# NSE publishes a daily zipped CSV with every traded F&O contract since 2005.
# Format: https://archives.nseindia.com/products/content/derivatives/equities/fo{DDMMMYYYY}bhav.csv.zip
# Note: post Jul-2024 NSE migrated to a new bhavcopy format
# (BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip). This script tries both.
#
# Per-day flow:
#   1. Download zip → extract CSV
#   2. Filter SYMBOL=HDFCBANK and current-month expiry only
#   3. Reconstruct the chain → call OptionsFetcher's existing _calculate_* helpers
#   4. Insert into options_snapshots with data_source='bhavcopy_historical'
#
# IV percentile is not available in bhavcopy. We mark it 50.0 (neutral) and the
# data_source column lets the audit step weight it correctly.
#
# Run: python -m data.historical.options_backfill --start=2020-01-01

import argparse
import csv
import io
import logging
import sqlite3
import time
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/octet-stream",
    "Referer": "https://www.nseindia.com/",
}

# URL priority: historical archive (2005-2023) → modern archive (2024+)
HISTORICAL_URL = "https://archives.nseindia.com/content/historical/DERIVATIVES/{yyyy}/{mmm}/fo{ddmmmyyyy}bhav.csv.zip"
MODERN_URL     = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"
SLEEP_SEC      = 1.0


def _session() -> requests.Session:
    s = requests.Session()
    try:
        s.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
    except Exception:
        pass
    return s


def _fetch_bhavcopy_csv(s: requests.Session, dt: datetime) -> List[Dict]:
    """Try historical archive first (2005-2023), then modern (2024+)."""
    candidates = [
        HISTORICAL_URL.format(
            yyyy=dt.strftime("%Y"),
            mmm=dt.strftime("%b").upper(),
            ddmmmyyyy=dt.strftime("%d%b%Y").upper(),
        ),
        MODERN_URL.format(yyyymmdd=dt.strftime("%Y%m%d")),
    ]
    for url in candidates:
        try:
            resp = s.get(url, headers=NSE_HEADERS, timeout=20)
            if resp.status_code != 200 or len(resp.content) < 1000:
                continue
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                name = zf.namelist()[0]
                with zf.open(name) as fh:
                    text = io.TextIOWrapper(fh, encoding="utf-8")
                    reader = csv.DictReader(text)
                    return list(reader)
        except Exception as e:
            logger.debug(f"bhavcopy {url[-40:]} fetch error: {e}")
    return []


def _is_hdfc_option(row: Dict) -> bool:
    sym = (row.get("SYMBOL") or row.get("TckrSymb") or "").strip().upper()
    inst = (row.get("INSTRUMENT") or row.get("FinInstrmTp") or "").strip().upper()
    return sym == "HDFCBANK" and inst in ("OPTSTK", "STO")


def _pick_nearest_expiry(rows: List[Dict], today: datetime) -> str:
    """Find the current-month expiry (smallest expiry date >= today)."""
    expiries = set()
    for r in rows:
        exp = (r.get("EXPIRY_DT") or r.get("XpryDt") or "").strip()
        if exp:
            expiries.add(exp)

    def _parse(e: str) -> datetime:
        for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(e, fmt)
            except ValueError:
                pass
        return today + timedelta(days=365)

    future = [(_parse(e), e) for e in expiries]
    future.sort(key=lambda x: x[0])
    for dt, e in future:
        if dt >= today:
            return e
    return future[-1][1] if future else ""


def _build_chain_dict(rows: List[Dict], today: datetime) -> Dict:
    """
    Reconstruct a chain-like dict matching the live NSE API shape so
    OptionsFetcher._calculate_* can consume it unchanged.

    Live API shape (records.data is a list of strike-level dicts):
        {"records": {"data": [{"strikePrice": K,
                                "CE": {"openInterest": ..., "impliedVolatility": ...},
                                "PE": {"openInterest": ..., "impliedVolatility": ...}}, ...],
                     "underlyingValue": spot}}
    """
    target_expiry = _pick_nearest_expiry(rows, today)
    if not target_expiry:
        return {}

    strikes: Dict[float, Dict] = {}
    spot = 0.0
    for r in rows:
        exp = (r.get("EXPIRY_DT") or r.get("XpryDt") or "").strip()
        if exp != target_expiry:
            continue
        try:
            k = float(r.get("STRIKE_PR") or r.get("StrkPric") or 0)
        except ValueError:
            continue
        if k <= 0:
            continue

        opt_type = (r.get("OPTION_TYP") or r.get("OptnTp") or "").strip().upper()
        try:
            oi    = float(r.get("OPEN_INT")  or r.get("OpnIntrst")    or 0)
            close = float(r.get("CLOSE")     or r.get("ClsPric")      or 0)
            udl   = float(r.get("SETTLE_PR") or r.get("UndrlygPric")  or 0)
        except ValueError:
            continue

        spot = udl if udl > 0 else spot
        slot = strikes.setdefault(k, {"strikePrice": k, "CE": {}, "PE": {}})
        side = "CE" if opt_type in ("CE", "CALL") else "PE"
        slot[side] = {
            "openInterest":      oi,
            "lastPrice":         close,
            "impliedVolatility": 0.0,   # bhavcopy lacks IV
        }

    return {
        "records": {
            "data": list(strikes.values()),
            "underlyingValue": spot,
        }
    }


def backfill(start: str, end: str, db_path: str = "database/trading.db") -> int:
    """Backfill options_snapshots from NSE F&O bhavcopy zips."""
    from data.options_fetcher import OptionsFetcher

    of = OptionsFetcher(db_path=db_path)

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")

    logger.info(f"Starting options backfill {start} → {end}")
    s = _session()
    conn = sqlite3.connect(db_path)

    cur = start_dt
    total = 0
    attempted = 0
    while cur <= end_dt:
        if cur.weekday() >= 5:
            cur += timedelta(days=1)
            continue

        # Skip if this day already backfilled
        existing = conn.execute("""
            SELECT 1 FROM options_snapshots
            WHERE  data_source = 'bhavcopy_historical'
              AND  date(fetched_at) = ?
            LIMIT  1
        """, (cur.strftime("%Y-%m-%d"),)).fetchone()
        if existing:
            cur += timedelta(days=1)
            continue

        attempted += 1
        if attempted % 10 == 1:
            logger.info(
                f"options backfill: scanning {cur.strftime('%Y-%m-%d')} "
                f"(saved={total} so far)..."
            )

        rows = _fetch_bhavcopy_csv(s, cur)
        if not rows:
            logger.debug(f"no bhavcopy data for {cur.date()}")
            cur += timedelta(days=1)
            time.sleep(SLEEP_SEC)
            continue

        hdfc_rows = [r for r in rows if _is_hdfc_option(r)]
        if not hdfc_rows:
            logger.debug(f"no HDFCBANK rows in bhavcopy for {cur.date()}")
            cur += timedelta(days=1)
            time.sleep(SLEEP_SEC)
            continue

        chain = _build_chain_dict(hdfc_rows, cur)
        if not chain.get("records", {}).get("data"):
            cur += timedelta(days=1)
            time.sleep(SLEEP_SEC)
            continue

        try:
            spot = chain["records"]["underlyingValue"]
            pcr        = of._calculate_pcr(chain)
            max_pain   = of._calculate_max_pain(chain)
            max_call   = of._get_max_call_oi_strike(chain)
            max_put    = of._get_max_put_oi_strike(chain)
            total_ce   = of._sum_oi(chain, "CE")
            total_pe   = of._sum_oi(chain, "PE")

            conn.execute("""
                INSERT INTO options_snapshots
                (pcr, pcr_roc, max_pain, iv_percentile, atm_iv,
                 max_call_oi_strike, max_put_oi_strike,
                 implied_move_pct, total_call_oi, total_put_oi,
                 spot_price, fetched_at, ticker, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pcr, 0.0, max_pain, 50.0, 0.0,
                max_call, max_put,
                0.0, total_ce, total_pe,
                spot, cur.strftime("%Y-%m-%d 15:30:00"),
                "HDFCBANK.NS", "bhavcopy_historical",
            ))
            total += 1
            conn.commit()
            logger.info(
                f"options backfill: saved {cur.strftime('%Y-%m-%d')} "
                f"pcr={pcr:.2f} spot={spot:.0f} (total={total})"
            )
        except Exception as e:
            logger.warning(f"options reconstruction error {cur.date()}: {e}")

        time.sleep(SLEEP_SEC)
        cur += timedelta(days=1)

    conn.commit()
    conn.close()
    logger.info(f"Options backfill complete — {total} snapshots written")
    return total


def _cli() -> None:
    p = argparse.ArgumentParser(description="Backfill HDFCBANK options from NSE bhavcopy")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end",   default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument("--db",    default="database/trading.db")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    backfill(args.start, args.end, args.db)


if __name__ == "__main__":
    _cli()
