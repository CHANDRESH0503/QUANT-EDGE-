# data/historical/fii_excel_import.py
# Import historical FII/DII data from downloaded files.
#
# SUPPORTED FILE FORMATS (auto-detected):
#   1. NSE FII/DII daily report  — .xlsx/.xls, daily rows, buy+sell+net for FII & DII
#      Source: https://www.nseindia.com/reports/fii-dii  (preferred — daily granularity)
#
#   2. NSDL FPI monthly report   — .xls served as HTML, monthly rows, net only
#      Source: https://www.fpi.nsdl.co.in/web/Reports/ReportYear.aspx?RptType=6
#      NOTE: monthly granularity (12 rows/year) — less useful for daily ML features.
#            Use NSE daily files for better model accuracy.
#
# NAMING: fii_2020.xls, fii_2021.xlsx, ... (year in filename is used for NSDL date parsing)
#
# Run: python -m data.historical.fii_excel_import
#      python -m data.historical.fii_excel_import --dry-run   (preview only)
#
# Idempotent — re-running skips dates already in fii_data (INSERT OR REPLACE).

import argparse
import calendar
import logging
import re
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DIR = "data/historical/fii_excel"
DB_PATH     = "database/trading.db"

# NSE daily format — column name aliases across years
DATE_COLS     = ["Date", "DATE", "date", "Trade Date", "TRADE DATE"]
FII_BUY_COLS  = ["FII Buy", "FII_BUY", "FPI Buy", "FPI_BUY",
                 "Buy(FII/FPI)", "Buy Value(FII/FPI)", "FII BUY", "FPI BUY"]
FII_SELL_COLS = ["FII Sell", "FII_SELL", "FPI Sell", "FPI_SELL",
                 "Sell(FII/FPI)", "Sell Value(FII/FPI)", "FII SELL", "FPI SELL"]
DII_BUY_COLS  = ["DII Buy", "DII_BUY", "Buy(DII)", "Buy Value(DII)", "DII BUY"]
DII_SELL_COLS = ["DII Sell", "DII_SELL", "Sell(DII)", "Sell Value(DII)", "DII SELL"]

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3,    "april": 4,
    "may": 5,     "june": 6,     "july": 7,      "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _require_openpyxl() -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise SystemExit("openpyxl required: pip install openpyxl")


def _is_html(path: str) -> bool:
    """Check if a file is actually HTML despite its .xls/.xlsx extension."""
    with open(path, "rb") as f:
        head = f.read(512).lstrip()
    return head.startswith(b"<") or b"<html" in head.lower() or b"<table" in head.lower()


def _find_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    col_lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand.lower().strip() in col_lower:
            return col_lower[cand.lower().strip()]
    return None


def _parse_float(val) -> float:
    if val is None:
        return 0.0
    try:
        return float(str(val).replace(",", "").strip() or 0)
    except (ValueError, TypeError):
        return 0.0


def _parse_date(val) -> Optional[str]:
    """Convert Excel date cell to YYYY-MM-DD string."""
    if val is None:
        return None
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d")
    s = str(val).strip()
    if not s:
        return None
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _year_from_path(path: str) -> Optional[int]:
    """Extract year from filename, e.g. fii_2020.xls → 2020."""
    m = re.search(r"(20\d{2})", Path(path).name)
    return int(m.group(1)) if m else None


# ─────────────────────────────────────────────────────────────
# NSDL MONTHLY HTML PARSER
# ─────────────────────────────────────────────────────────────

def _read_nsdl_html(path: str) -> List[Dict]:
    """
    Parse NSDL FPI monthly HTML-as-XLS.
    Columns: Calendar Year | Equity | Debt | Debt-VRR | Hybrid | Total (INR Crores)
    Each row = one month. Date is set to the last day of that month.
    Only fii_net_cr is populated (equity net). dii values = 0 (NSDL has no DII data).
    """
    import pandas as pd

    year = _year_from_path(path)

    try:
        tables = pd.read_html(path, header=None)
    except Exception as e:
        logger.error(f"{path}: pd.read_html failed — {e}")
        return []

    if not tables:
        logger.error(f"{path}: no HTML tables found")
        return []

    # Pick the table with the most data rows
    df = max(tables, key=len)

    rows = []
    for _, row in df.iterrows():
        # First cell is the month name (or a header/total/footnote — skip those)
        first = str(row.iloc[0]).strip().lower()
        month_num = MONTH_MAP.get(first)
        if month_num is None:
            continue

        # Infer year from filename; if missing, try to parse from HTML header
        if year is None:
            logger.warning(f"{path}: cannot determine year — include year in filename, e.g. fii_2020.xls")
            continue

        # Last day of the month
        last_day = calendar.monthrange(year, month_num)[1]
        trade_date = date(year, month_num, last_day).strftime("%Y-%m-%d")

        # Equity column (index 1) is the FPI equity net investment
        equity_net = _parse_float(row.iloc[1] if len(row) > 1 else None)

        rows.append({
            "trade_date":  trade_date,
            "fii_buy_cr":  round(equity_net, 2) if equity_net >= 0 else 0.0,
            "fii_sell_cr": round(abs(equity_net), 2) if equity_net < 0 else 0.0,
            "fii_net_cr":  round(equity_net, 2),
            "dii_buy_cr":  0.0,
            "dii_sell_cr": 0.0,
            "dii_net_cr":  0.0,
        })

    if rows:
        logger.warning(
            f"{path}: NSDL monthly format detected — {len(rows)} monthly rows imported. "
            "For daily granularity (better for ML), download from: "
            "https://www.nseindia.com/reports/fii-dii"
        )

    return rows


# ─────────────────────────────────────────────────────────────
# NSE DAILY EXCEL PARSER
# ─────────────────────────────────────────────────────────────

def _read_nse_excel(path: str) -> List[Dict]:
    """Parse NSE FII/DII daily Excel file (true binary .xlsx or .xls)."""
    import pandas as pd

    try:
        try:
            df = pd.read_excel(path, engine="openpyxl")
        except Exception:
            df = pd.read_excel(path)
    except Exception as e:
        logger.error(f"Cannot read {path}: {e}")
        return []

    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)

    date_col     = _find_col(cols, DATE_COLS)
    fii_buy_col  = _find_col(cols, FII_BUY_COLS)
    fii_sell_col = _find_col(cols, FII_SELL_COLS)
    dii_buy_col  = _find_col(cols, DII_BUY_COLS)
    dii_sell_col = _find_col(cols, DII_SELL_COLS)

    if not date_col:
        logger.error(f"{path}: date column not found. Columns: {cols}")
        return []

    if not fii_buy_col or not fii_sell_col:
        fii_net_col = _find_col(cols, ["FII Net", "FPI Net", "FII NET", "FPI NET", "Net(FII/FPI)"])
        dii_net_col = _find_col(cols, ["DII Net", "DII NET", "Net(DII)"])
        if not fii_net_col:
            logger.error(f"{path}: no usable FII columns. Columns: {cols}")
            return []
        # net-only path
        rows = []
        for _, row in df.iterrows():
            iso = _parse_date(row.get(date_col))
            if not iso:
                continue
            net = _parse_float(row.get(fii_net_col))
            dnet = _parse_float(row.get(dii_net_col)) if dii_net_col else 0.0
            if net == 0 and dnet == 0:
                continue
            rows.append({
                "trade_date":  iso,
                "fii_buy_cr":  round(net, 2) if net >= 0 else 0.0,
                "fii_sell_cr": round(abs(net), 2) if net < 0 else 0.0,
                "fii_net_cr":  round(net, 2),
                "dii_buy_cr":  round(dnet, 2) if dnet >= 0 else 0.0,
                "dii_sell_cr": round(abs(dnet), 2) if dnet < 0 else 0.0,
                "dii_net_cr":  round(dnet, 2),
            })
        return rows

    rows = []
    for _, row in df.iterrows():
        iso = _parse_date(row.get(date_col))
        if not iso:
            continue
        fii_buy  = _parse_float(row.get(fii_buy_col))
        fii_sell = _parse_float(row.get(fii_sell_col))
        dii_buy  = _parse_float(row.get(dii_buy_col)) if dii_buy_col else 0.0
        dii_sell = _parse_float(row.get(dii_sell_col)) if dii_sell_col else 0.0

        if fii_buy == 0 and fii_sell == 0 and dii_buy == 0 and dii_sell == 0:
            continue

        rows.append({
            "trade_date":  iso,
            "fii_buy_cr":  round(fii_buy,  2),
            "fii_sell_cr": round(fii_sell, 2),
            "fii_net_cr":  round(fii_buy - fii_sell, 2),
            "dii_buy_cr":  round(dii_buy,  2),
            "dii_sell_cr": round(dii_sell, 2),
            "dii_net_cr":  round(dii_buy - dii_sell, 2),
        })
    return rows


# ─────────────────────────────────────────────────────────────
# DISPATCHER
# ─────────────────────────────────────────────────────────────

def _read_file(path: str) -> List[Dict]:
    """Auto-detect file format and parse accordingly."""
    if _is_html(path):
        logger.info(f"  {Path(path).name}: HTML-as-XLS detected → NSDL monthly parser")
        return _read_nsdl_html(path)
    logger.info(f"  {Path(path).name}: binary Excel → NSE daily parser")
    return _read_nse_excel(path)


# ─────────────────────────────────────────────────────────────
# DB WRITE
# ─────────────────────────────────────────────────────────────

def _save(conn: sqlite3.Connection, rows: List[Dict]) -> int:
    written = 0
    for r in rows:
        conn.execute("""
            INSERT OR REPLACE INTO fii_data
            (trade_date, fii_buy_cr, fii_sell_cr, fii_net_cr,
             dii_buy_cr, dii_sell_cr, dii_net_cr, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            r["trade_date"],
            r["fii_buy_cr"], r["fii_sell_cr"], r["fii_net_cr"],
            r["dii_buy_cr"], r["dii_sell_cr"], r["dii_net_cr"],
        ))
        written += 1
    return written


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────

def run_import(
    excel_dir: str = DEFAULT_DIR,
    db_path:   str = DB_PATH,
    dry_run:   bool = False,
) -> int:
    """Import all .xlsx/.xls files in excel_dir into fii_data. Returns rows written."""
    _require_openpyxl()

    excel_path = Path(excel_dir)
    if not excel_path.exists():
        logger.error(f"Directory not found: {excel_dir}")
        return 0

    files = sorted(excel_path.glob("*.xlsx")) + sorted(excel_path.glob("*.xls"))
    if not files:
        logger.error(f"No .xlsx or .xls files found in {excel_dir}")
        return 0

    logger.info(f"Found {len(files)} file(s) in {excel_dir}")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    total = 0

    for f in files:
        logger.info(f"Reading {f.name} ...")
        rows = _read_file(str(f))

        if not rows:
            logger.warning(f"  {f.name}: 0 rows parsed — check column names or format")
            continue

        logger.info(
            f"  {f.name}: {len(rows)} rows "
            f"({rows[0]['trade_date']} → {rows[-1]['trade_date']})"
        )

        if dry_run:
            logger.info(f"  DRY RUN — skipping DB write")
            total += len(rows)
            continue

        n = _save(conn, rows)
        conn.commit()
        total += n
        logger.info(f"  {f.name}: {n} rows written")

    conn.close()

    if dry_run:
        logger.info(f"DRY RUN complete — {total} rows would be written")
    else:
        conn2 = sqlite3.connect(db_path)
        distinct   = conn2.execute("SELECT COUNT(DISTINCT fii_net_cr) FROM fii_data").fetchone()[0]
        total_rows = conn2.execute("SELECT COUNT(*) FROM fii_data").fetchone()[0]
        conn2.close()
        logger.info(
            f"Import complete — {total} rows written. "
            f"DB: {total_rows} total rows, {distinct} distinct fii_net_cr values. "
            f"{'✅' if distinct > 50 else '⚠️  Low diversity — check files'}"
        )

    return total


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Import FII/DII historical data from Excel/HTML files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sources (in order of preference):
  1. NSE daily (best):  https://www.nseindia.com/reports/fii-dii
     Set date range per year → Search → Download CSV/Excel
     Save as: fii_2020.xlsx, fii_2021.xlsx, ...

  2. NSDL monthly (fallback):  https://www.fpi.nsdl.co.in/web/Reports/ReportYear.aspx?RptType=6
     Select year → Download → Save as: fii_2020.xls, ...
     Note: monthly granularity only (12 rows/year)
        """,
    )
    p.add_argument("--dir",     default=DEFAULT_DIR)
    p.add_argument("--db",      default=DB_PATH)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    run_import(args.dir, args.db, args.dry_run)


if __name__ == "__main__":
    _cli()
