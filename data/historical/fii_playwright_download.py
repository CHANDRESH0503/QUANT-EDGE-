# data/historical/fii_playwright_download.py
# Downloads NSE FII/DII historical data using a real browser (Playwright).
#
# WHY THIS EXISTS:
#   All NSE FII/DII date-range APIs return today-only data or 503.
#   The data exists behind a JS-rendered page that needs a real browser.
#   NSE also blocks Playwright's bundled Chromium via TLS fingerprint —
#   this script uses real Chrome (channel='chrome') to bypass that.
#
# SETUP (one-time):
#   pip install playwright
#   playwright install chrome       # real Chrome — required to bypass NSE blocking
#   playwright install chromium     # fallback only
#
# RUN:
#   python -m data.historical.fii_playwright_download
#   python -m data.historical.fii_playwright_download --years 2022 2023 2024
#   python -m data.historical.fii_playwright_download --import   # download + import
#   python -m data.historical.fii_playwright_download --manual   # show manual steps

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DOWNLOAD_DIR  = Path("data/historical/fii_excel")
NSE_HOME      = "https://www.nseindia.com"
NSE_REPORTS   = "https://www.nseindia.com/reports/fii-dii"
DEFAULT_YEARS = list(range(2020, 2027))

_STEALTH_JS = """
    Object.defineProperty(navigator, 'webdriver',    { get: () => undefined });
    Object.defineProperty(navigator, 'plugins',       { get: () => [1,2,3,4,5] });
    Object.defineProperty(navigator, 'languages',     { get: () => ['en-US','en'] });
    window.chrome = { runtime: {} };
"""


def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
        return sync_playwright, PWTimeout
    except ImportError:
        print(
            "\n[ERROR] playwright is not installed.\n"
            "Run:\n"
            "  pip install playwright\n"
            "  playwright install chrome\n"
        )
        _print_manual_instructions()
        sys.exit(1)


def _launch_browser(p):
    """Use real Chrome to bypass NSE's TLS fingerprint detection of Chromium."""
    args = [
        "--no-sandbox",
        "--disable-blink-features=AutomationControlled",
        "--disable-dev-shm-usage",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-infobars",
    ]
    try:
        b = p.chromium.launch(channel="chrome", headless=True, args=args)
        logger.info("Browser: real Chrome (channel=chrome)")
        return b
    except Exception as e:
        logger.warning(f"Real Chrome not available ({e})")
        logger.warning("Falling back to Chromium — NSE may block it. Run: playwright install chrome")
        return p.chromium.launch(headless=True, args=args)


# ─────────────────────────────────────────────────────────────
# CORE DOWNLOAD
# ─────────────────────────────────────────────────────────────

def download_years(years: list, download_dir: Path = DOWNLOAD_DIR) -> list:
    """Open a real Chrome browser, navigate NSE FII/DII reports page,
    download a CSV for each year. Returns list of saved file paths."""
    sync_playwright, PWTimeout = _require_playwright()
    download_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []

    with sync_playwright() as p:
        browser = _launch_browser(p)
        ctx = browser.new_context(
            accept_downloads=True,
            viewport={"width": 1366, "height": 768},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="Asia/Kolkata",
        )
        ctx.add_init_script(_STEALTH_JS)
        page = ctx.new_page()

        # ── seed NSE session cookie via homepage ──────────────
        logger.info("Seeding NSE session (homepage visit)...")
        try:
            page.goto(NSE_HOME, wait_until="domcontentloaded", timeout=30_000)
            time.sleep(3)
        except Exception as e:
            logger.warning(f"Homepage load issue (non-fatal): {e}")

        # ── navigate to FII/DII reports page ─────────────────
        logger.info(f"Loading: {NSE_REPORTS}")
        try:
            page.goto(NSE_REPORTS, wait_until="domcontentloaded", timeout=30_000)
            time.sleep(4)
        except Exception as e:
            logger.error(
                f"Could not load NSE reports page: {e}\n"
                "NSE may be blocking this browser. Run with --manual for instructions."
            )
            browser.close()
            return []

        title = page.title()
        logger.info(f"Page: {title!r}")
        if any(kw in title.lower() for kw in ("403", "blocked", "access denied", "error")):
            logger.error("NSE blocked this browser. Run: playwright install chrome  then retry.")
            browser.close()
            return []

        logger.info("Page loaded. Starting downloads...")

        for year in years:
            dest = download_dir / f"fii_{year}.xlsx"
            if dest.exists():
                logger.info(f"  {year}: already exists — skip")
                downloaded.append(dest)
                continue

            path = _download_one_year(page, year, dest, PWTimeout)
            if path:
                downloaded.append(path)
            else:
                logger.warning(f"  {year}: failed — run with --manual to download manually")
            time.sleep(2)

        browser.close()

    return downloaded


def _react_fill(page, selector: str, value: str, timeout: int = 5_000) -> bool:
    """Fill a React-controlled input by typing (triggers React's synthetic onChange)."""
    try:
        el = page.wait_for_selector(selector, timeout=timeout, state="visible")
        if not el:
            return False
        el.click()
        time.sleep(0.2)
        el.triple_click()          # select all existing text
        time.sleep(0.1)
        page.keyboard.type(value, delay=40)
        page.keyboard.press("Tab")
        time.sleep(0.3)
        return True
    except Exception:
        return False


def _download_one_year(page, year: int, dest: Path, PWTimeout) -> Path | None:
    """Fill date range, click Search, click Download CSV, save file."""
    from_val = f"01-Jan-{year}"
    to_val   = f"31-Dec-{year}"
    logger.info(f"  {year}: {from_val} → {to_val}")

    # ── fill date inputs ──────────────────────────────────────
    selector_pairs = [
        ("input[placeholder*='From']",  "input[placeholder*='To']"),
        ("input[placeholder*='from']",  "input[placeholder*='to']"),
        ("input[id*='from' i]",         "input[id*='to' i]"),
        ("input[name*='from' i]",       "input[name*='to' i]"),
    ]

    filled = False
    for from_sel, to_sel in selector_pairs:
        if _react_fill(page, from_sel, from_val) and _react_fill(page, to_sel, to_val):
            filled = True
            break

    # Last resort: find all date-like inputs
    if not filled:
        try:
            all_inputs = page.query_selector_all("input[type='text'], input[type='date']")
            date_inputs = [
                el for el in all_inputs
                if any(
                    kw in (el.get_attribute(attr) or "").lower()
                    for attr in ("placeholder", "id", "name")
                    for kw in ("date", "from", "to", "start", "end")
                )
            ]
            if len(date_inputs) >= 2:
                for el, val in zip(date_inputs[:2], [from_val, to_val]):
                    el.triple_click()
                    page.keyboard.type(val, delay=40)
                    page.keyboard.press("Tab")
                    time.sleep(0.2)
                filled = True
        except Exception:
            pass

    if not filled:
        logger.warning(f"  {year}: date fields not found (page layout may have changed)")
        return None

    # ── click Search ──────────────────────────────────────────
    for btn in ["Search", "Apply", "Go", "Submit", "Get Data", "Filter"]:
        try:
            page.click(f"button:has-text('{btn}')", timeout=3_000)
            time.sleep(3)
            break
        except Exception:
            continue

    # ── download CSV ──────────────────────────────────────────
    download_selectors = [
        "button:has-text('CSV')",
        "a:has-text('CSV')",
        "button:has-text('Download')",
        "a:has-text('Download')",
        "[class*='download' i]",
        "button:has-text('Export')",
        "a[href*='.csv' i]",
    ]

    try:
        with page.expect_download(timeout=20_000) as dl_info:
            for sel in download_selectors:
                try:
                    page.click(sel, timeout=3_000)
                    break
                except Exception:
                    continue
            else:
                logger.warning(f"  {year}: download button not found")
                return None

        dl = dl_info.value
        dl.save_as(str(dest))
        logger.info(f"  {year}: saved → {dest.name}")
        return dest

    except PWTimeout:
        logger.warning(f"  {year}: download timed out")
        return None
    except Exception as e:
        logger.warning(f"  {year}: error — {e}")
        return None


# ─────────────────────────────────────────────────────────────
# MANUAL INSTRUCTIONS FALLBACK
# ─────────────────────────────────────────────────────────────

def _print_manual_instructions():
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MANUAL FII/DII DOWNLOAD (if automation is blocked)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION A — NSE Reports Page (preferred: buy+sell split):
  1. Open: https://www.nseindia.com/reports/fii-dii
  2. Set From: 01-Jan-2020  To: 31-Dec-2020 → Search → Download CSV
  3. Repeat for 2021, 2022, 2023, 2024, 2025, 2026
  4. Save files to: data/historical/fii_excel/
     Names: fii_2020.xlsx, fii_2021.xlsx, ...

OPTION B — NSDL FPI Reports (net values only):
  1. Open: https://www.fpi.nsdl.co.in/web/Reports/ReportYear.aspx?RptType=6
  2. Select Year → Download for 2020–2026
  3. Save to: data/historical/fii_excel/

After downloading, run:
  python3 -m data.historical.fii_excel_import --dry-run
  python3 -m data.historical.fii_excel_import
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="Download NSE FII/DII history via Playwright (real Chrome)",
    )
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--dir",   default=str(DOWNLOAD_DIR))
    parser.add_argument("--import", dest="do_import", action="store_true",
                        help="Run fii_excel_import after downloading")
    parser.add_argument("--manual", action="store_true",
                        help="Print manual download instructions and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    if args.manual:
        _print_manual_instructions()
        return

    dl_dir = Path(args.dir)
    logger.info(f"Years: {args.years}  |  Dir: {dl_dir.resolve()}")

    files = download_years(args.years, dl_dir)
    logger.info(f"Downloaded {len(files)} file(s): {[f.name for f in files]}")

    if not files:
        logger.warning(
            "No files downloaded.\n"
            "  • If Chrome is missing: playwright install chrome\n"
            "  • If NSE is blocking:   run with --manual"
        )
        return

    if args.do_import:
        logger.info("Running fii_excel_import...")
        from data.historical.fii_excel_import import run_import
        total = run_import(str(dl_dir))
        logger.info(f"Import complete — {total} rows written to fii_data")


if __name__ == "__main__":
    _cli()
