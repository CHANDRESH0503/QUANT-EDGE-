# data/nse_session.py
# Shared NSE access hardening.
#
# NSE's strict endpoints — option-chain-equities, corporate-share-holdings-master,
# block-deal — only return data when the session carries cookies minted by first
# visiting the matching HTML page (nsit/nseappid/bm_sv), and those cookies EXPIRE
# in a long-running process. A single homepage prime at __init__ (what the
# fetchers did) is enough for the lenient fiidiiTradeReact endpoint but NOT for
# these — they 401 once the boot-time cookie lapses, which is why options_snapshots
# and insider tables stay empty on the VPS while FII works.
#
# This module primes the correct page and retries ONCE after re-priming when a
# call comes back 401/403 (the auth-cookie-miss signal). It is intentionally
# defensive: any failure returns None and the caller's existing empty-path runs.
# The lenient FII fetcher is deliberately left on its own session — don't fix
# what isn't broken.

import logging
import requests

logger = logging.getLogger(__name__)

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

_HTML_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
_HOME = "https://www.nseindia.com"


def prime_session(session: requests.Session, page_url: str = _HOME) -> None:
    """
    Mint NSE cookies the way a browser would: hit the homepage, then the
    specific data page (its Referer is the homepage). Best-effort — failures
    are swallowed; the subsequent API call will surface the real status.
    """
    hdrs = {**BROWSER_HEADERS, "Accept": _HTML_ACCEPT}
    try:
        session.get(_HOME, headers=hdrs, timeout=10)
        if page_url and page_url != _HOME:
            session.get(page_url, headers={**hdrs, "Referer": _HOME + "/"}, timeout=10)
    except Exception as e:
        logger.debug(f"NSE prime failed ({page_url}): {e}")


def nse_get_json(
    session: requests.Session,
    url: str,
    referer: str,
    page_url: str = _HOME,
    timeout: int = 12,
):
    """
    GET a strict NSE API endpoint, re-priming and retrying ONCE on 401/403.

    Returns parsed JSON (dict or list) on success, or None on any failure
    (non-200 after retry, network error, or non-JSON body). Callers keep their
    existing "empty → skip" handling.
    """
    hdrs = {**BROWSER_HEADERS, "Referer": referer}
    for attempt in (1, 2):
        try:
            resp = session.get(url, headers=hdrs, timeout=timeout)
        except Exception as e:
            logger.debug(f"NSE get error ({url.split('?')[0]}): {e}")
            return None

        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception:
                logger.debug(f"NSE non-JSON body from {url.split('?')[0]}")
                return None

        if attempt == 1 and resp.status_code in (401, 403):
            logger.info(
                f"NSE {resp.status_code} on {url.split('?')[0]} "
                f"— re-priming session and retrying once"
            )
            prime_session(session, page_url=page_url)
            continue

        logger.warning(f"NSE returned {resp.status_code} for {url.split('?')[0]}")
        return None
    return None
