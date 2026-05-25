# QUANT EDGE — Autonomous Trading Intelligence System

## Project Identity
- **Universe:** Top-5 private banks — HDFC, ICICI, Kotak, Axis, IndusInd
- **Primary stock:** HDFC Bank (HDFCBANK.NS · NSE India)
- **Language:** Python 3.13 · **DB:** SQLite (`database/trading.db`)
- **Signals:** Telegram Bot (@PrinterPennyBot, Chat ID `7873846599`) · **Hosting:** Railway.app
- **Phase:** Paper-trading — collecting outcome data before real capital

---

## Architecture & 6-Gate Pipeline (per-category, since 2026-05-22)
```
data/ → processing/ → features/ → models/ → signals/ → risk/ → alerts/
```
Pre-Gate-1 safety: stale price (≤0) or invalid ATR (≤0) → FLAT.

| Gate | Name | Scope | Logic |
|------|------|-------|-------|
| 1 | Regime | global | CHOPPY → block all. Stability <25% → block. Direction never vetoes (`signal_direction=None`). |
| 2 | Rule Filter | global | 11 checks, need 8+. Hard fails: earnings <3d, VIX ≥28, anomaly HIGH. Check 11 (RBI day) soft-fail. |
| 3 | Universe Rank | global | Ranks 5 banks; size_mult by rank. Disqualifies only on falling-knife score (<-5). |
| 4 | ML Models | global | Passes iff ≥1 model non-FLAT. Alignment (A+/A/B/C/F) is INFORMATIONAL — boosts confidence and size, never vetoes. |
| 5 | S/R Validator | **per-category** | Entry quality A/B/C/D for each model's direction (SHORT inverts R:R). Grade-D override: conf ≥62%. |
| 6 | Confidence | **per-category** | Swing 60% / positional 55% / intraday 65% (FULL). VIX adjustment +5/+10pp. |

**Per-category emission:** any category that clears Gates 4+5+6 emits its own signal, Telegram alert, and paper trade. Counter-regime trades (LONG in BEAR / SHORT in BULL) allowed at 0.5× size.

---

## ML Models — Latest CV Scores (retrained 2026-05-23 with DII data)
| Bank | Swing CV | Positional CV | Intraday CV |
|------|----------|---------------|-------------|
| HDFCBANK | 0.595 | 0.641 | 0.764 |
| ICICIBANK | 0.499 | 0.556 | 0.796 |
| KOTAKBANK | 0.518 | 0.556 | 0.719 |
| AXISBANK | 0.454 | 0.472 | 0.716 |
| INDUSINDBK | 0.462 | 0.492 | 0.657 |

Each bank has 3 variants on disk: technical (25yr) · full (2020+) · prod (25yr + verified features). Gate 4 priority: prod → full → technical. All models calibrated (isotonic or Platt scaling if <150 cal samples).

**Scale capital when:** Win rate >52%, Profit factor >1.5, Sharpe >1.0, Max DD <20%, 50+ trades.

---

## Capital Modes & Risk
| Mode | Capital | Risk/Trade | Swing thr | Positional thr | Intraday thr |
|------|---------|------------|-----------|----------------|--------------|
| SMALL | <₹50K | 1% | 68% | 65% | 72% |
| GROWING | ₹50K–₹2L | 1.5% | 60% | 60% | 68% |
| FULL | >₹2L | 2% | 60% | 55% | 65% |

Alignment veto removed (per-category mode, 2026-05-22). Threshold is the conviction gate.
VIX >20: +5pp · VIX >25: +10pp · VIX ≥28: HALT_AND_FLATTEN.
Exit: 1R→book 50% | 2R→book 30% | trail 20% chandelier stop (3×ATR).
DD multipliers: ≤−2%→1.0× | −4%→0.75× | −6%→0.50× | <−6%→0.25×.

---

## Architecture Fixes Applied (cumulative)

### Signal Pipeline
- **Per-category pipeline** (2026-05-22): 3 independent paths (swing/positional/intraday) through Gates 4–6. Any clearing all 6 gates emits independently.
- **Gate 1**: stability block raised 0.40→0.25; LOW_STAB_MULT=0.65 for 25–40% range; direction veto dropped.
- **Gate 4**: alignment="F" no longer blanket blocks. Degeneracy threshold 0.95→0.92 routes collapsed isotonic output to technical fallback.
- **Gate 5**: SHORT entry quality now uses inverted R:R (`sup_dist/res_dist`). GRADE_D_OVERRIDE_CONF 0.75→0.62. Per-category mode drops alignment requirement in D-override.
- **Gate 6**: `skip_alignment=True` in per-category mode. GROWING swing threshold lowered 0.63→0.60.
- **Pre-Gate-1 safety**: stale price (≤0) or invalid ATR (≤0) → FLAT.
- **ATR fix** (2026-05-22): `feature_builder` now pulls `atr_14` directly from raw `tech_df` instead of the named-feature dict (which never included it), fixing wildly wrong stops across all non-HDFC banks.
- **F-alignment size veto** (2026-05-22): was returning `size_mult=0.0` hardcoded, emitting zero-share alerts. Now reads `ALIGNMENT_SIZE["F"] = 0.50` from lookup table. Defensive guard in `signal_engine` refuses signals with `shares=0 / stop=0 / target=0`.

### Exit Engine & Orchestrator (2026-05-25) — Critical Production Bugs Fixed

**Bug 1 — Cross-ticker price contamination (KOTAKBANK exiting at ₹392)**
- **Root cause**: `ExitEngine.check_all_positions(price)` had no ticker filter. During the exit monitor loop, INDUSINDBK's price (~₹392) was passed to KOTAKBANK's positions (actual price ~₹1800). The KOTAKBANK position's highest_price watermark was ₹1301 (correct), but the 1.5% trailing stop ₹1281 was compared against ₹392 → immediate false exit.
- **Fix** (`risk/exit_engine.py`): Added `ticker: str = None` to `_get_open_positions()` and `check_all_positions()`. When `ticker` is provided, the DB query filters `WHERE ticker=?`.
- **Fix** (`signals/signal_engine.py`): `check_exits()` now passes `ticker=self.ticker` so each engine only evaluates its own ticker's positions. Also guards `price <= 0` → returns `[]` before touching the DB.

**Bug 2 — Exit alerts never closing positions (spam every 60s)**
- **Root cause**: `_run_exit_checks()` in orchestrator called `send_exit()` but never called `close_position()`. Position stayed OPEN in DB → re-triggered on every 60s exit monitor cycle.
- **Fix** (`orchestrator.py`): `_run_exit_checks()` now calls `close_position()` / `close_position_partial()` first, then `send_exit()`. Added `_closed_this_cycle: set` guard so two engines can't double-close the same position_id.

**Bug 3 — Reversal alert spam every 5 minutes (AXIS POSITIONAL REGIME_ADVERSE)**
- **Root cause**: `_sent_reversals` was keyed by `position_id` (int). When a POSITIONAL position closed and immediately reopened (new DB id), the prune-by-active_ids logic deleted the old key and the fresh id bypassed dedup → alert fired every cycle. Additionally, `_dispatch_reversal_alerts` had no category filter, so the 5-min intraday-only cycle was checking POSITIONAL/SWING reversals 12×/hr instead of 4×/hr.
- **Fix** (`orchestrator.py`):
  - `_sent_reversals` now uses `(ticker, category, reversal_type)` tuple key instead of `pos_id`. Once `(AXISBANK.NS, positional, REGIME_CHANGE)` fires, it won't re-fire until that ticker's position actually closes.
  - `_dispatch_reversal_alerts(ticker, signal, categories)` accepts the cycle's `categories` list and skips any reversal whose `cat` is not in scope. 5-min intraday cycles skip POSITIONAL/SWING reversals entirely.
  - `_run_pipeline_cycle` passes `categories` down to `_dispatch_reversal_alerts`.
  - `_run_exit_checks` clears all `_sent_reversals` entries for that ticker on position close (by iterating tuple keys where `key[0] == ticker`).

**Bug 4 — `close_position_partial()` missing ticker in INSERT**
- **Root cause**: `closed_trades` INSERT didn't include `ticker` column, causing silent data loss or schema error.
- **Fix** (`risk/exit_engine.py`): `ticker = pos.get("ticker") or "HDFCBANK.NS"` extracted and included in INSERT.

### Data
- **FII daily**: 1,620 rows loaded from CSVs (2026-05-22).
- **DII data** (2026-05-23): 1,577 rows of MF equity net flows loaded from `dii_cash_daily_data/` into `fii_data.dii_net_cr`. Verified 0% overlap with FII values. 1,525 rows updated; 1,619 total rows now have DII. `dii_3d_norm` and `flow_confluence` features now non-zero.
- **DII API parser** (`fii_fetcher.py`): expanded field-name formats + category detection for MF/AMC/BANK/INSURANCE.
- **GlobalFetcher**: error logging added; 1 row confirmed saving.
- **FinBERT**: continuous `p_positive − p_negative` scoring (was returning 0.0 for ~54% of articles). `process_unscored()` rescores zero-scored recent articles.
- **News cooldown**: reduced 1800→900s to match 15-min scheduler interval.

### Dashboard & API
- **Market status**: fully client-side via `getNseMarketStatus()` — computes IST from UTC offset, checks NSE 2026 holidays. Updates every 1s via `updateClock()`. No server dependency.
- **Price fetch**: variable TTL (15s open / 300s closed). Angel One LTP only during market hours.
- **Non-trading hours**: signal pipeline and data-fetch tasks gated on `schedule.is_trading_day()` in `task_runner.py`. `is_pre_market_analysis_time()` checks `is_trading_day()` first.
- **FII query**: ORDER BY `(dii_net_cr IS NULL) ASC, trade_date DESC` — rows with DII sort first.
- **Macro**: `is not None` guards on all macro fields to prevent 0.0 being suppressed.
- **Model card features**: all 18 feature rows across swing/positional/intraday computed live from chart candles, order_flow, news, fundamentals.
- **Signal logging**: 20-col CSV schema with per-category rows (one row per passing category per cycle).
- **FII/DII panel dynamic** (2026-05-23): FII 5D NET, DII 5D NET, confluence badge, and 5-day sparkline (FII green/red + DII blue/amber bars) all live from API.
- **Portfolio heatmap dynamic** (2026-05-23): open exposure and trade count from `open_trades` DB, with `signal_log.csv` fallback (last row per `(ticker, category)` where `status=SIGNAL`, `shares>0`, `stop>0`). Risk badge (SAFE/WARNING/DANGER) from consecutive losses and month PnL.
- **System Performance dynamic** (2026-05-23): capital mode, open exposure, open trades count, Sharpe ratio, win rate, profit factor all live from DB + signal_log.
- **nan/JSON crash fix** (2026-05-25): `OrderFlowAnalyzer` returned `numpy.nan`, causing `json.dumps` ValueError → HTTP 500 on all `/api/*` endpoints. Fix: `_clean_json()` recursive sanitizer in `dashboard_api.py` converts nan/inf/numpy types to 0.0 before `JSONResponse`.
- **IST timestamp fix** (2026-05-25): VPS runs UTC. `datetime.now()` was displaying UTC labeled as IST. Fix: added `_now_ist = datetime.now(_pytz.timezone("Asia/Kolkata"))` for display fields only; naive `now` retained for DB queries to avoid `TypeError: can't subtract offset-naive and offset-aware`.
- **API_BASE dynamic prefix** (2026-05-25): Dashboard at `/quant/` path was sending API calls to `http://origin/api/...` (no `/quant` prefix) → 404. Fix: `API_BASE` now derived from `location.pathname` at runtime so it works on both localhost and VPS regardless of path prefix.
- **Paper Trading Ledger modal** (2026-05-25): clicking the Portfolio Risk Heat Map card opens a full-screen modal. Tabs: OPEN POSITIONS (ticker, category badge, direction, entry/current/stop/target, shares, unrealized P&L, stop dist, target dist, R:R remaining, progress bar, days held) | TRADE HISTORY (last 50 closed with reason badge). Summary strip: 7 live stats. Wired to `/api/trades` endpoint. Auto-refreshes every 30s; Escape closes.
- **`/api/trades` endpoint** (2026-05-25): returns open positions enriched with live price from `_quotes_cache` (unrealized P&L, distances, progress %) + last 50 closed trades with friendly reason labels + summary (win rate, profit factor, net P&L). Only queries columns that exist in DB schema — removed `confidence, alignment, regime, entry_quality` from SELECT.

### News
- **RSS feed audit** (2026-05-23): ET topic feeds removed (0 entries), Business Standard removed (403). Added LiveMint (`/rss/markets`, 35 entries) and CNBCTV18 (200 entries) as shared feeds. Google News remains primary (100 entries, ~93 bank-relevant).
- **Auto-fetch from API** (2026-05-23): `_trigger_news_fetch_bg()` in `dashboard_api.py` fires on every `/api/live` call if news is stale >15 min. Non-blocking daemon thread with `threading.Lock` to prevent concurrent fetches. Runs NewsFetcher + FinBERT scoring in background.
- **Clickable news** (2026-05-23): each headline is an `<a>` link opening the source article in a new tab; clicking the card row also opens it.
- **WEAK+/WEAK- bands** (2026-05-23): scores 0.02–0.10 show faint green (WEAK+), −0.02 to −0.10 show faint red (WEAK−) instead of collapsing to NEUTRAL.
- **Recency fix** (2026-05-23): Google News default sorts by *relevance*, returning articles from 2018-2026 mixed together. Added `+when:7d` operator to query — restricts to last 7 days AND sorts by date. `news_fetcher.py:_parse_published()` parses RFC822/ISO 8601 published timestamps; articles older than `MAX_ARTICLE_AGE_DAYS=7` at insert time are rejected. New `published_iso` column + index `idx_news_published`; 11,512 historical rows backfilled from the legacy `published` text.
- **Live signal uses publication date** (2026-05-23): `FinBERTSentiment.get_daily_sentiment()` now filters by `COALESCE(published_iso, created_at)` instead of `created_at`, so an article published 3 days ago but freshly ingested today no longer pollutes the 24h window. Backtest path (`get_daily_sentiment_at`) keeps `created_at <= as_of_date` for lookahead safety.
- **Dashboard news query** (2026-05-23): orders + filters by `COALESCE(published_iso, created_at)` (last 7 days, 20 rows). API returns explicit `scored` boolean so UI distinguishes "unscored" from "actually neutral."
- **Background scorer decoupled** (2026-05-23): `_trigger_news_fetch_bg()` runs FinBERT on any `processed=0` rows even when no new fetch happens — previously, freshly-inserted articles could stay unscored indefinitely when news was within the 15-min freshness window.
- **Tightened NEUTRAL band** (2026-05-23): dashboard JS dead-zone narrowed from ±0.02 to ±0.005, so FinBERT's subtle ±0.01–0.02 lean renders as WEAK+/WEAK− rather than NEUTRAL.

---

## Current State (2026-05-25) ✅
- Per-category pipeline live. All 3 categories evaluated every cycle.
- All 5 banks retrained with FII + DII data (prod variant, 28 features).
- Dashboard: 3 stacked panels (SWING/POSITIONAL/INTRADAY), client-side market badge, live model features.
- Dashboard: FII/DII panel, Portfolio heatmap, System Performance all fully dynamic (wired to DB + signal_log).
- Dashboard: Paper Trading Ledger modal — click Portfolio Heat Map → full open/closed trade view with live P&L.
- News: `+when:7d` date-restricted query, publication-date filter at insert, `published_iso` column, dashboard + signal pipeline both filter by publication date, NEUTRAL band tightened to ±0.005.
- Dedup: per-category (`_last_signal_by_cat`). Paper trades tagged by `trade_type`.
- Exit engine: ticker-scoped position queries (no cross-bank price contamination). Positions closed in DB before Telegram alert.
- Reversal dedup: `(ticker, category, reversal_type)` tuple key — survives position ID churn. 5-min cycle skips POSITIONAL/SWING reversals.
- VPS updated and running (2026-05-25).

---

## What's Still Not Done ❌

### Immediate (Deploy blockers)
0. **Standard deploy procedure** (NEVER delete open_trades blindly):
   ```bash
   ssh root@165.22.220.126
   cd ~/TradingBot && git pull
   # ⚠️  DO NOT run DELETE FROM open_trades — that destroys live paper positions
   # Only close positions if the code that manages them changed fundamentally.
   # To gracefully close all before restart:
   #   sqlite3 database/trading.db "UPDATE open_trades SET status='CLOSED' WHERE status='OPEN';"
   #   sqlite3 database/trading.db "INSERT INTO closed_trades SELECT NULL,signal_uuid,ticker,signal,trade_type,entry_price,entry_price,shares,0,0,'DEPLOY_RESTART',datetime('now'),'CLOSED' FROM open_trades WHERE status='CLOSED' AND close_date IS NULL;"
   sudo systemctl restart quantedge-signal quantedge-api
   ```

### High Priority
1. **5 alpha features** — wire into `feature_builder.py` + `*_FEATURES_PROD` lists, retrain Sunday:
   - `finbert_momentum_3d` = `score_24h − score_72h`
   - `fii_flow_surprise` = z-score of FII flow vs 20d baseline
   - `banknifty_relative_momentum_5d` = `banknifty_5d_pct − nifty_5d_pct`
   - `atr_percentile_252` = ATR rank vs 1y
   - `banks_above_50dma_pct` = breadth of 5-bank universe

### Medium Priority
2. **GlobalFetcher** — 1 row saved; needs 7+ for rolling macro features to be meaningful
3. **Fundamentals table** — wired to weekly scheduler (Sunday) but currently 0 rows
4. **`closed_trades.trade_type` column** — missing; `open_trades` has it. Needed for per-category historical analysis.
5. **News scoring backlog** — ~15 min lag after fetch (acceptable for swing/positional, marginal for intraday)

### Low Priority
6. **Gate 3 dashboard** — still uses live universe data; should read from `gate_results.gate3` for consistency

---

## Running the System
```bash
python3 main.py --mode=train --model=swing   # retrain swing only (~30s)
python3 main.py --mode=train                 # retrain all models
python3 main.py --mode=signal                # test one signal cycle
python3 main.py --ticker=HDFCBANK.NS         # live 24x7 for ONE bank
python3 orchestrator.py                      # live 24x7 for ALL 5 banks (single process)
python3 orchestrator.py --once               # single cycle for all 5 banks then exit
python3 load_dii_and_retrain.py              # load DII CSVs + retrain all 5 banks
uvicorn dashboard_api:app --host 0.0.0.0 --port 8000
```
Multi-bank: prefer `orchestrator.py` — one process, shared FinBERT model (~1.6 GB
vs ~8 GB across 5 main.py processes), shared global data fetchers. Falls back to
5 separate `main.py --ticker=...` processes if you need per-bank isolation.

---

## Key Design Rules (never break these)
1. `TimeSeriesSplit` always — never random split (lookahead bias)
2. Calibrated probabilities in production — raw XGBoost overconfident
3. FLAT is the default — only trade when 5+ independent signals align
4. ATR-based stops/targets/sizing — never fixed rupee stops
5. Win rate >70% in backtest = bug (lookahead or overfit). Target: 55–65%
6. Social sentiment is INVERTED — retail euphoria = contrarian bear signal
7. Retrain only after 8–12 weeks of live outcome data
8. All training getters use `<= as_of_date` — never peek at future data
9. Gate 2 raw pass-throughs (`days_to_rbi`, `india_vix_level`, `usdinr_5d_pct`, `nifty_5d_pct`) must stay as raw integers/floats in `feature_builder.py`

## Deferred
Bank Nifty futures hedging · shadow-mode deployment · Kelly sizing · `scale_pos_weight` balancing · pooled multi-ticker model with ticker embedding · P3 features (LLM earnings score, social contrarian, Google Trends alt data)
