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

### Regime Feature Consistency Fix (2026-05-25) — Counter-regime SHORT in BULL — Fixed

**Root cause:** `feature_builder.py` computed `regime_bull/bear/choppy` features using only `_adx > 22` threshold. After IndusInd's recovery from crash (April 2026: ADX briefly hit 53 then decayed back to 20 as consolidation set in), the EMA20/50 spread was +1.7% (bullish) and R20=+2.8% (positive), but ADX=20.2 < 22 → `regime_bull=0`, `regime_choppy=1` fed to ML model. The model predicts SHORT at **69.7%** with regime_choppy=1 (above 65% Gate 6 threshold → EMITTED) but SHORT at only **56.4%** with regime_bull=1 (below 65% → blocked). Gate 1 correctly identified BULL_TRENDING via EMA override, but the ML model was receiving stale CHOPPY features — a hidden Gate 1/Gate 4 inconsistency.

**Fix (`features/feature_builder.py`):** Added EMA override path mirroring `regime_detector.py`:
```python
_ema_bull_override = (_esp >  0.015 and _r20 > 0)
_ema_bear_override = (_esp < -0.015 and _r20 < 0)
_rule_bull = int((_adx > 22 and _esp >  0.01 and _r20 > 0) or _ema_bull_override)
_rule_bear = int((_adx > 22 and _esp < -0.01 and _r20 < 0) or _ema_bear_override)
```

**Effect on each bank:**
- **INDUSINDBK**: choppy→bull. Intraday SHORT confidence 69.7%→56.4% → blocked by Gate 6 ✅
- **HDFCBANK**: choppy→bear (EMA spread -3.08% confirms). Now consistent with Gate 1 BEAR_TRENDING ✅
- **ICICIBANK/KOTAKBANK/AXISBANK**: unchanged (ADX already ≥22 or spread within override range) ✅

**Design rule added**: `regime_bull/bear/choppy` features in `feature_builder.py` MUST stay in sync with the EMA override logic in `regime_detector.py` — same threshold (±1.5% spread + confirming returns).

### Regime Gate (2026-05-25) — INDUSINDBK always CHOPPY_SIDEWAYS — Fixed

**Root cause (3 stacked bugs):**
1. **Single shared HMM** (`regime_model.pkl`) trained on HDFCBANK's 25yr data — applied to all banks. IndusInd's post-crash return/volatility distribution is out-of-distribution for the HDFCBANK scaler → HMM consistently mapped IndusInd to CHOPPY state.
2. **`regime_snapshots` had no `ticker` column** — all 5 banks wrote to the same table. `get_recent_regime_trend()` mixed all banks' snapshots → inflated change counts → further suppressed stability.
3. **EMA override threshold ±2% too wide** — IndusInd in post-crash consolidation had EMA20/50 spread of +1.7%, inside the ±2% band → CHOPPY rescue never fired.

**Fixes:**
- **Per-ticker regime models** (`models/saved/regime_model_INDUSINDBK.pkl`, etc.): `RegimeDetector` and `RegimeModelTrainer` both accept `ticker` param. `_model_path(ticker)` returns per-bank path; falls back to legacy shared model until each bank is retrained.
- **`ticker` column added to `regime_snapshots`** with idempotent `ALTER TABLE` migration. `get_recent_regime_trend()` now filters `WHERE ticker=?`.
- **EMA override threshold ±2% → ±1.5%**: IndusInd's +1.7% spread now triggers BULL_TRENDING rescue correctly.
- **200-day EMA safety net added**: if price is ≥10% below 200d EMA → override CHOPPY→BEAR_TRENDING. Catches post-crash banks still deeply underwater even if EMA20/50 spread is tight.
- **`hv10` scalar bug fixed** in `train_regime.py _rule_based`: was missing `.iloc[-1]`, passing a Series to `float()`.
- **Wired throughout**: `train_all.py` passes `ticker` to `RegimeModelTrainer`. `feature_builder.py` and `signal_engine.py` pass `ticker` to `RegimeDetector`.
- **`/api/live` regime query** now ticker-scoped: `WHERE ticker=? OR ticker IS NULL`.
- **Verified result**: IndusInd now correctly reads `BULL_TRENDING 65%` (recovering from crash, price ₹923, EMA spread +1.7%). Was permanently CHOPPY before.

### Dashboard & API Mismatch Fix (2026-05-25)

**Root cause:** `signal_log.csv` fallback was used in both `/api/live` (for `open_trades_count`) and `/api/trades` (for modal rows) when `open_trades` DB was empty. Local machine (empty DB) fell back to 3 old CSV signal entries → showed 3. VPS (1 real DB row) used actual DB → showed 1. Permanent environment-dependent mismatch.

**Fix (`dashboard_api.py`):**
- `/api/live`: `open_trades_count` and `open_exposure` now use `open_trades` DB **only**. `signal_log.csv` is still read — for `capital_mode` (a string, not a count) — but never used as a position fallback.
- `/api/trades`: signal_log fallback block removed entirely. Modal always reflects true `open_trades` DB state.
- `tradingDashboard.html`: LOG badge and `opacity:0.75` removed — positions are always DB-sourced.
- **Result**: Local shows 0 positions (accurate — no orchestrator running locally), VPS shows its actual DB count. Both environments consistent within themselves.

### Crash / Stability Fixes (2026-05-25)

**Uvicorn segfault on macOS / Python 3.13+ (`loky` semaphore leak):**
- **Root cause**: `tokenizers` Rust extension (pulled in by `transformers`) spawned a loky thread pool. POSIX semaphore was leaked at process exit → `zsh: segmentation fault uvicorn ...`.
- **Fix**: Set `TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` **before any import** in `dashboard_api.py` and `orchestrator.py`. In `processing/finbert_sentiment.py`: same env vars inside the `try:` block + `torch.set_num_threads(1)` / `torch.set_num_interop_threads(1)` before `pipeline()` call.

**Deploy — safe position close utility:**
- Added `graceful_close_for_deploy.py`: interactive script that closes all open positions into `closed_trades` with `exit_reason='DEPLOY_RESTART'` before service restart. Use instead of `DELETE FROM open_trades` which silently destroys trade history.

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
- **Portfolio heatmap dynamic** (2026-05-23): open exposure and trade count from `open_trades` DB only (signal_log fallback removed 2026-05-25). Risk badge (SAFE/WARNING/DANGER) from consecutive losses and month PnL.
- **System Performance dynamic** (2026-05-23): capital mode, open exposure, open trades count, Sharpe ratio, win rate, profit factor all live from DB + signal_log.
- **nan/JSON crash fix** (2026-05-25): `OrderFlowAnalyzer` returned `numpy.nan`, causing `json.dumps` ValueError → HTTP 500 on all `/api/*` endpoints. Fix: `_clean_json()` recursive sanitizer in `dashboard_api.py` converts nan/inf/numpy types to 0.0 before `JSONResponse`.
- **IST timestamp fix** (2026-05-25): VPS runs UTC. `datetime.now()` was displaying UTC labeled as IST. Fix: added `_now_ist = datetime.now(_pytz.timezone("Asia/Kolkata"))` for display fields only; naive `now` retained for DB queries to avoid `TypeError: can't subtract offset-naive and offset-aware`.
- **API_BASE dynamic prefix** (2026-05-25): Dashboard at `/quant/` path was sending API calls to `http://origin/api/...` (no `/quant` prefix) → 404. Fix: `API_BASE` now derived from `location.pathname` at runtime so it works on both localhost and VPS regardless of path prefix.
- **Paper Trading Ledger modal** (2026-05-25): clicking the Portfolio Risk Heat Map card opens a full-screen modal. Tabs: OPEN POSITIONS (ticker, category badge, direction, entry/current/stop/target, shares, unrealized P&L, stop dist, target dist, R:R remaining, progress bar, days held) | TRADE HISTORY (last 50 closed with reason badge). Summary strip: 7 live stats. Wired to `/api/trades` endpoint. Auto-refreshes every 30s; Escape closes.
- **`/api/trades` endpoint** (2026-05-25): returns open positions enriched with live price from `_quotes_cache` (unrealized P&L, distances, progress %) + last 50 closed trades with friendly reason labels + summary (win rate, profit factor, net P&L). Only queries columns that exist in DB schema.
- **Mismatch fix** (2026-05-25): `/api/trades` and `/api/live` open_trades_count both use DB only — no signal_log fallback. Local always shows 0 (correct), VPS shows real DB state. No more local≠VPS confusion.

### News
- **RSS feed audit** (2026-05-23): ET topic feeds removed (0 entries), Business Standard removed (403). Added LiveMint (`/rss/markets`, 35 entries) and CNBCTV18 (200 entries) as shared feeds. Google News remains primary (100 entries, ~93 bank-relevant).
- **Browser User-Agent for VPS** (2026-05-25): `feedparser.parse(url)` sent no UA — Google News and Indian news sites blocked VPS IPs (DigitalOcean ranges). Added `_fetch_feed_bytes(url)` helper that fetches with Chrome-like UA + Accept headers; feedparser parses the raw bytes. Detailed per-feed logging: 0-entry feeds now log HTTP status + bozo_exception for easy diagnosis.
- **ET Banking RSS re-added** (2026-05-25): `_ET_BANKING_RSS` shared feed (banking sector) + per-bank ET topic RSS added back to `_build_feeds()`. Works from VPS when browser UA is sent. Serves as Google News fallback when VPS IP is blocked.
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
- All 5 banks pass Gate 1 (regime) — IndusInd unblocked (was permanent CHOPPY).
- **Per-bank regime HMMs trained and saved** (`regime_model_{BANK}.pkl` for all 5 banks).
- **regime_bull/bear/choppy feature consistency fix**: `feature_builder.py` now mirrors the EMA override logic from `regime_detector.py` — prevents counter-regime ML predictions sneaking through when ADX is transitionally low (e.g. IndusInd BULL regime but old logic set regime_choppy=1 → SHORT at 69.7% wrongly passing Gate 6; now regime_bull=1 → 56.4% → correctly blocked).
- Dashboard: 3 stacked panels (SWING/POSITIONAL/INTRADAY), client-side market badge, live model features.
- Dashboard: FII/DII panel, Portfolio heatmap, System Performance all fully dynamic (DB only, no signal_log fallback).
- Dashboard: Paper Trading Ledger modal — click Portfolio Heat Map → full open/closed trade view with live P&L.
- Dashboard: local vs VPS mismatch eliminated — both show true DB state.
- News: browser UA fetch on VPS, ET banking RSS backup, `+when:7d` date-restricted query, `published_iso` column, NEUTRAL band ±0.005.
- Dedup: per-category (`_last_signal_by_cat`). Paper trades tagged by `trade_type`.
- Exit engine: ticker-scoped position queries (no cross-bank price contamination). Positions closed in DB before Telegram alert.
- Reversal dedup: `(ticker, category, reversal_type)` tuple key — survives position ID churn. 5-min cycle skips POSITIONAL/SWING reversals.
- Regime: per-ticker HMM model files (`regime_model_{BANK}.pkl`). `regime_snapshots` ticker-scoped. EMA override ±1.5% + 200d EMA safety net.
- DB schema verified: `open_trades` (14 cols), `closed_trades` (13 cols, incl. `trade_type`), `regime_snapshots` (9 cols, incl. `ticker`) — all correct.
- uvicorn segfault fixed: `TOKENIZERS_PARALLELISM=false` + `torch.set_num_threads(1)` in dashboard_api, orchestrator, finbert_sentiment.
- `graceful_close_for_deploy.py` added — safe position close before deploy.
- VPS running. Deploy: `git pull && systemctl restart quantedge-signal quantedge-api`.

---

## What's Still Not Done ❌

### Standard Deploy Procedure (NEVER delete open_trades blindly)
```bash
ssh root@165.22.220.126
cd ~/TradingBot
# Option A — safe deploy (positions survive restart):
git pull && sudo systemctl restart quantedge-signal quantedge-api

# Option B — need to close positions first (schema change etc.):
python3 graceful_close_for_deploy.py   # interactive, writes closed_trades record
git pull && sudo systemctl restart quantedge-signal quantedge-api
```

### High Priority
1. **5 alpha features** — wire into `feature_builder.py` + `*_FEATURES_PROD` lists, retrain Sunday:
   - `finbert_momentum_3d` = `score_24h − score_72h`
   - `fii_flow_surprise` = z-score of FII flow vs 20d baseline
   - `banknifty_relative_momentum_5d` = `banknifty_5d_pct − nifty_5d_pct`
   - `atr_percentile_252` = ATR rank vs 1y
   - `banks_above_50dma_pct` = breadth of 5-bank universe

### Medium Priority
3. **GlobalFetcher** — 1 row saved; needs 7+ for rolling macro features to be meaningful
4. **Fundamentals table** — wired to weekly scheduler (Sunday) but currently 0 rows
5. **News on VPS** — browser UA fix deployed but Google News may still block DigitalOcean IPs. Monitor logs: `journalctl -u quantedge-signal | grep "0 entries"`. ET banking RSS serves as fallback.

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
python3 graceful_close_for_deploy.py         # safely close open positions before deploy
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
10. `open_trades` DB is the ONLY source of truth for open position counts — never fall back to signal_log for UI counts

## Deferred
Bank Nifty futures hedging · shadow-mode deployment · Kelly sizing · `scale_pos_weight` balancing · pooled multi-ticker model with ticker embedding · P3 features (LLM earnings score, social contrarian, Google Trends alt data)
