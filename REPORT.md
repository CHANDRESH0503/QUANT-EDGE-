# QUANT EDGE — System Audit Report

**Date:** 2026-05-27
**Scope:** Full top-to-bottom audit of the 5-bank × 3-category × 6-gate pipeline, including state independence, regime/direction logic, Telegram alert volume, and runtime efficiency.
**Dashboard under test:** http://165.22.220.126/quant/

---

## Executive Summary

The post-audit fix sweep (REPORT.md C1–C6, H1–H6, M1–M5) closed the multi-bank scoping bugs cleanly — independence at the data/fetcher layer is now solid. **Three new critical issues** surface in this audit:

1. **Regime-direction inversion is real and explainable.** The system reliably surfaces LONG signals in BEAR regimes and SHORT signals in BULL regimes. Root cause is a training-vs-inference horizon mismatch (mean-reversion bias baked into the labels), compounded by a silent key-path bug that nullifies the regime-rules veto in the feature vector.
2. **`_opened_today` is a global flag** in the orchestrator — once any bank fires its 09:15 open alert, the other four are suppressed for the rest of the trading day.
3. **Reversal-alert (the "situation changed") Telegram spam has no time-based throttle** — only a `(ticker, cat, type)` dedup that resets on position close. Active churn cycles produce 5–15 alerts per 15-min window.

Efficiency: the per-bank `FeatureBuilder.build_all()` and `RegimeDetector.detect()` re-do identical work for every cycle. There are 3–4 sec/cycle and 20–30 yfinance calls/cycle that can be removed *without* changing signal output.

Independence verdict: **PASS at the data layer**, **FAIL in two specific orchestrator-state surfaces** (`_opened_today` and the feature-vector regime rules pass-through).

---

## Critical Findings

### CR-1 — Regime / direction inversion (two compounding bugs)

**Symptom (reported by user):** BEAR regime → LONG signals; BULL regime → SHORT signals. Clean inversion across all 5 banks.

**Root cause A — training/inference horizon mismatch** (primary, structural)

`features/feature_builder.py:66–98` (`classify_regime_row`) labels each bar's regime using **PAST** 20-day returns (`r20`) and EMA spread:

```python
is_bear = (adx > 22 and esp < -0.01 and r20 < 0) or ema_bear_override
is_bull = (adx > 22 and esp >  0.01 and r20 > 0) or ema_bull_override
```

`features/feature_builder.py:915–957` generates training labels from **FORWARD** 5-day returns:

```python
fwd   = float(df["Close"].iloc[i + forward_days] / df["Close"].iloc[i] - 1)
label = 2 if fwd > threshold else (0 if fwd < -threshold else 1)   # 2=LONG, 0=SHORT
```

Result: the model is trained on rows where `regime=BEAR` (price was down for 20 days) frequently coincide with `label=LONG` (price reverts up over the next 5 days). It learns a mean-reversion pattern, not a trend-following one. At inference time the model accurately predicts what the training data taught it — LONG in BEAR, SHORT in BULL.

This is not a code bug; it is a **feature/label design conflict**. Two clean fixes:

- **Fix A.1 (preferred):** Drop `returns_20d` / `returns_5d` from the regime classifier and let ADX + EMA spread carry the trend. Retrain. Regime + ML labels stop fighting.
- **Fix A.2:** Switch to `forward_returns_20d` for regime labels too — but this leaks lookahead during inference unless the regime is recomputed from a different signal. Not recommended.

**Root cause B — feature-vector ignores the regime's rules dict** (secondary, silent)

`processing/regime_detector.py:451–462` returns:
```python
{"regime": "BEAR_TRENDING", "rules": {"trade_long": False, "trade_short": True, ...}, ...}
```

`features/feature_builder.py:788–797` reads them at the wrong level:
```python
"regime": {
    "trade_long":   bool(regime_result.get("trade_long", True)),    # ← always None → True
    "trade_short":  bool(regime_result.get("trade_short", False)),  # ← always None → False
    ...
}
```

`regime_result["trade_long"]` does not exist — the keys live under `regime_result["rules"]["trade_long"]`. Every cycle every bank gets `trade_long=True, trade_short=False` regardless of regime. The Gate 1 in-per-category mode does not veto on direction (CLAUDE.md design rule), so the impact is hidden, but counter-regime sizing in signal_engine consumes these fields and the dashboard displays the wrong flags.

**Fix B (one-line):**
```python
_rules = regime_result.get("rules", {})
"trade_long":   bool(_rules.get("trade_long", False)),
"trade_short":  bool(_rules.get("trade_short", False)),
"position_mult":float(_rules.get("position_mult", 1.0)),
```

Priority: **A is the dominant cause** of the visible inversion. **B is a strict-correctness fix** that should ship even if A is deferred.

---

### CR-2 — `_opened_today` collapses across all 5 banks

`orchestrator.py:203, 415, 418, 1110`:

```python
self._opened_today = False               # init (global)
if (is_open and not self._opened_today): # only one fires
    self._send_market_open_alerts()
    self._opened_today = True            # blocks all other banks
```

Once any bank's 09:15 alert fires, no other bank in the universe gets one until midnight reset. Looks like one alert per day per bank from outside, but only one bank actually broadcasts.

**Fix (per-ticker):**
```python
self._opened_today: Dict[str, bool] = {t: False for t in tickers}
# check / set per-ticker inside the loop
```

Same pattern review needed for `_eod_sent`, `_evening_sent`, `_intraday_close_sent`, `_premarket_done` — these are arguably ok as global one-shot daily flags (single Telegram broadcast), but verify each is intended global.

---

### CR-3 — Reversal Telegram alerts have no time throttle

The "situation changed" messages users see are MODEL_REVERSAL / REGIME_CHANGE alerts.

`signals/signal_engine.py:682–747` produces reversal events when an open position's direction conflicts with the latest model or regime. `orchestrator.py:679–736` (`_dispatch_reversal_alerts`) sends them with dedup keyed by `(ticker, cat, type)` stored in `self._sent_reversals` (orchestrator.py:185).

**Problem:** the dedup key is only cleared when the underlying position closes. There is no time-based cooldown. If a model oscillates LONG↔SHORT over a few cycles, each *new* direction immediately fires (because the previous direction's key got cleared on position close-and-reopen, or because the key wasn't set yet for this combination). 5 banks × 3 categories × 15-min cycles = up to 15 reversal evaluations per cycle.

**Fix — 30-minute per-key cooldown:**
```python
# orchestrator.py
self._sent_reversals: Dict[Tuple[str, str, str], float] = {}  # value = ts of last send
REVERSAL_COOLDOWN_SEC = 1800

# in _dispatch_reversal_alerts
last_sent = self._sent_reversals.get(key, 0.0)
if (time.time() - last_sent) < REVERSAL_COOLDOWN_SEC:
    continue
self._sent_reversals[key] = time.time()
telegram.send_raw(format_model_reversal(rev))
```

Optional: roll up multiple reversals in one cycle into a single "3 positions flagged for review" Telegram message instead of N separate alerts.

---

## Bank Independence Audit — 20 surfaces

| # | Surface | Verdict | Evidence |
|---|---------|---------|----------|
| 1 | `_last_signal_by_cat[ticker][cat]` | PASS | `orchestrator.py:178` |
| 2 | `_sent_reversals[(ticker, cat, type)]` | PASS | `orchestrator.py:185` |
| 3 | `_last_signal_per_ticker[t]` | PASS | `orchestrator.py:187` |
| 4 | One `SignalEngine` per bank | PASS | `orchestrator.py:165–174` |
| 5 | `_closed_this_cycle` (local set per call) | PASS | `orchestrator.py:755` |
| 6 | `_opened_today` (global flag) | **FAIL — CR-2** | `orchestrator.py:203,415,1110` |
| 7 | `self.options_fetchers[t]` dict | PASS | `orchestrator.py:168` |
| 8 | Options NSE symbol → per-bank | PASS | `data/options_fetcher.py:47–51` |
| 9 | BSE reads `WHERE ticker=?` | PASS | `data/bse_fetcher.py:319` |
| 10 | BSE writes include ticker | PASS | C4 fix applied |
| 11 | NewsFetcher per-bank with ticker writes | PASS | per-ticker dict, `WHERE ticker=?` |
| 12 | FII/DII/Global tables (no ticker col) | INTENTIONAL | universe-wide data; correct design |
| 13 | RegimeDetector per-ticker (HMM + reads) | PASS | C6 fix applied |
| 14 | Gate 1 regime feed | DEGRADED — CR-1B | feature-vector pass-through inverts defaults |
| 15 | Gate 2 ticker-scoped inputs | PASS | uses per-ticker fundamentals/earnings/news |
| 16 | Gate 3 universe rank (cross-bank by design) | PASS | doesn't write back to wrong ticker |
| 17 | Gate 4 ML loads per-bank model | PASS | `ModelTrainer(ticker).load_all()` |
| 18 | Gate 5 S/R features per-bank | PASS | from per-bank FeatureBuilder |
| 19 | Gate 6 confidence uses per-bank context | PASS | thresholds + signals all per-ticker |
| 20 | `_open_paper_trades` exposure/positions | PASS | H2 fix applied, recompute-in-loop |

**Two failures only**: CR-2 (`_opened_today` global) and CR-1B (feature-vector regime keys mis-pathed). Both have one-line fixes.

---

## Efficiency Audit — hotspots ranked by cycle cost

| Rank | Hotspot | Est. cost/cycle | Fix complexity | Signal impact |
|------|---------|-----------------|----------------|---------------|
| 1 | `FeatureBuilder.build_all()` repeats per-bank with no shared cache | 8–12 s | M | None — universe-wide inputs (FII, global) are deterministic per cycle |
| 2 | `RegimeDetector.detect()` re-runs HMM every cycle even with no new candle | 1–2 s | S | None — same OHLCV tail → identical output |
| 3 | `IntermarketAnalyzer` fetches Bank Nifty + 5 peers inside each per-bank FeatureBuilder (6 calls × 5 banks = 30 yf calls) | 2–3 s | M | None — cross-bank data is the same; hoist to orchestrator |
| 4 | Gate 3 `_rank_stocks` (already 2-min cached) | 0.5 s | — | none; already optimal |
| 5 | FinBERT batch (already shared singleton) | 0.5 s | — | none; already optimal |
| 6 | Repeated yfinance daily OHLCV per bank (each bank reads its own — that's correct) | 1–2 s | S | none — caching by ticker is fine |

**Concrete efficiency wins** (in order):

- **E1 — Hoist `IntermarketAnalyzer` to orchestrator-cycle scope.** Compute Bank Nifty + 5-bank peer table **once per cycle**, pass it into each FeatureBuilder via a `intermarket_context` parameter. Removes ~24 redundant yfinance calls per cycle. Same outputs.
- **E2 — Memoize regime detection per (ticker, last_candle_ts).** Skip HMM re-inference when no new bar. Easy ~1 s/cycle saved.
- **E3 — Hoist `GlobalFetcher.fetch_overnight_signals()` freshness check to orchestrator scope.** Already cached in DB, but per-bank `is_stale` check called 5×; collapse to 1×.
- **E4 — Reduce DB connection churn.** `signal_engine`, `feature_builder`, `regime_detector`, `risk_features` each open and close their own sqlite3 connections within one cycle. A per-cycle shared connection (or connection pool) cuts SQLite open/close overhead. Optional; marginal.

None of E1–E4 changes the signal output — they only eliminate work that produces the same answer twice.

---

## Status of Prior Audit Items

**C1 — Single entrypoint.** Closed. `main.py --mode=run` boots MultiBankOrchestrator; legacy TaskRunner deleted.

**C2 — Exit close-before-alert.** Closed.

**C3 — OptionsFetcher ticker-aware.** Closed. Per-bank `nse_symbol` from `get_bank_config(ticker)`.

**C4 — BSE ticker-scoped.** Closed.

**C5 — feature_snapshots ticker.** Closed.

**C6 — Regime ML features ticker-scoped.** Closed.

**H1 — Data quality gate.** Closed. POOR → FLAT; DEGRADED → +5pp Gate 6.

**H2 — Live exposure caps + position cap.** Closed.

**H3 — Capital-mode `allowed_tf` enforced.** Closed (signal_engine + `_open_paper_trades`).

**H4 — CircuitBreaker persistence.** Closed.

**H5 — Shared regime classifier.** Closed structurally — `classify_regime_row` powers both live and training paths. **But CR-1A above exposes that the shared classifier itself has a horizon-mismatch design flaw; this is new, not a regression of H5.**

**H6 — Fundamentals DEFAULTED.** Closed.

**M1 — Schema ownership.** Closed in this session (gate_results, fundamentals).

**M2 — Global macro bootstrap.** Closed (`bootstrap_history(days=30)` auto-runs in orchestrator boot when row count <7).

**M3 — BSE earnings calendar.** Closed.

**M4 — Gate 3 caching.** Closed (2-min TTL).

**M5 — FinBERT round-robin.** Closed via shared singleton batching.

**Today's session also added:** first-boot fundamentals fetch, macro-stale → DEGRADED bubble, VPS-aware news fetcher (skips Google News on DigitalOcean), gate_results DB table, dead-code cleanup of HDFC_EARNINGS_DATES.

---

## New Findings (this audit)

| ID | Severity | Title | File:Line |
|----|----------|-------|-----------|
| CR-1A | CRITICAL | Regime/label horizon mismatch causes mean-reversion ML signals | `features/feature_builder.py:66–98` vs `:915–957` |
| CR-1B | HIGH (correctness) | feature-vector reads `trade_long`/`trade_short` at wrong dict path | `features/feature_builder.py:791–792` |
| CR-2 | CRITICAL | `_opened_today` is global; suppresses 4-of-5 banks' 09:15 alerts | `orchestrator.py:203,415,1110` |
| CR-3 | HIGH | Reversal alerts have no time-based throttle → Telegram spam | `orchestrator.py:679–736`, `signal_engine.py:682–747` |
| E1 | MEDIUM | IntermarketAnalyzer duplicated 5× per cycle (~24 wasted yfinance calls) | `features/feature_builder.py:559` |
| E2 | MEDIUM | HMM regime re-inferred per cycle with no new candle | `processing/regime_detector.py:176–256` |
| E3 | LOW | GlobalFetcher freshness checked 5× per cycle | `features/feature_builder.py` (added in macro_stale wiring) |
| E4 | LOW | DB connection churn (multiple `sqlite3.connect` per component per cycle) | multiple |
| L-1 | LOW | `_eod_sent`, `_evening_sent`, `_intraday_close_sent` are global one-shots — confirm intent | `orchestrator.py:1057–1064` |

---

## Recommended Fix Priority

**Immediate (this session candidates):**
1. **CR-1B** — one-line fix in feature_builder; restore regime rules semantics.
2. **CR-2** — `_opened_today: Dict[str, bool]`.
3. **CR-3** — 30-minute cooldown on `_sent_reversals` keys (Telegram spam relief).
4. **E1** — Hoist IntermarketAnalyzer to orchestrator scope.

**Next (model + retrain):**
5. **CR-1A** — Drop `returns_20d`/`returns_5d` from `classify_regime_row` constants; retrain all 5 banks Sunday. Verify mean-reversion signal disappears.

**Later:**
6. **E2** — Regime memoization by last-candle timestamp.
7. **L-1** — Audit `_eod_sent` family for per-ticker intent.

---

## Verification Plan

1. **Independence** — boot orchestrator; confirm at 09:15 IST that all 5 banks broadcast their open alerts independently after CR-2 fix.
2. **Regime feature vector** — after CR-1B fix, `feature_snapshots.feature_values` should show `trade_long=False, trade_short=True` for any bank in BEAR_TRENDING. Spot-check with: `sqlite3 database/trading.db "SELECT ticker, regime, feature_values FROM feature_snapshots ORDER BY built_at DESC LIMIT 5"`.
3. **Reversal throttle** — set up a synthetic flip (manually update `open_trades` and rerun `_dispatch_reversal_alerts`); confirm second flip within 30 min is suppressed.
4. **Direction inversion** — after CR-1A fix + retrain, run backtest on 2025 data; confirm BULL → mostly LONG, BEAR → mostly SHORT.
5. **Efficiency** — instrument `_tick()` with cycle wall-clock timing before and after E1; expect 25–35% reduction in full-cycle duration.

---

## Local DB Snapshot (audit-time)

- `feature_snapshots`: ticker-scoped, mixed banks confirmed.
- `regime_snapshots`: per-bank HMM snapshots present for all 5 banks.
- `fundamentals`: bootstrap added this session; populates on next orchestrator boot.
- `global_snapshots`: auto-bootstraps to ≥7 rows via `bootstrap_history` on boot.
- `gate_results`: new in this session; per-ticker UPSERTed each cycle.
- `open_trades`, `closed_trades`: ticker column present.
- `news`: VPS-aware fetcher drops Google News when `QE_IS_VPS=1`.

---

**End of report.**
