# QUANT EDGE — System Audit & Enhancement Backlog (2026-05-31)

## Context
A-Z review of the live system from a 20-yr trader's lens, during the paper-trading phase.
The ENTRY side (6-gate per-bank/per-category pipeline) is sound and was hardened this week
(single-ownership sizing, regime-correct labels, news, price cache, data-quality, outcome
attribution). The remaining gaps are in RISK, RESULT-REALISM, and EXIT discipline — exactly
where undertested systems blow up. Paper trading is the correct phase to close them BEFORE
risking capital. This file is the prioritized backlog.

## Verdict
Structurally sound on entries; NOT yet safe for real capital. Three classes of gap remain:
(A) portfolio concentration across 5 correlated banks, (B) paper results don't model cost,
(C) the full strategy was never validated out-of-sample. None are crashes — all are the
kind of silent risk a veteran refuses to fund.

## Already solid — do NOT redo
- 6-gate pipeline, regime-aware thresholds, single-ownership sizing (audit SIZE-1).
- Core exits: hard stop, target, 1R/2R partial booking, chandelier trail, intraday 15:15
  square-off, basic time-exit — all wired and called every 60s for all 5 banks.
- Attributable outcome loop: R-multiple, MFE/MAE, per category × regime_match × grade.
- Circuit breaker (monthly DD, mode-aware) + per-name/total exposure caps enforced pre-open.

---

## P0 — Blockers before scaling capital

### P0-1  Concentration / correlation risk  (the single biggest gap)
The universe is 5 private banks (HDFC/ICICI/Kotak/Axis/IndusInd) — correlation ≈ 0.8+ to
Bank Nifty. They are not 5 picks; they are 5 copies of ONE macro bet.
- Evidence: `position_sizer.check_exposure()` caps only per-name 40% and total 80% by
  position *value* — no direction, no sector, no correlation. `risk_features._get_portfolio_heat`
  SUMS independent risks. So all 5 can open LONG in one 15-min cycle → ~80–100% deployed,
  same direction. A Bank Nifty −3% gap hits all 5 stops together → −10–15% in one candle;
  the circuit breaker only halts the NEXT trade, not this cluster.
- Enhancements:
  1. Net-exposure cap across the universe (e.g. max 60% net long or short).
  2. Same-direction cluster cap: max 2 simultaneous same-direction bank positions; 3rd at
     0.5×, 4th blocked. (Mode-scaled: SMALL 1, GROWING 2, FULL 3.)
  3. Correlation-adjusted heat: treat positions as ~0.8 correlated, so effective heat ≈
     Σ risk (no diversification credit) → tightens the existing heat gate honestly.
  4. Bank Nifty intraday cluster guard: if >50% deployed same-direction AND `^NSEBANK`
     moves −2% intraday, halve/halt the cluster (not just pause new entries).
- Files: `risk/position_sizer.py` (check_exposure), `orchestrator._open_paper_trades`,
  `features/risk_features.py` (_get_portfolio_heat), `risk/circuit_breaker.py`. Reuse the
  existing `^NSEBANK` fetch in `processing/intermarket.py` — don't add new yfinance calls.

### P0-2  Cost realism in paper trades
Paper P&L = `(exit−entry)/entry` with ZERO brokerage/STT/slippage (`exit_engine.close_position`
/ `close_position_partial`). Backtest models 0.2% round-trip but paper does not → paper
results overstate edge by ~20–40 bps/trade. A 51% strategy can be a net loser after cost;
right now paper trading cannot tell the difference.
- Enhancement: apply slippage to entry/exit fills and deduct brokerage+STT at close; store
  BOTH gross and net P&L (and net R-multiple) so attribution reflects reality. Centralize the
  cost constants (reuse backtest `TRANSACTION_COST`/`SLIPPAGE` in `backtest/engine.py`).
- Files: `risk/exit_engine.py` (close + partial), shared cost helper, `risk/outcome_tracker.py`
  (record net alongside gross).

### P0-3  Validate the FULL pipeline out-of-sample before scaling  ✅ DONE (2026-06-01)
`backtest/engine.py` runs only the single ML model + a 0.55 threshold — NOT the 6-gate
pipeline (no regime/rules/rank/S-R/confidence). `backtest/walk_forward.py` exists but there
is no evidence it was ever run, and there is no held-out period. CV accuracy (swing ~0.46,
intraday ~0.67 on 3 classes) is NOT strategy edge.
- Enhancement: make the 6-gate `signal_engine` callable offline over history; run walk-forward
  on a held-out 12-month window WITH costs; report WR / PF / Sharpe / MaxDD per category and
  per regime; gate capital scaling on it (target WR>52, PF>1.5, Sharpe>1, MaxDD<20, 50+ trades).
- Files: `backtest/engine.py`, `backtest/walk_forward.py`, `signals/signal_engine.py`.
- SHIPPED: `backtest/pipeline_backtest.py` (`PipelineBacktest`) replays the price-derivable
  gates over history using the SAME live gate code — Gate 1 (`classify_regime_row` + REGIME_RULES:
  CHOPPY block, counter-regime/HIGH_VOL positional hard blocks), Gate 4 (per-fold technical
  model, no lookahead), Gate 5 (`Gate5SRValidator` on real S/R levels), Gate 6 (`Gate6Confidence`
  with the live category thresholds + regime-aware boosts), regime-aware `max_hold_days`, and
  ~0.2% round-trip costs. Trades are tagged `regime_at_entry` for per-regime slicing.
  `WalkForwardValidator(use_pipeline=True)` + new `validate_holdout()` reserve the last
  `holdout_months` as a strict OOS window (train-on-past / test-on-tail). CLI:
  `python3 -m backtest.walk_forward --ticker HDFCBANK.NS --holdout-months 12` → per-category &
  per-regime WR/PF/Sharpe/MaxDD + the scale-up go/no-go, written to
  `models/evaluation/pipeline_validation.json`.
- OFFLINE LIMITATION (documented in the module): Gate 2 (external VIX/macro/earnings/fundamentals
  snapshots — not stored historically) and Gate 3 (needs all 5 banks per bar) default to neutral
  offline; no historical VIX → 0pp VIX boost. Gate 2 only ever BLOCKS live, so the offline trade
  count is a slight UPPER bound — it never admits a trade live would reject on price grounds.

---

## P1 — Exit & regime discipline

### P1-1  Regime-aware max_hold_days is dead config
`exit_engine` uses hardcoded `MAX_HOLD {swing:7, intraday:1, positional:28}` and ignores the
regime `max_hold_days` (BULL 21 / BEAR 4 / HIGH_VOL 2) that Gate 1 computes. A BEAR short
meant to close in 4 days runs 7 — straight into the mean-reversion bounce.
- Enhancement: persist `max_hold_days` on `open_trades` at entry; enforce it at close.
- Files: `risk/exit_engine.py` (time-exit), `database/db_setup.py` (open_trades column),
  `orchestrator._open_paper_trades` (pass through; value already in `g1_ctx`).

### P1-2  Auto-exit on regime-flip / model-reversal
`check_model_reversals` only sends advisory Telegram alerts; a LONG caught by a confirmed
BEAR flip is not closed. Wire it to flatten (or halve) positions whose regime/model turned
against them, with the existing 30-min reversal cooldown to avoid whipsaw.
- Files: `risk/exit_engine.py`, `orchestrator` exit cycle.

### P1-3  Gap-fill exit is dead code
`_check_gap_fill()` is robust but never receives `open_price`, so overnight gaps past the
stop are unguarded until the next 60s tick. Pass the session open into the exit cycle.
- Files: `risk/exit_engine.py` (check_all_positions), `signals/signal_engine.py` (check_exits).

---

## P2 — Edge quality & hygiene
- ✅ DONE (2026-06-01) Weak swing/positional model edge (~0.46–0.48 CV, 3-class). Added
  `Gate6Confidence.PROVISIONAL_EDGE_PREMIUM = {swing:+3pp, positional:+5pp, intraday:0}` —
  an additive Gate-6 threshold premium on the weak categories until attribution proves edge.
  Applies live AND in the P0-3 offline pipeline (same Gate 6 code), folded into the +15pp cap,
  surfaced in the gate dict (`edge_premium`) and fail-reason. Set to `{}` / zero a category once
  its attribution clears the scale-up bar. After 8–12 wks outcome data + the 5 planned alpha
  features, retrain and try `scale_pos_weight` for class imbalance.
- ✅ DONE G6-5: threshold-boost reason string now shows the magnitude + cause split
  (`+x% thr-adj`, `+x% edge-premium`, `+x% VIX`) instead of falsely labelling everything "DQ".
- ✅ DONE 7 stale unit tests updated (`tests/test_risk.py`, `test_features.py`, `test_signals.py`):
  capital-mode tests neutralise `FORCE_CAPITAL_MODE` (setUp/tearDown) to test the detection bands;
  CB halt test uses -16% (FULL halt=15%); Gate 2 keys `days_to_earnings`/`vix_halt`; Gate 6 CB
  test exercises `circuit_breaker_level`. Whole suite green.
- ⏳ OPEN (VPS-side, not codeable locally) Confirm FII / options / fundamentals tables actually
  populate on the VPS (many features read 0 locally → inflates data-quality miss% toward the POOR
  cliff). Verify via `sqlite3 .../trading.db` row counts after a few live cycles.

---

## Verification (per item)
- P0-1: unit test — 5 same-direction signals in one cycle cap at the net/cluster limit;
  simulate `^NSEBANK` −2% with cluster open → guard fires.
- P0-2: assert `closed_trades.net_pnl < gross_pnl` by exactly the modeled cost; reconcile a
  sample trade against backtest cost.
- P0-3: walk-forward over holdout produces a metrics JSON; compare its WR/PF to live paper.
- P1-1: BEAR position past day 4 auto-closes; P1-3: gap-down open triggers gap-fill exit.

## Sequencing & guardrail
P0-1 → P0-2 → P0-3 (make it SAFE and make results TRUE before more trading), then P1, then P2.
Keep `FORCE_CAPITAL_MODE=FULL` paper override until P0 is done. Do NOT move real capital until
P0-3 holdout passes the scale-up bar AND P0-1/P0-2 ship — otherwise you scale a concentrated,
cost-blind, unvalidated book.
