# QUANT EDGE — Autonomous Trading Intelligence System

## Project Identity
- **Universe:** Top-5 private banks — HDFC, ICICI, Kotak, Axis, IndusInd (all `.NS`)
- **Primary:** HDFC Bank · **Language:** Python 3.13 · **DB:** SQLite (`database/trading.db`)
- **Signals:** Telegram (@PrinterPennyBot, Chat ID `7873846599`)
- **Hosting:** DigitalOcean VPS at `165.22.220.126` · dashboard at `http://165.22.220.126/quant/`
- **Phase:** Paper-trading — collecting outcome data before real capital

---

## Architecture & 6-Gate Pipeline (per-category, per-bank)
```
data/ → processing/ → features/ → models/ → signals/ → risk/ → alerts/
```
One `SignalEngine` per bank; all 3 categories evaluated each cycle. Pre-Gate-1: stale price (≤0) or invalid ATR (≤0) → FLAT.

| Gate | Name | Scope | Logic |
|------|------|-------|-------|
| 1 | Regime | per-bank | CHOPPY → block. Stability <25% → block. `regime_match` computed BEFORE Gate 5/6. Aligned: normal threshold + full size. Counter-regime positional: HARD BLOCK. Positional+HIGH_VOL: HARD BLOCK (max_hold=2d). Counter-regime swing: +7pp Gate 6. Counter-regime intraday: +10pp Gate 6. HIGH_VOL aligned: +5pp Gate 6. low_stability (25–40%): +5pp Gate 6. regime_changes ≥2: +5pp Gate 6. Counter-regime size: 0.5×. |
| 2 | Rule Filter | per-bank | 11 checks, need 8+. Hard fails: earnings <3d, VIX ≥28, anomaly HIGH. Fundamentals DEFAULTED → soft-pass + bubble `fundamentals_stale`. Regime-aware: BEAR flips checks 5/8/9/10 (POOR fundamentals, hostile macro, global risk-off, rupee stress all CONFIRM short thesis). Stale macro → treated as neutral. |
| 3 | Universe Rank | per-bank | Ranks 5 banks; per-regime cache. BULL: strongest bank gets 1.00× (rank-1). BEAR: score inverted — weakest bank (most negative momentum) gets 1.00× (best short). Disqualifies score <−5: in BULL = falling knife; in BEAR = rising stock (short-squeeze risk). should_switch = ticker_rank > 2. |
| 4 | ML Models | per-bank | Passes iff ≥1 model non-FLAT. Alignment grades: A+(+15pp/1.20×) · A(+10pp/1.0×) · B(+5pp/0.85×) · B−(−5pp/0.75×, 2-vs-1 split) · C(0pp/0.70×) · F(−20pp/0.50×, 1-vs-1 split only). Conf floor 0.0. All grades INFORMATIONAL. |
| 5 | S/R Validator | per-category | Entry quality A/B/C/D per direction (SHORT inverts R:R). Hard block: S/R R:R < 0.5:1. Soft penalty: 0.5–2.0. Grade-D override: positional≥68% / swing≥65% / intraday≥62%. SHORT near_breakout: 0.80× size penalty (buyers aggressive = breakout risk). empty_sr() returns Grade D. |
| 6 | Confidence | per-category | FULL: swing 60% / pos 55% / intra 65%. Boosts: VIX +5/+10pp · DEGRADED features +5pp · positional + DEFAULTED fundamentals +5pp · capped +15pp. Reads `circuit_breaker_level` directly. |

**Data quality gate** (pre-Gate-4): POOR → hard FLAT. DEGRADED → +5pp Gate 6 threshold. `macro_stale=1` from `GlobalFetcher.is_stale(30h)` floor-clamps quality to DEGRADED.

**Capital-mode `allowed_tf` enforced** at signal_engine AND `_open_paper_trades` (SMALL=swing · GROWING=swing+intra · FULL=all). Disallowed categories return structured FLAT with reason for dashboard.

**Counter-regime trades:** positional always blocked. Swing/intraday allowed at 0.5× size with raised Gate 6 threshold (+7pp swing / +10pp intraday). `regime_match` evaluated BEFORE Gate 5/6 so threshold boost is baked in — Gate 6 is no longer regime-blind.
**Counter-regime + Grade D double-jeopardy hard block** — if `regime_match=False` AND `entry_quality=D`, the category is blocked regardless of confidence.
**Positional in HIGH_VOLATILITY hard block** — `max_hold_days=2` in HIGH_VOL is structurally incompatible with a 14–28 day positional hold. Hard blocked in `signal_engine.py` before Gate 5. Swing/intraday fine with +5pp Gate 6 boost.
**BEAR_TRENDING `position_mult` = 1.0** (symmetric with BULL_TRENDING = 1.2). Aligned BEAR SHORTs no longer penalised vs aligned BULL LONGs.
**Gate 6 instability boosts (G1-4):** `low_stability` (25–40% of last 10 bars agree on regime) → +5pp. `regime_changes ≥ 2` in last 10d (HMM oscillating) → +5pp. Both stack with counter-regime and other boosts; capped at +15pp total.

---

## ML Models — Latest CV (prod variant, retrained 2026-05-27 with CR-1A regime fix)
| Bank | Swing | Positional | Intraday |
|------|------|------------|----------|
| HDFCBANK | 0.598 | 0.641 | 0.764 |
| ICICIBANK | 0.497 | 0.551 | 0.797 |
| KOTAKBANK | 0.523 | 0.568 | 0.721 |
| AXISBANK | 0.457 | 0.499 | 0.733 |
| INDUSINDBK | 0.463 | 0.485 | 0.667 |

3 variants per bank: technical (25yr) · full (2020+) · prod (25yr + verified features). Gate 4 priority: prod→full→technical. All calibrated (isotonic, Platt if <150 cal samples). Per-ticker HMM regime models: `models/saved/regime_model_{BANK}.pkl`.

**Scale capital when:** Win >52%, PF >1.5, Sharpe >1.0, Max DD <20%, 50+ trades.

---

## Capital Modes & Risk
| Mode | Capital | Risk/Trade | Max Pos | Allowed TF | Swing/Pos/Intra Thresholds |
|------|---------|------------|---------|------------|----------------------------|
| SMALL | <₹50K | 1% | 2 | swing only | 68%/65%/72% |
| GROWING | ₹50K–₹2L | 1.5% | 3 | swing+intra | 60%/60%/68% |
| FULL | >₹2L | 2% | 5 | all three | 60%/55%/65% |

**Paper-trading override:** `FORCE_CAPITAL_MODE=FULL` in `/etc/quantedge.env` (default since 2026-05-27). Bypasses ₹ auto-detection so all 3 timeframes stay active regardless of configured capital. `CapitalMode.detect()` checks this first; consumed by `risk_features.py` → Gate 6. Remove or set to `""` before going live.

Exposure caps (live-enforced): 40% per-name · 80% total. VIX >20: +5pp · >25: +10pp · ≥28: HALT_AND_FLATTEN.
Exit: 1R→book 50% | 2R→book 30% | trail 20% chandelier (3×ATR).
DD multipliers (mode-aware, shared with `CircuitBreaker.THRESHOLDS`): SMALL halt −8% / GROWING −12% / FULL −15%.

---

## Latest Audit Fixes (2026-06-01)

**P0-3 — Out-of-sample validation of the FULL gate pipeline (the go-live gate).** `backtest/engine.py` only ran the bare ML model + a flat 0.55 threshold — NOT the gates that actually decide a live trade — so CV accuracy was being mistaken for strategy edge. New `backtest/pipeline_backtest.py` (`PipelineBacktest`) replays the **price-derivable** gates over history using the **same live gate code** (no divergence possible): Gate 1 via shared `classify_regime_row` + `REGIME_RULES` (CHOPPY block, counter-regime & HIGH_VOL positional hard blocks), Gate 4 (per-fold technical model, no lookahead), Gate 5 via `Gate5SRValidator` on real `SupportResistanceEngine` levels, Gate 6 via `Gate6Confidence` with the live category thresholds + regime-aware boosts (counter-regime +7/+10pp, HIGH_VOL +5pp), regime-aware `max_hold_days`, and ~0.2% round-trip costs. Every trade is tagged `regime_at_entry` for per-regime slicing. `WalkForwardValidator(use_pipeline=True)` routes folds through it; new `validate_holdout()` reserves the last N months as a strict train-on-past/test-on-tail OOS window. CLI: `python3 -m backtest.walk_forward --holdout-months 12` → per-category **and** per-regime WR/PF/Sharpe/MaxDD + the capital-scaling go/no-go (`scale_ready`), written to `models/evaluation/pipeline_validation.json`. **Offline limitation (documented in the module):** Gate 2 (external VIX/macro/earnings/fundamentals — not stored historically) and Gate 3 (needs all 5 banks/bar) default neutral; no historical VIX → 0pp VIX boost. Gate 2 only ever BLOCKS live, so the offline trade count is a slight UPPER bound — it never admits a trade live would reject on price grounds. **Do not scale capital until this holdout clears WR>52 / PF>1.5 / Sharpe>1 / MaxDD<20 / 50+ trades per category.**

**P2 edge premium — provisional confidence gate on weak-edge categories.** Swing (~0.46–0.60 CV) and positional (~0.49–0.64 CV) have the weakest 3-class model edge; intraday (~0.67–0.80) is strong. `Gate6Confidence.PROVISIONAL_EDGE_PREMIUM = {swing:+3pp, positional:+5pp, intraday:0}` adds an additive threshold premium on the weak categories until the attribution loop proves edge — applied everywhere Gate 6 runs (live `signal_engine` AND the P0-3 offline pipeline, so validation reflects live gating), folded into the same +15pp cap, surfaced in the gate dict (`edge_premium`) and the fail-reason. **Tradeoff:** slows paper-data collection on swing/positional. Set to `{}` or zero a category once its attribution clears the scale-up bar. Effective FULL thresholds now: swing 0.63 / positional 0.60 / intraday 0.65.

## Latest Audit Fixes (2026-05-31)

**SIZE-1 — Position-sizing chain triple-counted risk factors (critical, was silently killing signals).** A-Z audit of the sizing path found `risk_features.final_size_mult` was a mega-composite that re-applied factors the gates and the position sizer already own, compounding them multiplicatively:
- **S/R grade (was G6-4):** Gate 5's `GRADE_SIZE_MAP {A:1.0,B:0.85,C:0.65,D:0.40}` is applied as `g5_ctx.size_mult` in `signal_engine`, AND `risk_features` folded its own `sr_mult {A:1.0,B:0.85,C:0.65,D:0.45}` into `final_size_mult` (also multiplied in). Net: Grade D = 0.40×0.45 = **0.18×** (intended 0.40×), Grade C = 0.42×, Grade B = 0.72×. Grade C/D collapsed to 0–1 shares → tripped the position sanity guard → **discarded AFTER passing all 6 gates**. The system passed signals then sized them to zero — paper trading opened almost nothing.
- **Capital mode:** `position_sizer` applies `max_risk_pct` (1/1.5/2%) — the correct mechanism — but `risk_features` also folded `cap_mult {0.5/0.75/1.0}` in, double-penalising SMALL/GROWING (SMALL effectively risked 0.5%, not 1%). Dormant in FULL paper mode; bites on go-live.
- **Drawdown:** latent — `position_sizer` has its own DD band but `monthly_dd_pct` is never passed, so it's dormant (single-counted via risk_features). Left as-is, documented.

Fix: established **single-ownership** — `final_size_mult` now contains ONLY portfolio-state factors (heat × dd × anomaly × loss × meta); `sr_mult`/`cap_mult` still returned for transparency but removed from the product. Verified: effective S/R sizing now Grade B 0.85× / C 0.65× / D 0.40× (was 0.72/0.42/0.18). New Rule #29.

**G6-1 — VIX threshold now category-aware** (`gate6_confidence.py`). The +5/+10pp VIX penalty was identical for all categories. 20yr rule: elevated VIX is the SOURCE of alpha for intraday (2–3%/session moves) — penalising it like positional blocks valid same-session trades in the best environment; a 4-week positional hold THROUGH that vol is the real risk. Now `(vix>25, vix>20)` boosts: intraday `(0.03, 0.00)`, swing `(0.10, 0.05)`, positional `(0.15, 0.08)`. Verified: at VIX 28/conf 70%, intraday passes, positional doesn't.

**G6-2 — `circuit_breaker_level` was dead code in Gate 6.** `feature_builder`'s `risk_context` dict omitted the key, so `gate6.check()`'s `risk_context.get("circuit_breaker_level","OK")` was always "OK" and CB-halt reasons logged empty (the halt still worked via `trading_allowed=0`). Added `circuit_breaker_level` + `circuit_breaker_reason` (from `risk_feats`) to the risk_context. Verified: CB HALT now blocks with reason "Circuit breaker HALT: monthly loss limit".

**G6-3 — already fixed** (WARN drawdown band at 0.75× exists in `risk_features` lines 101-105). Removed from the to-do list.

**FE-1 — Dashboard static/fake data eliminated (no hardcoded info in frontend).** Several dashboard panels rendered hardcoded mock data as if live — misleading during paper trading and wrong for non-HDFC banks (literal "HDFC" labels regardless of selected ticker). Fixed at both layers: (1) **API** — `dashboard_api.py` hardcoded `psychology = {discipline:80, adherence:88, …}`; now computed by `PsychologyTracker.analyse()`, honestly returning `INSUFFICIENT_DATA` + trade count when <10 closed trades exist (no fake discipline score). (2) **HTML** — `tradingDashboard.html` had fully static panels with no `id`/JS wiring: PSYCHOLOGY bars (82%/91%/…), SYSTEM ALERTS (3 fake alerts incl. "HDFC debt raising plan"), GOOGLE TRENDS ("+18% ▲" etc.), HIRING ("EXPANSION +24%"), INSIDER ("25.8% NET BUY"). Replaced with id-anchored containers + three new render fns (`updatePsychology`/`updateAltData`/`updateSystemAlerts`) wired into `updateDashboard`. Psychology renders real bias flags or honest empty state; Google Trends renders the API's real `alt_data` (from `alt_trends`); System Alerts derive from live state (paper mode, risk badge, BSE/strong-sentiment news); hiring/insider show honest "no data" until a real per-bank source exists (`InsiderFetcher` is HDFC-only, intentionally NOT exposed as per-bank). Added `esc()` for safe innerHTML. Verified: JS node-checked, all 5 banks return 200 with real psychology=INSUFFICIENT_DATA, root serves HTML.

**OUTCOME-1 — Attributable outcome loop (the paper-trading deliverable).** Paper trading exists to measure *which edges are real* before risking capital — but the learning loop was blind end-to-end. `open_position()` accepted `confidence/alignment/regime/entry_quality` then **dropped them at the INSERT**; `closed_trades` had no context columns; `OutcomeTracker._fetch_trade_metadata()` queried columns that didn't exist → every recorded outcome had empty confidence/alignment/regime/grade and no R-multiple. Fix: (a) schema — entry context (`confidence, alignment, regime_at_entry, entry_quality, regime_match, size_mult, atr_at_entry, reward_risk`) persisted on `open_trades`, carried to `closed_trades`; exit analytics (`r_multiple, hold_hours, mfe_pct, mae_pct`) computed at close; `signal_outcomes` gains `category/entry_quality/regime_match/r_multiple/hold_hours/exit_reason/mfe_pct/mae_pct` (all idempotent ALTERs in `_migrate_columns`). (b) `exit_engine.open_position/close_position/close_position_partial` persist + compute the metrics (MFE/MAE from tracked high/low — reveals stop/target misplacement). (c) `OutcomeTracker.record()` writes rich rows from the trade dict directly (no broken re-query). (d) New `attribution()` / `attribution_report()` slice realised win-rate / **profit-factor / expectancy-R** / MFE / MAE by **category × aligned-vs-counter × S/R grade × alignment grade**, wired into the evening Telegram report. This is what tells you e.g. "counter-regime intraday has PF 0.6 → tighten" vs "aligned A+ swing has PF 2.1 → scale". Verified end-to-end: open→excursion→close→outcome→attribution produces correct R-multiples and per-slice PF.

**PERF-1 — Uncached yfinance was the lag AND the silent FLAT-cascade (critical).** `build_all()` re-downloaded ~300 daily bars per bank on **every** cycle — including the 5-min intraday-only cycles, even though daily bars are immutable intraday. 5 banks fetched sequentially → a burst yfinance rate-limits and returns an **empty** frame → `build_all()` aborts to `_empty_result()` (price=0, ATR=0, quality=POOR) → that bank FLATs the whole cycle. Measured cold build = 4.0s/bank; this was the system "lagging" and intermittently "failing". Fix: shared TTL cache + last-good fallback in `data/price_fetcher.py` (`yf_safe_download`). TTL by interval (1d=15min, 1wk=6h, 5m=2min); on an empty fetch, serve the last good frame instead of cascading to FLAT. Verified: warm build = **0.2s/bank (20×)**, last-good fallback returns full history when yfinance returns empty. A 5-bank cycle drops ~20s → ~1s warm.

**DQ-1 — Data-quality counted flags as "missing", risking POOR→hard-FLAT.** `_data_quality()` computed `missing_pct` as the fraction of *all* numeric features equal to 0.0. But ~30 features are binary flags / one-hot regime states / healthy-zero risk fields (`is_monday`, `regime_choppy`, `chart_golden_cross`, `monthly_dd_flag`, `consecutive_loss_halt`…) where 0 is a **valid signal, not missing data**. This inflated miss% by ~6pp, pushing otherwise-healthy builds toward the 50% POOR cliff that hard-FLATs every bank (worse on a fresh VPS with sparse options/FII/fundamentals tables). Fix: `_is_dq_flag()` excludes flag/one-hot/`_flag`-suffix/`is_`/`chart_`/`regime_` features from the denominator. Genuinely-continuous features (price/technical, options, sentiment, flow magnitudes) are still counted, so a real source outage still degrades quality. Verified: miss% 43.9% → 38.0%, remaining zeros are all genuinely-empty sources.

**VPS-1 — News pipeline dead on VPS (gzip never decompressed).** `_fetch_feed_bytes` in `data/news_fetcher.py` advertised `Accept-Encoding: gzip, deflate` but passed the raw compressed response straight to `feedparser.parse()`, which cannot parse compressed bytes and silently returned **0 entries** for every feed. Locally this was masked because Google News replied uncompressed; on the VPS (`QE_IS_VPS=1` disables Google News) it meant *zero* news ingested. Fixed: decompress based on the `Content-Encoding` response header (gzip / deflate, with raw-deflate fallback) before parsing. Verified: Moneycontrol/LiveMint/CNBC now yield 4/35/200 entries; full `fetch_all_banks` cycle inserted real articles.

**VPS-2 — Stale-model deployment gap (root cause of "BEAR_TRENDING → LONG").** `models/saved/*.pkl` is gitignored, so `git pull` on the VPS ships new code against the *old* pre-CR-1A models that learned mean-reversion (LONG in BEAR, SHORT in BULL). The live inference path was verified direction-correct (forward-return labels, shared `classify_regime_row` one-hot, Gate 4 argmax→{0:SHORT,1:FLAT,2:LONG}) — the bug was purely that the corrected 2026-05-27 models never reached the VPS. Fix: (a) new `deploy_models.sh` rsyncs models to the VPS; (b) `orchestrator._audit_models()` runs at boot and alerts (log CRITICAL + Telegram) if any bank's directional models are missing or predate the CR-1A fix epoch. Documented in the deploy section.

## Latest Audit Fixes (2026-05-27)

**CR-1A — Regime/label horizon mismatch (RETRAIN-REQUIRED).** `classify_regime_row` no longer reads `returns_20d`. Regime is now purely ADX + EMA spread. Previously, BEAR was labeled from past-20d-down while ML labels used forward-5d returns — the model learned mean-reversion and surfaced LONGs in BEAR and SHORTs in BULL. Vectorised training path aligned. All 5 banks retrained 2026-05-27.

**CR-1B — Regime rules path bug.** `RegimeDetector.detect()` returns rules under `regime_result["rules"]`, but `feature_builder.py` was reading `regime_result.get("trade_long")` at the top level, silently defaulting `trade_long=True, trade_short=False` regardless of regime. Fixed to read from the nested `rules` dict.

**CR-2 — Per-ticker `_opened_today` flag.** Was a single bool — once HDFCBANK fired the 09:15 alert, ICICI/Kotak/Axis/IndusInd were suppressed for the rest of the day. Now `Dict[str, bool]` keyed by ticker; `_send_market_open_alerts` no longer breaks after first non-FLAT bank.

**CR-3 — Reversal alert 30-min cooldown.** `_sent_reversals` was keyed by `(ticker, cat, type)` with no time floor — model oscillations spammed Telegram. Cooldown via `REVERSAL_COOLDOWN_SEC=1800` (30 min). Position-close still clears entries for that ticker.

**E1 — Intermarket cache.** `_fetch_peer_data` and `_sector_rotation` in `processing/intermarket.py` were called inside every per-bank FeatureBuilder, doing 7 yfinance calls × 5 banks = 35/cycle. Now class-level TTL cache (`_INTERMARKET_CACHE_TTL_SEC=300`) → 7/cycle. Same outputs, ~5× yfinance reduction.

**PT-1 — Paper-trading capital mode.** `FORCE_CAPITAL_MODE` was defined in `config.py` but never consumed — a dead config variable. Now wired into `CapitalMode.detect()` (checked before ₹ auto-detection) and consumed by `risk_features.py`. Default set to `"FULL"` so paper trading runs with all 3 timeframes unlocked regardless of `STARTING_CAPITAL`. `STARTING_CAPITAL` default bumped to ₹5,00,000.

**PT-2 — Gate 5 S/R R:R hard floor.** Previous code only reduced `size_mult` for poor S/R R:R — a signal with R:R 0.2:1 (resistance 5× closer than support) was passing Gate 5 with just a size penalty. Added `RR_HARD_BLOCK = 0.5`: any signal where the S/R geometry means you risk more than 2× to make 1× is now a hard block, not a soft penalty. Soft penalty (0.5× size) preserved for the 0.5–2.0 range.

**PT-3 — Counter-regime + Grade D double-jeopardy block.** Gate 5 Grade D override allowed a LONG-in-BEAR signal at 82% confidence. Two compounding negatives — (1) regime adversarial, (2) entry quality D (far from support, near resistance) — that no confidence level compensates for. `signal_engine.py` now explicitly blocks: if `regime_match=False AND entry_quality=D`, the category is rejected with reason "double jeopardy" before position sizing runs. Counter-regime Grade C or better still allowed at 0.5× size.

**PT-4 — Dashboard capital mode banner.** `tradingDashboard.html` had `● SMALL CAPITAL MODE · ₹10,000` hardcoded (line 865). Replaced with dynamic `id=capitalModeBadge` element; `updateCapitalBadge()` JS function reads `capital_mode` + `is_paper_trading` + `starting_capital` from the API response (exposed at top-level in `dashboard_api.py`). Now shows `🧪 PAPER · FULL MODE · ALL TIMEFRAMES · ₹X.XL` during paper phase; switches to `💰 LIVE` when `QE_PAPER_TRADING=0`.

**PT-5 — Gate 1 docstring corrected.** Docstring claimed "BEAR_TRENDING → SHORT signals only" — completely wrong since "User directive 4B" was implemented (Gate 1 called with `signal_direction=None`, direction check is dead code). Rewritten to document actual behavior: hard blocks only on CHOPPY + low stability; all other regimes apply size multipliers; counter-regime trades penalized 0.5× in `signal_engine.py` not here.

**G1-1 — Gate 6 was regime-blind (root cause of BEAR+LONG issue).** `regime_match` was computed AFTER Gate 6 in the per-category loop — Gate 6 used the same 60% threshold for aligned and counter-regime trades, making size the ONLY penalty for fighting the trend. Moved `regime_match` before Gate 5/6 so threshold boosts take effect before the signal is evaluated. Counter-regime swing: +7pp. Counter-regime intraday: +10pp. HIGH_VOL aligned: +5pp. All stack with existing dq/macro/fundamentals boosts (capped +15pp).

**G1-2 — Positional counter-regime hard block.** 2–4 week positional hold against confirmed BEAR (or SHORT in BULL) was only penalised at 0.4× size before. Hard block added in signal_engine per-category loop before Gate 5 — no confidence threshold compensates for a multi-week hold against the macro regime. Applies to all 5 banks.

**G1-3 — BEAR position_mult asymmetry fixed.** `REGIME_RULES["BEAR_TRENDING"]["position_mult"]` was 0.8, meaning a confirmed-regime SHORT in BEAR got 0.8× size while a confirmed-regime LONG in BULL got 1.2×. Asymmetric and unfair to aligned SHORTs. Changed to 1.0 — symmetric baseline for aligned trades in either trending regime.

**G1-4 — Gate 6 threshold blind to regime stability.** Gate 1 surfaced `low_stability` (stability 25–40%) and `regime_changes` (HMM flips in last 10 days) in its return dict, but `signal_engine.py` only used them for position sizing (already baked into `position_mult`). Gate 6 never received a threshold boost. Fixed: two new variables `regime_low_stability` + `regime_changes_count` extracted from `g1_ctx` before the per-category loop; each adds +5pp to `cat_boost` when true (stacks with other boosts, capped at +15pp). 20-yr rule: a weakly established regime is NOT a clean trend — demand extra confirmation before entering, regardless of direction.

**G1-5 — Positional blocked in HIGH_VOLATILITY.** `REGIME_RULES["HIGH_VOLATILITY"]["max_hold_days"] = 2`. A positional trade holds 14–28 trading days. Entering a 4-week thesis in a regime where the market may gap violently for only 2 days means the stop gets hit before any trend develops. This is a structural incompatibility that no confidence threshold can compensate. Hard block added in `signal_engine.py` per-category loop (before Gate 5) for `cat == "positional" and regime_name == "HIGH_VOLATILITY"`. Swing and intraday in HIGH_VOL are fine with +5pp Gate 6 boost.

**G1-6 — max_hold_days wired end-to-end.** `max_hold_days` was defined in `REGIME_RULES` but never surfaced past `RegimeDetector`. Fixed: `_build_result()` now returns `max_hold_days` in the rules dict; `gate1_regime.check()` reads and returns it in `g1_ctx`; `signal_engine.py` can enforce it for future hold-time expiry logic. BULL: 8 → 21 days (trend is exactly when multi-week positionals are most profitable). BEAR: 3 → 4 days (short-side moves are violent and fast; don't overstay).

**G2-1 — Gate 2 BULL-centric filters blocked valid BEAR short entries.** Checks 5, 8, 9, 10 used identical thresholds for all regimes. In BEAR_TRENDING, hostile macro, global risk-off, rupee stress, and weak fundamentals are all SHORT CONFIRMATIONS — the exact catalysts that cause banks to sell off. All four checks are now regime-aware: (a) Check 5 (POOR fundamentals) → PASS in BEAR with `fundamentals_bear_confirm=True` flag; (b) Check 8 (macro hostile) → PASS in BEAR; fails only if macro is strongly positive (>0.5 = genuine headwind for shorts); (c) Check 9 (intermarket headwinds) → PASS in BEAR; fails only if intermarket very bullish (>0.5); (d) Check 10 (rupee stress) → PASS in BEAR up to 3.5% (vs 2.0% in BULL). Stale macro data (macro_stale=1) now overrides to neutral 0.0 so stale GlobalFetcher data never blocks a live trade. Return dict surfaces five bear-confirmation flags for dashboard explainability.

**G2-2 — Stale macro silently failing Gate 2.** When `GlobalFetcher` data was >30h old, `macro_stale=1` was set in raw_features and Gate 6 added +5pp via DEGRADED quality — but Gate 2 Check 8 still evaluated the stale `macro_score` directly and could fail on an outdated hostile reading. Fixed: `macro_eff = 0.0 if macro_stale_flag else macro_score`. Stale data is treated as neutral at Gate 2; Gate 6 penalty still applies.

**G3-1 — Gate 3 scoring and sizing completely BULL-biased.** Scoring formula `mom_5d × 0.40 + rs_bn × 0.30 + rsi_long_zone × 0.20` always ranked the strongest (most positive momentum) stock #1 and assigned it 1.00× size. In BEAR regime this means: the BEST LONG candidate gets the most capital for a SHORT trade, while the weakest bank (best short candidate) gets rank-5 = 0.40× size. Completely backwards for SHORT trading. Fixed: in BEAR_TRENDING, signs are inverted — `(−mom_5d) × 0.40 + (−rs_bn) × 0.30` so the most-negative-momentum stock ranks #1 (weakest bank = best short gets full size). RSI zone also regime-aware: BULL sweet spot 40–65 (momentum not overbought); BEAR sweet spot ≥65 (overbought = bear-flag) OR ≤35 (confirmed breakdown) = 1.0. Disqualification in BEAR: score < −5 = stock RISING strongly = short-squeeze risk.

**G3-2 — Per-regime cache separation.** Previously `_RANK_CACHE` was a single `(ts, list)` tuple. If HDFC cycled in BULL and ICICI cycled in BEAR, the second bank would incorrectly reuse the first bank's scoring (BULL vs BEAR rankings are completely different). Changed to `Dict[regime: (ts, list)]` so each regime gets its own TTL-controlled cache entry.

**G3-3 — should_switch hardcoded to HDFC's rank.** `should_switch = hdfc_rank > 2` compared HDFC's rank regardless of which bank's engine was running. When evaluating ICICI's engine in BEAR (ICICI = rank-1 best short), `hdfc_rank = 5 > 2` incorrectly fired `should_switch=True` for ICICI. Fixed to `ticker_rank > 2` — each bank's engine evaluates its OWN rank against the top-2 threshold.

**G4-1 — F-grade blocked valid 2-vs-1 majority signals.** Grade F fired for BOTH "2 models agree, 1 dissents" AND "1 LONG vs 1 SHORT" conflicts, applying −20pp to ALL models in both cases. When swing+positional both said LONG but intraday dissented SHORT, F-grade dropped LONG confidence from 0.68 → 0.48 — below every Gate 6 threshold. Missed entry. Fixed by adding **Grade B−** for 2-vs-1 conflicts: conf_boost=−5pp, size=0.75×. The majority direction is mildly penalised and still proceeds. Grade F now reserved for true 1-vs-1 splits (balanced disagreement, no majority). 20yr rule: if two higher-timeframe models agree and one intraday model dissents, trust the timeframe consensus.

**G4-2 — Confidence floor missing.** `min(0.95, conf + boost)` could produce negative confidence values when F-grade (−0.20) hit a low-confidence model (e.g., 0.10 − 0.20 = −0.10). Fixed: `max(0.0, min(0.95, ...))`.

**G4-3 — MIN_CONF dead code.** `MIN_CONF` dict defined in Gate 4 but never consumed there — thresholds are enforced in Gate 6. Marked as DOCUMENTATION ONLY with a comment.

**G5-1 — `_empty_sr()` Grade C inconsistency.** `SupportResistanceEngine._empty_sr()` hardcoded `entry_quality="C"` / `entry_quality_score=0.2`. When sup_dist=res_dist=5%, `_entry_quality(0.05, 0.05, 1, 1)` computes rr=1.0 < 1.5 → Grade D / -0.2. The inconsistency silently promoted no-data entries to a better-than-warranted grade. Fixed: `_empty_sr()` now returns `"D"` / `-0.2`, consistent with the actual formula.

**G5-2 — Grade D override uniform threshold.** `GRADE_D_OVERRIDE_CONF` was a single float (0.62). Grade D = far from support (LONG) or near support (SHORT) — poor entry geometry. A positional trade carries that geometry for 3-4 weeks; intraday is out by end-of-day. Same threshold for both is structurally wrong. Fixed: `GRADE_D_OVERRIDE_CONF` → category-aware dict: `positional=0.68`, `swing=0.65`, `intraday=0.62`. `check()` signature gains `category="swing"` parameter; signal_engine passes `category=cat`. Error messages now show the threshold that was compared.

**G5-3 — SHORT `near_breakout` not penalised.** LONG had a `near_breakout` boost (+10% size, "confirm with volume"). SHORT had no corresponding check. `near_breakout=True` means volume is spiking at resistance, buyers are aggressive — that's the worst possible moment to enter a SHORT (potential breakout, not a rollover). Added: SHORT `near_breakout` → `size_mult × 0.80` + warning note. Symmetric with the LONG check but opposite direction (risk instead of opportunity).

---

## Previously Closed (compressed)

**C1–C6:** Single entrypoint (`main.py --mode=run` → MultiBankOrchestrator). TaskRunner deleted. OptionsFetcher per-bank (NSE symbol from `get_bank_config`). BSE announcements ticker-scoped. feature_snapshots ticker. Regime ML features ticker-scoped.

**H1–H6:** Data quality gate (POOR→FLAT, DEGRADED→+5pp). Exposure caps + position cap live-enforced. Capital-mode `allowed_tf` dual-enforced. CircuitBreaker persisted in `circuit_breaker_state` table. Shared `classify_regime_row` between live and training. Fundamentals DEFAULTED bubbles + positional +5pp.

**M1–M5:** Schema ownership in `database/db_setup.py` (incl. `fundamentals`, `gate_results`). `global_snapshots` auto-bootstraps to ≥7 rows on boot. BSE earnings calendar `UNIQUE(ticker, result_date)`. Gate 3 ranking 2-min TTL. FinBERT shared-singleton batching.

**Recent (this week):** First-boot fundamentals fetch · macro-stale → DEGRADED bubble · VPS-aware news fetcher (`QE_IS_VPS=1` skips Google News) · per-source news cycle summary + empty-cycle Telegram alarm · `gate_results` DB table (signal_engine UPSERTs; dashboard reads DB) · dead HDFC_EARNINGS_DATES removed.

### Independence guarantees
- **Per-bank state**: `_last_signal_by_cat[ticker][cat]`, `_sent_reversals[(ticker, cat, type)]`, `_last_signal_per_ticker[t]`, `_opened_today[t]`, per-bank fetcher dicts. One `SignalEngine` per bank.
- **Per-category state**: `cat_dir`, `cat_conf`, `per_category` all keyed by cat. Gate 5/6 results local per iteration.
- **Shared singletons** (`exit_engine`, `circuit_breaker`, `portfolio_tracker`, FinBERT, `_intermarket_cache`) only access DB with ticker filters or are universe-wide by design.

---

## What's Still To Do

> **Paper-trading phase active (started 2026-05-27).** Target: 1–2 months of live outcome data before scaling capital or retraining. Scale capital when: Win >52%, PF >1.5, Sharpe >1.0, Max DD <20%, 50+ closed trades.

### High Priority
1. **5 alpha features** — wire into `feature_builder.py` + `*_FEATURES_PROD`, retrain after 8–12 weeks:
   - `finbert_momentum_3d` = `score_24h − score_72h`
   - `fii_flow_surprise` = z-score of FII flow vs 20d baseline
   - `banknifty_relative_momentum_5d` = `banknifty_5d_pct − nifty_5d_pct`
   - `atr_percentile_252` = ATR rank vs 1y
   - `banks_above_50dma_pct` = breadth of 5-bank universe

### Medium Priority
2. **E2 — Regime HMM memoization.** Cache `RegimeDetector.detect()` result keyed by last-candle timestamp; skip re-inference when no new bar (~1 s/cycle saved).
3. **GlobalFetcher rolling rows** — `bootstrap_history(days=30)` runs on boot; once 7+ rows accumulate naturally, the `macro_stale` flag drops.
4. **Fundamentals population** — first-boot fetch added; verify Sunday weekly refresh keeps growing the table for trend features.
5. **News on VPS** — `QE_IS_VPS=1` drops Google News, monitor per-source counters: `journalctl -u quantedge-signal | grep "news cycle:"`.

### Low Priority
6. **Gate 3 dashboard** — already reads `gate_results.gate3` from DB after JSON→DB migration. Verify panel renders identically.
7. **L-1** — confirm `_eod_sent`, `_evening_sent`, `_intraday_close_sent` are intentionally global one-shots (vs per-ticker like `_opened_today`).
8. **E3/E4** — collapse 5× `GlobalFetcher.is_stale` per cycle; share sqlite3 connections within a cycle.

### Before Going Live (remove paper-trading overrides)
- Set `FORCE_CAPITAL_MODE=""` (or remove) in `/etc/quantedge.env` — restores ₹-based auto-detection
- Set `QE_PAPER_TRADING=0` — switches dashboard banner from 🧪 PAPER to 💰 LIVE
- Set `STARTING_CAPITAL` to actual account size
- Verify DD thresholds match account risk tolerance (Rule #22)

---

## Running the System (local)
```bash
python3 main.py                              # live 24x7, all 5 banks (MultiBankOrchestrator)
python3 main.py --mode=train                 # retrain HDFCBANK
python3 main.py --mode=train --ticker=ICICIBANK.NS
python3 main.py --mode=train --model=swing   # swing only (~30s)
python3 main.py --mode=signal --ticker=ICICIBANK.NS
python3 orchestrator.py --once               # one cycle, then exit
python3 graceful_close_for_deploy.py         # safe position close before deploy
uvicorn dashboard_api:app --host 0.0.0.0 --port 8000
```

---

## VPS Hosting (DigitalOcean — 165.22.220.126)

### One-time setup (fresh VPS)
```bash
# 1. SSH in
ssh root@165.22.220.126

# 2. System deps
apt update && apt install -y python3.13 python3.13-venv python3-pip git sqlite3 nginx
cd /root && git clone <repo-url> TradingBot && cd TradingBot

# 3. Python deps
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Environment file — IMPORTANT: set QE_IS_VPS=1 so Google News is skipped
cat > /etc/quantedge.env <<EOF
QE_IS_VPS=1
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
TELEGRAM_TOKEN=<your_token>
TELEGRAM_CHAT_ID=7873846599
# Paper-trading phase — keeps all 3 timeframes active regardless of capital.
# Remove or set FORCE_CAPITAL_MODE="" before going live with real money.
FORCE_CAPITAL_MODE=FULL
QE_PAPER_TRADING=1
STARTING_CAPITAL=500000
EOF
chmod 600 /etc/quantedge.env

# 5. Initialise DB schema
.venv/bin/python -c "from database.db_setup import DatabaseSetup; DatabaseSetup().setup_all()"

# 6. systemd unit — signal pipeline
cat > /etc/systemd/system/quantedge-signal.service <<'EOF'
[Unit]
Description=QUANT EDGE Signal Engine
After=network.target

[Service]
Type=simple
WorkingDirectory=/root/TradingBot
EnvironmentFile=/etc/quantedge.env
ExecStart=/root/TradingBot/.venv/bin/python orchestrator.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# 7. systemd unit — dashboard API
cat > /etc/systemd/system/quantedge-api.service <<'EOF'
[Unit]
Description=QUANT EDGE Dashboard API
After=network.target

[Service]
Type=simple
WorkingDirectory=/root/TradingBot
EnvironmentFile=/etc/quantedge.env
ExecStart=/root/TradingBot/.venv/bin/uvicorn dashboard_api:app --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 8. nginx reverse proxy at /quant/
cat > /etc/nginx/sites-available/quant <<'EOF'
server {
    listen 80;
    server_name 165.22.220.126;

    location /quant/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF
ln -sf /etc/nginx/sites-available/quant /etc/nginx/sites-enabled/quant
nginx -t && systemctl reload nginx

# 9. Enable + start
systemctl daemon-reload
systemctl enable quantedge-signal quantedge-api
systemctl start quantedge-signal quantedge-api

# 10. Verify
journalctl -u quantedge-signal -f      # follow logs
curl -s http://127.0.0.1:8000/api/live  # quick health check
# Browser: http://165.22.220.126/quant/
```

### Standard deploy (after one-time setup)
```bash
ssh root@165.22.220.126
cd /root/TradingBot

# Safe deploy — positions survive restart:
git pull && systemctl restart quantedge-signal quantedge-api

# Schema change — close positions first:
.venv/bin/python graceful_close_for_deploy.py
git pull && systemctl restart quantedge-signal quantedge-api
journalctl -u quantedge-signal -n 50    # confirm clean boot
```

### Deploying models (after a retrain) — CRITICAL
`models/saved/*.pkl` is **gitignored**, so `git pull` does NOT update the
directional ML models — the VPS keeps whatever .pkl files are already on disk.
After any retrain you MUST sync the models separately, **from the training
machine** (not the VPS):
```bash
./deploy_models.sh                 # rsync models → VPS, then restart signal svc
./deploy_models.sh --no-restart    # sync only
```
Skipping this is what caused the "BEAR_TRENDING giving LONG" bug: new code ran
against the *old pre-CR-1A models* that learned mean-reversion. The orchestrator
now runs `_audit_models()` at boot and alerts (log CRITICAL + Telegram) if any
bank's directional models are missing or predate the 2026-05-27 CR-1A fix —
watch for `Model audit ✓` in the boot log.

### Health checks
```bash
journalctl -u quantedge-signal | grep "news cycle:"        # per-source news counts
journalctl -u quantedge-signal | grep "fundamentals"       # bootstrap status
sqlite3 /root/TradingBot/database/trading.db ".tables"     # schema sanity
sqlite3 /root/TradingBot/database/trading.db \
  "SELECT ticker, COUNT(*) FROM feature_snapshots GROUP BY ticker;"
```

---

## Key Design Rules (never break)
1. `TimeSeriesSplit` always — never random split (lookahead bias).
2. Calibrated probabilities in production — raw XGBoost overconfident.
3. FLAT is default — only trade when 5+ independent signals align.
4. ATR-based stops/targets/sizing — never fixed rupee stops.
5. Win rate >70% in backtest = bug. Target: 55–65%.
6. Social sentiment INVERTED — retail euphoria = contrarian bear.
7. Retrain only after 8–12 weeks of live outcome data (exception: CR-1A required immediate retrain since the bug was structural, not data-driven).
8. All training getters use `<= as_of_date` — never peek at future.
9. Gate 2 raw pass-throughs (`days_to_rbi`, `india_vix_level`, `usdinr_5d_pct`, `nifty_5d_pct`) stay as raw values in `feature_builder.py`.
10. `open_trades` DB is the ONLY source of truth for open-position counts — never fall back to signal_log.
11. **Multi-bank tables MUST include `ticker`** — never rely on a global default. Tables: `feature_snapshots`, `options_snapshots`, `bse_announcements`, `fundamentals`, `regime_snapshots`, `open_trades`, `closed_trades`, `signal_outcomes`, `news`, `gate_results`.
12. **Live + training regime features MUST share `classify_regime_row`** (top of `feature_builder.py`). Both paths use the same `_REGIME_*` constants — change only the constants. **As of 2026-05-27 the function reads ADX + EMA spread only — no `returns_20d` (audit CR-1A).**
13. **CircuitBreaker is single source of truth for halt state.** Persisted in `circuit_breaker_state`; consumed by `RiskFeatures` → Gate 6 via `risk_context["circuit_breaker_level"]`. Don't add a second halt mechanism.
14. **`PositionSizer.check_exposure()` MUST run before every `open_position()`** in the live path; recompute open positions inside the loop.
15. **Capital-mode `allowed_tf` enforced in BOTH** `signal_engine` (for dashboard explainability) AND `orchestrator._open_paper_trades` (belt-and-suspenders).
16. **Regime rules live under `regime_result["rules"]`** — read `trade_long`/`trade_short`/`position_mult` from there, not from the top-level dict (audit CR-1B).
17. **Reversal alerts have a 30-min cooldown per `(ticker, cat, type)`** (`REVERSAL_COOLDOWN_SEC` in `orchestrator.py`). Position-close clears entries for that ticker.
18. **VPS deployments must set `QE_IS_VPS=1`** in `/etc/quantedge.env` — otherwise Google News fetches waste cycles on a 403 from DigitalOcean IPs.
19. **Intermarket peer/sector fetches are universe-cached** (`_intermarket_cache` in `processing/intermarket.py`, 5-min TTL). Don't add per-bank yfinance calls there.
20. **Gate 5 S/R R:R hard floor = 0.5** (`RR_HARD_BLOCK` in `gate5_sr_validator.py`). R:R below 0.5:1 is a hard block regardless of confidence. Soft size-penalty only for 0.5–2.0 range. Don't lower this floor — it prevents statistically losing entry geometry.
21. **Counter-regime + Grade D = hard block** (`signal_engine.py` per-category loop). If `regime_match=False` AND `entry_quality=D`, reject the category before sizing. Counter-regime trades need at least Grade C entry quality to proceed (at 0.5× size).
22. **`FORCE_CAPITAL_MODE` must be explicitly unset (or set to `""`) before going live.** Currently defaults to `"FULL"` for paper trading. Leaving it as `"FULL"` in production would remove risk-of-ruin guardrails that SMALL/GROWING modes enforce on small accounts.
23. **`regime_match` MUST be computed before Gate 5/6** in the per-category loop — regime-aware threshold boosts only work if they're known before Gate 6 runs. Never move `regime_match` below the Gate 6 call.
24. **Positional counter-regime = always HARD BLOCK.** No exception, no confidence override. A 3–4 week hold against the macro regime is not a paper-trading probe — it's a structural loss. The hard block lives in `signal_engine.py` before Gate 5.
25. **BEAR_TRENDING `position_mult` = 1.0, BULL_TRENDING = 1.2.** Keep these symmetric for aligned trades. The BEAR asymmetry (0.8) was removed 2026-05-27 — don't reinstate it without reason.
26. **Gate 5 Grade D threshold is category-aware** (`GRADE_D_OVERRIDE_CONF` dict in `gate5_sr_validator.py`). positional=68%, swing=65%, intraday=62%. Never flatten to a single float — longer hold duration in a bad S/R setup needs proportionally more conviction. Pass `category=cat` from `signal_engine.py`.
27. **`_empty_sr()` returns Grade D / -0.2** (`SupportResistanceEngine` in `support_resistance.py`). When price data is insufficient (<30 bars), the SR engine has no information — Grade D is the honest grade, not Grade C. Returning C would silently promote no-data entries.
28. **SHORT `near_breakout` is a risk, not an opportunity.** For LONG it's a positive confirmation (+10% size). For SHORT it's a danger signal — aggressive buyers at resistance = breakout attempt, not rollover. Penalised 0.80× in Gate 5. Never remove this asymmetry.
29. **Each sizing factor is applied EXACTLY ONCE, by its owning component** (audit SIZE-1). S/R entry grade → Gate 5 (`GRADE_SIZE_MAP`, applied as `g5_ctx.size_mult` in signal_engine). Capital mode → `position_sizer` (`max_risk_pct`). Drawdown/heat/anomaly/loss-streak/meta → `risk_features.final_size_mult`. NEVER re-add `sr_mult` or `cap_mult` to the `risk_features` composite — that double-counts and collapses Grade C/D (and SMALL/GROWING) signals to 0–1 shares so they're discarded after passing all 6 gates. `sr_mult`/`cap_mult` are returned for dashboard transparency only.
30. **The offline pipeline backtest MUST reuse the live gate classes** (`backtest/pipeline_backtest.py`). Gate 5 = `Gate5SRValidator`, Gate 6 = `Gate6Confidence`, regime = shared `classify_regime_row` + `REGIME_RULES`. Never fork a parallel copy of gate logic into the backtest — the whole point of P0-3 is that the validated strategy is the SAME code that trades live. Gates 2/3 are the only documented offline neutralisations (no historical external/universe data).
31. **`PROVISIONAL_EDGE_PREMIUM` lives only in `Gate6Confidence`** (P2). Don't replicate the premium elsewhere (signal_engine, pipeline backtest) — Gate 6 owns it so it applies identically live and offline. It is provisional: zero a category once its attribution clears the scale-up bar (WR>52, PF>1.5). Removing it is a deliberate, attribution-driven decision, not a cleanup.


## Fixes to be done 
✅ G6-4 — FIXED (see SIZE-1 above). S/R double-count removed; Gate 5 owns S/R sizing.
✅ G6-1 — FIXED. VIX threshold now category-aware in gate6_confidence.py.
✅ G6-3 — ALREADY FIXED (WARN band at 0.75× exists in risk_features).
✅ G6-2 — FIXED. circuit_breaker_level/reason now in feature_builder risk_context.

✅ G6-5 — FIXED (2026-06-01). Threshold-boost reason now shows magnitude + cause split (`+x% thr-adj`, `+x% edge-premium`, `+x% VIX`) in `gate6_confidence.py` instead of falsely labelling all stacked boosts "+DQ boost".

✅ Test hygiene — FIXED (2026-06-01). All 7 stale unit tests updated to current behavior (not code bugs): capital-mode tests neutralise `FORCE_CAPITAL_MODE` via setUp/tearDown to test the detection bands; CB-halt test uses −16% (FULL halt = 15%; −12% is only PAUSE); Gate 2 tests use current feature key `days_to_earnings` + hard-fail key `vix_halt`; Gate 6 CB test exercises `circuit_breaker_level`. Full suite green.

## Deferred (long-horizon)
Bank Nifty futures hedging · shadow-mode deployment · Kelly sizing · `scale_pos_weight` balancing · pooled multi-ticker model with ticker embedding · P3 features (LLM earnings score, social contrarian, Google Trends alt data).
