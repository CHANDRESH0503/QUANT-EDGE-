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
| 1 | Regime | per-bank | CHOPPY → block. Stability <25% → block. `regime_match` computed BEFORE Gate 5/6. Aligned: normal threshold + full size. Counter-regime positional: HARD BLOCK. Positional+HIGH_VOL: HARD BLOCK (max_hold=2d). Counter-regime swing: +7pp Gate 6 / intraday: +10pp. HIGH_VOL aligned: +5pp. low_stability (25–40%): +5pp. regime_changes ≥2: +5pp. Counter-regime size: 0.5×. **Anti-whipsaw: a new regime must persist 3 consecutive cycles before committing (HIGH_VOL commits instantly = fast de-risk); held regime flagged low-stability.** |
| 2 | Rule Filter | per-bank | 11 checks, need 8+. Hard fails: earnings <3d, VIX ≥28, anomaly HIGH. Fundamentals DEFAULTED → soft-pass + `fundamentals_stale`. BEAR flips checks 5/8/9/10 (POOR fundamentals, hostile macro, risk-off, rupee stress all CONFIRM short). Stale macro → neutral. |
| 3 | Universe Rank | per-bank | Ranks 5 banks; per-regime cache. BULL: strongest gets 1.00× (rank-1). BEAR: score inverted — weakest (most negative momentum) gets 1.00× (best short). Disqualifies score <−5 (BULL=falling knife, BEAR=short-squeeze risk). should_switch = ticker_rank > 2. |
| 4 | ML Models | per-bank | Passes iff ≥1 model non-FLAT. Alignment grades: A+(+15pp/1.20×) · A(+10pp/1.0×) · B(+5pp/0.85×) · B−(−5pp/0.75×, 2-vs-1) · C(0pp/0.70×) · F(−20pp/0.50×, 1-vs-1 only). Conf floor 0.0. All grades INFORMATIONAL. |
| 5 | S/R Validator | per-category | Entry quality A/B/C/D per direction (SHORT inverts R:R). Hard block: R:R < 0.5:1. Soft penalty: 0.5–2.0. Grade-D override: pos≥68% / swing≥65% / intra≥62%. SHORT near_breakout: 0.80× (buyers aggressive). empty_sr() → Grade D. |
| 6 | Confidence | per-category | FULL base: swing 60% / pos 55% / intra 65%. **+ provisional edge premium: swing +3pp, pos +5pp → effective 63/60/65%.** Boosts: VIX category-aware · DEGRADED +5pp · positional+DEFAULTED fundamentals +5pp · capped +15pp. Reads `circuit_breaker_level` directly. |

**Data quality gate** (pre-Gate-4): POOR → hard FLAT. DEGRADED → +5pp Gate 6. `macro_stale=1` (`GlobalFetcher.is_stale(30h)`) clamps quality to DEGRADED.
**Capital-mode `allowed_tf`** enforced at signal_engine AND `_open_paper_trades` (SMALL=swing · GROWING=swing+intra · FULL=all). Disallowed → structured FLAT with reason.
**Counter-regime:** positional always HARD BLOCK; swing/intraday at 0.5× size + raised Gate 6 threshold. `regime_match` evaluated BEFORE Gate 5/6 so boost is baked in.
**Counter-regime + Grade D** = double-jeopardy HARD BLOCK regardless of confidence.

---

## ML Models — Latest CV (prod variant, retrained 2026-06-04, MODEL-1 class-balancing)
> **Balanced 3-class accuracy** (random floor 0.33). NOT comparable to the pre-MODEL-1 numbers, which were inflated by the FLAT base rate — e.g. the old "intraday 0.764" was ~the 77%-FLAT majority, not edge. These are the honest, de-biased numbers; **judge edge by the holdout** (`pipeline_validation.json`), not by this accuracy.
| Bank | Swing | Positional | Intraday |
|------|------|------------|----------|
| HDFCBANK | 0.500 | 0.383 | 0.531 |
| ICICIBANK | 0.478 | 0.419 | 0.550 |
| KOTAKBANK | 0.505 | 0.503 | 0.491 |
| AXISBANK | 0.439 | 0.381 | 0.478 |
| INDUSINDBK | 0.443 | 0.379 | 0.453 |

3 variants/bank: technical (25yr) · full (2020+) · prod (25yr + verified features). Gate 4 priority: prod→full→technical. All calibrated (isotonic, Platt if <150 cal samples). Per-ticker HMM regime: `models/saved/regime_model_{BANK}.pkl`.
**Scale capital when:** Win >52%, PF >1.5, Sharpe >1.0, Max DD <20%, 50+ trades.

---

## Capital Modes & Risk
| Mode | Capital | Risk/Trade | Max Pos | Allowed TF | Swing/Pos/Intra base thresholds |
|------|---------|------------|---------|------------|----------------------------|
| SMALL | <₹50K | 1% | 2 | swing only | 68%/65%/72% |
| GROWING | ₹50K–₹2L | 1.5% | 3 | swing+intra | 60%/60%/68% |
| FULL | >₹2L | 2% | 5 | all three | 60%/55%/65% (+edge premium → 63/60/65) |

**Paper-trading override:** `FORCE_CAPITAL_MODE=FULL` in `/etc/quantedge.env` (default since 2026-05-27). Bypasses ₹ auto-detection so all 3 timeframes stay active. `CapitalMode.detect()` checks this first. Remove/`""` before going live.
- Exposure caps (live-enforced): 40% per-name · 80% total. VIX >20: +5pp · >25: +10pp · ≥28: HALT_AND_FLATTEN.
- Exit: 1R→book 50% | 2R→book 30% | trail 20% chandelier (3×ATR).
- DD multipliers (mode-aware, = `CircuitBreaker.THRESHOLDS`): SMALL halt −8% / GROWING −12% / FULL −15%.

---

## Audit History (one-line; detail in git log + Key Design Rules)

**2026-06-04**
- **GATE1-1** (regime correctness) — Gate 1 used the **HMM** regime for gating/sizing/counter-regime while the ML models were trained on the **`classify_regime_row` (ADX+EMA)** regime — two classifiers that disagreed on **2/5 banks** live (Kotak/Axis: HMM said `BEAR_TRENDING` at **100% stability** on tapes that were objectively CHOPPY by ADX/EMA, ema spread −0.3%/−0.6%). The HMM systematically under-called CHOPPY (slight drift → "BEAR 100%"), and two extra overrides (±1.5% EMA rescue + 200d rescue) only ever pushed CHOPPY→trend — so Gate 1's single most important job (block choppy markets, the system's #1 rule) was being defeated, and the gate/model were inconsistent (model sees `regime_choppy=1`, gate enforces BEAR). Fix (user-chosen "unify on ADX/EMA"): `RegimeDetector.detect(df, tech_df=...)` now classifies via **`_classify_detect`** using the SHARED `classify_regime_row` thresholds (HIGH_VOL > BEAR/BULL > CHOPPY precedence), stability = rolling 10-bar fraction of the current regime; the HMM detection + EMA/200d overrides are **removed from the gating path** (HMM methods kept for offline/back-compat only). `feature_builder` passes `tech_df` so Gate 1 reads the EXACT `adx`/`ema_spread` the model features use → one regime end-to-end, finishing CR-1A. **Verified live (all 5):** HDFC/ICICI BEAR, INDUSIND BULL (all legit-trending, 100% stab), **Kotak now CHOPPY** (held BEAR 34% during the 3-cycle anti-whipsaw transition → low_stability caution fires → will block), **Axis now HIGH_VOL** (ADX 31 + >2.5% move → instant de-risk commit). The false "100%-stable BEAR on chop" is gone. Persistence/anti-whipsaw (REGIME-1) and per-category machinery (positional hard-block, swing/intraday +7/+10pp, counter-regime+Grade-D) unchanged — they were already sound; only the classifier feeding them was wrong. (Rule #12 rewritten) Code-only — `git pull && restart`.
- **GATE5-1** (root cause, next throttle after MODEL-1) — After MODEL-1 de-biased the models, the binding constraint was **Gate 5 S/R**, which graded ~59% of bars Grade-D and **hard-blocked ~37% on the R:R floor**. Root cause: `reward_risk_sr = nearest_resistance_dist / nearest_support_dist` was **disconnected from the actual trade** — the trade executes a fixed ATR geometry (stop 2×ATR, target 5×ATR = a real **2.5:1 on every trade**) and never uses S/R for stop/target. The S/R level set (swing extrema + round numbers every ₹50 + volume nodes) is *denser than ATR*, so price sits a median ~0.7% (~0.5 ATR) from a level on BOTH sides → the ratio was ~1.0 noise that blocked genuine 2.5:1 trades at random. It was a **mean-reversion rule ("enter near support") grafted onto a momentum + ATR-stop system**. Diagnosis ruled out window-shortening, strength filters, and ATR-capped walls (all still block 36–56% — the levels are real, the *metric* is the wrong tool). Fix (user-chosen "Real-R:R, S/R sizes not vetoes"): `reward_risk_sr` = the **real trade R:R (2.5)** so the hard floor never spuriously fires; `entry_quality` recomputed as an **ATR-relative runway grade** (`_runway_grade`, both directions) that ONLY modulates size (A=full runway → full size … D=strong wall in path → 0.40× probe); Grade-D hard-block + `GRADE_D_OVERRIDE_CONF` removed; `near_*` size penalties kept. **Verified (12-mo holdout, all 5):** `g5_rr_med` 0.9→**2.5**, Gate-5 death points (`gate5_RR_hardfloor`/`gate5_other`) **gone** from every funnel, grades now A-heavy. **The throttle moved cleanly downstream to Gate 6 confidence** (de-biased models are honestly ~0.50–0.55, Gate 6 wants 60–65%; signals now die ~10pp under threshold = the EDGE-1 boundary). Trade results mixed/tiny: INDUSINDBK positional **5 trades WR 60% PF 1.84 Sharpe 3.58** (first genuinely positive), but ICICI/Axis positional still PF~0.03 — no proven edge. (Rules #20/#26/#27 rewritten) **Honest end state: the gate stack is now clean (Gates 1–5 no longer spuriously block); remaining quietness is the models' genuine lack of high-confidence edge — which is what paper trading must MEASURE, not what to engineer away. Do NOT lower Gate 6 to force volume (Rule #35/#3).** Local only — `./deploy_models.sh` is models; this is code (`git pull`).
- **MODEL-1** (root cause) — Fixed the edge failure behind the 2026-06-03 verdict (positional structurally LONG-biased → dead market-wide in BEAR; swing/intraday degenerate). All three trainers fit `multi:softprob` with **no class weighting**, so 25yr secular-uptrend priors + heavy regularisation made positional default LONG on sparse/zero input (probe: 16/20 random→LONG, zero→LONG 0.69), and the 0.4%/30min threshold made FLAT ~77% of intraday labels → intraday collapsed to FLAT≈1.0 (the old "0.764 CV" was the FLAT base rate, **not skill** — real directional CV ≈0.53). Fix (no gate/logic change, revertible via `CLASS_BALANCE=None`): per-fold + final-fit `sample_weight=compute_sample_weight("balanced", y)` in `train_swing`/`train_positional`/`train_intraday`; calibration left **UNweighted** so probs stay true-frequency; eval_set unweighted so early-stopping tracks real perf. Added `PreFitSigmoidCalibrator` (Platt) + `make_calibrator(base, n_cal)` factory (Platt <150 cal samples, isotonic above) for the small positional/per-bank tails that collapsed isotonic. Retrained all 5 banks with `--force` (balanced accuracy is **lower** than the FLAT-inflated metric, so the CV save-gate would reject the de-biased model). **Verified (12-mo OOS holdout, all 5 banks):** positional flipped from ~all-LONG to balanced/varied on EVERY bank (ICICI dir 111 SHORT/29 LONG → opened 6 real BEAR shorts that were structurally impossible before; Axis real-bar dir 40S/52L/48F — the synthetic-input "still LONG" probe was an out-of-distribution artifact); swing balanced on all 5; HDFC trades 0→4. **The throttle moved off the model layer onto the gate stack** (swing dies at Gate 5 Grade-D + R:R<2; positional at Gate 1 CHOPPY 48% + counter-regime 35% by design; intraday at honest FLAT). (Rule #36) **Honest caveats (unchanged verdict — not yet scale-ready):** (1) still too quiet for the 30+/cat proof — next throttle is **Gate 5 S/R geometry** (Grade-D dominant, g5_rr_med 0.5–1.0), not the models; (2) the few trades that fire are losing on tiny samples (ICICI pos PF 0.81, Axis pos PF 0.03) — no demonstrated edge yet, consistent with the weak CV; (3) intraday direction is genuinely ~unpredictable at 0.4%/30min — its FLAT-collapse is calibration being honest; **deliberately NOT forced to fire** (Rule #35 — manufacturing intraday trades poisons the paper-trading proof). **Models retrained LOCALLY only — push to VPS via `./deploy_models.sh` before they take effect live (gitignored `*.pkl`).** Next: diagnose Gate 5 offline geometry (why Grade-D dominates the holdout); consider intraday label-threshold/abstention redesign only if a real intraday edge is wanted (deferred — may just be noise).

**2026-06-03**
- **DATA-1** — Diagnosed "only ICICI SHORT": mostly BY DESIGN (BEAR_TRENDING universe-wide → LONGs counter-regime, positional LONG hard-blocked; only ICICI's swing had SHORT conviction 0.919 above threshold). Underlying suppressor was persistent DEGRADED data quality (37–42/129 features zero/cycle). **NSE is NOT datacenter-blocking the VPS** (`/api/fiidiiTradeReact`→200; FII/fundamentals/global_snapshots populate) — earlier block hypothesis was wrong. Real cause: NSE **strict-endpoint cookie expiry** — `option-chain-equities` (9 features) + insider `corporate-share-holdings-master`/`block-deal` (5) need cookies minted by visiting the matching HTML page, and the boot-time homepage prime EXPIRES in the long-running orchestrator → silent 401 → empty (the lenient `fiidiiTradeReact` tolerates the stale cookie, so FII worked but options/insider didn't). Fix: new `data/nse_session.py` (`nse_get_json` = prime correct page + retry ONCE on 401/403), wired into `options_fetcher._fetch_options_chain` and `insider_fetcher._fetch_shareholding`/`_fetch_block_deals`. **FII fetcher left untouched** (works — don't fix what isn't broken). (Rule #34) Offline caveat: couldn't reproduce the live 401 from the dev box (market closed → endpoint returns 200-but-empty); verify on VPS during a market session (`journalctl … grep "Options:|re-priming"`, then count `options_snapshots`). insider/bulk may also be partly legitimately thin (block deals sparse per-bank; shareholding quarterly).
- **EDGE-1 / EDGE-2 / EDGE-3** — System was too quiet for paper-trading proof (only ICICI firing). Root cause through a trader's lens: it gated on *certainty* (P(win) ≥ 60–65%) and ignored *expectancy*, applied its highest threshold to its best model (intraday), let its LONG-biased positional model veto aligned shorts, and never reviewed near-misses. Three fixes (all revertible, no retrain): **EDGE-1 expectancy override in `Gate6Confidence`** — below the P(win) threshold a signal still passes if Gate-5 geometry is Grade A/B, R:R ≥ 2:1, conf ≥ 0.50 (leans right) and conf within 10pp of threshold (`EXPECTANCY_*` consts); EV(R)=conf·rr−(1−conf); `signal_engine` now threads `entry_quality`/`reward_risk` into Gate 6; surfaced as `expectancy_pass`/`expected_R`. (Rule #35) **EDGE-2 regime-aware Gate-4 alignment** — a model vote against what the regime permits (e.g. LONG-biased positional in BEAR) is neutralised to FLAT *for grading only* (`_calc_alignment(regime_trade_long/short)`), so it no longer drags an aligned signal to Grade F (−20pp); per-category direction/confidence untouched. **EDGE-3 `funnel.py`** — near-miss "tape review": reads the latest per-bank `gate_results` (zero new logging) and prints where each signal died + how close (gap to threshold, grade, R:R, "within expectancy reach"). Run `python3 funnel.py` on the VPS. Tune `EXPECTANCY_*` from what it shows. Note: the P2 `PROVISIONAL_EDGE_PREMIUM` was deliberately KEPT (Rule #31) — the expectancy path is the release valve, not premium removal. Open/deferred still: positional LONG-bias retrain (Rule #7).
- **DATA-2** — `InsiderFetcher` was HDFC-only (hardcoded `HDFC_NSE_SYM`, single-bank `shareholding_pattern`/`insider_block_deals` with UNIQUE on quarter alone) → the other 4 banks silently inherited HDFC's promoter/block-deal data. Made it **per-bank**: `InsiderFetcher(db_path, ticker)` resolves `nse_symbol`/`bse_code` via `get_bank_config`; both tables gained a `ticker` column + `UNIQUE(ticker, …)` (Rule #11) with a self-healing drop+recreate migration in `_setup_db` (tables hold only refetchable public cache, never trades — safe to drop; both empty on VPS anyway); all read/write paths ticker-scoped; orchestrator now builds a per-bank `insider_fetchers` dict and the weekly refresh loops it; `feature_builder` passes its ticker. Also fixed the shareholding fetch itself: NSE needs `&index=equities` (omitting → 200 "missing index"), the endpoint returns a **bare list** of ~87 historical quarters (old code assumed `{"data":[…]}` + wrong field names), and the real fields are `pr_and_prgrp`/`public_val`/`date` — now parsed, with the date normalised to ISO (`_iso_quarter`) so `quarter DESC` sorts chronologically, persisting the latest 8 quarters so the promoter trend works immediately. Verified live: HDFC/ICICI promoter 0% (widely held), **Kotak 25.87%**, **IndusInd 15.82%** — exactly the per-bank signal the HDFC proxy was hiding. FII/DII/pledge remain 0 (XBRL-only, not in the summary endpoint).
- **DQ-2** — Persistent DEGRADED was partly a measurement artifact. `_data_quality` counted any numeric `==0.0` as missing, miscounting legitimately-zero features (`day_of_week_norm`=0 on a Wednesday, neutral z-scores, and all portfolio/streak features that are 0 at paper-start) → false DEGRADED → blanket +5pp Gate 6 threshold every category every cycle, suppressing valid signals. Fix: added 10 date-derived/portfolio-state/meta names (`day_of_week_norm`, `month_seasonality`, `holiday_proximity`, `portfolio_heat`, `monthly_pnl_norm`, `monthly_dd_pct`, `signal_streak_norm`, `losing_streak_norm`, `tf_alignment_score`, `direction_consensus`) to `_DQ_FLAG_NAMES`. Externally-sourced `_norm`s (options/flow/sentiment) **still counted** so a real outage still degrades. Verified total 129→119, miss% 39.5→34.5%. Extends DQ-1.
- **BSE-1** — `bse_fetcher.fetch_bse_announcements` crashed every cycle (`'str' object has no attribute 'get'`) when BSE returns a bare JSON string instead of an object → `data.get("Table")` on a str. Fix: `isinstance(data, dict)` guard before `.get`; logs + returns 0 instead of throwing.
- **Open (deferred, user hold-off):** positional models default LONG with `prob_flat≈0` on sparse/zero input (16/20 random inputs→LONG; both prod & technical variants) — CR-1A pathology persists post-retrain, masked by the positional-LONG-in-BEAR hard block. Proper fix is a retrain with `scale_pos_weight` balancing, deferred per Rule #7. Also noted: Gate 4 alignment is regime-blind (a counter-regime LONG vote can drag an aligned SHORT to Grade F).

**2026-06-01**
- **P0-3** — Offline OOS validation of the FULL gate pipeline. `backtest/pipeline_backtest.py` (`PipelineBacktest`) replays price-derivable gates over history using the **same live gate code** (Gate 1 via shared `classify_regime_row`+`REGIME_RULES`, Gate 4 per-fold technical model no-lookahead, Gate 5 `Gate5SRValidator`, Gate 6 `Gate6Confidence` + regime boosts, regime-aware `max_hold_days`, ~0.2% costs). Trades tagged `regime_at_entry`. `WalkForwardValidator(use_pipeline=True)` + `validate_holdout()` (train-on-past / test-on-tail). CLI `python3 -m backtest.walk_forward --holdout-months 12` → per-category & per-regime WR/PF/Sharpe/MaxDD + `scale_ready`, written to `models/evaluation/pipeline_validation.json`. Offline limitation: Gates 2/3 + VIX neutralised (no historical external/universe data); offline trade count is an UPPER bound. **Don't scale until holdout clears the bar per category.** (Rule #30)
- **P2 edge premium** — `Gate6Confidence.PROVISIONAL_EDGE_PREMIUM={swing:+3pp,positional:+5pp,intraday:0}`, applied live + offline, in the +15pp cap, surfaced as `edge_premium`. Provisional until attribution proves edge. (Rule #31)
- **G6-5** — Gate 6 fail-reason shows magnitude + cause split (`+x% thr-adj/edge-premium/VIX`) not "+DQ boost".
- **Test hygiene** — 7 stale unit tests updated to current behavior (FORCE_CAPITAL_MODE neutralised in capital-mode tests; CB halt −16%; Gate 2 `days_to_earnings`/`vix_halt`; Gate 6 `circuit_breaker_level`). Suite green.
- **REGIME-2** — `REGIME_FLIP_EXIT` churned counter-regime probes (entry==exit, ~0.2% cost loop). `check_model_reversals` flagged ANY LONG-in-BEAR / SHORT-in-BULL as a regime flip and auto-closed it — but the entry side *deliberately* allows counter-regime swing/intraday at 0.5× + raised threshold, so those probes were adverse from birth and got killed the next cycle; the close let orchestrator Guard 2 reopen them → open→flip-exit→reopen loop bleeding cost (the dashboard's 4 Kotak/HDFC "Regime Flip Exit" trades). Fix: only fire `REGIME_CHANGE` when the position was opened ALIGNED (`regime_match=1`) AND `regime_at_entry` actually differs from the current regime (a genuine flip). Counter-regime probes are left to their own stop/target/time-exit. Verified: probe → no auto-exit; aligned-then-flipped → still exits. Pairs with REGIME-1 (persistence stops the whipsaw that caused aligned positions to flip in the first place).
- **REGIME-1** — Regime flipped too frequently (whipsaw). `RegimeDetector.detect()` re-classified independently every ~15-min cycle with no hysteresis, so one noisy cycle — usually the EMA override hugging the ±1.5% spread threshold on the partial daily candle — flipped the regime that gates Gate 6 thresholds, sizing, and the counter-regime hard blocks. Fix: **confirmation persistence** (`_apply_persistence` + in-memory `_committed_regime`/`_raw_history`, seeded from the last DB snapshot). A candidate regime must persist `REGIME_CONFIRM_CYCLES=3` consecutive detections before committing; **HIGH_VOLATILITY commits immediately** (fast de-risk). Unconfirmed flips hold the previous regime at `HOLD_STABILITY=0.34` (>25% so no hard-block, <40% so low_stability caution fires). Snapshots now store the COMMITTED regime → the flip-counter/ML features reflect genuine transitions. Verified: alternating BULL/CHOPPY jitter stays BULL; sustained BEAR commits on the 3rd cycle; HIGH_VOL instant. (Rule #32) Existing 3-bar HMM consensus + duplicate-collapsing flip counter retained.
- **NEWS-1** — VPS news `total=0` every cycle. Root cause: Google News IP-blocked (datacenter ASN) AND ET per-bank topic RSS now returns an empty 776-byte shell (0 entries) → no bank-specific source; generic feeds (moneycontrol/livemint/cnbc) work but rarely keyword-match the 5 banks. Fix: added **GDELT DOC 2.0 API** as the datacenter-friendly bank PRIMARY (`news_fetcher.py`), with two non-obvious gotchas solved: (1) GDELT 429s hard on per-bank bursts (5 banks ×3 retries) → fetch ONE **combined OR-query for the whole universe**, cached class-level (`_get_gdelt_entries`, 13-min TTL, shared across all 5 fetchers = 1 request/cycle; universe pattern, Rule #19); each bank keyword-filters the shared set. (2) GDELT's RSS `<pubDate>` is malformed → feedparser misreads it as month-old → recency filter dropped 248/250 → fetch in **JSON mode and parse `seendate`** instead. Dead ET feeds dropped (config kept for re-enable). Verified: 242/250 within 7d, per-bank coverage HDFC 33 / ICICI 3 / Kotak 3 / Axis 1 from one request. Gzip fix (VPS-1) confirmed already live on the VPS.

**2026-05-31**
- **SIZE-1** (critical) — sizing chain triple-counted. `risk_features.final_size_mult` re-applied `sr_mult`/`cap_mult` already owned by Gate 5/position_sizer → Grade D collapsed to 0.18× → signals discarded after passing all gates. Fix: single-ownership (Rule #29). `sr_mult`/`cap_mult` now transparency-only.
- **G6-1** — VIX threshold category-aware: intraday (0.03,0.00), swing (0.10,0.05), positional (0.15,0.08) for (vix>25, vix>20).
- **G6-2** — `circuit_breaker_level`/`reason` added to feature_builder `risk_context` (was dead code → CB reasons logged empty).
- **G6-3** — already fixed (WARN DD band 0.75× in risk_features).
- **FE-1** — dashboard static/fake data eliminated; psychology computed (honest INSUFFICIENT_DATA <10 trades), alt-data/alerts wired to live state, `esc()` for innerHTML.
- **OUTCOME-1** — attributable outcome loop: entry context persisted on open/closed_trades; MFE/MAE/R-multiple at close; `attribution()` slices WR/PF/expectancy-R by category × aligned-vs-counter × S/R grade × alignment, in evening Telegram.
- **PERF-1** (critical) — uncached yfinance caused lag + silent FLAT-cascade. Shared TTL cache + last-good fallback in `price_fetcher.yf_safe_download` (1d=15m,1wk=6h,5m=2m). Warm build 0.2s/bank.
- **DQ-1** — `_data_quality` counted flags/one-hots as "missing". `_is_dq_flag()` excludes them; miss% 43.9→38.0%.
- **VPS-1** — news pipeline dead on VPS (gzip never decompressed). Decompress by `Content-Encoding` before `feedparser`.
- **VPS-2** — stale-model deploy gap (root of "BEAR→LONG"): `*.pkl` gitignored. `deploy_models.sh` rsyncs; `orchestrator._audit_models()` alerts at boot if models missing/predate CR-1A.

**2026-05-27**
- **CR-1A** (retrain-req) — regime/label horizon mismatch: `classify_regime_row` now ADX+EMA only (no `returns_20d`); model had learned mean-reversion → LONG-in-BEAR. All 5 retrained.
- **CR-1B** — regime rules read from `regime_result["rules"]` (was silently defaulting at top level).
- **CR-2/CR-3** — `_opened_today` now per-ticker dict; reversal alerts 30-min cooldown per `(ticker,cat,type)`.
- **E1** — intermarket peer/sector fetches universe-cached (5-min TTL), 35→7 yfinance calls/cycle.
- **PT-1..5** — FORCE_CAPITAL_MODE wired (default FULL); Gate 5 `RR_HARD_BLOCK=0.5`; counter-regime+Grade D block; dashboard capital banner; Gate 1 docstring fix.
- **G1-1..6** — Gate 6 made regime-aware (regime_match before Gate 5/6; counter +7/+10pp, HIGH_VOL +5pp, instability +5pp); positional counter-regime & HIGH_VOL hard blocks; BEAR position_mult 0.8→1.0; max_hold_days wired (BULL 21/BEAR 4).
- **G2-1/2** — Gate 2 checks 5/8/9/10 regime-aware (BEAR confirmations); stale macro → neutral.
- **G3-1/2/3** — Gate 3 BEAR-inverted scoring; per-regime cache; `should_switch` uses own ticker_rank.
- **G4-1/2/3** — Grade B− for 2-vs-1 splits (F reserved for 1-vs-1); conf floor `max(0.0,…)`; MIN_CONF doc-only.
- **G5-1/2/3** — `_empty_sr()` → Grade D; `GRADE_D_OVERRIDE_CONF` category-aware; SHORT near_breakout 0.80×.

**Previously closed (C/H/M batches):** single entrypoint (`main.py --mode=run`); per-bank OptionsFetcher/BSE/feature_snapshots; data-quality gate; exposure+position caps; CircuitBreaker persisted; shared `classify_regime_row`; schema ownership in `db_setup.py`; `global_snapshots` bootstrap; Gate 3 2-min TTL; FinBERT singleton.

### Independence guarantees
- **Per-bank state**: `_last_signal_by_cat[ticker][cat]`, `_sent_reversals[(ticker,cat,type)]`, `_opened_today[t]`, per-bank fetcher dicts. One `SignalEngine` per bank.
- **Per-category state**: `cat_dir`/`cat_conf`/`per_category` keyed by cat; Gate 5/6 local per iteration.
- **Shared singletons** (`exit_engine`, `circuit_breaker`, `portfolio_tracker`, FinBERT, `_intermarket_cache`) only touch DB with ticker filters or are universe-wide by design.

---

## What's Still To Do
> **Paper-trading active (started 2026-05-27).** Target 1–2 months outcome data before scaling/retraining.

**High:** Wire 5 alpha features into `feature_builder` + `*_FEATURES_PROD`, retrain after 8–12 wks: `finbert_momentum_3d`, `fii_flow_surprise` (z-score vs 20d), `banknifty_relative_momentum_5d`, `atr_percentile_252`, `banks_above_50dma_pct`.
**Medium:** E2 regime HMM memoization (cache by last-candle ts); GlobalFetcher rolling rows (macro_stale drops at 7+); verify Sunday fundamentals refresh; monitor VPS news per-source counts.
**Low:** Gate 3 dashboard render parity; L-1 confirm `_eod_sent`/`_evening_sent`/`_intraday_close_sent` are intentional global one-shots; E3/E4 collapse 5× `is_stale` + share sqlite connections/cycle.
**Ops (P2 open):** confirm FII/options/fundamentals tables populate on VPS (`sqlite3` row counts after live cycles) — many read 0 locally, inflating data-quality miss%.

**Before going live (remove paper overrides):** `FORCE_CAPITAL_MODE=""` (Rule #22) · `QE_PAPER_TRADING=0` · set real `STARTING_CAPITAL` · verify DD thresholds vs account.

---

## Running the System (local)
```bash
python3 main.py                              # live 24x7, all 5 banks
python3 main.py --mode=train [--ticker=ICICIBANK.NS] [--model=swing]
python3 main.py --mode=signal --ticker=ICICIBANK.NS
python3 orchestrator.py --once               # one cycle, then exit
python3 graceful_close_for_deploy.py         # safe position close before deploy
python3 -m backtest.walk_forward --holdout-months 12   # P0-3 OOS pipeline validation
uvicorn dashboard_api:app --host 0.0.0.0 --port 8000
```

---

## VPS Hosting (DigitalOcean — 165.22.220.126)

**One-time setup (fresh VPS):** deps (`python3.13 venv`, `nginx`, `sqlite3`), clone repo, `pip install -r requirements.txt`, init schema (`DatabaseSetup().setup_all()`), create two systemd units (`quantedge-signal` → `orchestrator.py`; `quantedge-api` → `uvicorn dashboard_api:app --port 8000`), nginx reverse-proxy `/quant/` → `127.0.0.1:8000`. **Critical env file `/etc/quantedge.env` (chmod 600):**
```
QE_IS_VPS=1                 # skips Google News (403 on DO IPs) — Rule #18
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
TELEGRAM_TOKEN=<token>
TELEGRAM_CHAT_ID=7873846599
FORCE_CAPITAL_MODE=FULL     # paper-trading — remove/"" before going live (Rule #22)
QE_PAPER_TRADING=1
STARTING_CAPITAL=500000
```

**Standard deploy:** `git pull && systemctl restart quantedge-signal quantedge-api` (positions survive). **Schema change:** run `graceful_close_for_deploy.py` first.

**Deploy models after retrain — CRITICAL:** `*.pkl` is gitignored, so `git pull` does NOT update models. From the **training machine** run `./deploy_models.sh` (rsync→VPS+restart) / `--no-restart`. Skipping this caused "BEAR→LONG". `_audit_models()` alerts at boot if models missing/predate CR-1A — watch for `Model audit ✓`.

**Health checks:** `journalctl -u quantedge-signal | grep "news cycle:"` · `… grep fundamentals` · `sqlite3 database/trading.db ".tables"` · `… "SELECT ticker,COUNT(*) FROM feature_snapshots GROUP BY ticker;"`

---

## Key Design Rules (never break)
1. `TimeSeriesSplit` always — never random split (lookahead bias).
2. Calibrated probabilities in production — raw XGBoost overconfident.
3. FLAT is default — only trade when 5+ independent signals align.
4. ATR-based stops/targets/sizing — never fixed rupee stops.
5. Win rate >70% in backtest = bug. Target: 55–65%.
6. Social sentiment INVERTED — retail euphoria = contrarian bear.
7. Retrain only after 8–12 weeks of live outcome data (exception: CR-1A — structural bug).
8. All training getters use `<= as_of_date` — never peek at future.
9. Gate 2 raw pass-throughs (`days_to_rbi`, `india_vix_level`, `usdinr_5d_pct`, `nifty_5d_pct`) stay raw in `feature_builder.py`.
10. `open_trades` DB is the ONLY source of truth for open-position counts — never fall back to signal_log.
11. **Multi-bank tables MUST include `ticker`**: `feature_snapshots`, `options_snapshots`, `bse_announcements`, `fundamentals`, `regime_snapshots`, `open_trades`, `closed_trades`, `signal_outcomes`, `news`, `gate_results`, `shareholding_pattern`, `insider_block_deals` (last two made per-bank 2026-06-03, DATA-2 — were HDFC-only).
12. **Gate 1, live features, and training ALL share `classify_regime_row`** (top of `feature_builder.py`) — change only the `_REGIME_*` constants. Reads ADX+EMA spread only (no `returns_20d`, CR-1A). As of GATE1-1 the gating regime comes from `RegimeDetector._classify_detect` (same thresholds), NOT the HMM — the HMM over-called BEAR-100% on choppy tapes and disagreed with the model's own regime feature. Pass `tech_df` into `detect()` so Gate 1 reads the EXACT `adx`/`ema_spread` the feature vector uses. Don't reinstate the HMM (or the EMA/200d CHOPPY-rescue overrides) as the gating authority — that re-splits the regime.
13. **CircuitBreaker is single source of truth for halt state.** Persisted in `circuit_breaker_state`; consumed by `RiskFeatures` → Gate 6 via `risk_context["circuit_breaker_level"]`. No second halt mechanism.
14. **`PositionSizer.check_exposure()` MUST run before every `open_position()`** in the live path; recompute open positions inside the loop.
15. **Capital-mode `allowed_tf` enforced in BOTH** `signal_engine` AND `orchestrator._open_paper_trades`.
16. **Regime rules live under `regime_result["rules"]`** — read `trade_long`/`trade_short`/`position_mult` there, not top-level (CR-1B).
17. **Reversal alerts: 30-min cooldown per `(ticker, cat, type)`** (`REVERSAL_COOLDOWN_SEC`). Position-close clears entries for that ticker.
18. **VPS deployments must set `QE_IS_VPS=1`** — else Google News wastes cycles on 403 from DO IPs. On VPS the bank-news source is **GDELT** (not Google News / ET topic RSS — both dead from datacenter IPs). GDELT is fetched as ONE combined universe query, cached class-level and shared across all 5 banks (`_get_gdelt_entries`, 13-min TTL) — do NOT revert to per-bank GDELT calls (instant 429 storm). Parse the JSON `seendate`, never the RSS `<pubDate>` (malformed → misparsed as month-old → recency-dropped).
19. **Intermarket peer/sector fetches are universe-cached** (`_intermarket_cache`, 5-min TTL). No per-bank yfinance calls there.
20. **Gate 5 `reward_risk` = the REAL ATR trade R:R (2.5:1 = target 5×ATR / stop 2×ATR), NOT a S/R-distance ratio** (GATE5-1, supersedes the old nearest_res/nearest_sup floor). The trade never uses S/R for execution, so the old ratio was sub-ATR noise that hard-blocked ~37% of genuine 2.5:1 trades. `RR_HARD_BLOCK=0.5` is kept only as a defensive guard against malformed input — in prod `reward_risk`=2.5 so it never fires. Don't reintroduce the S/R-distance ratio as `reward_risk`.
21. **Counter-regime + Grade D = hard block** (`signal_engine.py`). `regime_match=False` AND `entry_quality=D` → reject before sizing. Counter-regime needs ≥Grade C (at 0.5×).
22. **`FORCE_CAPITAL_MODE` must be `""`/unset before going live.** Defaults `"FULL"` for paper — leaving it removes SMALL/GROWING risk-of-ruin guardrails.
23. **`regime_match` MUST be computed before Gate 5/6** — boosts only work if known before Gate 6. Never move below the Gate 6 call.
24. **Positional counter-regime = always HARD BLOCK.** No confidence override. Hard block lives in `signal_engine.py` before Gate 5.
25. **BEAR_TRENDING `position_mult` = 1.0, BULL_TRENDING = 1.2** — symmetric for aligned trades. Don't reinstate the 0.8 BEAR asymmetry.
26. **Gate 5 `entry_quality` is an ATR-relative RUNWAY grade that SIZES, never VETOES** (GATE5-1). Computed in `support_resistance.py._runway_grade` for both directions (`entry_quality` = LONG, `entry_quality_short` = SHORT): A = clear runway to the 5×ATR target, D = entering with a strong wall (strength ≥ `WALL_STRENGTH`=6) inside the path → 0.40× probe size. Grade D NO LONGER hard-blocks on a confidence threshold (the removed `GRADE_D_OVERRIDE_CONF`) — that conviction gate is Gate 6's category threshold. Only `WALL_STRENGTH`-confluence levels cap the runway (momentum breaks minor levels). Counter-regime + Grade D is still a hard block in `signal_engine`/backtest (Rule #21) — a regime safety on probes, independent of GATE5-1.
27. **`_empty_sr()` returns Grade D + real R:R (2.5)** — <30 bars = no info → 0.40× probe size, but S/R never vetoes so the trade passes small rather than being blocked on no data (GATE5-1; was Grade D / rr 1.0 which could trip the old floor).
28. **SHORT `near_breakout` is a risk** (0.80× in Gate 5), opposite to LONG's +10% confirmation. Never remove this asymmetry.
29. **Each sizing factor applied EXACTLY ONCE by its owner** (SIZE-1): S/R grade → Gate 5 (`GRADE_SIZE_MAP`); capital mode → `position_sizer` (`max_risk_pct`); DD/heat/anomaly/loss/meta → `risk_features.final_size_mult`. NEVER re-add `sr_mult`/`cap_mult` to the composite (double-counts → Grade C/D collapse to 0–1 shares). They're transparency-only.
30. **Offline pipeline backtest MUST reuse the live gate classes** (`backtest/pipeline_backtest.py`): Gate 5 = `Gate5SRValidator`, Gate 6 = `Gate6Confidence`, regime = shared `classify_regime_row`+`REGIME_RULES`. Never fork gate logic into the backtest. Gates 2/3 are the only documented offline neutralisations.
31. **`PROVISIONAL_EDGE_PREMIUM` lives only in `Gate6Confidence`** (P2) — applies identically live + offline. Provisional: zero a category once its attribution clears the scale-up bar (WR>52, PF>1.5). Removal is attribution-driven, not cleanup.
32. **Regime commits via confirmation persistence** (`RegimeDetector._apply_persistence`, REGIME-1). A new regime must persist `REGIME_CONFIRM_CYCLES=3` cycles before replacing the committed one; only HIGH_VOLATILITY commits instantly (fast de-risk). Don't bypass it (e.g., reading raw per-cycle classification) or the regime whipsaws and drags Gate 6 thresholds/sizing/counter-regime blocks with it. Snapshots persist the COMMITTED regime — keep it that way so the flip counter stays honest.
33. **`REGIME_FLIP_EXIT` only fires for ALIGNED positions whose regime genuinely flipped** (`check_model_reversals`, REGIME-2). Gate on `regime_match=1` at entry AND `regime_at_entry != current regime`. NEVER auto-exit a counter-regime probe (swing/intraday opened at 0.5× against the regime) just for being counter-regime — it was a deliberate bet; its stop/target/time-exit owns it. Reintroducing the raw "any LONG-in-BEAR" test churns probes open→exit→reopen at ~0.2%/loop.
35. **Gate 6 accepts on EXPECTANCY, not just P(win)** (EDGE-1). Below the confidence threshold a signal still passes iff Gate-5 grade ∈ {A,B} AND `reward_risk` ≥ `EXPECTANCY_MIN_RR` (2.0) AND conf ≥ `EXPECTANCY_CONF_FLOOR` (0.50) AND conf ≥ threshold − `EXPECTANCY_MAX_GAP` (0.10). Keep all four guards — dropping the grade/R:R/floor checks turns this into blanket threshold-lowering (garbage trades that poison the paper-trading proof). `signal_engine` MUST pass Gate 5's `entry_quality`+`reward_risk` into Gate 6 for this to work. Sizing is unchanged (Gate 5 grade mult already shrinks B/weaker entries). Tune the consts from `funnel.py`, not by feel.
34. **Strict NSE API calls go through `data/nse_session.nse_get_json`** (DATA-1) — `option-chain-equities`, `corporate-share-holdings-master`, `block-deal` need page-minted cookies that EXPIRE in the long-running orchestrator; the helper primes the matching HTML page and retries ONCE on 401/403. Don't revert these fetchers to a bare `_session.get` with only a boot-time homepage prime (→ silent 401 → empty options/insider tables). The lenient `fiidiiTradeReact` (FII) endpoint is deliberately left on its own session — it tolerates the stale cookie; don't refactor it in. NSE is NOT IP-blocking the VPS (unlike Google News, Rule #18) — this is cookie lifetime, not ASN.
36. **All three trading trainers MUST class-balance** (`CLASS_BALANCE="balanced"` → `compute_sample_weight` on BOTH per-fold and final fit; MODEL-1). Without it the 25yr LONG prior makes positional default-LONG on sparse input (CR-1A pathology → positional dead in BEAR) and the FLAT-heavy intraday labels collapse the model to FLAT. Calibration is fit **UNweighted** (true-frequency); choose the calibrator via `make_calibrator(base, n_cal)` — Platt <150 cal samples, isotonic above (small tails collapse isotonic). Balanced CV accuracy (~0.50, floor 0.33) is LOWER than the old FLAT-inflated metric and is the **honest** number — judge edge by the holdout, and **retrain with `--force`** (the accuracy save-gate in `_train_one_variant` would otherwise reject the de-biased model). Don't "fix" intraday's honest FLAT-collapse by forcing it to trade (Rule #35 / #3).

---

## Deferred (long-horizon)
Bank Nifty futures hedging · shadow-mode deployment · Kelly sizing · `scale_pos_weight` balancing · pooled multi-ticker model with ticker embedding · P3 features (LLM earnings score, social contrarian, Google Trends alt data).
