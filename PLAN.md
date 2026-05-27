# QUANT EDGE Reliability Plan

Date: 2026-05-26

Goal: make QUANT EDGE work correctly as a 5-bank, 3-category, 6-gate trading system without silent cross-bank contamination, stale data decisions, repeated exit alerts, or uncontrolled risk.

This plan is ordered by failure risk. Do not optimize or add alpha features until Phase 0 and Phase 1 are complete.

## Operating Principles

1. Correctness before signal frequency.
2. FLAT is safer than a contaminated signal.
3. Every bank-scoped datum must carry `ticker`.
4. Every open trade must pass portfolio risk checks immediately before opening.
5. Every exit alert must close or partially close the DB position first.
6. Missing, stale, or defaulted data must be visible to gates.
7. One production runner only. No stale parallel code paths.
8. Every fix must include a regression test or a deterministic check.

## Phase 0 - Stop Immediate Failure Modes

### 0.1 Make the correct runner the default

Problem:
- `python main.py` starts the old single-ticker `TaskRunner`, not the 5-bank orchestrator.
- The old runner still has stale behavior, including exit alerts without DB close.

Plan:
- Change `main.py --mode=run` to instantiate `MultiBankOrchestrator`.
- Keep old `TaskRunner` only behind an explicit `--mode=legacy-run` if truly needed.
- Update CLI examples and setup text to say `python orchestrator.py` or `python main.py --mode=run` runs all 5 banks.

Files:
- `main.py`
- `scheduler/task_runner.py`
- `DEPLOY.md`
- `CLAUDE.md`

Acceptance checks:
- `python main.py --mode=run --once` or equivalent dry-run path runs all 5 banks.
- No production docs point users to the old single-bank runner.
- Old runner cannot accidentally be started by the default command.

### 0.2 Disable or repair legacy TaskRunner exit handling

Problem:
- `TaskRunner` sends exit alerts but does not close positions.

Plan:
- Preferred: remove production use of `TaskRunner`.
- If retained: copy `orchestrator.py:_run_exit_checks()` logic into `TaskRunner`.
- Any exit alert must first call `close_position()` or `close_position_partial()`.

Files:
- `scheduler/task_runner.py`
- `risk/exit_engine.py`

Acceptance checks:
- Unit test: open one fake position, trigger stop, run legacy exit path, assert `open_trades=0` and `closed_trades=1`.
- Unit test: partial exit reduces shares and writes one closed row.

## Phase 1 - Fix Multi-Bank Isolation

### 1.1 Make options data ticker-scoped

Problem:
- Options are hardcoded to HDFCBANK.
- Non-HDFC banks can receive HDFC PCR, max pain, IV, and OI walls.

Plan:
- Add `ticker` to `OptionsFetcher.__init__`.
- Resolve NSE symbol from `get_bank_config(ticker)["nse_symbol"]`.
- Add `ticker` to `options_snapshots`.
- Filter all reads by ticker:
  - `get_latest()`
  - `get_snapshot_at()`
  - `_calculate_pcr_roc()`
  - `_calculate_iv_percentile()`
  - `_get_max_pain_distance()`
  - `_get_max_pain_direction()`
- In `FeatureBuilder`, instantiate `OptionsFetcher(db_path, ticker=ticker)`.
- In `orchestrator.py`, either fetch options once per bank or create a multi-bank options refresh loop.

Files:
- `data/options_fetcher.py`
- `features/feature_builder.py`
- `orchestrator.py`
- `database/db_setup.py`

Acceptance checks:
- DB has options rows for all 5 tickers after refresh.
- `OptionsFetcher("ICICIBANK.NS").get_latest()` never returns HDFCBANK rows.
- Test fails if `NSE_SYMBOL = "HDFCBANK"` style global constant returns.

### 1.2 Make BSE announcements ticker-scoped

Problem:
- BSE announcements have no ticker column.
- High-priority filings can be processed by the wrong bank's cycle.

Plan:
- Add `ticker` to `bse_announcements`.
- Insert `self.ticker` in `_save_announcements()` and `_save_corporate_actions()`.
- Replace title-only uniqueness with ticker-aware uniqueness.
- Filter these methods by ticker:
  - `get_high_priority_unanalyzed()`
  - `get_recent_announcements()`
  - `fetch_quarterly_results_calendar()`
  - `mark_llm_analyzed()`
- Include ticker in high-priority alert payloads.

Files:
- `data/bse_fetcher.py`
- `database/db_setup.py`
- `orchestrator.py`
- `processing/llm_analyzer.py` if it stores announcement-derived records

Acceptance checks:
- Insert same mock title for two banks; both rows survive.
- `BSEFetcher("AXISBANK.NS").get_high_priority_unanalyzed()` returns only AXIS rows.

### 1.3 Save feature snapshots with ticker

Problem:
- `feature_snapshots` defaults to HDFCBANK even for other bank feature builds.

Plan:
- Add `ticker` to `FeatureBuilder._save_snapshot()` insert.
- Add an index on `(ticker, built_at DESC)`.
- Mark old ambiguous rows as `UNKNOWN` unless they can be safely backfilled.

Files:
- `features/feature_builder.py`
- `database/db_setup.py`
- optional one-time migration script

Acceptance checks:
- Run one feature build for each bank.
- DB shows exactly one latest snapshot per bank with the correct ticker.

### 1.4 Scope regime ML features by ticker

Problem:
- `get_regime_features_for_ml()` reads the latest regime snapshot globally.

Plan:
- Filter by current ticker.
- Prefer current in-memory `regime_result["stability"]` in live build so the model does not depend on a stale DB row.

Files:
- `processing/regime_detector.py`
- `features/feature_builder.py`

Acceptance checks:
- Create two regime snapshots for two tickers. Each detector returns its own regime.
- Test verifies `regime_stability` cannot come from another bank.

## Phase 2 - Enforce Risk Before Opening Trades

### 2.1 Add pre-open risk gate in orchestrator

Problem:
- Signals pass gates, then paper trades open without final portfolio exposure checks.

Plan:
- Before every `open_position()` call:
  - Check duplicate ticker + category.
  - Check max open positions for capital mode.
  - Check per-name exposure cap.
  - Check total exposure cap.
  - Check portfolio heat.
  - Check allowed timeframe for capital mode.
- Re-fetch open positions after each successful open, because opening one category changes risk for the next category.

Files:
- `orchestrator.py`
- `risk/position_sizer.py`
- `risk/capital_mode.py`
- `dashboard/portfolio_heatmap.py` if shared risk helpers are reused

Acceptance checks:
- If total exposure is already above 80%, no new trade opens.
- If HDFCBANK already has 40% exposure, no second/third category opens for HDFCBANK.
- SMALL capital cannot open intraday or positional.
- GROWING capital cannot open positional unless policy is explicitly changed.

### 2.2 Integrate circuit breaker state into Gate 6

Problem:
- `CircuitBreaker.check()` alerts but does not necessarily halt new entries.
- `RiskFeatures` has separate hard-coded thresholds.

Plan:
- Create a persisted `risk_state` table:
  - `level`
  - `trading_allowed`
  - `size_multiplier`
  - `reason`
  - `checked_at`
- `orchestrator._run_circuit_breaker_check()` writes current state.
- `RiskFeatures.extract()` reads `risk_state` and includes it in `risk_context`.
- Gate 6 blocks if `risk_state.trading_allowed` is false.
- Use one shared threshold config from `RiskConfig`.

Files:
- `risk/circuit_breaker.py`
- `features/risk_features.py`
- `signals/gate6_confidence.py`
- `database/db_setup.py`
- `config.py`

Acceptance checks:
- Force circuit breaker PAUSE in DB, run signal cycle, assert no category passes Gate 6.
- Force VIX halt, assert new entries are blocked.

### 2.3 Define explicit VIX flatten behavior

Problem:
- Gate 2 says VIX >= 28 means flatten all positions, but code primarily blocks new signals.

Plan:
- On `HALT_AND_FLATTEN`:
  - Close all intraday positions immediately.
  - For swing/positional, either close or tighten stops according to a written policy.
  - Record exit reason `VIX_HALT` or stop-adjustment event.
- Alert once per halt event.

Files:
- `orchestrator.py`
- `risk/circuit_breaker.py`
- `risk/exit_engine.py`

Acceptance checks:
- Mock VIX 30 with open intraday positions; assert they close with `VIX_HALT`.
- Repeated ticks do not spam duplicate halt alerts.

## Phase 3 - Make Data Quality Actionable

### 3.1 Convert feature quality into a gate policy

Problem:
- Feature quality is logged but not enforced.

Plan:
- Add `data_quality_policy` before Gate 4:
  - `POOR`: hard FLAT.
  - `DEGRADED`: technical-only fallback or confidence threshold uplift.
  - `GOOD/EXCELLENT`: normal.
- Track source freshness separately:
  - price
  - options
  - news
  - FII/DII
  - global macro
  - fundamentals
  - regime
- Include source freshness in `gate_results`.

Files:
- `features/feature_builder.py`
- `signals/signal_engine.py`
- new helper, for example `features/data_quality.py`

Acceptance checks:
- If options are stale, options features are marked stale and either neutralized or confidence threshold is raised.
- If price is stale/missing, pre-check returns FLAT.
- If fundamentals are missing, positional category gets a penalty or block according to policy.

### 3.2 Stop treating missing fundamentals as GOOD

Problem:
- Empty fundamentals table returns sector defaults and a non-blocking grade.

Plan:
- Return `fundamental_grade = "MISSING"` or `DEFAULTED`.
- Gate 2 should handle it as:
  - Swing/intraday: soft fail.
  - Positional: hard fail or threshold uplift until real fundamentals exist.
- Add `fundamentals_fresh` boolean.

Files:
- `processing/fundamental.py`
- `signals/gate2_rule_filter.py`
- `features/fundamental_features.py`

Acceptance checks:
- Empty fundamentals table does not produce `GOOD`.
- Positional signal does not pass on missing fundamentals unless policy explicitly permits it.

### 3.3 Bootstrap macro and fundamentals

Problem:
- Local DB has only 1 global macro row and 0 fundamentals rows.

Plan:
- Add a bootstrap command:
  - fetch global macro several times or backfill where available
  - fetch fundamentals for all 5 banks
  - verify minimum row counts
- Startup should warn loudly if minimum rows are missing.

Files:
- `main.py`
- `orchestrator.py`
- `data/global_fetcher.py`
- `processing/fundamental.py`

Acceptance checks:
- `python main.py --mode=healthcheck` reports missing macro/fundamental rows.
- Production service refuses live mode or runs in observation-only mode if critical data is missing.

## Phase 4 - Align Training and Live Inference

### 4.1 Share regime feature logic

Problem:
- Live regime feature override and training bulk regime logic differ.

Plan:
- Extract one function:
  - input: ADX, EMA spread, returns_20d, returns_5d
  - output: `regime_bull`, `regime_bear`, `regime_high_vol`, `regime_choppy`
- Use it in:
  - live `FeatureBuilder.build_all()`
  - historical `FeatureBuilder._compute_bulk_regime_features()`
  - any model training utilities

Files:
- `features/feature_builder.py`
- possible new `features/regime_rules.py`

Acceptance checks:
- Test fixed cases:
  - ADX low, EMA spread +1.7%, return positive => bull
  - ADX low, EMA spread -3%, return negative => bear
  - no trend => choppy
- Training and live helpers return identical outputs for the same inputs.

### 4.2 Version model feature contracts

Problem:
- Model artifacts and feature definitions can drift.

Plan:
- Save a feature contract with each trained model:
  - ticker
  - category
  - variant
  - feature list hash
  - training code version
  - training data window
- At load time, validate contract against current feature registry.
- If mismatch, fallback to FLAT or technical-only mode.

Files:
- `models/train_swing.py`
- `models/train_intraday.py`
- `models/train_positional.py`
- `signals/gate4_ml_predictor.py`

Acceptance checks:
- Tamper with feature list hash in a test artifact; load refuses it.

## Phase 5 - Simplify Schema Ownership

### 5.1 Make `database/db_setup.py` the only schema owner

Problem:
- Multiple modules create partial versions of the same tables.

Plan:
- Keep all table definitions and migrations in `database/db_setup.py`.
- Other modules may call `DatabaseSetup.setup_all()`, but should not create their own partial schemas.
- Remove trading-table creation from `FeatureBuilder`.
- Keep only local helper tables where module ownership is clear and non-overlapping.

Files:
- `database/db_setup.py`
- `features/feature_builder.py`
- `risk/exit_engine.py`
- `data/news_fetcher.py`
- `data/options_fetcher.py`
- `data/bse_fetcher.py`
- `processing/fundamental.py`

Acceptance checks:
- Fresh empty DB can run `DatabaseSetup.setup_all()` and every module works.
- Fresh empty DB cannot get a partial `open_trades` table from importing `FeatureBuilder`.

### 5.2 Add healthcheck mode

Plan:
- Add `python main.py --mode=healthcheck`.
- It should verify:
  - all 5 banks have 3 model categories
  - all 5 banks have regime model files
  - DB schema has required ticker columns
  - options rows exist per bank
  - news rows exist per bank
  - fundamentals rows exist per bank
  - latest global macro is fresh
  - latest feature snapshot per bank has correct ticker
  - no open trade violates exposure rules

Files:
- `main.py`
- new `monitoring/healthcheck.py`

Acceptance checks:
- Healthcheck exits non-zero on missing critical wiring.
- Healthcheck output is short enough to paste into Telegram/logs.

## Phase 6 - Test Suite for Failure Prevention

Add focused tests before adding new trading features.

Required test groups:

1. Multi-bank data isolation:
   - Options per ticker.
   - BSE announcements per ticker.
   - Regime snapshots per ticker.
   - Feature snapshots per ticker.

2. Per-category gating:
   - A bank can emit swing only.
   - A bank can emit positional only.
   - A bank can emit intraday only.
   - FLAT category does not open a trade.

3. Risk enforcement:
   - Exposure cap blocks opens.
   - Max position cap blocks opens.
   - SMALL/GROWING allowed timeframe rules are enforced.
   - Circuit breaker pause blocks Gate 6.

4. Exit lifecycle:
   - Stop closes position and writes closed trade.
   - Partial 1R exit reduces shares.
   - Same exit does not alert twice.
   - Ticker-specific price cannot close another bank's position.

5. Data quality:
   - POOR data returns FLAT.
   - DEGRADED data applies the configured policy.
   - Missing fundamentals do not masquerade as GOOD.

6. Entry point:
   - `main.py --mode=run` launches multi-bank orchestrator.
   - Old runner is not default.

Tooling:
- Add `pytest` to development requirements.
- Use temporary SQLite DBs for tests.
- Mock network fetchers; never call yfinance/NSE in unit tests.

## Phase 7 - Operational Guardrails

### 7.1 Startup safety checks

Before starting live mode:
- Run healthcheck.
- If critical checks fail, start in observation-only mode.
- Observation-only mode can build signals and log them but cannot open trades.

### 7.2 Daily pre-market checklist automation

At 06:00 IST:
- Verify data freshness.
- Verify model files.
- Verify no exposure violations.
- Send one compact Telegram health summary.

### 7.3 Audit every signal

Every signal row should include:
- ticker
- category
- gate pass/fail reasons
- feature quality
- source freshness
- model variant used
- model fallback used or not
- size multiplier components
- risk cap decision
- whether paper trade opened
- reason if not opened

## Implementation Order

Recommended order:

1. Fix entrypoint and retire/repair `TaskRunner`.
2. Fix options ticker scoping.
3. Fix BSE ticker scoping.
4. Save feature snapshot ticker.
5. Scope regime ML features.
6. Enforce exposure and capital-mode rules before opening trades.
7. Persist circuit breaker state into Gate 6.
8. Add data quality gate policy.
9. Fix fundamentals missing/default behavior.
10. Unify regime feature logic.
11. Consolidate schema ownership.
12. Add healthcheck.
13. Add regression tests.
14. Run one full dry cycle for all five banks.
15. Deploy only after healthcheck and tests pass.

## Definition of Done

The system is considered reliable enough for continued paper trading when:

- The default runner evaluates all 5 banks and all 3 categories.
- No bank receives another bank's options, BSE, regime, or snapshot data.
- Every emitted paper trade passes final portfolio risk checks.
- Exit alerts mutate the DB before Telegram is sent.
- Circuit breakers can actually block new trades.
- Missing critical data cannot silently become GOOD.
- `REPORT.md` critical findings C1 through C6 are closed.
- `PLAN.md` Phase 0 through Phase 3 are complete.
- Healthcheck passes.
- Regression tests pass locally.

Real capital should not be considered until the above is complete and at least 50 clean, correctly-attributed closed paper trades exist with no open audit exceptions.

## Senior Trader Assessment - Will This System Be Effective?

Short answer: yes, this system can become effective as a disciplined paper-trading and decision-support engine if `REPORT.md` and the earlier phases of this plan are implemented. But it should not be considered a proven trading system yet. Right now, the biggest risk is not that one gate is wrong. The bigger risk is that the system can look sophisticated while still taking trades from contaminated data, stale features, unproven model confidence, and incomplete risk enforcement.

From a 20+ year trading perspective, the system has the right instincts:
- It starts with regime.
- It has multiple independent filters.
- It separates swing, positional, and intraday logic.
- It tries to control size, drawdown, exits, and event risk.
- It records paper trades before real capital.

But professional trading systems do not fail only because of missing code. They fail because the edge is assumed instead of proven, because execution costs are underestimated, because regimes change, and because risk limits are advisory instead of absolute. The current plan fixes the wiring. The next layer must prove that the wiring actually creates positive expectancy.

## What The System Still Lags As A Trading System

### 1. Proven expectancy is still missing

The system has model accuracy numbers, but accuracy is not expectancy. A 60% accurate model can still lose money if losses are larger than wins, if entries are late, if stops are too tight, or if costs/slippage eat the edge.

Needed:
- Per-bank, per-category expectancy:
  - average win
  - average loss
  - win rate
  - profit factor
  - expectancy per trade
  - max adverse excursion
  - max favorable excursion
  - average hold time
- Separate reports for swing, positional, intraday.
- Separate reports for LONG and SHORT.
- Separate reports by regime.

Rule:
- No category should graduate to real capital until it has positive expectancy over at least 50 clean closed trades, and preferably 100+ for intraday.

### 2. Gate count can create false confidence

Six gates sound safe, but if several gates are fed by the same underlying price movement, they are not independent. RSI, EMA spread, ADX, returns, support/resistance, trend regime, and ML features can all be different views of the same price action.

Loophole:
- The system may believe "6 gates passed" when in reality only one or two independent sources passed.

Needed:
- Classify gates by information source:
  - price/trend
  - volume/order flow
  - options
  - news/sentiment
  - macro
  - fundamentals
  - risk state
- Require at least 3 genuinely independent confirmations for full-size trades.
- If all confirmations come from price-derived features, cap size.

### 3. Intraday edge is the weakest and easiest to overfit

Intraday bank-stock prediction is noisy. A 5-minute candle model can look strong in backtests and fail live because of:
- spread
- slippage
- delayed yfinance fallback
- API latency
- lunch-hour chop
- opening volatility
- sudden news
- low depth near stops

Needed:
- Intraday should have stricter rules than swing:
  - trade only liquid windows
  - avoid first 5-10 minutes unless specifically modeled
  - avoid lunch chop
  - force end-of-day close
  - require real-time Angel price, not delayed fallback
- If Angel One is unavailable, intraday should be FLAT, not yfinance fallback.

### 4. Execution realism is not strong enough

Paper trades that fill at exact current price, exact stop, and exact target can overstate performance. Real trading has:
- bid/ask spread
- slippage
- gap-through stops
- partial fills
- brokerage, STT, GST, exchange charges
- latency between signal and actual entry

Needed:
- Add simulated execution costs:
  - intraday: higher slippage
  - swing/positional: gap risk
  - short trades: borrow/availability realism if applicable
- Paper ledger should record:
  - signal price
  - assumed fill price
  - slippage
  - fees
  - net P&L after costs

Rule:
- Evaluate the system only on net P&L after realistic costs.

### 5. Short-side assumptions need special treatment

Bank stocks can gap hard against short positions due to RBI, earnings, management news, index moves, or global risk-on events. The system allows counter-regime shorts at reduced size, but the short side must be held to a higher standard.

Needed:
- Separate SHORT playbook:
  - tighter event filters
  - lower max hold
  - mandatory market-wide confirmation
  - no short into major support unless breakdown confirmed
  - no positional shorts near earnings/RBI windows
- Track long expectancy and short expectancy separately.

Rule:
- If SHORT expectancy is not independently positive, disable SHORTs first instead of weakening the entire system.

### 6. Regime can be late at turning points

Regime filters protect capital in chop, but they often lag near turning points. HMM/EMA/ADX regimes can say BEAR after most of the fall is done or BULL after the first strong move is already gone.

Loophole:
- A regime gate can block early high-quality reversal trades and permit late trend-following trades.

Needed:
- Add a separate "transition regime" state:
  - early bull recovery
  - early bear breakdown
  - volatility compression
  - post-crash base
- In transition regime:
  - reduce size
  - require stronger S/R confirmation
  - avoid full conviction labels

### 7. Model confidence may not be calibrated to money risk

The system uses model confidence thresholds, but model confidence is not the same as probability of profitable trade after costs.

Needed:
- Reliability curves per model/category.
- Brier score by ticker and category.
- Bucketed outcome study:
  - 55-60% confidence
  - 60-65%
  - 65-70%
  - 70-75%
  - 75%+
- If 70% confidence bucket does not outperform 60% bucket, confidence is not trustworthy for sizing.

Rule:
- Position size should scale only with empirically validated confidence buckets.

### 8. Feature importance can hide unstable edges

If the model depends heavily on features that are often stale or defaulted, it may work in backtest and fail live.

Needed:
- SHAP/feature importance by model.
- Stability analysis:
  - which features drive live predictions?
  - are those features fresh?
  - are those features available for all banks?
  - do they have point-in-time history?
- Disable or penalize any feature that is not reliably available live.

### 9. Correlation risk across five banks is high

Five private banks are not five independent bets. They are highly correlated through Bank Nifty, RBI policy, FII flows, liquidity, and macro risk.

Loophole:
- The system may open multiple "different" trades that are actually one large banking-sector bet.

Needed:
- Sector exposure cap, not just per-name cap.
- Directional net exposure:
  - total long banking exposure
  - total short banking exposure
  - net long/short beta
- If three banks fire the same direction, choose the best one or reduce all sizes.

Rule:
- Full-size exposure should go to the best risk-adjusted setup, not every setup that passes.

### 10. News and fundamentals are not yet strong enough for positional conviction

Fundamentals are currently missing locally, and news can be noisy. Positional trades need stronger slow-data quality than intraday trades.

Needed:
- Positional category should require:
  - fresh fundamentals
  - latest earnings date
  - no unresolved high-priority announcement
  - macro not stale
  - clear sector regime
- If fundamentals are missing, positional should be observation-only.

### 11. No "do not trade" market-wide kill switch yet

Some days are simply bad trading days:
- budget day
- surprise RBI action
- election shock
- global crash
- exchange/feed instability
- VIX explosion
- major bank earnings cluster

Needed:
- Manual and automated kill switch:
  - `risk_state = HALT`
  - reason
  - expiry time
  - manual reset required for severe events
- Dashboard and Telegram must show the halt clearly.

### 12. The system lacks a post-trade learning loop that changes behavior

It records outcomes, but the plan should ensure outcomes change the system.

Needed:
- Weekly review that answers:
  - Which bank is losing?
  - Which category is losing?
  - Which gate passed too easily?
  - Which confidence bucket failed?
  - Which regime produced losses?
  - Did stops fail or entries fail?
- Automatically reduce size or disable a category when recent expectancy turns negative.

## Additional Improvement Plan

### A. Add an Expectancy Dashboard

Add a dashboard/report that shows:
- expectancy per bank
- expectancy per category
- expectancy per direction
- expectancy per regime
- expectancy by confidence bucket
- average R captured
- max adverse excursion before win
- max favorable excursion before loss

This should be more important than raw accuracy.

### B. Add a Trade Selection Layer

Currently, every passing category can open. A more professional system should rank all passing trades and choose only the best risk-adjusted opportunities.

Selection score should include:
- expected R
- confidence bucket quality
- liquidity
- regime match
- S/R quality
- source freshness
- correlation with existing positions
- current portfolio heat

Policy:
- In SMALL mode: max 1 active trade.
- In GROWING mode: max 2 active trades, no duplicate same-bank categories unless strongly justified.
- In FULL mode: max 3-5 active trades, but sector cap always applies.

### C. Add Observation-Only Shadow Mode

Before real capital:
- Run all signals.
- Do not open even paper trades automatically for a week if healthcheck fails.
- Compare theoretical signal price vs realistic fill price.
- Record missed trades and rejected trades.

Shadow mode helps identify whether the system is creating good trades or just many alerts.

### D. Build a "Why Did We Lose?" Analyzer

For every losing trade, classify:
- bad regime
- bad entry
- stop too tight
- target too ambitious
- news shock
- macro shock
- model reversal ignored
- poor liquidity/slippage
- data stale
- correlated sector drawdown

After 20 losses, patterns will show where the system really fails.

### E. Use Walk-Forward and Purged Validation

For trading data, normal cross-validation can overstate edge. Use:
- walk-forward validation
- purged time-series split
- embargo periods around labels
- out-of-sample year holdouts
- post-2024/post-2025 stress tests

No model should be trusted only because it has good CV accuracy.

### F. Add Market Regime Playbooks

Each regime should have explicit behavior:
- Bull trending:
  - prefer pullback longs
  - avoid low-quality shorts
- Bear trending:
  - prefer rallies into resistance for shorts
  - reduce long size
- High volatility:
  - smaller size
  - fewer trades
  - wider stops or no trade
- Choppy:
  - no new trades
  - only manage exits
- Transition:
  - probe size only
  - require fresh confirmation

### G. Add Liquidity and Slippage Gate

Before opening any trade:
- spread must be acceptable
- volume must be above threshold
- position size must be small relative to recent volume
- no trade if data provider is delayed for intraday

### H. Add Event Risk Calendar

Track and block/reduce around:
- RBI policy
- bank earnings
- budget
- US Fed events
- major inflation prints
- election results
- monthly expiry
- bank-specific management/regulatory events

Event risk should affect both entries and exits.

## The Deepest Loophole

The deepest loophole is not one Python bug. It is this:

The system can pass many gates without proving that the specific trade has positive expectancy after costs, in the current regime, for that bank, in that category, with fresh data.

That is the standard that matters.

Therefore, after fixing wiring and data contamination, the next goal is not "more signals." The next goal is "fewer but proven signals." The system should become more willing to say FLAT, more strict about stale data, and more selective when multiple banks fire together.

## Final Trader's Verdict

If Phase 0 through Phase 3 are implemented, QUANT EDGE can become a reliable paper-trading engine.

If Phase 4 through Phase 7 and the senior-trader improvements above are implemented, it can become a serious decision-support system.

For real capital, the required standard is higher:
- clean data lineage
- no cross-bank contamination
- realistic execution accounting
- positive expectancy by bank/category/direction
- strict risk kill switch
- proven behavior across at least 50-100 clean closed trades
- no unresolved critical audit findings

Until then, the system should stay in paper mode. The job now is not to make it louder. The job is to make it harder for a bad trade to get through.
