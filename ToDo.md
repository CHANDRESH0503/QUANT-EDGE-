 Current Status: 🔴 ACTIVELY BLOCKING — regime=BEAR_TRENDING,─stability=30%──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  
       ---
       GATE 2: RULE FILTER (signals/gate2_rule_filter.py)

       ALL Blocking Conditions:
       1. Has trend: adx ≤ 18 AND |ema_spread| ≤ 0.01% → No clear direction
       2. Volume: volume_ratio < 0.7 → Low volume confirmation
       3. Market direction (regime-aware):
         - BULL: nifty_5d_pct ≤ -3.0% (market crashed)
         - BEAR: nifty_5d_pct ≥ +3.0% (market surged)
       4. Earnings: days_to_earnings ≤ 3 → Hard fail within 3 days
       5. Fundamentals: fundamental_grade in ("POOR", "UNKNOWN") → Blocks trades
       6. High anomaly: anomaly_severity == "HIGH" → Hard fail regardless of other checks
       7. VIX halt: india_vix_level ≥ 28 → Hard fail, flatten all
       8. Macro hostile: macro_score ≤ -0.5
       9. Intermarket headwinds: intermarket_score ≤ -0.4
       10. Rupee collapse: usdinr_5d_pct ≥ 2.0% → USD/INR stressed
       11. RBI day: days_to_rbi ≤ 0 → Soft fail only, reduces size to 0.4× via macro_event_mult

       Pass Requirement: n_passed ≥ 8 of 11 AND no hard_fails

       Data Risks:
       - All 11 checks use .get() with safe defaults (0.0 or True for booleans)
       - CRITICAL BUG: Line 162 evaluates checks.get(k, True) for HARD_FAILS — if a key is missing, it's treated as PASSED. However, the checks dict is built so missing keys won't be checked.
       - N  validation that context dict h s required keys ("regime", "fundamen al_gr de", "anomaly_severity")

       Known Blockers:
       - Feature quality is "DEGRADED" → some features are zero, but Gate 2 doesn't check this
       - If fund mentals table  s empty → fundamental_grade d faults  o "GOOD"
       - If global_snapshots has 0 row  → macro features default to neutral (0.0)

       Current Status: ⚠️   THEORETICAL — dep nds on f a u e data being available

       ---
       GATE 3: UNIVERSE RANK (signals/gate3_universe_rank.py)

       Blocking Condition:
       - Score threshold: ticker_score < MIN_SCORE_THRESHOLD (-5.0) → Sets size_mult = 0.0 (skip this ticker)
       - Note: Gate never returns passed=False; it always passes but may set size_mult=0 for weak stocks

       Critical Data Risk — INDEX OUT OF BOUNDS:
       best_ticker = rankings[0]["ticker"]  # Line 96
       best_name   = rankings[0]["name"]    # Line 97
       best_scor   = rankings[0]["score"]   # Line 98
       BUG: If rankings list is empty (all 5 banks failed to download or have <10 bars):
       - Direct array indexing rankings[0] raises IndexError
       - No guards; the early return only handles empty list after initial ranking attempt

       Data Dependency Issues:
       - Requires successful yfinance download for ^NSEBANK (Bank Nifty) to calculate bn_5d
       - If Bank Nifty is empty: bn_5d = 0.0 (fallback), propagates neutral score
       - Requires 5 banks to rank; if <5 have 10+ bars: only partial rankings, still attempts rankings[0] access
       - HDFCBANK-specific risk: Other banks missing "prod" model variant → only full+technical used; HDFCBANK has prod variant so it's stronger

       Ranking Formula Issue:
       - RSI zone score: rsi_zone * 20 * 0.20 scales to 4.0 max; can skew ranking if other factors are small

       Current Status: 🟡 THEORETICAL HIGH RISK — IndexError if universe fetch fails

       ---
       GATE 4: ML PREDICTOR (signals/gate4_ml_predictor.py)

       Blocking Conditions:
       1. Timeframe conflict: alignment == "F" (LONG vs SHORT conflict) → Hard fail
       2. Primary signal FLAT: All models predict FLAT → No directional edge
       3. Swing confidence too low:
         - BEAR_TRENDING: primary_conf < 0.55 (5pp reduction from 0.60)
         - Other regimes: primary_conf < 0.60
       4. Degenerate fallback: If full model has prob_flat/prob_long/prob_short ≥ 0.95, falls back to technical variant

       Feature Quality Risk — CRITICAL BUG:
       - Gate 4 does NOT check data_quality or zero feature count
       - When feature quality is DEGRADED (>25% features are zero), XGBoost predictions become unreliable
       - All-zero features (structural zeros from missing data sources like DII daily, LLM, social) degrade calibration
       - Models trained on PROD variant (25yr history, clean features) but fed DEGRADED full feature set → model mismatch
       - Result: Confidence boosting happens on unreliable predictions

       Model Loading Order Risk:
       - Tries prod → full → technical fallback
       - HDFCBANK has prod models; other banks only full+technical
       - If prod fails to load silently, uses full (which may be missing some data sources)

       Confidence Threshold Issues:
       - BEAR_TRENDING SHORT signals get 5pp threshold reduction (line 200) to avoid missing shorts
       - But if features are DEGRADED, this reduction is masking poor prediction quality

       Data Access Risks:
       - Line 130: raw = feature_vector.get("raw_features", {}) — safe
       - Lines 171-173: Direct dictionary access swing_pred["signal"], swing_pred["confidence"] — UNSAFE if model.predict() returns incomplete dict
       - If model is never loaded, self.swing_model.predict() may fail

       Current Status: 🔴 ACTIVELY PROBLEMATIC — Feature quality=DEGRADED + no validation gate + unreliable predictions

       ---
       GATE 5: SUPPORT/RESISTANCE (signals/gate5_sr_validator.py)

       Blocking Conditions:
       1. Grade D entry: entry_quality == "D" blocks unless ml_confidence ≥ 0.75 AND alignment in ("A+", "A")
       2. Poor R:R ratio: reward_risk < 2.0 → Size multiplied by 0.5 (becomes more restrictive with other multipliers)

       Size Reduction (Not Hard Blocks):
       - LONG near resistance: size_mult *= 0.75
       - SHORT near support: size_mult *= 0.75
       - Both R:R + position reductions cascade: size_mult *= 0.5 * 0.75... can shrink position dangerously

       Data Risks:
       - Line 56: entry_quality = sr_levels.get("entry_quality", "C") — safe default to C
       - Lines 57-62: All use .get() with safe defaults
       - sr_levels comes from SupportResistanceEngine — if S/R detection fails, all distances are 0.0

       Current Status: ⚠️   CONDITIONAL — Blocks only Grade D entries with low confidence

       ---
       GATE 6: CONFIDENCE + CAPITAL MODE (signals/gate6_confidence.py)

       Hard Blocking Conditions:
       1. Circuit breakers:
         - trading_allowed == False → Hard block
         - monthly_dd_flag == True → Monthly loss limit hit
         - consecutive_loss_halt == True → 3 consecutive losses
       2. Alignment mismatch:
         - SMALL capital: alignment must be A+ only
         - GROWING capital: alignment must be A+ or A
         - FULL capital: alignment must be A+, A, or B
         - Any other alignment (C, F) blocks
       3. Confidence threshold (VIX-adaptive):
         - Base thresholds by capital mode × model type (see THRESHOLDS dict, lines 27-30)
         - VIX > 25: threshold += 10pp (capped at 0.90)
         - VIX 20-25: threshold += 5pp (capped at 0.90)

       Threshold Examples:
       - FULL capital + Swing + VIX=15: need 60% confidence
       - FULL capital + Swing + VIX=26: need 70% confidence (60% + 10pp)
       - SMALL capital + Intraday + VIX=15: need 72% confidence

       Data Risks:
       - Line 58: risk_context.get("trading_allowed", True) — defaults to allow (risky if missing)
       - Line 70: allowed_alignments = self.MIN_ALIGNMENT.get(capital_mode, {"A+", "A", "B"}) — safe default to FULL mode
       - Line 86: india_vix = float(risk_context.get("india_vix", 0.0)) — defaults to 0 (no VIX adjustment if missing)

       Current Status: 🟡 THEORETICAL — Conditional on risk circuit breakers; VIX adjustment works if data present

       ---
       SIGNAL ENGINE (signals/signal_engine.py)

       Pipeline Flow Issues:

       1. Gate 1 called TWICE (lines 168 & 239):
         - First with regime-expected direction (SHORT for BEAR, LONG for BULL)
         - Second with actual ML signal direction
         - Risk: If first pass uses wrong direction assumption, blocks unnecessarily
       2. Feature Quality Never Checked:
         - feature_vector["data_quality"] is built (line 629 in FeatureBuilder) but never used in signal_engine
         - Should trigger a "DEGRADED quality" block before Gate 4
         - Currently ML models run on DEGRADED features without warning
       3. Gate 3 Disqualification Logic (lines 202-204):
         - Only checks g3_ctx.get("disqualified") — if True, returns FLAT
         - But Gate 3 sets size_mult=0.0 for weak stocks without setting disqualified=True
         - Result: Size multiplier is 0 but signal still proceeds, wastes CPU cycles
       4. Size Multiplier Cascade (lines 290-296):
         - Multiplies: g1 * g3 * g4 * g5 * g6
         - If ANY gate sets 0.0 (even as soft rejection), entire position is zeroed
         - But signal still passes through all 6 gates with confidence=60%+ and alignment=A
         - Result: SIGNAL output created with shares=0 (wasteful)
       5. Context Passing to Gate 2 (lines 182-187):
         - fundamental_grade comes from raw.get() — if feature build failed, defaults to "GOOD"
         - anomaly_severity comes from risk_context.get() which is built from anomaly_raw
         - If anomaly detection returns no result, defaults to "LOW"
       6. Gate 1 Re-check Logic (lines 238-248):
         - Re-validates direction post-Gate4, but if BEAR regime blocks LONG signal returned by ML:
         - Returns FLAT with reason "Regime {regime} blocks {signal} signals"
         - But this is redundant if Gate 1 logic is correct — indicates potential inconsistency
       7. Missing Early-Exit for DEGRADED Quality:
       # MISSING IN signal_engine.run():
       if feature_vector.get("data_quality", {}).get("quality") in ("DEGRADED", "POOR"):
           return self._flat("Feature quality degraded — {zero_features} of {total}", ...)

       ---
       KNOWN ACTIVE BLOCKERS (RIGHT NOW)

       ┌────────┬───────────────────────────────────┬─────────────────┬────────────────────────────────────────────────────────────┐
       │  Gate  │             Condition             │     Status      │                           Impact                           │
       ├────────┼───────────────────────────────────┼─────────────────┼────────────────────────────────────────────────────────────┤
       │ Gate 1 │ Stability 30% < 40% threshold     │ 🔴 ACTIVE       │ Blocks all signals in BEAR_TRENDING                        │
       ├────────┼───────────────────────────────────┼─────────────────┼────────────────────────────────────────────────────────────┤
       │ Gate 4 │ Feature quality DEGRADED          │ 🔴 ACTIVE       │ Models unreliable; confidence inflated via fallback        │
       ├────────┼───────────────────────────────────┼─────────────────┼────────────────────────────────────────────────────────────┤
       │ Gate 2 │ Depends on feature data freshness │ ⚠️   CONDITIONAL │ If macro/fundamental fetch fails, defaults to GOOD/NEUTRAL │
       └────────┴───────────────────────────────────┴─────────────────┴────────────────────────────────────────────────────────────┘

       ---
       CRITICAL BUGS FOUND

       1. Gate 3 IndexError Risk (line 96-98): rankings[0] with no bounds check
       2. Gate 4 + Feature Degradation (no validation): Runs ML on DEGRADED features without checking
       3. Signal Engine Size Cascading (line 290-296): Signals proceed with 0 shares (wasteful, not a hard block)
       4. Missing Feature Quality Gate: No early exit when data_quality is DEGRADED or POOR

       ---
       HARDCODED THRESHOLDS THAT MAY BE WRONG

       - Gate 1: MIN_STABILITY = 0.40 (40%) — seems reasonable but currently blocking valid BEAR_TRENDING signals
       - Gate 2: MIN_PASSES = 8 of 11 — allows 3 checks to fail, reasonable
       - Gate 2: vix < 28 — hard threshold, but no soft warning at 25-28 range (Gate 6 compensates)
       - Gate 3: MIN_SCORE_THRESHOLD = -5.0 — only blocks bottom decile; reasonable
       - Gate 5: MIN_REWARD_RISK = 2.0 — requires 2:1 minimum, reasonable but restrictive in choppy markets
       - Gate 6: SMALL capital requires A+ only — reasonable protection for beginners

       ---
       SUMMARY: What's Blocking Valid Signals

       ✅ Working as intended: Gates 1, 2, 5, 6 have proper guards
       🔴 Broken/Missing: Gate 4 + Feature Quality validation
       ⚠️   Fragile: Gate 3 index access, cascading size multipliers
