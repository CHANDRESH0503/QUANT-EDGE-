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
| 1 | Regime | per-bank | CHOPPY → block. Stability <25% → block. Direction never vetoes per-category (informational). |
| 2 | Rule Filter | per-bank | 11 checks, need 8+. Hard fails: earnings <3d, VIX ≥28, anomaly HIGH. Fundamentals DEFAULTED → soft-pass + bubble `fundamentals_stale`. |
| 3 | Universe Rank | global | Ranks 5 banks; size_mult by rank. Disqualifies on falling-knife score <−5. 2-min TTL cache. |
| 4 | ML Models | per-bank | Passes iff ≥1 model non-FLAT. Alignment INFORMATIONAL — boosts conf+size, never vetoes. |
| 5 | S/R Validator | per-category | Entry quality A/B/C/D per direction (SHORT inverts R:R). Grade-D override: conf ≥62%. |
| 6 | Confidence | per-category | FULL: swing 60% / pos 55% / intra 65%. Boosts: VIX +5/+10pp · DEGRADED features +5pp · positional + DEFAULTED fundamentals +5pp · capped +15pp. Reads `circuit_breaker_level` directly. |

**Data quality gate** (pre-Gate-4): POOR → hard FLAT. DEGRADED → +5pp Gate 6 threshold. `macro_stale=1` from `GlobalFetcher.is_stale(30h)` floor-clamps quality to DEGRADED.

**Capital-mode `allowed_tf` enforced** at signal_engine AND `_open_paper_trades` (SMALL=swing · GROWING=swing+intra · FULL=all). Disallowed categories return structured FLAT with reason for dashboard.

**Counter-regime trades** (LONG in BEAR / SHORT in BULL) allowed at 0.5× size.

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

Exposure caps (live-enforced): 40% per-name · 80% total. VIX >20: +5pp · >25: +10pp · ≥28: HALT_AND_FLATTEN.
Exit: 1R→book 50% | 2R→book 30% | trail 20% chandelier (3×ATR).
DD multipliers (mode-aware, shared with `CircuitBreaker.THRESHOLDS`): SMALL halt −8% / GROWING −12% / FULL −15%.

---

## Latest Audit Fixes (2026-05-27)

**CR-1A — Regime/label horizon mismatch (RETRAIN-REQUIRED).** `classify_regime_row` no longer reads `returns_20d`. Regime is now purely ADX + EMA spread. Previously, BEAR was labeled from past-20d-down while ML labels used forward-5d returns — the model learned mean-reversion and surfaced LONGs in BEAR and SHORTs in BULL. Vectorised training path aligned. All 5 banks retrained 2026-05-27.

**CR-1B — Regime rules path bug.** `RegimeDetector.detect()` returns rules under `regime_result["rules"]`, but `feature_builder.py` was reading `regime_result.get("trade_long")` at the top level, silently defaulting `trade_long=True, trade_short=False` regardless of regime. Fixed to read from the nested `rules` dict.

**CR-2 — Per-ticker `_opened_today` flag.** Was a single bool — once HDFCBANK fired the 09:15 alert, ICICI/Kotak/Axis/IndusInd were suppressed for the rest of the day. Now `Dict[str, bool]` keyed by ticker; `_send_market_open_alerts` no longer breaks after first non-FLAT bank.

**CR-3 — Reversal alert 30-min cooldown.** `_sent_reversals` was keyed by `(ticker, cat, type)` with no time floor — model oscillations spammed Telegram. Cooldown via `REVERSAL_COOLDOWN_SEC=1800` (30 min). Position-close still clears entries for that ticker.

**E1 — Intermarket cache.** `_fetch_peer_data` and `_sector_rotation` in `processing/intermarket.py` were called inside every per-bank FeatureBuilder, doing 7 yfinance calls × 5 banks = 35/cycle. Now class-level TTL cache (`_INTERMARKET_CACHE_TTL_SEC=300`) → 7/cycle. Same outputs, ~5× yfinance reduction.

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

### High Priority
1. **5 alpha features** — wire into `feature_builder.py` + `*_FEATURES_PROD`, retrain Sunday:
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

## Deferred (long-horizon)
Bank Nifty futures hedging · shadow-mode deployment · Kelly sizing · `scale_pos_weight` balancing · pooled multi-ticker model with ticker embedding · P3 features (LLM earnings score, social contrarian, Google Trends alt data).
