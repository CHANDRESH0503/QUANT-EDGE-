# QUANT EDGE — Current Progress

## System
Python 3.10+ · SQLite · Telegram · Railway.app · Demo/paper-trading phase

## Universe
5 private banks: HDFCBANK.NS, ICICIBANK.NS, KOTAKBANK.NS, AXISBANK.NS, INDUSINDBK.NS  
SBI excluded (PSU dynamics differ). Top-2 ranked per 15-min cycle; best signal wins.

## All 12 Tier 1+2 Tasks — COMPLETE

| # | Task | File(s) |
|---|------|---------|
| 1.5 | `.env.example` + `.gitignore` verified | `.env.example` |
| 2.7 | `signal_uuid` FK + UUID in signal_engine | `db_setup.py`, `signal_engine.py` |
| 1.1 | Outcome tracker auto-writes `signal_outcomes` from `close_position()` | `risk/outcome_tracker.py`, `risk/exit_engine.py` |
| 1.3 | Multi-asset: UNIVERSE config, DB ticker columns, `run_universe()`, exposure caps | `config.py`, `db_setup.py`, all fetchers, `signal_engine.py`, `position_sizer.py` |
| 1.2 | `CalibratedClassifierCV` isotonic on all 3 models | `models/train_*.py`, `gate_4_ml_predictor.py` |
| 1.4 | Gap-fill risk in live exit_engine + backtest engine | `risk/exit_engine.py`, `backtest/engine.py` |
| 2.1 | Event-aware position sizing (earnings + RBI MPC multiplier) | `processing/earnings_predictor.py`, `risk/position_sizer.py` |
| 2.2 | VIX-spike auto-flatten via circuit_breaker | `risk/circuit_breaker.py`, `scheduler/task_runner.py` |
| 2.3 | VIX-adaptive confidence thresholds in gate6 | `signals/gate_6_confidence.py` |
| 2.5 | Drawdown-adaptive sizing (4-tier DD table) | `risk/position_sizer.py` |
| 2.4 | Multi-target exit ladder: 50% at 1R, 30% at 2R, chandelier trail 20% | `risk/exit_engine.py` |
| 2.6 | PSI drift monitor + Brier score model health | `monitoring/drift_monitor.py` |

## Live Dashboard — COMPLETE
- `dashboard_api.py`: FastAPI `/api/live` endpoint, all 5 banks in universe, AXISBANK + INDUSINDBK quotes added, `days_held` + `ticker` in response
- `tradingDashboard.html`: 50+ `id` attributes added, full `fetchLive()` polling every 15s, real chart from API OHLCV, dynamic ticker strip, news, universe ranking (5 banks only — SBIN/BANDHANBNK removed)
- Run: `uvicorn dashboard_api:app --host 0.0.0.0 --port 8000` then open `tradingDashboard.html`

## Key Architecture
- 6-gate pipeline → `signals/signal_engine.py:run_universe()`
- 3 calibrated XGBoost models (swing/intraday/positional)
- `signal_uuid` links `feature_snapshots → open_trades → closed_trades → signal_outcomes`
- `PRAGMA foreign_keys=ON` on every SQLite connection

## Deferred (after 8–12 week demo phase)
Bank Nifty hedging · shadow-mode deployment · LLM devil's-advocate · Kelly sizing · pooled multi-ticker model training
