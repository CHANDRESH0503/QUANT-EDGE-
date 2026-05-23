#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# QUANT EDGE — one-shot installer for Ubuntu 22.04 / 24.04 LTS (DigitalOcean
# droplets, AWS Lightsail, etc).
#
# Usage (as a sudo-capable user — NOT root, see DEPLOY.md):
#   curl -fsSL https://your-repo/install.sh | bash
#   # or, after cloning:
#   bash install.sh
#
# What it does (idempotent — safe to re-run):
#   1. apt update + system packages (Python 3.11, build tools, ta-lib C lib,
#      nginx, sqlite, fonts for playwright)
#   2. Creates a Python virtualenv at ./.venv
#   3. Installs every Python package from requirements.txt
#   4. Downloads playwright's chromium browser
#   5. Caches the FinBERT model (~440 MB) so first signal cycle isn't slow
#   6. Initialises the SQLite DB
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "▶ [1/6] System packages"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    build-essential gcc g++ make cmake pkg-config \
    libssl-dev libffi-dev libxml2-dev libxslt1-dev libjpeg-dev zlib1g-dev \
    libta-lib0-dev \
    sqlite3 libsqlite3-dev \
    nginx \
    git curl wget ca-certificates \
    fonts-liberation libnss3 libatk1.0-0 libatk-bridge2.0-0 libxcomposite1 \
    libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 \
    libcairo2 libasound2 \
    supervisor

echo "▶ [2/6] Python virtualenv (.venv)"
if [ ! -d ".venv" ]; then
    python3.11 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

echo "▶ [3/6] Python dependencies (requirements.txt)"
# Install torch CPU build first — pulling the default GPU wheel wastes ~2 GB
# of disk on a CPU-only droplet.
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1+cpu
pip install --no-cache-dir -r requirements.txt

echo "▶ [4/6] Playwright chromium"
python -m playwright install chromium
python -m playwright install-deps chromium 2>/dev/null || true

echo "▶ [5/6] Pre-cache FinBERT model"
python - <<'PY'
import os, logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from transformers import pipeline
print("Downloading ProsusAI/finbert (~440 MB)…")
pipeline("sentiment-analysis", model="ProsusAI/finbert", revision="main", device=-1)
print("✓ FinBERT cached")
PY

echo "▶ [6/6] Initialise SQLite schema"
mkdir -p database logs models
python -c "from data.news_fetcher import NewsFetcher; NewsFetcher()" || true

echo ""
echo "✅ Install complete."
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  cp .env.example .env       # then edit API keys"
echo "  uvicorn dashboard_api:app --host 127.0.0.1 --port 8000"
echo ""
echo "For production (systemd + nginx) see DEPLOY.md"
