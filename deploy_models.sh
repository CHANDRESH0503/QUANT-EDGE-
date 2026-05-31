#!/usr/bin/env bash
# deploy_models.sh — push locally-trained ML models to the VPS.
#
# WHY THIS EXISTS:
#   models/saved/*.pkl is gitignored (artifacts too large for git), so a plain
#   `git pull` on the VPS ships new CODE but keeps whatever model .pkl files are
#   already on disk. After the 2026-05-27 CR-1A regime retrain this caused a
#   silent, dangerous bug: the VPS ran the *old* pre-fix models that predict
#   LONG in a BEAR regime (and SHORT in BULL). New code, stale models.
#
#   This script rsyncs the trained models so the deployed brain matches the
#   deployed code. Run it whenever you retrain, BEFORE restarting the service.
#
# USAGE:
#   ./deploy_models.sh                 # sync models, then restart signal service
#   ./deploy_models.sh --no-restart    # sync only, restart yourself later
#
set -euo pipefail

VPS_HOST="${VPS_HOST:-root@165.22.220.126}"
VPS_DIR="${VPS_DIR:-/root/TradingBot}"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)/models/saved"
RESTART=1
[[ "${1:-}" == "--no-restart" ]] && RESTART=0

if [[ ! -d "$LOCAL_DIR" ]]; then
    echo "ERROR: $LOCAL_DIR not found — train models first (python3 main.py --mode=train)." >&2
    exit 1
fi

n_models=$(find "$LOCAL_DIR" -name '*.pkl' | wc -l | tr -d ' ')
echo ">> Syncing $n_models model files ($(du -sh "$LOCAL_DIR" | cut -f1)) to $VPS_HOST:$VPS_DIR/models/saved/"

# --delete so models removed locally are removed on the VPS too (no stale
# variants lingering). Checksums (-c) guard against clock-skew false matches.
rsync -avzc --delete \
    --include='*.pkl' --include='*/' --exclude='*' \
    "$LOCAL_DIR/" "$VPS_HOST:$VPS_DIR/models/saved/"

echo ">> Model sync complete."

if [[ "$RESTART" == "1" ]]; then
    echo ">> Restarting quantedge-signal service..."
    ssh "$VPS_HOST" "systemctl restart quantedge-signal && sleep 3 && systemctl is-active quantedge-signal"
    echo ">> Done. Tail logs with: ssh $VPS_HOST 'journalctl -u quantedge-signal -f'"
else
    echo ">> Skipped restart (--no-restart). Restart manually when ready."
fi
