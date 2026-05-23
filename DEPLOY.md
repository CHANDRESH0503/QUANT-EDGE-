# QUANT EDGE — DigitalOcean Deployment Guide

This guide deploys both the **FastAPI backend** (`dashboard_api.py`) and the
**HTML frontend** (`tradingDashboard.html`) to a single DigitalOcean droplet.
The HTML is already served by FastAPI at `/`, so a single Nginx reverse proxy
fronts everything.

---

## 0. What you'll end up with

```
Internet  ──→  Nginx :80/:443  ──→  uvicorn 127.0.0.1:8000  ──→  FastAPI
                                                              ├── /            → tradingDashboard.html
                                                              ├── /api/live    → JSON
                                                              └── /api/chart   → PNG via Playwright

                                    orchestrator.py (separate systemd service)
                                    └── ONE process, all 5 banks, every 15 min
                                        ├── shared FinBERT pipeline (~1.6 GB)
                                        ├── shared global data fetchers
                                        ├── reads database/trading.db
                                        └── writes signals to Telegram
```

Two systemd services, one Nginx server block, one SQLite file, one TLS cert.

**Why one orchestrator process instead of five `main.py --ticker=...` processes?**
Five processes load FinBERT five times — ~8 GB of RAM for nothing. The orchestrator
loads it once (class-level cache in `FinBERTSentiment._shared_pipeline`) and routes
every bank through it sequentially. Resident memory: ~2.0–2.5 GB total for all five
banks combined, instead of ~8 GB.

---

## 1. Provision the droplet

DigitalOcean → Create Droplet:

| Setting | Value |
|---|---|
| Image | **Ubuntu 24.04 (LTS) x64** |
| Plan | **Basic · Regular · 4 GB RAM / 2 vCPU / 80 GB SSD** ($24/mo) |
| Region | **BLR1 (Bangalore)** — lowest latency to NSE feeds |
| Authentication | SSH key (paste your `~/.ssh/id_ed25519.pub`) |
| Hostname | `quantedge` |
| Firewall | Add a DO firewall (Networking → Firewalls): inbound 22, 80, 443 only |

> **Why 4 GB?** FinBERT alone uses ~1.6 GB RSS once loaded. XGBoost training
> spikes briefly to ~2 GB. The $12/mo 2 GB plan OOMs during model retrain.

---

## 2. First login + base hardening

```bash
ssh root@<droplet-ip>

# create a non-root user
adduser quant                       # set a password
usermod -aG sudo quant
rsync --archive --chown=quant:quant ~/.ssh /home/quant

# disable root SSH + password auth
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart ssh

# enable OS firewall
ufw allow OpenSSH && ufw allow 'Nginx Full' && ufw --force enable
```

Log out and back in as the new user:
```bash
ssh quant@<droplet-ip>
```

---

## 3. Pull the code

```bash
cd ~
git clone https://github.com/<you>/TradingBot.git
cd TradingBot
```

If your repo is private, either:
- Use a deploy key: `ssh-keygen -t ed25519 -f ~/.ssh/deploy_key`, add the
  public key as a deploy key on GitHub, then clone via `git@github.com:...`.
- Or `rsync` the project up from your laptop:
  ```bash
  rsync -av --exclude=.venv --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='database/trading.db-journal' \
        ./ quant@<droplet-ip>:~/TradingBot/
  ```

---

## 4. Run the installer

```bash
bash install.sh
```

This single command:
- Installs Python 3.11, ta-lib, nginx, sqlite, supervisor, chromium deps
- Creates `.venv/` and installs every Python package from `requirements.txt`
- Installs torch's CPU wheel (saves ~2 GB vs the default GPU wheel)
- Downloads playwright's chromium
- Pre-caches the FinBERT model (~440 MB) so the first signal cycle is fast
- Initialises the SQLite schema

Takes ~8–12 min on a 2 vCPU droplet.

---

## 5. Configure secrets

```bash
cp .env.example .env 2>/dev/null || touch .env
nano .env
```

Set at minimum:
```env
ANTHROPIC_API_KEY=sk-ant-...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=7873846599
ANGEL_API_KEY=...
ANGEL_CLIENT_CODE=...
ANGEL_PASSWORD=...
ANGEL_TOTP_SECRET=...
HF_HUB_DISABLE_TELEMETRY=1
```

Then:
```bash
chmod 600 .env
```

---

## 6. systemd: dashboard API (uvicorn)

```bash
sudo tee /etc/systemd/system/quantedge-api.service >/dev/null <<'UNIT'
[Unit]
Description=QUANT EDGE — dashboard FastAPI
After=network.target

[Service]
Type=simple
User=quant
Group=quant
WorkingDirectory=/home/quant/TradingBot
EnvironmentFile=/home/quant/TradingBot/.env
ExecStart=/home/quant/TradingBot/.venv/bin/uvicorn dashboard_api:app \
          --host 127.0.0.1 --port 8000 \
          --workers 2 --proxy-headers --forwarded-allow-ips='*'
Restart=always
RestartSec=3
StandardOutput=append:/home/quant/TradingBot/logs/api.log
StandardError=append:/home/quant/TradingBot/logs/api.err

# Resource limits — FinBERT + chromium can spike memory
MemoryMax=3G
LimitNOFILE=4096

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now quantedge-api
sudo systemctl status quantedge-api --no-pager
```

Sanity check:
```bash
curl -s http://127.0.0.1:8000/api/health
```

---

## 7. systemd: signal pipeline (all 5 banks, one process)

```bash
sudo tee /etc/systemd/system/quantedge-signal.service >/dev/null <<'UNIT'
[Unit]
Description=QUANT EDGE — multi-bank orchestrator (all 5 banks, single process)
After=network.target quantedge-api.service

[Service]
Type=simple
User=quant
Group=quant
WorkingDirectory=/home/quant/TradingBot
EnvironmentFile=/home/quant/TradingBot/.env
ExecStart=/home/quant/TradingBot/.venv/bin/python orchestrator.py
Restart=always
RestartSec=10
StandardOutput=append:/home/quant/TradingBot/logs/signal.log
StandardError=append:/home/quant/TradingBot/logs/signal.err

# All 5 banks share one FinBERT pipeline; 3 GB cap is comfortable headroom
MemoryMax=3G

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now quantedge-signal
```

That single unit runs all 5 banks. Want a custom subset?

```bash
# Edit the service:
sudo systemctl edit quantedge-signal
# Add an override that replaces ExecStart, e.g.:
#   [Service]
#   ExecStart=
#   ExecStart=/home/quant/TradingBot/.venv/bin/python orchestrator.py \
#             --tickers HDFCBANK.NS,ICICIBANK.NS --interval 900
sudo systemctl restart quantedge-signal
```

Sanity check — run one cycle in the foreground first to make sure models
load and Telegram is wired up:
```bash
source .venv/bin/activate
python orchestrator.py --once
```

---

## 8. Nginx reverse proxy + TLS

Buy/point a domain (e.g. `quantedge.yourdomain.com`) at the droplet's IP.

```bash
sudo tee /etc/nginx/sites-available/quantedge >/dev/null <<'NGINX'
server {
    listen 80;
    server_name quantedge.yourdomain.com;

    # Let’s Encrypt http-01 challenge passes through here before TLS is set up
    location /.well-known/acme-challenge/ { root /var/www/html; }

    # Everything else proxies to uvicorn
    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
        client_max_body_size 5m;
    }

    # Gzip for the JSON API — knocks ~25 KB /api/live response down to ~7 KB
    gzip on;
    gzip_types application/json text/css application/javascript text/html;
    gzip_min_length 1024;
}
NGINX

sudo ln -sf /etc/nginx/sites-available/quantedge /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

Visit `http://quantedge.yourdomain.com` — you should see the dashboard.
Now provision TLS:

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d quantedge.yourdomain.com \
     --non-interactive --agree-tos -m you@yourdomain.com --redirect
```

certbot will edit the Nginx config in place and add a renewal cron.

---

## 9. Day-2 operations

| Action | Command |
|---|---|
| Tail API logs | `journalctl -u quantedge-api -f` |
| Tail signal logs | `journalctl -u quantedge-signal -f` |
| Restart API | `sudo systemctl restart quantedge-api` |
| Restart orchestrator | `sudo systemctl restart quantedge-signal` |
| Run one cycle now | `source .venv/bin/activate && python orchestrator.py --once` |
| Run subset of banks | `python orchestrator.py --tickers HDFCBANK.NS,AXISBANK.NS` |
| Memory/CPU | `htop` or `systemctl status quantedge-api` |
| DB backup | `sqlite3 database/trading.db ".backup db-$(date +%F).bak"` |
| Pull updates | `cd ~/TradingBot && git pull && sudo systemctl restart quantedge-api quantedge-signal` |
| Retrain models | `source .venv/bin/activate && python main.py --mode=train` |

### Automated nightly DB backup
```bash
sudo tee /etc/cron.daily/quantedge-backup >/dev/null <<'CRON'
#!/usr/bin/env bash
cd /home/quant/TradingBot
sudo -u quant sqlite3 database/trading.db ".backup database/backups/db-$(date +\%F).bak"
find database/backups/ -name 'db-*.bak' -mtime +14 -delete
CRON
sudo chmod +x /etc/cron.daily/quantedge-backup
sudo -u quant mkdir -p /home/quant/TradingBot/database/backups
```

### Updating dashboard HTML only
The HTML is served from disk by FastAPI's `FileResponse` (no in-memory cache),
so editing `tradingDashboard.html` does **not** require a restart — just
hard-refresh the browser (Cmd-Shift-R).

---

## 10. Common issues

**FinBERT OOM at first signal cycle.** First load is ~1.6 GB. On a 2 GB
droplet this can OOM. Either upgrade to 4 GB, or add 2 GB of swap:
```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**ta-lib build fails.** The `libta-lib0-dev` package isn't in every Ubuntu
release. Build from source:
```bash
wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz
tar -xzf ta-lib-0.6.4-src.tar.gz && cd ta-lib-0.6.4
./configure --prefix=/usr && make && sudo make install
pip install TA-Lib
```

**Playwright chromium missing libs.** Run `python -m playwright install-deps`.

**Angel One DNS fails.** Some droplet regions block Angel One's API; pick BLR1.

**API returns 502 Bad Gateway through Nginx.** uvicorn isn't running. Check:
```bash
sudo systemctl status quantedge-api
journalctl -u quantedge-api -n 100 --no-pager
```

**Frontend loads but `/api/live` hangs.** Usually FinBERT is downloading the
model on first call. Pre-cache it via the installer (step 4) or watch
`journalctl -u quantedge-api -f` for "FinBERT loaded successfully".

---

## 11. Cost estimate

| Item | Monthly |
|---|---|
| Droplet (4 GB / 2 vCPU / 80 GB) — BLR1 | **$24** |
| Domain (.com via Cloudflare/Namecheap) | ~$1 |
| TLS (Let's Encrypt) | $0 |
| DO backups (optional, +20%) | +$4.80 |
| **Total** | **~$25–30/mo** |

Bandwidth: 4 TB included on every droplet. Dashboard payload is ~30 KB —
nowhere near the limit.
