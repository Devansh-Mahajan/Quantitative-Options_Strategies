# Binance Quant Bot v2.0

A self-learning, GPU-accelerated crypto trading system for Binance. Trades spot, USDM futures perpetuals, and European options (BTC/ETH). Combines 8 dynamic strategies with HMM regime detection, LSTM+Transformer price prediction, and a PPO reinforcement-learning strategy allocator. Retrains itself every weekend on fresh data.

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        Orchestrator (async)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │  WebSocket   │  │  ML Models   │  │  Strategy Suite     │  │
│  │  Streams     │  │  HMM + LSTM  │  │  8 active strategies│  │
│  │  (klines/    │  │  + RL Alloc  │  │  momentum           │  │
│  │   orderbook/ │  │  + GARCH     │  │  mean_reversion     │  │
│  │   liquidations│ │  + IV Surface│  │  funding_arb        │  │
│  └──────┬───────┘  └──────┬───────┘  │  basis_trade        │  │
│         │                 │          │  pairs_arb           │  │
│  ┌──────▼─────────────────▼───────┐  │  options_vol        │  │
│  │        Risk Engine             │  │  order_flow         │  │
│  │  VaR / CVaR / Drawdown Guard  │  │  breakout           │  │
│  │  Position Sizer (Kelly/Vol)    │  └─────────────────────┘  │
│  └──────────────┬────────────────┘                             │
│                 │                                               │
│  ┌──────────────▼────────────────┐                             │
│  │       Execution Layer         │                             │
│  │  Limit→Market fallback        │                             │
│  │  Auto-reprice, TWAP           │                             │
│  └───────────────────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘

Parallel processes:
  run_bot.py          — main trading loop (60s cycles)
  risk_monitor.py     — independent risk alerts (60s poll)
  weekend_training.py — weekly self-retraining (cron Saturday 01:00 UTC)
```

---

## ML Models

| Model | Purpose | Algorithm |
| ----- | ------- | --------- |
| `RegimeHMM` | 4-state market regime detection | Gaussian HMM (hmmlearn) |
| `CryptoPricePredictor` | 1h/4h/24h price direction + magnitude | BiLSTM + Multi-head Attention |
| `RLAllocator` | Dynamic strategy weight allocation | PPO (stable-baselines3) |
| `GARCHVolModel` | Conditional volatility (position sizing) | GARCH(1,1) (arch) |
| `IVSurface` | Options implied vol surface | 2D Linear interpolation |

**GPU**: All PyTorch models train and infer on CUDA (RTX 5090). Weekend training uses full GPU capacity for LSTM + RL.

---

## Strategies

| Strategy | Market | Regime | Edge |
| -------- | ------ | ------ | ---- |
| `momentum` | Futures | Bull/Bear | EMA crossover + MACD + volume |
| `mean_reversion` | Futures | Ranging | Bollinger Band + RSI extremes |
| `funding_arb` | Futures + Spot | Any | Collect extreme funding rates |
| `basis_trade` | Spot + Futures | Any | Spot-futures basis convergence |
| `pairs_arb` | Futures | Any | BTC/ETH Z-score mean reversion |
| `options_vol` | Options | Ranging/Volatile | Short/long straddles on IV/RV ratio |
| `order_flow` | Futures | Any | Buy/sell imbalance + VWAP deviation |
| `breakout` | Futures | Bull/Volatile | Donchian channel + volume surge |

The **PPO RL Allocator** dynamically reweights these strategies each cycle based on current market state and portfolio performance.

---

## Broker Selection — Alpaca vs Binance

The single flag that controls everything is **`BROKER`** in your `.env` file.
The file that reads it is [bot/config.py](bot/config.py) (`cfg.broker`).
The factory that acts on it is [exchange/\_\_init\_\_.py](exchange/__init__.py).

| `BROKER=` | Client used | What works | What doesn't |
| --------- | ----------- | ---------- | ------------ |
| `alpaca` | [exchange/alpaca_client.py](exchange/alpaca_client.py) | Spot crypto (BTC/ETH/SOL…), momentum, mean-reversion, pairs-arb, order-flow, breakout, all ML models | Funding-arb, basis-trade, options-vol (return zero signals — safe), real futures leverage |
| `binance` | [exchange/client.py](exchange/client.py) | Everything — futures, options, funding rates, OI, liquidations | Nothing (full feature set) |

### Switching brokers

Open [`.env`](.env) and change one line:

```dotenv
# Test on Alpaca paper trading (no real money, uses existing Alpaca account)
BROKER=alpaca
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
ALPACA_PAPER=true

# Switch to Binance when ready
# BROKER=binance
# BINANCE_API_KEY=your_key
# BINANCE_API_SECRET=your_secret
# BINANCE_TESTNET=true   # start on testnet first
```

### Install Alpaca dependency

```bash
pip install -e ".[alpaca]"
# or just:
pip install alpaca-py
```

### What to test on Alpaca

Alpaca paper trading gives you $100,000 simulated USD and real market data,
making it ideal for verifying:

- Strategy signal generation (momentum, mean-reversion, breakout, pairs, order-flow)
- Order placement, reprice logic, TWAP
- Risk guard circuit breakers (drawdown halt, daily loss limit)
- ML model inference (HMM regime, LSTM predictions, RL allocator weights)
- Discord/Telegram notifications
- Weekend training pipeline (same code, uses Alpaca REST for historical data)

Strategies that silently produce **no signals** on Alpaca (futures-only logic):

- `funding_arb` — requires perpetual funding rates (always 0.0 on Alpaca)
- `basis_trade` — requires spot-futures basis
- `options_vol` — requires Binance EAPI options

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo>
cd <repo>
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[alpaca]"         # for Alpaca paper trading (BROKER=alpaca)
# For GPU (RTX 5090 / CUDA 12):
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. Configure credentials

```bash
cp .env.example .env
nano .env    # set BROKER=alpaca, fill ALPACA_API_KEY / ALPACA_API_SECRET
```

The default is **`BROKER=alpaca`** with **`ALPACA_PAPER=true`** — simulated money
on real market data.  Switch to `BROKER=binance` once you are satisfied.

### 3. Pre-flight check

```bash
system-check
# or:
python -m scripts.system_check
```

### 4. Run the bot

```bash
run-bot
# or:
python -m scripts.run_bot
```

### 5. Run the risk monitor (separate terminal / process)

```bash
risk-monitor
# or:
python -m scripts.risk_monitor
```

---

## Ubuntu Server Deployment

### Directory layout

```text
/opt/binance-bot/
├── .env                    # credentials (chmod 600)
├── .venv/                  # virtualenv
├── models/
│   ├── live/               # models used by run_bot.py
│   └── training/           # output of weekend training
├── logging/
│   ├── bot.log
│   └── errors.log
└── .runtime/
    └── bot_state.db        # SQLite state
```

### systemd services

**`/etc/systemd/system/binance-bot.service`**

```ini
[Unit]
Description=Binance Quant Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/binance-bot
EnvironmentFile=/opt/binance-bot/.env
ExecStart=/opt/binance-bot/.venv/bin/run-bot
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal
# Allow access to CUDA GPU
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=multi-user.target
```

**`/etc/systemd/system/binance-risk-monitor.service`**

```ini
[Unit]
Description=Binance Bot Risk Monitor
After=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/binance-bot
EnvironmentFile=/opt/binance-bot/.env
ExecStart=/opt/binance-bot/.venv/bin/risk-monitor
Restart=on-failure
RestartSec=60
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable binance-bot binance-risk-monitor
sudo systemctl start binance-bot binance-risk-monitor
sudo systemctl status binance-bot
```

View logs:

```bash
journalctl -u binance-bot -f
journalctl -u binance-risk-monitor -f
```

---

## Crontab Configuration

Edit crontab with `crontab -e` as the bot user:

```cron
# ============================================================
# Binance Quant Bot — Crontab Configuration
# All times are UTC. Adjust path to your virtualenv.
# ============================================================

SHELL=/bin/bash
VENV=/opt/binance-bot/.venv/bin
BOT=/opt/binance-bot

# --- Weekend self-retraining (Saturday 01:00 UTC) ---
# Trains HMM, LSTM, GARCH, and RL allocator on fresh data.
# Deploys new models to models/live/ when training completes.
0 1 * * 6  cd $BOT && $VENV/weekend-train >> $BOT/logging/training.log 2>&1

# --- Daily system health check (every day 00:05 UTC) ---
5 0 * * *  cd $BOT && $VENV/system-check >> $BOT/logging/healthcheck.log 2>&1

# --- Log rotation (daily, keep 30 days) ---
# Handled automatically by TimedRotatingFileHandler in bot/logger.py

# --- Restart bot service if not running (watchdog every 5 min) ---
# Uncomment if NOT using systemd (e.g. tmux / screen setup):
# */5 * * * *  systemctl is-active --quiet binance-bot || systemctl restart binance-bot
```

### tmux-based deployment (alternative to systemd)

```bash
# Create a persistent tmux session
tmux new-session -d -s bot
tmux new-window -t bot:0 -n trading
tmux new-window -t bot:1 -n risk
tmux new-window -t bot:2 -n logs

# Start in each window
tmux send-keys -t bot:0 'cd /opt/binance-bot && source .venv/bin/activate && run-bot' Enter
tmux send-keys -t bot:1 'cd /opt/binance-bot && source .venv/bin/activate && risk-monitor' Enter
tmux send-keys -t bot:2 'tail -f /opt/binance-bot/logging/bot.log' Enter

# Attach to view
tmux attach -t bot
```

---

## Risk Controls

All limits are set in `.env` and enforced in real-time:

| Control | Default | Description |
| ------- | ------- | ----------- |
| `MAX_RISK_PER_TRADE` | 2% | Max equity per single signal |
| `MAX_PORTFOLIO_RISK` | 20% | Max total gross exposure |
| `DAILY_LOSS_LIMIT` | 5% | Auto-halt trading for the day |
| `MAX_DRAWDOWN` | 15% | Auto-halt until manual resume |
| `MAX_LEVERAGE` | 10x | Hard leverage cap on futures |
| `KELLY_FRACTION` | 0.25 | Quarter-Kelly position sizing |

The `RiskGuard` will send Discord/Telegram alerts at 80% of each threshold, and halt trading at 100%.

---

## Weekend Training Internals

```text
Saturday 01:00 UTC — cron triggers weekend_training.py
  │
  ├─ [1] Download 90 days × 10 symbols × 1h candles (~90,000 bars)
  ├─ [2] Retrain HMM (4-state regime detector) per symbol
  ├─ [3] Retrain BiLSTM price predictor per symbol (GPU, ~80 epochs)
  ├─ [4] Refit GARCH(1,1) volatility model per symbol
  ├─ [5] Retrain PPO RL allocator (200k timesteps, GPU)
  └─ [6] Deploy: copy models/training/ → models/live/

Bot auto-reloads new models on next startup (Sunday 00:00 UTC restart via cron/watchdog).
```

---

## Notifications

Set in `.env`:

- **Discord**: `DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...`
- **Telegram**: `TELEGRAM_BOT_TOKEN=...` + `TELEGRAM_CHAT_ID=...`

Alerts are sent for: startup/shutdown, trades executed, risk threshold warnings, drawdown halts, training completion.

---

## Environment Variables Reference

See [.env.example](.env.example) for the full annotated list.

---

## Disclaimer

This bot trades real money on live markets. Cryptocurrency trading carries substantial risk of loss. The authors provide no guarantee of profitability. Always:

- Start on **testnet** (`BINANCE_TESTNET=true`)
- Run paper trading for at least 2–4 weeks before going live
- Monitor risk metrics daily
- Never risk capital you cannot afford to lose
