# Operations & Maintenance Guide

This guide explains how to deploy, operate, and maintain the automated options strategy safely.

---

## 1) Purpose and Scope

This system automates multiple options workflows (wheel, spreads, straddles, hedges) with risk gates and model-driven signal routing.

Use this document for:
- environment setup,
- run procedures,
- health checks,
- troubleshooting,
- maintenance and change management.

---

## 2) Environment Setup

## 2.1 Prerequisites

- Python 3.11+
- `uv` installed
- Alpaca account with options enabled
- API credentials provisioned

## 2.2 Install

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## 2.3 Secrets

Create `.env`:

```env
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
IS_PAPER=true
```

## 2.4 Symbol Universe

Edit:
- `config/symbol_list.txt` (baseline universe)
- optional `config/volatile_symbols.txt` (weekend/special universe override)

Pick liquid names you are comfortable owning.

---

## 3) Runtime Model

Primary entrypoint:

```bash
run-strategy
```

High-level flow:
1. Manage/close existing positions.
2. Recompute account risk and buying power.
3. Pull model signals and macro regime posture.
4. Allocate capital according to risk controls.
5. Deploy eligible strategy modules.
6. Log outcomes and emit alerts.

---

## 4) Day-to-Day Operation

## 4.1 Recommended Command

```bash
run-strategy --strat-log --log-level INFO --log-to-file
```

First clean run:

```bash
run-strategy --fresh-start --strat-log --log-level DEBUG --log-to-file
```

Manage-only mode (no new entries):

```bash
run-strategy --manage-only --strat-log
```

## 4.2 Suggested Schedule

Weekdays during market hours (example):
- 10:00
- 13:00
- 15:30

Use cron/systemd/Kubernetes schedules based on your infrastructure.

---

## 5) Risk Controls You Should Understand

Core knobs in `config/params.py`:
- `RISK_ALLOCATION`
- `MAX_RISK_PER_SPREAD`
- `MAX_SPREADS_PER_SYMBOL`
- `MAX_NEW_TRADES_PER_CYCLE`
- `MIN_SIGNAL_CONFIDENCE`
- `LOW_CONFIDENCE_RISK_MULTIPLIER`
- `PROFIT_TARGET`
- `STOP_LOSS`

Operational interpretation:
- weak signals => smaller fresh deployment,
- high volatility => risk scaling down,
- daily drawdown thresholds => kill switch / reduced sizing behavior.

---

## 6) Monitoring and Health Checks

Daily checks:
1. Did the process run at scheduled times?
2. Are alerts flowing?
3. Any repeated order placement errors?
4. Any stale open orders?
5. Any positions near expiry with no exits?

Log hotspots:
- risk gating messages,
- order execution exceptions,
- market-hours close failures,
- repeated symbol-specific rejects.

---

## 7) Incident Response

If behavior looks unsafe:
1. Switch to `--manage-only` immediately.
2. Optionally use `--fresh-start` to reset positions (paper first; live only with caution).
3. Lower `RISK_ALLOCATION` and `MAX_RISK_PER_SPREAD`.
4. Raise confidence threshold / lower multiplier aggressiveness.
5. Review logs before reenabling new entries.

If API issues occur:
- retry later if transient,
- validate credentials and permissions,
- confirm market session state.

---

## 8) Maintenance Cadence

Daily:
- review runtime logs and alerts.

Weekly:
- refresh symbol universe,
- review per-strategy outcomes,
- run recalibration if using model retraining workflows.

Monthly:
- parameter review and stress test assumptions,
- check dependency updates and breaking changes.

After every code change:
1. Run static checks/compile.
2. Paper trade soak.
3. Promote to live gradually.

---

## 9) Change Management Best Practices

- Make small incremental changes.
- Keep one risk-related change per release where possible.
- Attach before/after metrics (hit-rate, premium capture, drawdown, fill quality).
- Roll back quickly if error rate or risk profile degrades.

---

## 10) Security and Compliance Notes

- Never commit API keys to git.
- Rotate credentials periodically.
- Treat webhook URLs and logs as sensitive.
- Ensure your usage complies with broker and jurisdictional requirements.

---

## 11) Quick Troubleshooting Matrix

- **No trades placed**: check buying power, confidence gates, VIX regime, and symbol eligibility.
- **Too many rejections**: inspect spread/oi/delta filters and market liquidity.
- **Unexpected risk-on behavior**: verify confidence and max-trade settings in `config/params.py`.
- **Orders not closing**: confirm market is open and inspect `safe_close_position` warnings.

---

## 12) Disclaimer

This software does not guarantee profits and can lose money. Test thoroughly in paper mode and use position sizes appropriate for your risk tolerance.

