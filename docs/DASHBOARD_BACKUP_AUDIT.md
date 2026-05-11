# Dashboard Backup Audit

## Summary

The frontend backups in `dashboard/static` are not complete functional backups. They are visual snapshots / experiments and should not be used as restore sources for the trading UI.

The current working legacy frontend is:

- `dashboard/static/index.html`

It preserves the original backend-facing functionality and adds the terminal visual shell plus the Digital Twin tab.

## File Inventory

| File | Lines | Nav Tabs | Tab Panels | API Call Sites | WebSocket Refs | Can Restore Full UI? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `dashboard/static/index.html` | 6309 | 17 | 17 | 63 | 3 | Yes |
| `dashboard/static/index.html.bak` | 808 | 16 | 1 | 1 | 0 | No |
| `dashboard/static/index.terminal.bak` | 663 | 0 | 0 | 0 | 1 | No |

## Backup Findings

### `dashboard/static/index.html.bak`

This file looks like the pasted "Elite Terminal" visual mock. It has the nav bar labels, but only the Live tab panel exists.

Missing from this file:

- Options tab panel and loader.
- Stocks/Pairs tab panel and loader.
- Trade Odds tab panel and loader.
- Backtests, Strategy Params, Risk, Sim Lab, Strategy Lab, Forward Test.
- ML Training, Intelligence, Trade Analysis.
- Command Center, GS Quant, Live Logs.
- Real `TAB_INIT` map for all tabs.
- Real `loadLive`, `connectLiveWs`, and `triggerMasterRecal`; these are stubs.

Conclusion: useful as a visual reference only.

### `dashboard/static/index.terminal.bak`

This is a smaller React/CDN terminal experiment. It has no legacy tab panels and no full dashboard loader map.

Conclusion: useful as an aesthetic reference only.

## Current Frontend Coverage

`dashboard/static/index.html` currently has:

- 17 nav tabs and 17 matching panels.
- 63 API call sites.
- 3 websocket refs.
- 27 Chart.js/canvas surfaces.
- Full tab loader registry.
- CLI routing.
- Digital Twin extension tab.

Required legacy loaders present:

- `loadLive`
- `connectLiveWs`
- `loadOptions`
- `loadStocks`
- `loadOdds`
- `loadBacktests`
- `loadParams`
- `loadRisk`
- `loadSimLab`
- `loadStratLab`
- `loadTraining`
- `loadIntelligence`
- `loadTradeAnalysis`
- `loadCmdCenter`
- `loadGsQuant`
- `initLogs`
- `triggerMasterRecal`

## Backend Surface

`dashboard/server.py` exposes 87 routes/websocket endpoints, including:

- Core account/equity: `/api/health`, `/api/config`, `/api/equity`, `/api/live/tape`, `/api/positions`, `/api/daily_pnl`.
- Execution truth source: `/api/execution/ledger`, `/api/execution/quality`.
- Risk: `/api/risk/snapshot`, `/api/risk/engine`, `/api/risk/heatmap`, `/api/risk/correlations`, `/api/risk/var_surface`, `/api/risk/events`.
- Strategy/config control: `/api/strategies/*`, `/api/config/risk`, `/api/config/universe`, `/api/trading/kill_switch`.
- Research/backtesting: `/api/backtest/*`, `/api/forwardtest/*`, `/api/permutations/*`, `/api/training/*`.
- Market/research panels: `/api/options/*`, `/api/stocks/overview`, `/api/trades/odds`, `/api/research/desk`, `/api/intelligence`, `/api/trades/analysis`.
- GS Quant analytics: `/api/gsquant/*`.
- Model zoo: `/api/model_zoo/tft/{symbol}`, `/api/model_zoo/xgb/{symbol}`, `/api/model_zoo/vae_vol/{symbol}`, `/api/model_zoo/hmm/{symbol}`, `/api/model_zoo/kill_switch/*`.
- Live streams: `/ws/live`, `/ws/logs/trader`, `/ws/logs/risk`.

## Guardrail

Use this before and after any visual overhaul:

```bash
python3 scripts/verify_dashboard_contract.py dashboard/static/index.html
```

The guard fails if a file has nav labels but missing tab panels, backend hooks, or enough API connectors. It correctly passes the current full dashboard and fails both legacy backup files.

## Recommendation

Do not restore either backup. Keep `dashboard/static/index.html` as the active legacy dashboard until the React migration reaches feature parity.

Use the backups only as visual references:

- `index.html.bak`: terminal CSS/header/nav inspiration.
- `index.terminal.bak`: compact React/CDN terminal inspiration.

Use `dashboard/react-terminal` as the new component-driven migration path. It should pull real backend data and clearly label any synthetic research/demo data.
