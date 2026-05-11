# Quant Alpha React Terminal

Component-driven migration scaffold for the legacy `dashboard/static/index.html`.

```text
dashboard/react-terminal/
  src/
    components/
      panels/
        LiveOperationsPanel.tsx           # real endpoint mirror: equity, positions, risk, strategies
        LegacyParityPanels.tsx            # broad endpoint hooks for every legacy tab
        AlphaRegimeMatrix.tsx
        SimulationForwardTesting.tsx
        DerivativesVolatilityLab.tsx      # next: R3F vol surface + Greeks radar
        MicrostructureExecution.tsx       # next: L3 book + latency scatter
        FeatureParityMatrix.tsx           # migration checklist; legacy remains production
    data/
      alphaMockData.ts
      simulationMockData.ts
    lib/
      quant.ts
    App.tsx
    main.tsx
    styles.css
```

Design language:

- Deep dark background: `#0A0A0A`.
- Muted gridlines: `#1A1A1A`.
- Action-only accents: cyan, emerald, crimson, amber.
- JetBrains Mono for all numeric data.
- Recharts for 2D graph work.
- React Three Fiber for 3D volatility surfaces.
- AG Grid for future high-density ledgers.

Reality/provenance policy:

- `dashboard/static/index.html` remains the active functional dashboard until React reaches feature parity.
- `dashboard/static/index.html.bak` and `dashboard/static/index.terminal.bak` are visual references only; do not restore them over the active dashboard.
- Broker fills are never mocked. `MicrostructureExecution` reads `/api/execution/ledger` and displays an empty state if no confirmed fills are available.
- Live operations are never mocked. `LiveOperationsPanel` reads `/api/equity`, `/api/positions`, `/api/risk/engine`, and `/api/strategies/status`; missing data renders as unavailable.
- Legacy parity modules are endpoint-backed first. `LegacyParityPanels` wires every legacy tab family to its backend contract before we refine each view into a dedicated production-grade panel.
- Header status reads `/api/health`; no fake uptime/latency status is printed.
- Model and simulation panels may use deterministic synthetic research data only when the live model-zoo or equity analytics endpoints are unavailable, and those panels label the fallback as synthetic/demo.
- Current live endpoint mappings:
  - `/api/health`
  - `/api/equity?window=1w`
  - `/api/positions`
  - `/api/risk/engine`
  - `/api/strategies/status`
  - `/api/execution/ledger`
  - `/api/equity/analytics`
  - `/api/options/overview`
  - `/api/options/chain`
  - `/api/stocks/overview`
  - `/api/trades/odds`
  - `/api/backtest/list`
  - `/api/strategies/params`
  - `/api/config/risk`
  - `/api/research/desk`
  - `/api/permutations/strategies`
  - `/api/training/scripts`
  - `/api/training/jobs`
  - `/api/ml/alpha`
  - `/api/intelligence`
  - `/api/trades/analysis`
  - `/api/system/health`
  - `/api/system/master_recalibrate`
  - `/api/risk/events`
  - `/api/strategies/pnl`
  - `/api/gsquant/summary`
  - `/api/model_zoo/hmm/SPY`
  - `/api/model_zoo/xgb/SPY`
  - `/ws/logs/trader`
  - `/ws/logs/risk`

Migration workflow:

1. Keep backend-facing fixes in `dashboard/static/index.html` first.
2. Port one legacy tab or backend contract at a time into isolated React panels.
3. Update `FeatureParityMatrix.tsx` when a tab reaches partial or full parity.
4. Run `npm run verify` before treating React changes as safe.
5. Do not replace the legacy HTML until every tab in the parity matrix is migrated and checked against real endpoints.

Build note:

- The first build intentionally imports Recharts, React Three Fiber, Three, and AG Grid in one shell so the visual scaffold is easy to inspect.
- Before production rollout, lazy-load `DerivativesVolatilityLab` and `MicrostructureExecution` with `React.lazy` or manual Rollup chunks to keep the initial bundle small.
