import { AlphaRegimeMatrix } from "./components/panels/AlphaRegimeMatrix";
import { DerivativesVolatilityLab } from "./components/panels/DerivativesVolatilityLab";
import { FeatureParityMatrix } from "./components/panels/FeatureParityMatrix";
import { LegacyParityPanels } from "./components/panels/LegacyParityPanels";
import { LiveOperationsPanel } from "./components/panels/LiveOperationsPanel";
import { MicrostructureExecution } from "./components/panels/MicrostructureExecution";
import { SimulationForwardTesting } from "./components/panels/SimulationForwardTesting";
import { useApiResource } from "./hooks/useApiResource";
import type { HealthResponse } from "./lib/api";

export function App() {
  const health = useApiResource<HealthResponse>("/api/health", 15_000);
  const live = health.data?.status === "ok";

  return (
    <main className="min-h-screen bg-terminal-bg p-3 text-terminal-ink">
      <header className="mb-3 grid grid-cols-[1fr_auto] items-center border border-terminal-grid bg-black px-3 py-2">
        <div>
          <div className="text-xs font-black uppercase tracking-[0.24em] text-terminal-cyan">
            QUANT/ALPHA REACT TERMINAL
          </div>
          <div className="mt-1 text-[10px] uppercase tracking-[0.16em] text-terminal-muted">
            Migration target only · production remains dashboard/static/index.html until feature parity
          </div>
        </div>
        <div className="text-right text-[10px] text-terminal-muted">
          <div>
            API <span className={live ? "text-terminal-emerald" : "text-terminal-crimson"}>{live ? "LIVE" : "OFFLINE"}</span>
          </div>
          <div>
            Broker <span className="text-terminal-cyan">{health.data?.broker?.toUpperCase() ?? "UNKNOWN"}</span>
          </div>
        </div>
      </header>

      <section className="mb-3">
        <LiveOperationsPanel />
      </section>

      <section className="mb-3">
        <AlphaRegimeMatrix />
      </section>

      <section className="mb-3">
        <DerivativesVolatilityLab />
      </section>

      <section className="mb-3">
        <SimulationForwardTesting />
      </section>

      <section>
        <MicrostructureExecution />
      </section>

      <section className="mt-3">
        <LegacyParityPanels />
      </section>

      <section className="mt-3">
        <FeatureParityMatrix />
      </section>
    </main>
  );
}
