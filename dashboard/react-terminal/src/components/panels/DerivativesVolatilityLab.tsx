import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { BufferAttribute, BufferGeometry, DoubleSide } from "three";
import {
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip
} from "recharts";
import { greekExposure, scenarioMatrix, volSurface } from "../../data/derivativesMockData";

function VolSurfaceMesh() {
  const expiries = [...new Set(volSurface.map((point) => point.expiry))];
  const strikes = [...new Set(volSurface.map((point) => point.strike))];
  const positions: number[] = [];
  const indices: number[] = [];

  expiries.forEach((expiry, y) => {
    strikes.forEach((strike, x) => {
      const point = volSurface.find((item) => item.expiry === expiry && item.strike === strike)!;
      positions.push((x - strikes.length / 2) * 0.22, (point.iv - 0.18) * 8, (y - expiries.length / 2) * 0.22);
    });
  });

  for (let y = 0; y < expiries.length - 1; y += 1) {
    for (let x = 0; x < strikes.length - 1; x += 1) {
      const a = y * strikes.length + x;
      const b = a + 1;
      const c = a + strikes.length;
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(new Float32Array(positions), 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  return (
    <mesh geometry={geometry} rotation={[0.05, -0.6, 0]}>
      <meshStandardMaterial color="#00E5FF" wireframe opacity={0.62} transparent side={DoubleSide} />
    </mesh>
  );
}

function VolSurfacePanel() {
  return (
    <section className="panel min-h-[360px] lg:col-span-2">
      <div className="panel-title">
        <span>3D Implied Volatility Surface</span>
        <span className="text-terminal-cyan">strike · expiry · IV mesh</span>
      </div>
      <div className="h-[312px]">
        <Canvas camera={{ position: [0, 2.4, 5.2], fov: 48 }}>
          <ambientLight intensity={0.45} />
          <pointLight position={[4, 5, 3]} intensity={1.6} color="#00E5FF" />
          <VolSurfaceMesh />
          {volSurface
            .filter((point) => point.anomaly)
            .slice(0, 18)
            .map((point) => (
              <mesh key={`${point.strike}-${point.expiry}`} position={[(point.strike - 5200) / 270, (point.iv - 0.18) * 8, (point.expiry - 105) / 64]}>
                <sphereGeometry args={[0.035, 12, 12]} />
                <meshStandardMaterial emissive="#00E5FF" color="#00E5FF" />
              </mesh>
            ))}
          <OrbitControls enablePan={false} />
        </Canvas>
      </div>
    </section>
  );
}

function GreekRadar() {
  const data = greekExposure.map((row) => ({
    greek: row.greek,
    exposure: Math.abs(row.exposure),
    limit: Math.abs(row.limit)
  }));

  return (
    <section className="panel min-h-[280px]">
      <div className="panel-title">
        <span>Greek Exposure Radar</span>
        <span className="text-terminal-emerald">risk limit normalized</span>
      </div>
      <div className="h-[236px]">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={data}>
            <PolarGrid stroke="#1A1A1A" />
            <PolarAngleAxis dataKey="greek" tick={{ fill: "#6B7280", fontSize: 10 }} />
            <Radar dataKey="limit" stroke="#6B7280" fill="rgba(107,114,128,0.08)" isAnimationActive={false} />
            <Radar dataKey="exposure" stroke="#00FF88" fill="rgba(0,255,136,0.18)" isAnimationActive={false} />
            <Tooltip contentStyle={{ background: "#0F0F0F", border: "1px solid #1A1A1A", fontFamily: "JetBrains Mono" }} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function ScenarioMatrix() {
  const spots = [...new Set(scenarioMatrix.map((cell) => cell.spotShock))];
  const vols = [...new Set(scenarioMatrix.map((cell) => cell.volShock))];
  const maxAbs = Math.max(...scenarioMatrix.map((cell) => Math.abs(cell.pnl)));

  return (
    <section className="panel min-h-[280px]">
      <div className="panel-title">
        <span>Scenario Matrix</span>
        <span className="text-terminal-amber">spot shock x vol shock PnL</span>
      </div>
      <div className="grid p-2 text-[10px]" style={{ gridTemplateColumns: `72px repeat(${spots.length}, 1fr)` }}>
        <div />
        {spots.map((spot) => (
          <div key={spot} className="px-1 py-1 text-center text-terminal-muted">
            {spot > 0 ? "+" : ""}
            {spot}%
          </div>
        ))}
        {vols.map((vol) => (
          <div key={vol} className="contents">
            <div className="border-r border-terminal-grid px-1 py-2 text-terminal-muted">
              {vol > 0 ? "+" : ""}
              {vol}% vol
            </div>
            {spots.map((spot) => {
              const cell = scenarioMatrix.find((item) => item.spotShock === spot && item.volShock === vol)!;
              const alpha = Math.max(0.06, Math.abs(cell.pnl) / maxAbs);
              const color = cell.pnl >= 0 ? `rgba(0,255,136,${alpha})` : `rgba(255,51,95,${alpha})`;
              return (
                <div key={`${spot}-${vol}`} className="border border-black px-1 py-2 text-right" style={{ background: color }}>
                  {(cell.pnl / 1000).toFixed(1)}k
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </section>
  );
}

export function DerivativesVolatilityLab() {
  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <VolSurfacePanel />
      <GreekRadar />
      <ScenarioMatrix />
    </div>
  );
}
