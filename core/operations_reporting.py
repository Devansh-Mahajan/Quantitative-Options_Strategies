from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _backtest_summary_lines(payload: dict, *, title: str, generated_at: str, source_label: str) -> list[str]:
    overview = payload.get("massive_overview") or {}
    movement = (payload.get("movement_suite") or {}).get("summary") or {}
    pairs = (payload.get("pairs_suite") or {}).get("summary") or {}
    regime = (payload.get("regime_suite") or {}).get("summary") or {}
    profiles = (payload.get("strategy_profile_suite") or {}).get("summary") or {}
    return [
        f"# {title}",
        "",
        f"Generated: {generated_at}",
        f"Source: {source_label}",
        "",
        "## Executive Summary",
        f"- Predictive score: {overview.get('predictive_score', 'n/a')}",
        f"- Consensus market state: {profiles.get('consensus_state') or overview.get('consensus_market_state', 'n/a')}",
        f"- Consensus strategy profile: {profiles.get('consensus_profile') or overview.get('consensus_strategy_profile', 'n/a')}",
        "",
        "## Diagnostics",
        f"- Movement accuracy: {movement.get('avg_accuracy', 'n/a')}",
        f"- Pairs win-rate: {pairs.get('win_rate', 'n/a')}",
        f"- Regime directional score: {regime.get('directional_accuracy_proxy', 'n/a')}",
        f"- Strategy profile score: {profiles.get('avg_best_profile_score', 'n/a')}",
        "",
        "## Note",
        "- This report summarizes predictive/backtest diagnostics. It is not a guarantee of future trading performance.",
    ]


def archive_backtest_artifacts(report_payload: dict, *, output_path: Path, reports_root: Path) -> dict[str, str]:
    timestamp = _utc_now()
    stamp = timestamp.strftime("%Y%m%d_%H%M%S")
    backtests_dir = reports_root / "backtests"
    backtests_dir.mkdir(parents=True, exist_ok=True)

    archive_json = backtests_dir / f"massive_backtest_{stamp}.json"
    archive_md = backtests_dir / f"massive_backtest_{stamp}.md"
    latest_json = reports_root / "latest_backtest_report.json"
    latest_md = reports_root / "latest_backtest_summary.md"

    payload = dict(report_payload)
    payload.setdefault("generated_at_utc", timestamp.isoformat())
    payload["source_output_path"] = str(output_path)

    archive_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_lines = _backtest_summary_lines(
        payload,
        title="Massive Backtest Summary",
        generated_at=timestamp.isoformat(),
        source_label=str(output_path),
    )
    archive_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    latest_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "archive_json": str(archive_json),
        "archive_md": str(archive_md),
        "latest_json": str(latest_json),
        "latest_md": str(latest_md),
    }


def write_daily_ops_report(
    *,
    repo_root: Path,
    report_path: Path,
    latest_report_path: Path,
    context: dict | None = None,
) -> dict:
    context = dict(context or {})
    reports_root = repo_root / "reports"
    runtime_root = repo_root / ".runtime"
    timestamp = _utc_now()
    date_key = timestamp.date().isoformat()

    risk = _safe_read_json(runtime_root / "risk_snapshot.json")
    system = _safe_read_json(runtime_root / "system_resource_snapshot.json")
    foundry = _safe_read_json(reports_root / "quant_foundry_report.json")
    latest_backtest = _safe_read_json(reports_root / "latest_backtest_report.json")
    maintenance = _safe_read_json(reports_root / "daily_model_maintenance_report.json")
    market_policy = _safe_read_json(repo_root / "config" / "market_regime_policy.json")

    risk_metrics = {
        "daily_pnl_pct": risk.get("daily_pnl_pct"),
        "daily_pnl_dollars": risk.get("daily_pnl_dollars"),
        "total_equity": risk.get("total_equity"),
        "buying_power_budget": risk.get("buying_power_budget"),
        "portfolio_delta": risk.get("portfolio_delta"),
        "portfolio_theta": risk.get("portfolio_theta"),
        "portfolio_vega": risk.get("portfolio_vega"),
        "target_delta": risk.get("target_delta"),
        "target_theta": risk.get("target_theta"),
        "target_vega": risk.get("target_vega"),
        "movement_bias": risk.get("movement_bias"),
        "macro_regime": risk.get("macro_regime"),
        "runtime_market_state": risk.get("runtime_market_state"),
        "runtime_profile": risk.get("runtime_profile"),
    }

    host_metrics = system.get("host_metrics") or {}
    memory = host_metrics.get("memory") or {}
    disk = host_metrics.get("disk") or {}
    latest_backtest_overview = latest_backtest.get("massive_overview") or {}
    foundry_summary = foundry.get("summary") or {}

    payload = {
        "generated_at_utc": timestamp.isoformat(),
        "date": date_key,
        "context": context,
        "risk": risk_metrics,
        "system": {
            "pressure": (system.get("status") or {}).get("pressure"),
            "normalized_cpu_load_pct": host_metrics.get("normalized_cpu_load_pct"),
            "memory_usage_pct": memory.get("usage_pct"),
            "disk_usage_pct": disk.get("usage_pct"),
        },
        "foundry": {
            "mode": foundry.get("mode"),
            "ensemble_accuracy": foundry_summary.get("ensemble_accuracy"),
            "estimated_strategy_daily_return": foundry_summary.get("estimated_strategy_daily_return"),
        },
        "latest_backtest": {
            "predictive_score": latest_backtest_overview.get("predictive_score"),
            "consensus_market_state": latest_backtest_overview.get("consensus_market_state"),
            "consensus_strategy_profile": latest_backtest_overview.get("consensus_strategy_profile"),
        },
        "market_policy": {
            "current_regime_label": market_policy.get("current_regime_label"),
            "current_market_state": market_policy.get("current_market_state"),
            "selected_profile": market_policy.get("selected_profile"),
            "risk_multiplier": market_policy.get("risk_multiplier"),
            "deployment_multiplier": market_policy.get("deployment_multiplier"),
            "trade_intensity_multiplier": market_policy.get("trade_intensity_multiplier"),
        },
        "model_maintenance": {
            "mode": maintenance.get("mode"),
            "steps": len(maintenance.get("steps") or []),
        },
        "note": "Daily operating report for the automated stack. It summarizes system state and the latest available research outputs.",
    }

    lines = [
        "# Daily Automation Report",
        "",
        f"Generated: {timestamp.isoformat()}",
        "",
        "## Portfolio And Risk",
        f"- Daily P/L: {risk_metrics.get('daily_pnl_pct', 'n/a')}% / ${risk_metrics.get('daily_pnl_dollars', 'n/a')}",
        f"- Equity / buying-power budget: ${risk_metrics.get('total_equity', 'n/a')} / ${risk_metrics.get('buying_power_budget', 'n/a')}",
        f"- Portfolio Greeks: delta={risk_metrics.get('portfolio_delta', 'n/a')} theta={risk_metrics.get('portfolio_theta', 'n/a')} vega={risk_metrics.get('portfolio_vega', 'n/a')}",
        f"- Target Greeks: delta={risk_metrics.get('target_delta', 'n/a')} theta={risk_metrics.get('target_theta', 'n/a')} vega={risk_metrics.get('target_vega', 'n/a')}",
        f"- Macro / regime: {risk_metrics.get('macro_regime', 'n/a')} | {risk_metrics.get('runtime_market_state', 'n/a')} | {risk_metrics.get('runtime_profile', 'n/a')}",
        "",
        "## System Health",
        f"- Host pressure: {(system.get('status') or {}).get('pressure', 'n/a')}",
        f"- CPU load: {host_metrics.get('normalized_cpu_load_pct', 'n/a')}%",
        f"- Memory usage: {memory.get('usage_pct', 'n/a')}%",
        f"- Disk usage: {disk.get('usage_pct', 'n/a')}%",
        "",
        "## Model Maintenance",
        f"- Foundry mode: {foundry.get('mode', 'n/a')}",
        f"- Ensemble accuracy: {foundry_summary.get('ensemble_accuracy', 'n/a')}",
        f"- Estimated daily return proxy: {foundry_summary.get('estimated_strategy_daily_return', 'n/a')}",
        f"- Daily maintenance steps: {len(maintenance.get('steps') or [])}",
        "",
        "## Latest Backtest",
        f"- Predictive score: {latest_backtest_overview.get('predictive_score', 'n/a')}",
        f"- Consensus market state: {latest_backtest_overview.get('consensus_market_state', 'n/a')}",
        f"- Consensus strategy profile: {latest_backtest_overview.get('consensus_strategy_profile', 'n/a')}",
        "",
        "## Runtime Policy",
        f"- Regime label: {market_policy.get('current_regime_label', 'n/a')}",
        f"- Risk / deploy / intensity multipliers: {market_policy.get('risk_multiplier', 'n/a')} / {market_policy.get('deployment_multiplier', 'n/a')} / {market_policy.get('trade_intensity_multiplier', 'n/a')}",
        "",
        "## Notes",
        "- The stack is automated by day/time, but these outputs are operational diagnostics, not profit guarantees.",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    latest_report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (latest_report_path.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
