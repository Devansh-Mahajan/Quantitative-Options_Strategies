from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


def _safe_cpu_count() -> int:
    return max(1, int(os.cpu_count() or 1))


def _detect_memory_gb() -> float:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
        return (page_size * page_count) / (1024 ** 3)
    except (AttributeError, OSError, ValueError):
        return 0.0


def _detect_disk_gb(path: Path) -> float:
    try:
        usage = shutil.disk_usage(path)
        return usage.total / (1024 ** 3)
    except OSError:
        return 0.0


@dataclass(frozen=True)
class ResourceProfile:
    cpu_count: int
    memory_gb: float
    disk_gb: float
    reserved_cores: int
    controller_blas_threads: int
    backtest_workers: int
    research_rf_jobs: int
    model_parallelism: int
    daily_training_max_symbols: int
    risk_interval_seconds: int
    regime_interval_seconds: int
    strategy_interval_seconds: int
    telemetry_interval_seconds: int

    def to_env(self) -> dict[str, str]:
        thread_cap = str(self.controller_blas_threads)
        return {
            "OMP_NUM_THREADS": thread_cap,
            "OPENBLAS_NUM_THREADS": thread_cap,
            "MKL_NUM_THREADS": thread_cap,
            "NUMEXPR_NUM_THREADS": thread_cap,
            "VECLIB_MAXIMUM_THREADS": thread_cap,
            "RAYON_NUM_THREADS": thread_cap,
            "OPTIONS_STACK_CPU_COUNT": str(self.cpu_count),
            "OPTIONS_STACK_MEMORY_GB": f"{self.memory_gb:.2f}",
            "OPTIONS_STACK_DISK_GB": f"{self.disk_gb:.2f}",
            "OPTIONS_STACK_BACKTEST_WORKERS": str(self.backtest_workers),
            "OPTIONS_STACK_RF_JOBS": str(self.research_rf_jobs),
            "OPTIONS_STACK_MODEL_PARALLELISM": str(self.model_parallelism),
        }

    def to_dict(self) -> dict:
        return asdict(self)


def build_resource_profile(
    *,
    repo_root: Path,
    cpu_count: int | None = None,
    memory_gb: float | None = None,
    disk_gb: float | None = None,
) -> ResourceProfile:
    cpu = max(1, int(cpu_count or _safe_cpu_count()))
    mem = float(memory_gb if memory_gb is not None else _detect_memory_gb())
    disk = float(disk_gb if disk_gb is not None else _detect_disk_gb(repo_root))

    reserved_cores = max(2, min(4, cpu // 6 or 2))
    available_cores = max(1, cpu - reserved_cores)
    memory_headroom_factor = max(1, int(mem // 4)) if mem > 0 else max(1, cpu // 4)

    controller_blas_threads = max(1, min(6, available_cores // 2 or 1))
    backtest_workers = max(2, min(16, available_cores, max(2, int(mem // 2)) if mem > 0 else available_cores))
    research_rf_jobs = max(2, min(10, available_cores // 2 or 2, memory_headroom_factor * 2))
    model_parallelism = max(1, min(3, available_cores // 6 or 1))
    daily_training_max_symbols = max(12, min(40, available_cores + int(mem // 2) if mem > 0 else available_cores + 8))

    # Faster monitoring near 24/7 operations, but still bounded to avoid churn.
    risk_interval_seconds = 180 if cpu >= 16 and mem >= 24 else 240
    regime_interval_seconds = 480 if cpu >= 16 else 600
    strategy_interval_seconds = 900 if cpu >= 16 else 1200
    telemetry_interval_seconds = 300

    return ResourceProfile(
        cpu_count=cpu,
        memory_gb=round(mem, 2),
        disk_gb=round(disk, 2),
        reserved_cores=reserved_cores,
        controller_blas_threads=controller_blas_threads,
        backtest_workers=backtest_workers,
        research_rf_jobs=research_rf_jobs,
        model_parallelism=model_parallelism,
        daily_training_max_symbols=daily_training_max_symbols,
        risk_interval_seconds=risk_interval_seconds,
        regime_interval_seconds=regime_interval_seconds,
        strategy_interval_seconds=strategy_interval_seconds,
        telemetry_interval_seconds=telemetry_interval_seconds,
    )


def load_resource_profile(repo_root: Path) -> ResourceProfile:
    cpu_override = os.environ.get("OPTIONS_STACK_CPU_COUNT")
    memory_override = os.environ.get("OPTIONS_STACK_MEMORY_GB")
    disk_override = os.environ.get("OPTIONS_STACK_DISK_GB")
    return build_resource_profile(
        repo_root=repo_root,
        cpu_count=int(float(cpu_override)) if cpu_override else None,
        memory_gb=float(memory_override) if memory_override else None,
        disk_gb=float(disk_override) if disk_override else None,
    )
