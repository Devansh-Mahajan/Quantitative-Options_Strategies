from __future__ import annotations

import hashlib
import importlib
import json
import py_compile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_PATH = REPO_ROOT / ".runtime" / "preflight_state.json"
SOURCE_GLOBS = (
    "*.py",
    "core/**/*.py",
    "scripts/**/*.py",
    "config/**/*.py",
    "logging/**/*.py",
    "tests/**/*.py",
)
JSON_GLOBS = (
    "config/*.json",
    ".runtime/*.json",
)
CRITICAL_IMPORTS = (
    "core.signal_fusion",
    "core.runtime_calibration",
    "core.strategy_regime",
    "core.equity_overlay",
    "scripts.run_strategy",
    "scripts.automation_controller",
    "scripts.model_maintenance",
    "scripts.weekend_recalibration",
    "scripts.massive_backtest_engine",
)
REQUIRED_ARTIFACTS = (
    "config/brain_weights.pth",
    "config/hmm_macro_model.pkl",
    "config/correlation_alpha_model.pkl",
    "data/brain_dataset.pt",
)
OPTIONAL_ARTIFACTS = (
    "config/regime_movement_models.pkl",
    "config/market_regime_policy.json",
    "config/quant_strategy_pack.json",
)


@dataclass
class PreflightIssue:
    severity: str
    check: str
    detail: str


@dataclass
class PreflightResult:
    ok: bool
    skipped: bool
    summary: str
    completed_at_utc: str
    signature: str
    source_files: int
    json_files: int
    artifacts_checked: int
    issues: list[PreflightIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["issues"] = [asdict(issue) for issue in self.issues]
        return payload


ProgressCallback = Callable[[int, str, str | None], None]


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _discover_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(set(files))


def _artifact_paths(root: Path) -> list[Path]:
    candidates = [root / relative for relative in (*REQUIRED_ARTIFACTS, *OPTIONAL_ARTIFACTS)]
    return sorted(set(candidates))


def _build_signature(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        rel = str(path)
        if path.exists():
            stat = path.stat()
            digest.update(rel.encode("utf-8"))
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
        else:
            digest.update(rel.encode("utf-8"))
            digest.update(b"missing")
    return digest.hexdigest()


def _report(callback: ProgressCallback | None, percent: int, message: str, detail: str | None = None) -> None:
    if callback:
        callback(percent, message, detail)


def _relative_label(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _compile_sources(
    paths: list[Path],
    issues: list[PreflightIssue],
    callback: ProgressCallback | None,
    *,
    root: Path,
) -> None:
    total = len(paths)
    for idx, path in enumerate(paths, start=1):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            issues.append(
                PreflightIssue(
                    severity="error",
                    check="compile",
                    detail=f"{_relative_label(path, root)}: {exc.msg}",
                )
            )
        if idx == total or idx % 25 == 0:
            _report(callback, 30, "Compiling Python sources", detail=f"{idx}/{total}")


def _import_modules(modules: Iterable[str], issues: list[PreflightIssue], callback: ProgressCallback | None) -> None:
    modules = list(modules)
    total = len(modules)
    for idx, module_name in enumerate(modules, start=1):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - import surfaces repo-specific issues
            issues.append(
                PreflightIssue(
                    severity="error",
                    check="import",
                    detail=f"{module_name}: {exc}",
                )
            )
        _report(callback, 55, "Importing critical modules", detail=f"{idx}/{total} | {module_name}")


def _validate_json(
    paths: list[Path],
    issues: list[PreflightIssue],
    callback: ProgressCallback | None,
    *,
    root: Path,
) -> None:
    total = len(paths)
    if total == 0:
        _report(callback, 70, "Validating JSON payloads", detail="0 files")
        return
    for idx, path in enumerate(paths, start=1):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(
                PreflightIssue(
                    severity="error",
                    check="json",
                    detail=f"{_relative_label(path, root)}: {exc}",
                )
            )
        _report(callback, 70, "Validating JSON payloads", detail=f"{idx}/{total}")


def _validate_artifacts(
    paths: list[Path],
    issues: list[PreflightIssue],
    callback: ProgressCallback | None,
    *,
    root: Path,
    deep_model_checks: bool,
) -> None:
    total = len(paths)
    for idx, path in enumerate(paths, start=1):
        rel = _relative_label(path, root)
        rel_name = Path(rel).name
        if str(rel) in REQUIRED_ARTIFACTS and not path.exists():
            issues.append(
                PreflightIssue(
                    severity="error",
                    check="artifact",
                    detail=f"Missing required artifact: {rel}",
                )
            )
            _report(callback, 85, "Checking model artifacts", detail=f"{idx}/{total} | missing {rel}")
            continue

        if str(rel) in OPTIONAL_ARTIFACTS and not path.exists():
            issues.append(
                PreflightIssue(
                    severity="warning",
                    check="artifact",
                    detail=f"Optional artifact not present yet: {rel}",
                )
            )
            _report(callback, 85, "Checking model artifacts", detail=f"{idx}/{total} | optional {rel}")
            continue

        if not path.exists() or not deep_model_checks:
            _report(callback, 85, "Checking model artifacts", detail=f"{idx}/{total} | {rel_name}")
            continue

        try:
            if path.suffix == ".pkl":
                import joblib

                payload = joblib.load(path)
                if payload is None:
                    raise ValueError("joblib payload is empty")
            elif path.suffix in {".pt", ".pth"}:
                import torch

                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload is None:
                    raise ValueError("torch payload is empty")
            elif path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - model loader errors depend on local artifacts
            issues.append(
                PreflightIssue(
                    severity="error" if str(rel) in REQUIRED_ARTIFACTS else "warning",
                    check="artifact",
                    detail=f"{rel}: {exc}",
                )
            )
        _report(callback, 85, "Checking model artifacts", detail=f"{idx}/{total} | {rel_name}")


def run_preflight(
    *,
    root: Path = REPO_ROOT,
    state_path: Path = DEFAULT_STATE_PATH,
    max_age_seconds: int = 300,
    progress_callback: ProgressCallback | None = None,
    deep_model_checks: bool = True,
) -> PreflightResult:
    source_files = _discover_files(root, SOURCE_GLOBS)
    json_files = _discover_files(root, JSON_GLOBS)
    artifact_files = _artifact_paths(root)
    signature = _build_signature([*source_files, *json_files, *artifact_files])
    cached = _load_state(state_path)
    now_epoch = time.time()

    if (
        cached.get("ok") is True
        and cached.get("signature") == signature
        and (now_epoch - float(cached.get("completed_at_epoch", 0.0) or 0.0)) <= max(30, int(max_age_seconds))
    ):
        return PreflightResult(
            ok=True,
            skipped=True,
            summary="Preflight reused cached success state.",
            completed_at_utc=cached.get("completed_at_utc", _utc_now()),
            signature=signature,
            source_files=len(source_files),
            json_files=len(json_files),
            artifacts_checked=len(artifact_files),
            issues=[PreflightIssue(**issue) for issue in cached.get("issues", [])],
        )

    issues: list[PreflightIssue] = []
    _report(progress_callback, 5, "Discovering automation health targets", detail=f"{len(source_files)} source files")
    _compile_sources(source_files, issues, progress_callback, root=root)
    _import_modules(CRITICAL_IMPORTS, issues, progress_callback)
    _validate_json(json_files, issues, progress_callback, root=root)
    _validate_artifacts(
        artifact_files,
        issues,
        progress_callback,
        root=root,
        deep_model_checks=deep_model_checks,
    )

    hard_failures = [issue for issue in issues if issue.severity == "error"]
    summary = (
        "Preflight passed."
        if not hard_failures
        else f"Preflight blocked automation with {len(hard_failures)} error(s)."
    )
    result = PreflightResult(
        ok=not hard_failures,
        skipped=False,
        summary=summary,
        completed_at_utc=_utc_now(),
        signature=signature,
        source_files=len(source_files),
        json_files=len(json_files),
        artifacts_checked=len(artifact_files),
        issues=issues,
    )
    _report(progress_callback, 100, summary, detail=f"errors={len(hard_failures)} warnings={len(issues) - len(hard_failures)}")
    _save_state(
        state_path,
        {
            **result.to_dict(),
            "completed_at_epoch": now_epoch,
        },
    )
    return result
