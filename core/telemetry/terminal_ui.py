from __future__ import annotations

import logging
from dataclasses import dataclass


def _clamp_progress(value: int) -> int:
    return max(0, min(100, int(value)))


@dataclass
class ProgressTracker:
    logger: logging.Logger
    label: str
    total_steps: int
    enabled: bool = True
    completed_steps: int = 0

    def advance(self, message: str, *, detail: str | None = None) -> int:
        self.completed_steps = min(self.total_steps, self.completed_steps + 1)
        percent = int(round((self.completed_steps / max(1, self.total_steps)) * 100.0))
        self.emit(percent, message, detail=detail)
        return percent

    def emit(self, percent: int, message: str, *, detail: str | None = None) -> None:
        if not self.enabled:
            return
        suffix = f" | {detail}" if detail else ""
        self.logger.info("%s %3d%% | %s%s", self.label, _clamp_progress(percent), message, suffix)

    def substep(self, current: int, total: int, message: str, *, detail: str | None = None) -> int:
        base_progress = (self.completed_steps / max(1, self.total_steps)) * 100.0
        span = 100.0 / max(1, self.total_steps)
        fraction = max(0.0, min(1.0, float(current) / max(1, total)))
        percent = int(round(base_progress + (span * fraction)))
        self.emit(percent, message, detail=detail)
        return percent


def format_status_line(label: str, state: str, **metrics) -> str:
    fragments = [f"{label}: {state}"]
    for key, value in metrics.items():
        fragments.append(f"{key}={value}")
    return " | ".join(fragments)
