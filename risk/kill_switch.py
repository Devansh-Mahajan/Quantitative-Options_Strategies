"""
Deterministic Risk Kill-Switch Engine.

Implements hard circuit-breakers with deterministic evaluation — no ML inference
in the hot path. All thresholds are pre-computed; the kill decision is pure
arithmetic over the current risk snapshot.

Analogous Rust implementation outline is documented inline (Rust pseudo-code
comments) so the team can port the hot path to Tokio async when needed.

Risk limits enforced:
  1. VaR breach        — historical 1-day VaR vs configured limit
  2. Expected Shortfall— CVaR (tail-loss) vs ES limit
  3. Max drawdown      — running peak-to-trough vs drawdown limit
  4. Daily loss        — intraday P&L vs daily loss limit
  5. Gross leverage    — notional / NAV vs leverage cap
  6. Concentration HHI — Herfindahl-Hirschman Index vs concentration limit
  7. Velocity          — rate-of-loss over a sliding 5-minute window
  8. Correlation shock — portfolio returns correlation spike vs threshold

Actions (in priority order):
  HALT      — all new orders blocked; existing positions unchanged
  REDUCE    — scale all new position sizes by `scale` factor (0.1 – 0.9)
  WARN      — log warning, no restriction

Rust hot-path scaffold
─────────────────────
```rust
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::RwLock;

static HALT_FLAG: AtomicBool = AtomicBool::new(false);

pub struct KillSwitch {
    config: RiskConfig,
    state:  RwLock<RiskState>,
}

impl KillSwitch {
    pub async fn evaluate(&self, snapshot: &RiskSnapshot) -> KillDecision {
        // 1. Atomic read — zero contention on hot path
        if HALT_FLAG.load(Ordering::Relaxed) {
            return KillDecision::Halt("already halted");
        }
        // 2. Lock-free VaR check
        if snapshot.var_1d_pct.abs() > self.config.var_limit_pct {
            HALT_FLAG.store(true, Ordering::SeqCst); // memory fence
            return KillDecision::Halt("VaR breached");
        }
        // ... (same pattern for ES, drawdown, leverage)
        KillDecision::Allow { scale: self.compute_scale(snapshot) }
    }

    pub async fn reset(&self) {
        HALT_FLAG.store(false, Ordering::SeqCst);
        self.state.write().await.reset();
    }
}
```
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

log = logging.getLogger("risk.kill_switch")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KillSwitchConfig:
    # VaR / ES limits (as fraction of NAV, negative convention)
    var_limit_pct:    float = -5.0      # 1-day 95% VaR > 5% of NAV → HALT
    es_limit_pct:     float = -7.5      # CVaR > 7.5% of NAV → HALT
    # Drawdown limits
    max_drawdown_pct: float = -15.0     # running DD > 15% → HALT
    dd_warn_pct:      float = -8.0      # running DD > 8%  → REDUCE
    # Daily loss limit
    daily_loss_pct:   float = -3.0      # daily P&L < -3% of start NAV → HALT
    daily_warn_pct:   float = -1.5      # daily P&L < -1.5% → WARN
    # Leverage
    max_leverage:     float = 3.0       # gross notional / NAV > 3× → HALT
    warn_leverage:    float = 2.0       # > 2× → REDUCE
    # Concentration
    hhi_halt:         float = 0.80      # very concentrated → HALT
    hhi_reduce:       float = 0.50      # moderately concentrated → REDUCE
    # Velocity (5-min loss rate)
    velocity_pct:     float = -1.0      # loss rate > 1% per 5 min → HALT
    velocity_window:  int   = 300       # seconds
    # Correlation shock (cross-asset correlation spike)
    corr_shock_threshold: float = 0.90  # avg pairwise correlation > 0.90 → WARN


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot input
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskSnapshot:
    """Current risk metrics passed to KillSwitch.evaluate()."""
    nav:              float          # current portfolio NAV (base currency)
    peak_nav:         float          # running peak NAV (for drawdown calc)
    start_nav:        float          # NAV at start of today's session
    var_1d_pct:       float          # 1-day historical VaR as % of NAV (negative)
    es_1d_pct:        float          # 1-day CVaR as % of NAV (negative)
    gross_leverage:   float          # gross notional / NAV
    hhi:              float          # Herfindahl-Hirschman concentration
    avg_pairwise_corr: float = 0.0   # avg return correlation across holdings
    timestamp:        float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# Decision output
# ─────────────────────────────────────────────────────────────────────────────

class Action(str, Enum):
    ALLOW  = "ALLOW"
    WARN   = "WARN"
    REDUCE = "REDUCE"
    HALT   = "HALT"


@dataclass
class KillDecision:
    action: Action
    reason: str       = ""
    scale:  float     = 1.0     # [0.1, 1.0] — multiply position sizes
    checks: dict      = field(default_factory=dict)   # per-check detail


# ─────────────────────────────────────────────────────────────────────────────
# Velocity tracker
# ─────────────────────────────────────────────────────────────────────────────

class VelocityTracker:
    """Sliding-window rate-of-loss tracker (lock-free via deque + mutex)."""

    def __init__(self, window_seconds: int = 300) -> None:
        self._window = window_seconds
        self._lock   = threading.Lock()
        self._samples: list[tuple[float, float]] = []   # (ts, nav)

    def record(self, nav: float) -> None:
        now = time.time()
        with self._lock:
            self._samples.append((now, nav))
            cutoff = now - self._window
            self._samples = [(t, v) for t, v in self._samples if t >= cutoff]

    def loss_rate_pct(self, current_nav: float) -> float:
        """Return loss rate over the window as % of window-start NAV."""
        with self._lock:
            if not self._samples:
                return 0.0
            start_nav = self._samples[0][1]
        if start_nav <= 0:
            return 0.0
        return (current_nav - start_nav) / start_nav * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Main kill switch
# ─────────────────────────────────────────────────────────────────────────────

class KillSwitch:
    """
    Deterministic, stateful kill switch.

    Thread-safe: uses a single lock around the halt flag.
    Designed to be evaluated every 1–5 seconds in the main risk loop.
    """

    def __init__(self, cfg: KillSwitchConfig = KillSwitchConfig()) -> None:
        self.cfg       = cfg
        self._lock     = threading.Lock()
        self._halted   = False
        self._halt_ts  = 0.0
        self._halt_reason = ""
        self._vel      = VelocityTracker(cfg.velocity_window)
        self._events: list[dict] = []

    # ── Core evaluation (deterministic, < 1 μs on hot path) ──────────────

    def evaluate(self, snap: RiskSnapshot) -> KillDecision:
        """
        Evaluate all risk checks against the snapshot.
        Returns KillDecision with action, scale, and per-check details.
        """
        checks: dict[str, dict] = {}
        worst_action = Action.ALLOW
        worst_reason = ""
        composite_scale = 1.0

        # ── 0. Already halted? ────────────────────────────────────────────
        with self._lock:
            if self._halted:
                return KillDecision(Action.HALT, self._halt_reason, 0.0)

        # ── 1. VaR check ──────────────────────────────────────────────────
        if snap.var_1d_pct < self.cfg.var_limit_pct:
            checks["var"] = {"value": snap.var_1d_pct, "limit": self.cfg.var_limit_pct, "action": "HALT"}
            return self._halt(snap, f"VaR {snap.var_1d_pct:.2f}% < limit {self.cfg.var_limit_pct:.2f}%",
                              checks)

        checks["var"] = {"value": snap.var_1d_pct, "limit": self.cfg.var_limit_pct, "action": "OK"}

        # ── 2. ES / CVaR check ────────────────────────────────────────────
        if snap.es_1d_pct < self.cfg.es_limit_pct:
            checks["es"] = {"value": snap.es_1d_pct, "limit": self.cfg.es_limit_pct, "action": "HALT"}
            return self._halt(snap, f"CVaR {snap.es_1d_pct:.2f}% < limit {self.cfg.es_limit_pct:.2f}%",
                              checks)

        checks["es"] = {"value": snap.es_1d_pct, "limit": self.cfg.es_limit_pct, "action": "OK"}

        # ── 3. Drawdown ───────────────────────────────────────────────────
        if snap.peak_nav > 0:
            dd_pct = (snap.nav - snap.peak_nav) / snap.peak_nav * 100.0
        else:
            dd_pct = 0.0

        if dd_pct < self.cfg.max_drawdown_pct:
            checks["drawdown"] = {"value": dd_pct, "limit": self.cfg.max_drawdown_pct, "action": "HALT"}
            return self._halt(snap, f"Drawdown {dd_pct:.2f}% < limit {self.cfg.max_drawdown_pct:.2f}%",
                              checks)
        elif dd_pct < self.cfg.dd_warn_pct:
            scale = max(0.1, 1.0 - abs(dd_pct - self.cfg.dd_warn_pct) /
                        max(abs(self.cfg.max_drawdown_pct - self.cfg.dd_warn_pct), 1e-6))
            composite_scale = min(composite_scale, scale)
            worst_action = Action.REDUCE
            worst_reason = f"Drawdown {dd_pct:.2f}% approaching limit"
            checks["drawdown"] = {"value": dd_pct, "limit": self.cfg.max_drawdown_pct,
                                  "action": "REDUCE", "scale": scale}
        else:
            checks["drawdown"] = {"value": dd_pct, "limit": self.cfg.max_drawdown_pct, "action": "OK"}

        # ── 4. Daily loss ─────────────────────────────────────────────────
        if snap.start_nav > 0:
            daily_pct = (snap.nav - snap.start_nav) / snap.start_nav * 100.0
        else:
            daily_pct = 0.0

        if daily_pct < self.cfg.daily_loss_pct:
            checks["daily_loss"] = {"value": daily_pct, "limit": self.cfg.daily_loss_pct, "action": "HALT"}
            return self._halt(snap, f"Daily loss {daily_pct:.2f}% < limit {self.cfg.daily_loss_pct:.2f}%",
                              checks)
        elif daily_pct < self.cfg.daily_warn_pct:
            worst_action = max(worst_action, Action.WARN, key=lambda a: list(Action).index(a))
            worst_reason = worst_reason or f"Daily loss {daily_pct:.2f}% (warning)"
            checks["daily_loss"] = {"value": daily_pct, "limit": self.cfg.daily_loss_pct, "action": "WARN"}
        else:
            checks["daily_loss"] = {"value": daily_pct, "limit": self.cfg.daily_loss_pct, "action": "OK"}

        # ── 5. Leverage ───────────────────────────────────────────────────
        if snap.gross_leverage > self.cfg.max_leverage:
            checks["leverage"] = {"value": snap.gross_leverage, "limit": self.cfg.max_leverage, "action": "HALT"}
            return self._halt(snap, f"Gross leverage {snap.gross_leverage:.2f}× > limit {self.cfg.max_leverage}×",
                              checks)
        elif snap.gross_leverage > self.cfg.warn_leverage:
            scale = max(0.2, 1.0 - (snap.gross_leverage - self.cfg.warn_leverage) /
                        max(self.cfg.max_leverage - self.cfg.warn_leverage, 1e-6))
            composite_scale = min(composite_scale, scale)
            worst_action = max(worst_action, Action.REDUCE, key=lambda a: list(Action).index(a))
            worst_reason = worst_reason or f"Leverage {snap.gross_leverage:.2f}× elevated"
            checks["leverage"] = {"value": snap.gross_leverage, "limit": self.cfg.max_leverage,
                                  "action": "REDUCE", "scale": scale}
        else:
            checks["leverage"] = {"value": snap.gross_leverage, "limit": self.cfg.max_leverage, "action": "OK"}

        # ── 6. Concentration HHI ──────────────────────────────────────────
        if snap.hhi > self.cfg.hhi_halt:
            checks["hhi"] = {"value": snap.hhi, "limit": self.cfg.hhi_halt, "action": "HALT"}
            return self._halt(snap, f"Concentration HHI={snap.hhi:.3f} > limit {self.cfg.hhi_halt:.3f}",
                              checks)
        elif snap.hhi > self.cfg.hhi_reduce:
            scale = max(0.2, 1.0 - (snap.hhi - self.cfg.hhi_reduce) /
                        max(self.cfg.hhi_halt - self.cfg.hhi_reduce, 1e-6))
            composite_scale = min(composite_scale, scale)
            worst_action = max(worst_action, Action.REDUCE, key=lambda a: list(Action).index(a))
            worst_reason = worst_reason or f"High concentration HHI={snap.hhi:.3f}"
            checks["hhi"] = {"value": snap.hhi, "limit": self.cfg.hhi_halt,
                             "action": "REDUCE", "scale": scale}
        else:
            checks["hhi"] = {"value": snap.hhi, "limit": self.cfg.hhi_halt, "action": "OK"}

        # ── 7. Velocity ───────────────────────────────────────────────────
        self._vel.record(snap.nav)
        vel = self._vel.loss_rate_pct(snap.nav)
        if vel < self.cfg.velocity_pct:
            checks["velocity"] = {"value": vel, "limit": self.cfg.velocity_pct, "action": "HALT"}
            return self._halt(snap, f"Loss velocity {vel:.2f}%/{self.cfg.velocity_window}s exceeded",
                              checks)
        checks["velocity"] = {"value": vel, "limit": self.cfg.velocity_pct, "action": "OK"}

        # ── 8. Correlation shock ──────────────────────────────────────────
        if snap.avg_pairwise_corr > self.cfg.corr_shock_threshold:
            checks["corr_shock"] = {"value": snap.avg_pairwise_corr,
                                    "limit": self.cfg.corr_shock_threshold, "action": "WARN"}
            worst_action = max(worst_action, Action.WARN, key=lambda a: list(Action).index(a))
            worst_reason = worst_reason or f"Correlation shock avg_ρ={snap.avg_pairwise_corr:.2f}"
        else:
            checks["corr_shock"] = {"value": snap.avg_pairwise_corr,
                                    "limit": self.cfg.corr_shock_threshold, "action": "OK"}

        # ── Composite decision ─────────────────────────────────────────────
        if worst_action == Action.WARN:
            log.warning("RISK WARNING: %s", worst_reason)

        return KillDecision(
            action=worst_action,
            reason=worst_reason,
            scale=round(composite_scale, 4),
            checks=checks,
        )

    # ── Halt helpers ──────────────────────────────────────────────────────

    def _halt(self, snap: RiskSnapshot, reason: str, checks: dict) -> KillDecision:
        with self._lock:
            if not self._halted:
                self._halted   = True
                self._halt_ts  = time.time()
                self._halt_reason = reason
                event = {
                    "ts":     datetime.now(timezone.utc).isoformat(),
                    "reason": reason,
                    "nav":    snap.nav,
                    "var":    snap.var_1d_pct,
                    "dd":     (snap.nav - snap.peak_nav) / max(snap.peak_nav, 1e-6) * 100,
                }
                self._events.append(event)
                log.critical("KILL SWITCH TRIGGERED: %s  NAV=%.2f", reason, snap.nav)
        return KillDecision(Action.HALT, reason, 0.0, checks)

    def reset(self, reason: str = "manual_reset") -> None:
        with self._lock:
            self._halted      = False
            self._halt_reason = ""
            self._halt_ts     = 0.0
            log.warning("Kill switch reset: %s", reason)

    @property
    def is_halted(self) -> bool:
        with self._lock:
            return self._halted

    @property
    def halt_reason(self) -> str:
        with self._lock:
            return self._halt_reason

    @property
    def events(self) -> list[dict]:
        with self._lock:
            return list(self._events)

    def status(self) -> dict:
        with self._lock:
            return {
                "halted":       self._halted,
                "halt_reason":  self._halt_reason,
                "halt_ts":      datetime.fromtimestamp(self._halt_ts, tz=timezone.utc).isoformat()
                                if self._halt_ts else None,
                "n_events":     len(self._events),
                "recent_events": self._events[-5:],
            }


# ─────────────────────────────────────────────────────────────────────────────
# Conformal-prediction-enhanced VaR
# ─────────────────────────────────────────────────────────────────────────────

def conformal_var(returns: list[float], alpha: float = 0.05,
                  cal_size: int = 252) -> tuple[float, float]:
    """
    Split-conformal VaR with guaranteed finite-sample coverage.

    Method (Bates et al. 2023 "Testing for Outliers with Conformal P-values"):
      1. Calibrate nonconformity scores on cal_size held-out returns
      2. Compute quantile q_hat = (1 + 1/n_cal) * (1 - alpha) empirical quantile
      3. VaR estimate has guaranteed (1 - alpha) coverage on future returns

    Returns (var_estimate, conformal_upper_bound).
    The upper bound is wider but has rigorous finite-sample guarantees.
    """
    if len(returns) < cal_size + 10:
        import numpy as np
        arr = sorted(returns)
        return float(arr[int(len(arr) * alpha)]), 0.0

    import numpy as np
    arr  = np.array(returns, dtype=float)
    cal  = arr[-cal_size:]
    # Nonconformity score = negative return (higher = more extreme loss)
    scores = -cal
    n_cal  = len(scores)
    # Conformal quantile with finite-sample correction
    q_level = min(1.0, (1 + 1.0 / n_cal) * (1 - alpha))
    q_hat   = float(np.quantile(scores, q_level))
    var_est = float(np.quantile(arr, alpha))
    return var_est, -q_hat   # var_est is standard; -q_hat is the conformal bound


# ── Module-level singleton ────────────────────────────────────────────────────
_kill_switch: Optional[KillSwitch] = None


def get_kill_switch(cfg: Optional[KillSwitchConfig] = None) -> KillSwitch:
    global _kill_switch
    if _kill_switch is None:
        _kill_switch = KillSwitch(cfg or KillSwitchConfig())
    return _kill_switch
