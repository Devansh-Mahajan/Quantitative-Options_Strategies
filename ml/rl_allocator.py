"""
PPO-based strategy weight allocator.
State  = market features + regime + current portfolio weights + P&L
Action = strategy weights (continuous, softmax normalised)
Reward = portfolio Sharpe ratio increment
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from bot.config import cfg

log = logging.getLogger("ml.rl_allocator")

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False
    log.warning("stable-baselines3 not installed — RL allocator will use equal weights")


# --------------------------------------------------------------------------- #
# Custom Gym environment
# --------------------------------------------------------------------------- #

class StrategyAllocEnv(gym.Env):
    """
    Episodic environment over historical return sequences.
    One step = one trading bar.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: np.ndarray,        # (T, n_strategies)
        features: np.ndarray,       # (T, n_features) — market state
        regimes: np.ndarray,        # (T,) int — regime index
        episode_len: int = 200,
        neural_probs: np.ndarray | None = None,   # (T, 4) or None
    ) -> None:
        super().__init__()
        self.returns = returns.astype(np.float32)
        self.features = features.astype(np.float32)
        self.regimes = regimes.astype(np.int32)
        self.episode_len = min(episode_len, len(returns))
        self.n_strategies = returns.shape[1]
        self.n_features = features.shape[1]

        # Per-step neural macro probs: (T, 4); defaults to neutral if not supplied
        T = len(returns)
        if neural_probs is not None and len(neural_probs) == T:
            self._neural_probs_seq = np.asarray(neural_probs, dtype=np.float32)
        else:
            self._neural_probs_seq = np.full((T, 4), 0.25, dtype=np.float32)
        self._current_neural_probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

        # Action: allocation weights for each strategy
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_strategies,), dtype=np.float32)
        # Observation: features + weights + pnl + regime one-hot + neural_probs
        obs_dim = self.n_features + self.n_strategies + 1 + 4 + 4
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self._t = 0
        self._start = 0
        self._weights = np.ones(self.n_strategies, dtype=np.float32) / self.n_strategies
        self._portfolio_rets: list[float] = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        max_start = max(1, len(self.returns) - self.episode_len)
        self._start = self.np_random.integers(0, max_start)
        self._t = self._start
        self._weights = np.ones(self.n_strategies, dtype=np.float32) / self.n_strategies
        self._portfolio_rets = []
        return self._obs(), {}

    def step(self, action: np.ndarray):
        self._current_neural_probs = self._neural_probs_seq[min(self._t, len(self._neural_probs_seq) - 1)]
        weights = np.exp(action) / (np.exp(action).sum() + 1e-10)  # softmax
        step_ret = float(np.dot(weights, self.returns[self._t]))
        self._portfolio_rets.append(step_ret)
        self._weights = weights
        self._t += 1

        # Reward: incremental Sharpe contribution
        if len(self._portfolio_rets) >= 20:
            arr = np.array(self._portfolio_rets[-20:])
            sharpe = arr.mean() / (arr.std() + 1e-8) * np.sqrt(252)
        else:
            sharpe = step_ret * 100

        # Penalty for excessive concentration
        hhi = float(np.sum(weights ** 2))
        reward = float(sharpe) - 0.5 * hhi

        done = self._t >= self._start + self.episode_len or self._t >= len(self.returns) - 1
        return self._obs(), reward, done, False, {"portfolio_ret": step_ret}

    def _obs(self) -> np.ndarray:
        t = min(self._t, len(self.features) - 1)
        regime_ohe = np.zeros(4, dtype=np.float32)
        regime_ohe[min(self.regimes[t], 3)] = 1.0
        last_pnl = self._portfolio_rets[-1] if self._portfolio_rets else 0.0
        neural = self._current_neural_probs
        return np.concatenate([
            self.features[t],
            self._weights,
            [last_pnl],
            regime_ohe,
            neural,
        ]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Allocator wrapper
# --------------------------------------------------------------------------- #

class RLAllocator:
    def __init__(self, n_strategies: int) -> None:
        self.n_strategies = n_strategies
        self._model: Any = None
        self._equal = np.ones(n_strategies, dtype=np.float32) / n_strategies
        # Sidecar metadata written at save time: strategy_names (canonical
        # ordering the action vector is defined against), trained_on,
        # trained_at_utc. None when the model predates metadata (e.g. the old
        # mock-data-trained artifact) — such models must not drive live weights.
        self.metadata: dict | None = None

    @property
    def observation_dim(self) -> int | None:
        """Box observation size from the loaded PPO policy, if any."""
        if self._model is None:
            return None
        space = getattr(self._model, "observation_space", None)
        shape = getattr(space, "shape", None) if space is not None else None
        if not shape:
            return None
        return int(shape[0])

    def train(
        self,
        returns: np.ndarray,
        features: np.ndarray,
        regimes: np.ndarray,
        timesteps: int | None = None,
        neural_probs: np.ndarray | None = None,
    ) -> None:
        if not _SB3_AVAILABLE:
            log.warning("SB3 unavailable — skipping RL training")
            return

        timesteps = timesteps or cfg.rl_timesteps

        # If no neural-prob sequence is supplied, vary the dims randomly rather
        # than freezing them at 0.25 — frozen training dims made the live
        # neural-prob inputs out-of-distribution noise to the policy.
        if neural_probs is None:
            rng = np.random.default_rng(42)
            neural_probs = rng.dirichlet(np.ones(4) * 2.0, size=len(returns)).astype(np.float32)

        def make_env():
            return StrategyAllocEnv(returns, features, regimes, neural_probs=neural_probs)

        env = DummyVecEnv([make_env])
        device = cfg.device if _gpu_available() else "cpu"
        self._model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0,
            device=device,
        )
        log.info("Training PPO allocator for %d timesteps on %s ...", timesteps, device)
        self._model.learn(total_timesteps=timesteps)
        log.info("PPO training complete")

    def get_weights(self, obs: np.ndarray) -> np.ndarray:
        """
        obs: 1-D array matching the env observation space.
        Returns softmax-normalised strategy weights.
        """
        if self._model is None or not _SB3_AVAILABLE:
            return self._equal.copy()
        vec = np.asarray(obs, dtype=np.float32).reshape(-1)
        exp = int(self._model.observation_space.shape[0])
        if vec.shape[0] != exp:
            log.error(
                "RL observation length mismatch: got %d, policy expects %d — using equal weights",
                int(vec.shape[0]),
                exp,
            )
            return self._equal.copy()
        action, _ = self._model.predict(vec, deterministic=True)
        weights = np.exp(action) / (np.exp(action).sum() + 1e-10)
        return weights.astype(np.float32)

    def save(self, path: Path | str, metadata: dict | None = None) -> None:
        if self._model is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))
        if metadata is not None:
            import json

            self.metadata = dict(metadata)
            path.with_suffix(".meta.json").write_text(
                json.dumps(self.metadata, indent=2), encoding="utf-8"
            )
        log.info("RLAllocator saved to %s", path)

    def load(self, path: Path | str) -> "RLAllocator":
        if not _SB3_AVAILABLE:
            return self
        path = Path(path)
        if path.with_suffix(".zip").exists() or path.exists():
            try:
                self._model = PPO.load(str(path))
                log.info("RLAllocator loaded from %s", path)
            except Exception as exc:
                log.warning("Could not load RL model: %s", exc)
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                import json

                self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("Could not read RL metadata sidecar: %s", exc)
                self.metadata = None
        return self

    def is_valid_for(self, strategy_names: list[str]) -> bool:
        """
        True only when a model is loaded AND its metadata proves it was trained
        on real returns for exactly this (ordered) strategy list. Models without
        metadata — including the legacy mock-data-trained artifact — fail this
        check, and callers must fall back to equal weighting.
        """
        if self._model is None or not self.metadata:
            return False
        if self.metadata.get("trained_on") != "backtest_returns":
            return False
        return list(self.metadata.get("strategy_names") or []) == list(strategy_names)


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def load_or_init_allocator(
    n_strategies: int,
    expected_strategy_names: list[str] | None = None,
) -> RLAllocator:
    path = cfg.model_dir / "rl_allocator"
    allocator = RLAllocator(n_strategies)
    if (path.with_suffix(".zip")).exists() or path.exists():
        allocator.load(path)
        if expected_strategy_names is not None and not allocator.is_valid_for(expected_strategy_names):
            log.warning(
                "RL allocator model at %s has no valid metadata for the current "
                "strategy set — ignoring it (equal weights until retrained).",
                path,
            )
            allocator._model = None
    return allocator
