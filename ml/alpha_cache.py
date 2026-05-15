"""
AlphaCache: thin in-process cache for the ml_alpha snapshot JSON.

The weekend training pipeline writes .runtime/ml_alpha_snapshot.json once a
week.  This wrapper re-reads the file at most once per TTL (default 1 h) so
the 60-second trading cycle never hits disk on every tick.
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path

from core.ml_alpha import AlphaSignal

log = logging.getLogger("ml.alpha_cache")

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_PATH = _ROOT / ".runtime" / "ml_alpha_snapshot.json"


class AlphaCache:
    def __init__(
        self,
        cache_path: Path | str = DEFAULT_CACHE_PATH,
        ttl_seconds: float = 3600.0,
    ) -> None:
        self._path = Path(cache_path)
        self._ttl = float(ttl_seconds)
        self._loaded_at: float = 0.0
        self._data: dict[str, AlphaSignal] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def load(self, symbols: list[str]) -> dict[str, AlphaSignal]:
        """Return {symbol: AlphaSignal} for each requested symbol in cache."""
        if time.monotonic() - self._loaded_at > self._ttl:
            self._refresh()
        return {sym: self._data[sym] for sym in symbols if sym in self._data}

    def invalidate(self) -> None:
        """Force reload on next call (call after weekend training deploys)."""
        self._loaded_at = 0.0

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _refresh(self) -> None:
        if not self._path.exists():
            log.debug("AlphaCache: snapshot not found at %s", self._path)
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            self._data = {
                sig["symbol"]: AlphaSignal(**sig)
                for sig in payload.get("signals", [])
                if "symbol" in sig
            }
            self._loaded_at = time.monotonic()
            log.debug("AlphaCache refreshed: %d signals from %s", len(self._data), self._path)
        except Exception as exc:
            log.warning("AlphaCache refresh failed: %s", exc)
