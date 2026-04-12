from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

logger = logging.getLogger(f"strategy.{__name__}")

# Broker/runtime symbols on the left, Yahoo Finance equivalents on the right.
YAHOO_SYMBOL_ALIASES = {
    "SQ": "XYZ",
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}


def normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def dedupe_symbols(symbols: Iterable[str]) -> list[str]:
    return list(dict.fromkeys([normalize_symbol(symbol) for symbol in symbols if normalize_symbol(symbol)]))


def resolve_download_symbol(symbol: str) -> str:
    canonical = normalize_symbol(symbol)
    return YAHOO_SYMBOL_ALIASES.get(canonical, canonical)


def load_symbol_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    return dedupe_symbols(path.read_text(encoding="utf-8").splitlines())


def save_symbol_file(path: Path, symbols: Iterable[str]) -> None:
    cleaned = dedupe_symbols(symbols)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(cleaned)
    if body:
        body += "\n"
    path.write_text(body, encoding="utf-8")


def _extract_frame(raw, field: str, requested: list[str]) -> pd.DataFrame:
    if raw is None or getattr(raw, "empty", True):
        return pd.DataFrame()

    field_frame = raw.get(field) if isinstance(raw, pd.DataFrame) else None
    if field_frame is None or getattr(field_frame, "empty", True):
        return pd.DataFrame()

    if isinstance(field_frame, pd.Series):
        if len(requested) == 1:
            field_frame = field_frame.to_frame(name=requested[0])
        else:
            field_frame = field_frame.to_frame()

    return field_frame.dropna(how="all")


def download_close_matrix(
    symbols: Iterable[str],
    period: str = "10y",
    auto_adjust: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    canonical_symbols = dedupe_symbols(symbols)
    if not canonical_symbols:
        return pd.DataFrame()

    resolved_to_canonical = {resolve_download_symbol(symbol): symbol for symbol in canonical_symbols}
    requested = list(resolved_to_canonical.keys())

    try:
        raw = yf.download(
            requested,
            period=period,
            auto_adjust=auto_adjust,
            progress=progress,
            threads=False,
        )
    except Exception as exc:
        logger.warning("Bulk close download failed for %s: %s", period, exc)
        raw = pd.DataFrame()

    close = _extract_frame(raw, "Close", requested)
    if not close.empty:
        close = close.rename(columns=resolved_to_canonical)

    missing_symbols = [symbol for symbol in canonical_symbols if symbol not in close.columns]
    for symbol in missing_symbols:
        resolved = resolve_download_symbol(symbol)
        try:
            single = yf.download(
                resolved,
                period=period,
                auto_adjust=auto_adjust,
                progress=False,
                threads=False,
            )
            single_close = _extract_frame(single, "Close", [resolved])
            if not single_close.empty:
                series = single_close.iloc[:, 0]
                close[symbol] = series
        except Exception as exc:
            logger.debug("Single-symbol download failed for %s (%s): %s", symbol, resolved, exc)

    if close.empty:
        return pd.DataFrame()

    close = close.reindex(columns=[symbol for symbol in canonical_symbols if symbol in close.columns])
    return close.dropna(how="all").sort_index()


def validate_symbol_universe(
    symbols: Iterable[str],
    recent_period: str = "6mo",
    history_period: str = "2y",
    min_recent_rows: int = 40,
    min_history_rows: int = 120,
) -> dict:
    canonical_symbols = dedupe_symbols(symbols)
    recent = download_close_matrix(canonical_symbols, period=recent_period, auto_adjust=True, progress=False)
    history = download_close_matrix(canonical_symbols, period=history_period, auto_adjust=True, progress=False)

    valid_symbols: list[str] = []
    invalid_symbols: list[dict] = []

    for symbol in canonical_symbols:
        recent_rows = int(recent[symbol].dropna().shape[0]) if symbol in recent.columns else 0
        history_rows = int(history[symbol].dropna().shape[0]) if symbol in history.columns else 0

        if recent_rows >= min_recent_rows and history_rows >= min_history_rows:
            valid_symbols.append(symbol)
            continue

        reason = "download_failed"
        if recent_rows == 0 and history_rows > 0:
            reason = "stale_recent_data"
        elif recent_rows > 0 and history_rows < min_history_rows:
            reason = "insufficient_history"
        elif recent_rows < min_recent_rows and history_rows >= min_history_rows:
            reason = "insufficient_recent_rows"

        invalid_symbols.append(
            {
                "symbol": symbol,
                "resolved_symbol": resolve_download_symbol(symbol),
                "recent_rows": recent_rows,
                "history_rows": history_rows,
                "reason": reason,
            }
        )

    return {
        "symbols_input": len(canonical_symbols),
        "symbols_valid": len(valid_symbols),
        "symbols_invalid": len(invalid_symbols),
        "valid_symbols": valid_symbols,
        "invalid_symbols": invalid_symbols,
        "recent_period": recent_period,
        "history_period": history_period,
        "min_recent_rows": int(min_recent_rows),
        "min_history_rows": int(min_history_rows),
        "alias_map": {symbol: resolve_download_symbol(symbol) for symbol in canonical_symbols},
    }


def save_validation_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
