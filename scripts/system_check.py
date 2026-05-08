"""
Comprehensive pre-flight system check.

Catches every class of bug before you waste time running a full backtest or live bot.
Checks: packages, GPU, disk, config, exchange connectivity, strategy imports,
        backtester modules, and a 200-bar smoke-test backtest.

Usage:
    python -m scripts.system_check          # full check (requires live exchange)
    python -m scripts.system_check --quick  # skip connectivity + smoke test
"""

from __future__ import annotations
import asyncio
import importlib
import inspect
import logging
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.config import cfg
from bot.logger import setup_logging

setup_logging(cfg.log_dir, "INFO")
log = logging.getLogger("scripts.system_check")

# ─────────────────────────────────────────────────────────────────────────────
# Package manifest — (import_name, install_hint, fatal?)
# ─────────────────────────────────────────────────────────────────────────────

CORE_PACKAGES: list[tuple[str, str, bool]] = [
    ("numpy",             "pip install numpy",            True),
    ("pandas",            "pip install pandas",           True),
    ("scipy",             "pip install scipy",            True),
    ("sklearn",           "pip install scikit-learn",     True),
    ("torch",             "pip install torch",            False),
    ("hmmlearn",          "pip install hmmlearn",         False),
    ("httpx",             "pip install httpx",            True),
    ("tqdm",              "pip install tqdm",             False),
    ("optuna",            "pip install optuna",           False),
    ("pyarrow",           "pip install pyarrow",          True),   # Parquet cache
]

BROKER_PACKAGES: dict[str, list[tuple[str, str, bool]]] = {
    "binance": [
        ("binance",       "pip install python-binance",  True),
    ],
    "alpaca": [
        ("alpaca",        'pip install "alpaca-py"',     True),
    ],
}

# Strategy registry: (module_path, class_name)
STRATEGIES: list[tuple[str, str]] = [
    ("strategies.momentum",      "MomentumStrategy"),
    ("strategies.mean_reversion","MeanReversionStrategy"),
    ("strategies.funding_arb",   "FundingArbStrategy"),
    ("strategies.basis_trade",   "BasisTradeStrategy"),
    ("strategies.pairs_arb",     "PairsArbStrategy"),
    ("strategies.options_vol",   "OptionsVolatilityStrategy"),
    ("strategies.order_flow",    "OrderFlowStrategy"),
    ("strategies.breakout",      "BreakoutStrategy"),
]

# Core backtester/risk modules to import-check
INTERNAL_MODULES: list[str] = [
    "backtester.engine",
    "backtester.fill_model",
    "backtester.metrics",
    "backtester.report",
    "backtester.walk_forward",
    "backtester.monte_carlo_bt",
    "backtester.optimizer",
    "backtester.data_loader",
    "risk.monte_carlo",
    "risk.portfolio_risk",
    "execution.position_manager",
    "exchange.streams",
    "ml.features",
    "bot.config",
    "bot.state",
    "bot.notifications",
]

_PASS  = "[PASS]"
_WARN  = "[WARN]"
_FAIL  = "[FAIL]"


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def check_packages() -> bool:
    all_ok = True
    packages = CORE_PACKAGES + BROKER_PACKAGES.get(cfg.broker, [])
    for import_name, hint, fatal in packages:
        try:
            importlib.import_module(import_name)
            log.info("  %s %s", _PASS, import_name)
        except ImportError:
            level = log.error if fatal else log.warning
            tag   = _FAIL if fatal else _WARN
            level("  %s %-28s  →  %s", tag, import_name, hint)
            if fatal:
                all_ok = False
    return all_ok


def check_gpu() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info("  %s GPU: %s  (%.1f GB VRAM)", _PASS, name, vram)
        else:
            log.warning("  %s CUDA not available — CPU mode (training will be slower)", _WARN)
        return True
    except ImportError:
        log.warning("  %s PyTorch not installed", _WARN)
        return True   # non-fatal


def check_disk() -> bool:
    usage = shutil.disk_usage(".")
    free_gb = usage.free / 1e9
    if free_gb < 5.0:
        log.error("  %s Only %.1f GB free — need ≥ 5 GB", _FAIL, free_gb)
        return False
    log.info("  %s Disk: %.1f GB free", _PASS, free_gb)
    return True


def check_env() -> bool:
    try:
        cfg.validate()
        key = cfg.api_key or ""
        display = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "(short)"
        log.info("  %s .env loaded — broker=%s  key=%s", _PASS, cfg.broker, display)
        return True
    except ValueError as exc:
        log.error("  %s .env invalid: %s", _FAIL, exc)
        return False


def check_internal_modules() -> bool:
    all_ok = True
    for mod in INTERNAL_MODULES:
        try:
            importlib.import_module(mod)
            log.info("  %s %s", _PASS, mod)
        except Exception as exc:
            log.error("  %s %-40s  →  %s", _FAIL, mod, exc)
            all_ok = False
    return all_ok


def check_strategies() -> bool:
    """
    Import every strategy class, instantiate it, and verify the required interface.
    This catches: wrong class names, missing __init__ args, broken imports.
    """
    all_ok = True
    required_attrs = ["name", "generate_signals"]
    for mod_path, class_name in STRATEGIES:
        label = f"{mod_path}.{class_name}"
        try:
            mod = importlib.import_module(mod_path)
        except Exception as exc:
            log.error("  %s import %s  →  %s", _FAIL, label, exc)
            all_ok = False
            continue

        cls = getattr(mod, class_name, None)
        if cls is None:
            # Try to find what the module actually exports to give a helpful hint
            public = [n for n in dir(mod) if not n.startswith("_") and inspect.isclass(getattr(mod, n))]
            log.error("  %s %s not found in module. Classes found: %s", _FAIL, label, public)
            all_ok = False
            continue

        try:
            instance = cls()
        except Exception as exc:
            log.error("  %s instantiate %s  →  %s", _FAIL, label, exc)
            all_ok = False
            continue

        missing = [a for a in required_attrs if not hasattr(instance, a)]
        if missing:
            log.error("  %s %s missing attrs: %s", _FAIL, label, missing)
            all_ok = False
        else:
            log.info("  %s %-52s  name='%s'", _PASS, label, instance.name)

    return all_ok


def check_backtester_metrics() -> bool:
    """Validate that metrics module computes without error on synthetic data."""
    try:
        import numpy as np
        import pandas as pd
        from backtester.metrics import compute_metrics

        eq = pd.Series(
            [10000 * (1 + 0.001 * i + 0.002 * np.sin(i)) for i in range(200)],
            index=pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC"),
            name="equity",
        )
        rets  = [0.01, -0.005, 0.02, -0.03] * 20
        holds = [10, 5, 20, 8] * 20
        m = compute_metrics(eq, rets, holds)
        log.info(
            "  %s metrics smoke test  Sharpe=%.3f  MaxDD=%.2f%%",
            _PASS, m.sharpe, m.max_drawdown_pct,
        )
        return True
    except Exception as exc:
        log.error("  %s metrics smoke test failed: %s", _FAIL, exc)
        log.debug(traceback.format_exc())
        return False


def check_fill_model() -> bool:
    """Verify fill model processes a bar without error."""
    try:
        from backtester.fill_model import FillModel, make_market_order
        fm = FillModel()
        order = make_market_order("BTCUSDT", "BUY", 0.001, "test", 0)
        bar = {"open": 50000, "high": 50500, "low": 49800, "close": 50200,
               "volume": 100, "atr": 400}
        fills = fm.process_bar(bar, [order], 0)
        log.info("  %s fill_model smoke test  fills=%d", _PASS, len(fills))
        return True
    except Exception as exc:
        log.error("  %s fill_model smoke test failed: %s", _FAIL, exc)
        log.debug(traceback.format_exc())
        return False


def check_smoke_backtest() -> bool:
    """
    Run a 200-bar mini backtest on synthetic data.
    Catches runtime errors that only appear once the engine actually runs.
    """
    try:
        import numpy as np
        import pandas as pd
        from backtester.engine import BacktestEngine
        from strategies.momentum import MomentumStrategy
        from strategies.breakout import BreakoutStrategy

        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        np.random.seed(42)
        price = 50000 * np.cumprod(1 + np.random.normal(0.0001, 0.005, n))
        df = pd.DataFrame({
            "open_time": dates.astype(np.int64) // 10**6,
            "open":   price * (1 - 0.0002),
            "high":   price * (1 + 0.005),
            "low":    price * (1 - 0.005),
            "close":  price,
            "volume": np.random.uniform(100, 1000, n),
        })

        data = {"BTCUSDT": df}
        engine = BacktestEngine(
            data=data,
            interval="1h",
            initial_equity=10_000,
            strategies=[MomentumStrategy(), BreakoutStrategy()],
            lookback=60,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
        )
        result = engine.run()
        log.info(
            "  %s smoke backtest (200 bars)  return=%.2f%%  trades=%d",
            _PASS,
            result.metrics.total_return_pct,
            len(result.trades),
        )
        return True
    except Exception as exc:
        log.error("  %s smoke backtest failed: %s", _FAIL, exc)
        log.debug(traceback.format_exc())
        return False


async def check_broker_connectivity() -> bool:
    from exchange import get_client
    client = get_client()
    try:
        await client.start()
        ts    = await client.get_server_time()
        price = await client.get_price("BTCUSDT")
        await client.close()
        log.info(
            "  %s %s API  BTC=%.2f USD  server_time=%d",
            _PASS, cfg.broker.upper(), price, ts,
        )
        return True
    except Exception as exc:
        log.error("  %s %s API: %s", _FAIL, cfg.broker.upper(), exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

async def _run_checks(quick: bool = False) -> int:
    sep = "─" * 58
    def section(n: str, num: int) -> None:
        log.info("\n%s\n  [%d] %s\n%s", sep, num, n, sep)

    log.info("═" * 58)
    log.info("  SYSTEM PRE-FLIGHT CHECK%s", "  (quick mode)" if quick else "")
    log.info("═" * 58)

    failures = 0

    section("Python packages", 1)
    if not check_packages():
        failures += 1

    section("GPU / CUDA", 2)
    check_gpu()

    section("Disk space", 3)
    if not check_disk():
        failures += 1

    section("Environment (.env)", 4)
    if not check_env():
        failures += 1

    section("Internal modules (import check)", 5)
    if not check_internal_modules():
        failures += 1

    section("Strategy classes (import + instantiate)", 6)
    if not check_strategies():
        failures += 1

    section("Backtester — metrics smoke test", 7)
    if not check_backtester_metrics():
        failures += 1

    section("Backtester — fill model smoke test", 8)
    if not check_fill_model():
        failures += 1

    if not quick:
        section("Backtester — 200-bar smoke backtest", 9)
        if not check_smoke_backtest():
            failures += 1

        section(f"{cfg.broker.upper()} API connectivity", 10)
        if not await check_broker_connectivity():
            failures += 1

    log.info("\n" + "═" * 58)
    if failures == 0:
        log.info("  ALL CHECKS PASSED — system is ready")
    else:
        log.error("  %d CHECK(S) FAILED — fix the above before running", failures)
    log.info("═" * 58)
    return failures


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Pre-flight system check")
    p.add_argument("--quick", action="store_true",
                   help="Skip connectivity and smoke-backtest checks")
    args = p.parse_args()
    failures = asyncio.run(_run_checks(quick=args.quick))
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
