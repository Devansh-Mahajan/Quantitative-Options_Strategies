"""Central configuration — loads from .env and exposes typed fields."""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(_ROOT / ".env", override=False)


def _bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, default))


def _list(key: str, default: str = "") -> list[str]:
    raw = os.getenv(key, default)
    return [s.strip() for s in raw.split(",") if s.strip()]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        norm = item.upper()
        if norm in seen:
            continue
        seen.add(norm)
        ordered.append(norm)
    return ordered


def _interleave(primary: list[str], secondary: list[str], total_limit: int | None = None) -> list[str]:
    merged: list[str] = []
    max_len = max(len(primary), len(secondary))
    for idx in range(max_len):
        if idx < len(primary):
            merged.append(primary[idx])
        if idx < len(secondary):
            merged.append(secondary[idx])
    if total_limit is not None:
        merged = merged[:total_limit]
    return _dedupe(merged)


@dataclass
class Config:
    # --- Broker selection ---
    broker: str = field(default_factory=lambda: os.getenv("BROKER", "alpaca").lower())

    # --- Alpaca credentials ---
    alpaca_api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_api_secret: str = field(default_factory=lambda: os.getenv("ALPACA_API_SECRET", ""))
    alpaca_paper: bool = field(default_factory=lambda: _bool("ALPACA_PAPER", True))

    # --- Binance credentials ---
    api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    testnet: bool = field(default_factory=lambda: _bool("BINANCE_TESTNET", False))

    # --- Capital ---
    initial_capital: float = field(default_factory=lambda: _float("INITIAL_CAPITAL", 10_000))
    max_risk_per_trade: float = field(default_factory=lambda: _float("MAX_RISK_PER_TRADE", 0.02))
    max_portfolio_risk: float = field(default_factory=lambda: _float("MAX_PORTFOLIO_RISK", 0.20))
    daily_loss_limit: float = field(default_factory=lambda: _float("DAILY_LOSS_LIMIT", 0.05))
    max_drawdown: float = field(default_factory=lambda: _float("MAX_DRAWDOWN", 0.15))
    max_leverage: int = field(default_factory=lambda: _int("MAX_LEVERAGE", 10))
    kelly_fraction: float = field(default_factory=lambda: _float("KELLY_FRACTION", 0.25))

    # --- Markets ---
    enable_spot: bool = field(default_factory=lambda: _bool("ENABLE_SPOT", True))
    enable_futures: bool = field(default_factory=lambda: _bool("ENABLE_FUTURES", True))
    enable_options: bool = field(default_factory=lambda: _bool("ENABLE_OPTIONS", False))
    enable_coinm: bool = field(default_factory=lambda: _bool("ENABLE_COINM", False))

    # --- Strategies ---
    enable_momentum: bool = field(default_factory=lambda: _bool("ENABLE_MOMENTUM", True))
    enable_mean_reversion: bool = field(default_factory=lambda: _bool("ENABLE_MEAN_REVERSION", True))
    enable_funding_arb: bool = field(default_factory=lambda: _bool("ENABLE_FUNDING_ARB", True))
    enable_basis_trade: bool = field(default_factory=lambda: _bool("ENABLE_BASIS_TRADE", True))
    enable_pairs_arb: bool = field(default_factory=lambda: _bool("ENABLE_PAIRS_ARB", True))
    enable_options_vol: bool = field(default_factory=lambda: _bool("ENABLE_OPTIONS_VOL", False))
    enable_order_flow: bool = field(default_factory=lambda: _bool("ENABLE_ORDER_FLOW", True))
    enable_breakout: bool = field(default_factory=lambda: _bool("ENABLE_BREAKOUT", True))
    # Elite strategies (batch 1)
    enable_statistical_arb: bool = field(default_factory=lambda: _bool("ENABLE_STATISTICAL_ARB", True))
    enable_cross_sectional_momentum: bool = field(default_factory=lambda: _bool("ENABLE_CROSS_SECTIONAL_MOMENTUM", True))
    enable_liquidation_cascade: bool = field(default_factory=lambda: _bool("ENABLE_LIQUIDATION_CASCADE", True))
    enable_carry_portfolio: bool = field(default_factory=lambda: _bool("ENABLE_CARRY_PORTFOLIO", True))
    # Research-paper strategies (batch 2 — Kakushadze, Bloch, Cartea et al.)
    enable_tsmom: bool = field(default_factory=lambda: _bool("ENABLE_TSMOM", True))
    enable_quant_factors: bool = field(default_factory=lambda: _bool("ENABLE_QUANT_FACTORS", True))
    enable_contrarian_oi: bool = field(default_factory=lambda: _bool("ENABLE_CONTRARIAN_OI", True))
    enable_rma_strategy: bool = field(default_factory=lambda: _bool("ENABLE_RMA_STRATEGY", True))
    enable_vpin_flow: bool = field(default_factory=lambda: _bool("ENABLE_VPIN_FLOW", True))
    enable_knn_predictor: bool = field(default_factory=lambda: _bool("ENABLE_KNN_PREDICTOR", True))
    enable_pivot_sr: bool = field(default_factory=lambda: _bool("ENABLE_PIVOT_SR", True))
    enable_hp_trend: bool = field(default_factory=lambda: _bool("ENABLE_HP_TREND", True))
    enable_momentum_carry_combo: bool = field(default_factory=lambda: _bool("ENABLE_MOMENTUM_CARRY_COMBO", True))
    enable_microstructure_pressure: bool = field(default_factory=lambda: _bool("ENABLE_MICROSTRUCTURE_PRESSURE", True))
    enable_pullback_confluence: bool = field(default_factory=lambda: _bool("ENABLE_PULLBACK_CONFLUENCE", True))
    # High-alpha quantitative strategies (batch 3 — Avellaneda-Stoikov, Taleb, Gatheral)
    enable_market_making: bool = field(default_factory=lambda: _bool("ENABLE_MARKET_MAKING", True))
    enable_gamma_scalping: bool = field(default_factory=lambda: _bool("ENABLE_GAMMA_SCALPING", True))
    enable_vol_surface_arb: bool = field(default_factory=lambda: _bool("ENABLE_VOL_SURFACE_ARB", True))

    # --- Symbol universe ---
    futures_symbols: list[str] = field(
        default_factory=lambda: _list(
            "FUTURES_SYMBOLS",
            "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT,DOGEUSDT,AVAXUSDT",
        )
    )
    spot_symbols: list[str] = field(
        default_factory=lambda: _list("SPOT_SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT")
    )
    stock_symbols: list[str] = field(
        default_factory=lambda: _list(
            "STOCK_SYMBOLS",
            "SPY,QQQ,AAPL,MSFT,NVDA,AMZN,META,GOOGL",
        )
    )
    options_underlying: list[str] = field(
        default_factory=lambda: _list("OPTIONS_UNDERLYING", "BTC,ETH")
    )
    regime_anchor_symbol: str = field(default_factory=lambda: os.getenv("REGIME_ANCHOR_SYMBOL", "").strip().upper())
    benchmark_symbol: str = field(default_factory=lambda: os.getenv("BENCHMARK_SYMBOL", "").strip().upper())

    # --- ML ---
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "cuda"))
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("MODEL_DIR", str(_ROOT / "models" / "live"))))
    training_dir: Path = field(default_factory=lambda: Path(os.getenv("TRAINING_DIR", str(_ROOT / "models" / "training"))))
    retrain_lookback_days: int = field(default_factory=lambda: _int("RETRAIN_LOOKBACK_DAYS", 90))
    lstm_lookback: int = field(default_factory=lambda: _int("LSTM_LOOKBACK", 50))
    hmm_states: int = field(default_factory=lambda: _int("HMM_STATES", 4))
    rl_timesteps: int = field(default_factory=lambda: _int("RL_TIMESTEPS", 200_000))

    # --- Execution ---
    order_timeout_seconds: int = field(default_factory=lambda: _int("ORDER_TIMEOUT_SECONDS", 30))
    max_reprices: int = field(default_factory=lambda: _int("MAX_REPRICES", 3))
    maker_only: bool = field(default_factory=lambda: _bool("MAKER_ONLY", False))
    twap_slices: int = field(default_factory=lambda: _int("TWAP_SLICES", 5))
    min_notional: float = field(default_factory=lambda: _float("MIN_NOTIONAL", 10.0))

    # --- Notifications ---
    discord_webhook_url: str = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL", ""))
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    # --- Logging ---
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", str(_ROOT / "logging"))))

    @property
    def is_alpaca(self) -> bool:
        return self.broker == "alpaca"

    @property
    def is_binance(self) -> bool:
        return self.broker == "binance"

    @property
    def crypto_symbols(self) -> list[str]:
        return _dedupe(self.futures_symbols + self.spot_symbols)

    @property
    def asset_universe(self) -> list[str]:
        return _dedupe(self.stock_symbols + self.crypto_symbols)

    def balanced_symbols(
        self,
        *,
        stock_limit: int | None = None,
        crypto_limit: int | None = None,
        total_limit: int | None = None,
    ) -> list[str]:
        stocks = self.stock_symbols[:stock_limit] if stock_limit is not None else list(self.stock_symbols)
        crypto = self.crypto_symbols[:crypto_limit] if crypto_limit is not None else self.crypto_symbols
        return _interleave(stocks, crypto, total_limit=total_limit)

    @property
    def runtime_symbols(self) -> list[str]:
        return self.balanced_symbols(stock_limit=6, crypto_limit=6, total_limit=12)

    @property
    def model_symbols(self) -> list[str]:
        return self.balanced_symbols(stock_limit=4, crypto_limit=4, total_limit=8)

    @property
    def risk_symbols(self) -> list[str]:
        return self.balanced_symbols(stock_limit=5, crypto_limit=5, total_limit=10)

    @property
    def benchmark_symbols(self) -> list[str]:
        return self.balanced_symbols(stock_limit=4, crypto_limit=4, total_limit=8)

    @property
    def live_runtime_symbols(self) -> list[str]:
        return self.runtime_symbols if self.is_alpaca else self.crypto_symbols

    @property
    def live_model_symbols(self) -> list[str]:
        return self.model_symbols if self.is_alpaca else self.crypto_symbols[:4]

    @property
    def primary_regime_symbol(self) -> str:
        if self.regime_anchor_symbol:
            return self.regime_anchor_symbol
        if self.stock_symbols:
            return self.stock_symbols[0]
        if self.crypto_symbols:
            return self.crypto_symbols[0]
        return "SPY"

    @property
    def default_benchmark_symbol(self) -> str:
        if self.benchmark_symbol:
            return self.benchmark_symbol
        if self.benchmark_symbols:
            return self.benchmark_symbols[0]
        return self.primary_regime_symbol

    @property
    def live_primary_regime_symbol(self) -> str:
        if self.is_alpaca:
            return self.primary_regime_symbol
        if self.crypto_symbols:
            return self.crypto_symbols[0]
        return self.primary_regime_symbol

    def is_crypto_symbol(self, symbol: str) -> bool:
        sym = (symbol or "").strip().upper()
        if not sym:
            return False
        if sym in {s.upper() for s in self.crypto_symbols}:
            return True
        if "/" in sym:
            base, quote = sym.split("/", 1)
            crypto_bases = {
                s[:-4] for s in self.crypto_symbols if s.endswith("USDT")
            } | {
                s[:-3] for s in self.crypto_symbols if s.endswith("USD")
            } | {
                s[:-4] for s in self.crypto_symbols if s.endswith("BUSD")
            }
            return quote in {"USD", "USDT", "BUSD"} and base in crypto_bases
        return any(sym.endswith(suffix) and len(sym) > len(suffix) for suffix in ("USDT", "BUSD", "USD"))

    def is_stock_symbol(self, symbol: str) -> bool:
        sym = (symbol or "").strip().upper()
        if not sym or self.is_crypto_symbol(sym):
            return False
        if sym in {s.upper() for s in self.stock_symbols}:
            return True
        if "/" in sym:
            return False
        cleaned = sym.replace(".", "").replace("-", "")
        return cleaned.isalpha()

    def validate(self) -> None:
        if self.broker not in ("alpaca", "binance"):
            raise ValueError(f"BROKER must be 'alpaca' or 'binance', got '{self.broker}'")
        if self.is_alpaca:
            if not self.alpaca_api_key or self.alpaca_api_key == "your_alpaca_key_here":
                raise ValueError("ALPACA_API_KEY is not set. Copy .env.example to .env and fill in credentials.")
            if not self.alpaca_api_secret or self.alpaca_api_secret == "your_alpaca_secret_here":
                raise ValueError("ALPACA_API_SECRET is not set.")
        else:
            if not self.api_key or self.api_key == "your_binance_key_here":
                raise ValueError("BINANCE_API_KEY is not set. Copy .env.example to .env and fill in credentials.")
            if not self.api_secret or self.api_secret == "your_binance_secret_here":
                raise ValueError("BINANCE_API_SECRET is not set.")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# Singleton — imported everywhere
cfg = Config()
