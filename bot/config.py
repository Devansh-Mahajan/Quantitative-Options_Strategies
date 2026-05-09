"""Central configuration — loads from .env and exposes typed fields."""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=False)


def _bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, default))


def _list(key: str, default: str = "") -> list[str]:
    raw = os.getenv(key, default)
    return [s.strip() for s in raw.split(",") if s.strip()]


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
    # Elite strategies
    enable_statistical_arb: bool = field(default_factory=lambda: _bool("ENABLE_STATISTICAL_ARB", True))
    enable_cross_sectional_momentum: bool = field(default_factory=lambda: _bool("ENABLE_CROSS_SECTIONAL_MOMENTUM", True))
    enable_liquidation_cascade: bool = field(default_factory=lambda: _bool("ENABLE_LIQUIDATION_CASCADE", True))
    enable_carry_portfolio: bool = field(default_factory=lambda: _bool("ENABLE_CARRY_PORTFOLIO", True))

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
    options_underlying: list[str] = field(
        default_factory=lambda: _list("OPTIONS_UNDERLYING", "BTC,ETH")
    )

    # --- ML ---
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "cuda"))
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("MODEL_DIR", "./models/live")))
    training_dir: Path = field(default_factory=lambda: Path(os.getenv("TRAINING_DIR", "./models/training")))
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
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", "./logging")))

    @property
    def is_alpaca(self) -> bool:
        return self.broker == "alpaca"

    @property
    def is_binance(self) -> bool:
        return self.broker == "binance"

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
