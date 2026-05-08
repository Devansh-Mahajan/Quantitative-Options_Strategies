"""
Vectorized bar-by-bar backtester.
Feeds a SimulatedMarketDataStore with historical windows so existing live-trading
strategies run completely unchanged — no strategy code is duplicated here.
No look-ahead bias: strategies only see data up to bar i-1 at bar i.
"""

from __future__ import annotations
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backtester.fill_model import FillModel, Order, Fill, OrderType, make_market_order, make_limit_order
from backtester.metrics import PerformanceMetrics, compute_metrics
from exchange.streams import MarketDataStore, Candle, BookTicker
from ml.features import compute_features
from bot.config import cfg

log = logging.getLogger("backtester.engine")

STOP_LOSS_PCT = 0.05       # 5% hard stop per position
TAKE_PROFIT_PCT = 0.10     # 10% take-profit per position


# --------------------------------------------------------------------------- #
# Simulated Market Data Store
# --------------------------------------------------------------------------- #

class SimulatedMarketDataStore(MarketDataStore):
    """
    Extends MarketDataStore for backtest use.
    `set_window(symbol, interval, df, i)` loads data up to bar i.
    """

    def __init__(self) -> None:
        super().__init__()
        self._raw: dict[tuple[str, str], pd.DataFrame] = {}

    def load_symbol(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        self._raw[(symbol, interval)] = df.reset_index()

    def set_time_index(self, i: int) -> None:
        """Advance the store to bar i (strategies see only bars 0..i-1)."""
        for (symbol, interval), df in self._raw.items():
            window = df.iloc[:i]
            key = (symbol, interval)
            self.history[key] = deque(
                (
                    Candle(
                        symbol=symbol, interval=interval,
                        open_time=int(pd.Timestamp(row.get("open_time", 0)).timestamp() * 1000) if not isinstance(row.get("open_time", 0), (int, float)) else int(row.get("open_time", 0)),
                        open=float(row["open"]), high=float(row["high"]),
                        low=float(row["low"]), close=float(row["close"]),
                        volume=float(row["volume"]), closed=True,
                    )
                    for _, row in window.iterrows()
                ),
                maxlen=500,
            )
            if len(window):
                last = window.iloc[-1]
                self.candles[key] = Candle(
                    symbol=symbol, interval=interval,
                    open_time=int(pd.Timestamp(last.get("open_time", 0)).timestamp() * 1000) if not isinstance(last.get("open_time", 0), (int, float)) else int(last.get("open_time", 0)),
                    open=float(last["open"]), high=float(last["high"]),
                    low=float(last["low"]), close=float(last["close"]),
                    volume=float(last["volume"]), closed=True,
                )
                # Synthetic book ticker
                mid = float(last["close"])
                self.book[symbol] = BookTicker(symbol=symbol, bid=mid * 0.9999, ask=mid * 1.0001)


# --------------------------------------------------------------------------- #
# Open position tracker
# --------------------------------------------------------------------------- #

@dataclass
class SimPosition:
    symbol: str
    side: str        # LONG | SHORT
    quantity: float
    entry_price: float
    entry_bar: int
    strategy: str
    market: str = "futures"

    def pnl(self, current_price: float) -> float:
        if self.side == "LONG":
            return (current_price - self.entry_price) * self.quantity
        return (self.entry_price - current_price) * self.quantity

    def pnl_pct(self, current_price: float) -> float:
        return self.pnl(current_price) / (self.entry_price * self.quantity + 1e-10)


@dataclass
class BacktestResult:
    equity_curve: pd.Series                    # DatetimeIndex → equity value
    trades: pd.DataFrame                       # one row per closed trade
    metrics: PerformanceMetrics
    per_strategy_metrics: dict[str, PerformanceMetrics]
    signals_fired: int
    fills_executed: int
    config_snapshot: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

class BacktestEngine:
    """
    Usage:
        from backtester.engine import BacktestEngine
        from backtester.data_loader import load_multi
        from strategies.registry import build_registry

        data = load_multi(["BTCUSDT","ETHUSDT"], "1h", "2023-01-01", "2024-01-01")
        engine = BacktestEngine(
            data=data, interval="1h",
            initial_equity=10_000,
            strategies=build_registry(),
        )
        result = engine.run()
        result.metrics  # PerformanceMetrics
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        interval: str,
        initial_equity: float = 10_000,
        strategies=None,
        fill_model: FillModel | None = None,
        lookback: int = 60,              # bars to warm up before first signal
        max_open_positions: int = 8,
        stop_loss_pct: float = STOP_LOSS_PCT,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        max_risk_per_trade: float | None = None,
    ) -> None:
        self.data = data
        self.interval = interval
        self.initial_equity = initial_equity
        self.strategies = strategies or []
        self.fill_model = fill_model or FillModel()
        self.lookback = lookback
        self.max_open_positions = max_open_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_risk_per_trade = max_risk_per_trade or cfg.max_risk_per_trade

        # Validate data
        if not data:
            raise ValueError("data dict is empty")
        self._anchor_sym = next(iter(data))
        self._anchor_df = data[self._anchor_sym].reset_index()
        self._n_bars = len(self._anchor_df)

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def run(self) -> BacktestResult:
        log.info(
            "Backtest starting: %d symbols × %d bars (%s to %s)",
            len(self.data), self._n_bars,
            self._anchor_df["open_time"].iloc[0] if "open_time" in self._anchor_df.columns else "?",
            self._anchor_df["open_time"].iloc[-1] if "open_time" in self._anchor_df.columns else "?",
        )

        store = SimulatedMarketDataStore()
        for sym, df in self.data.items():
            for interval in ["1m", "5m", "15m", "1h", "4h"]:
                store.load_symbol(sym, interval, df)   # all use same df; strategies filter by interval

        # Build ATR series per symbol for fill model
        atr_series: dict[str, np.ndarray] = {}
        for sym, df in self.data.items():
            df = df.reset_index()
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"] - df["close"].shift()).abs(),
            ], axis=1).max(axis=1)
            atr_series[sym] = tr.rolling(14).mean().bfill().values

        equity = self.initial_equity
        equity_ts: list[tuple] = []
        positions: list[SimPosition] = []
        pending_orders: list[Order] = []
        closed_trades: list[dict] = []
        total_signals = 0

        for i in range(self.lookback, self._n_bars):
            bar_ts = self._anchor_df.get("open_time", pd.Series()).iloc[i] if i < self._n_bars else None

            # ---------------------------------------------------------
            # 1. Advance store window (no look-ahead)
            # ---------------------------------------------------------
            store.set_time_index(i)

            # ---------------------------------------------------------
            # 2. Process fills for pending orders using this bar's OHLC
            # ---------------------------------------------------------
            for sym in self.data:
                bar_row = self.data[sym].reset_index().iloc[i]
                bar_dict = {
                    "open": float(bar_row["open"]),
                    "high": float(bar_row["high"]),
                    "low": float(bar_row["low"]),
                    "close": float(bar_row["close"]),
                    "volume": float(bar_row["volume"]),
                    "atr": atr_series[sym][i],
                }
                sym_orders = [o for o in pending_orders if o.symbol == sym and not o.is_done]
                fills = self.fill_model.process_bar(bar_dict, sym_orders, i)
                for fill in fills:
                    equity -= fill.net_cost  # cost already signed (buy=negative cash flow)
                    self._apply_fill_to_positions(fill, positions)

            # Remove completed orders
            pending_orders = [o for o in pending_orders if not o.is_done]

            # ---------------------------------------------------------
            # 3. Exit existing positions (stop-loss / take-profit)
            # ---------------------------------------------------------
            for pos in list(positions):
                sym_row = self.data[pos.symbol].reset_index().iloc[i]
                price = float(sym_row["close"])
                pnl_pct = pos.pnl_pct(price)

                if pnl_pct <= -self.stop_loss_pct or pnl_pct >= self.take_profit_pct:
                    close_side = "SELL" if pos.side == "LONG" else "BUY"
                    exit_order = make_market_order(
                        pos.symbol, close_side, pos.quantity, "exit_manager", i
                    )
                    pending_orders.append(exit_order)
                    pnl_abs = pos.pnl(price)
                    closed_trades.append({
                        "symbol": pos.symbol,
                        "strategy": pos.strategy,
                        "side": pos.side,
                        "entry_price": pos.entry_price,
                        "exit_price": price,
                        "quantity": pos.quantity,
                        "pnl_pct": pnl_pct,
                        "pnl_abs": pnl_abs,
                        "holding_bars": i - pos.entry_bar,
                        "bar_index": i,
                    })
                    positions.remove(pos)

            # ---------------------------------------------------------
            # 4. Generate signals from strategies
            # ---------------------------------------------------------
            if len(positions) < self.max_open_positions:
                regime = self._estimate_regime(store, self._anchor_sym)
                predictions: dict[str, Any] = {}  # no ML during backtest (avoids look-ahead)

                for strategy in self.strategies:
                    try:
                        sigs = strategy.generate_signals(store, regime, predictions)
                        total_signals += len(sigs)
                        for sig in sigs:
                            if sig.quantity > 0 or True:  # always size internally
                                order = self._signal_to_order(sig, equity, i, atr_series)
                                if order:
                                    pending_orders.append(order)
                    except Exception as exc:
                        log.debug("Strategy %s error at bar %d: %s", strategy.name, i, exc)

            # ---------------------------------------------------------
            # 5. Mark-to-market all open positions
            # ---------------------------------------------------------
            unrealised = sum(
                pos.pnl(float(self.data[pos.symbol].reset_index().iloc[i]["close"]))
                for pos in positions
                if pos.symbol in self.data
            )
            total_equity = equity + unrealised

            if bar_ts is not None:
                equity_ts.append((bar_ts, total_equity))

            if i % 500 == 0:
                log.info("  bar %d/%d  equity=%.2f  positions=%d  pending=%d",
                         i, self._n_bars, total_equity, len(positions), len(pending_orders))

        # ---------------------------------------------------------
        # 6. Force-close any remaining positions at last bar
        # ---------------------------------------------------------
        last_i = self._n_bars - 1
        for pos in positions:
            price = float(self.data[pos.symbol].reset_index().iloc[last_i]["close"])
            pnl_pct = pos.pnl_pct(price)
            closed_trades.append({
                "symbol": pos.symbol, "strategy": pos.strategy,
                "side": pos.side, "entry_price": pos.entry_price,
                "exit_price": price, "quantity": pos.quantity,
                "pnl_pct": pnl_pct, "pnl_abs": pos.pnl(price),
                "holding_bars": last_i - pos.entry_bar, "bar_index": last_i,
            })
            equity += pos.pnl(price)

        # ---------------------------------------------------------
        # 7. Build results
        # ---------------------------------------------------------
        eq_series = pd.Series(
            [v for _, v in equity_ts],
            index=pd.DatetimeIndex([t for t, _ in equity_ts], tz="UTC"),
            name="equity",
        )
        trades_df = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()

        trade_rets = trades_df["pnl_pct"].tolist() if "pnl_pct" in trades_df.columns else []
        hold_bars = trades_df["holding_bars"].tolist() if "holding_bars" in trades_df.columns else []

        metrics = compute_metrics(eq_series, trade_rets, hold_bars)
        per_strat = self._per_strategy_metrics(trades_df, eq_series)

        log.info("Backtest complete — %.2f%% total return, Sharpe=%.3f, MaxDD=%.2f%%",
                 metrics.total_return_pct, metrics.sharpe, metrics.max_drawdown_pct)

        return BacktestResult(
            equity_curve=eq_series,
            trades=trades_df,
            metrics=metrics,
            per_strategy_metrics=per_strat,
            signals_fired=total_signals,
            fills_executed=len(closed_trades),
            config_snapshot={
                "symbols": list(self.data.keys()),
                "interval": self.interval,
                "initial_equity": self.initial_equity,
                "strategies": [s.name for s in self.strategies],
                "stop_loss_pct": self.stop_loss_pct,
                "take_profit_pct": self.take_profit_pct,
            },
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _signal_to_order(
        self, sig, equity: float, bar_index: int, atr_series: dict
    ) -> Order | None:
        """Convert a live Signal to a backtester Order with realistic sizing."""
        price = sig.price or float(self.data.get(sig.symbol, self.data[self._anchor_sym])
                                   .reset_index().iloc[bar_index]["close"])
        if price <= 0:
            return None

        # Size: risk-based
        notional = equity * self.max_risk_per_trade * sig.confidence
        notional = max(notional, cfg.min_notional)
        quantity = notional / price

        if sig.price > 0 and not cfg.maker_only:
            return make_limit_order(sig.symbol, sig.side, quantity, sig.price, sig.strategy, bar_index)
        return make_market_order(sig.symbol, sig.side, quantity, sig.strategy, bar_index)

    def _apply_fill_to_positions(self, fill: Fill, positions: list[SimPosition]) -> None:
        """Open or close a position based on a fill."""
        existing = next((p for p in positions if p.symbol == fill.order.symbol), None)

        if fill.order.side == "BUY":
            if existing and existing.side == "SHORT":
                positions.remove(existing)   # closing short
            else:
                positions.append(SimPosition(
                    symbol=fill.order.symbol,
                    side="LONG",
                    quantity=fill.quantity,
                    entry_price=fill.fill_price,
                    entry_bar=fill.bar_index,
                    strategy=fill.order.strategy,
                ))
        else:  # SELL
            if existing and existing.side == "LONG":
                positions.remove(existing)   # closing long
            else:
                positions.append(SimPosition(
                    symbol=fill.order.symbol,
                    side="SHORT",
                    quantity=fill.quantity,
                    entry_price=fill.fill_price,
                    entry_bar=fill.bar_index,
                    strategy=fill.order.strategy,
                ))

    def _estimate_regime(self, store: SimulatedMarketDataStore, symbol: str) -> str:
        """Lightweight regime proxy during backtest (no HMM to avoid look-ahead)."""
        try:
            closes = store.get_closes(symbol, self.interval, 30)
            if len(closes) < 20:
                return "ranging"
            arr = np.array(closes)
            rets = np.diff(np.log(arr))
            vol = rets.std()
            trend = (arr[-1] - arr[-20]) / arr[-20]
            if vol > 0.03:
                return "volatile"
            if trend > 0.05:
                return "bull"
            if trend < -0.05:
                return "bear"
            return "ranging"
        except Exception:
            return "ranging"

    def _per_strategy_metrics(
        self, trades_df: pd.DataFrame, eq_series: pd.Series
    ) -> dict[str, PerformanceMetrics]:
        if trades_df.empty or "strategy" not in trades_df.columns:
            return {}
        result = {}
        for strat in trades_df["strategy"].unique():
            strat_trades = trades_df[trades_df["strategy"] == strat]
            rets = strat_trades["pnl_pct"].tolist()
            holds = strat_trades["holding_bars"].tolist()
            result[strat] = compute_metrics(eq_series, rets, holds)
        return result
