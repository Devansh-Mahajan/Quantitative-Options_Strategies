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
from risk.position_sizer import PositionSizer
from bot.config import cfg

log = logging.getLogger("backtester.engine")

STOP_LOSS_PCT = 0.05       # fallback hard stop (used only if PositionSizer produces no plan)
TAKE_PROFIT_PCT = 0.10     # fallback take-profit


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
        self._prebuilt: dict[tuple[str, str], list[Candle]] = {}
        self._last_i: int = 0

    def load_symbol(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        self._raw[(symbol, interval)] = df.reset_index()

    def _ts_to_ms(self, val) -> int:
        if isinstance(val, (int, float)):
            return int(val)
        return int(pd.Timestamp(val).timestamp() * 1000)

    def prebuild(self) -> None:
        """Pre-convert all raw DataFrames to Candle lists once — O(n) total."""
        for (symbol, interval), df in self._raw.items():
            candles = []
            for _, row in df.iterrows():
                candles.append(Candle(
                    symbol=symbol, interval=interval,
                    open_time=self._ts_to_ms(row.get("open_time", 0)),
                    open=float(row["open"]), high=float(row["high"]),
                    low=float(row["low"]), close=float(row["close"]),
                    volume=float(row["volume"]), closed=True,
                ))
            self._prebuilt[(symbol, interval)] = candles
            self.history[(symbol, interval)] = deque(maxlen=500)

    def set_time_index(self, i: int) -> None:
        """Incrementally advance store to bar i — O(1) per call after prebuild."""
        if not self._prebuilt:
            self.prebuild()

        for (symbol, interval), candles in self._prebuilt.items():
            key = (symbol, interval)
            # Append only newly visible bars since last call
            appended = False
            for j in range(self._last_i, min(i, len(candles))):
                self.history[key].append(candles[j])
                appended = True
            if appended:
                # CRITICAL: invalidate the DataFrame cache. Appending to the
                # deque directly bypasses the live ingest path that flags
                # _df_dirty, so get_history_df() previously returned a FROZEN
                # early-window frame for the whole backtest — every strategy
                # computing indicators from get_history_df saw static data,
                # silently producing zero signals in historical backtests.
                self._df_dirty[key] = True
            if i > 0 and i <= len(candles):
                last = candles[i - 1]
                self.candles[key] = last
                self.book[symbol] = BookTicker(
                    symbol=symbol,
                    bid=last.close * 0.9999,
                    ask=last.close * 1.0001,
                )
        self._last_i = i


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
    stop_loss_pct: float = STOP_LOSS_PCT
    take_profit_pct: float = TAKE_PROFIT_PCT

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
        if fill_model is None:
            # Charge what the LIVE venue charges (Alpaca crypto is ~6x Binance
            # futures) — a backtest at the wrong fee tier flatters turnover.
            from backtester.fill_model import venue_fees

            maker, taker = venue_fees(cfg.broker, "spot" if cfg.is_alpaca else "futures")
            fill_model = FillModel(maker_fee=maker, taker_fee=taker)
        self.fill_model = fill_model
        self.lookback = lookback
        self.max_open_positions = max_open_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_risk_per_trade = max_risk_per_trade or cfg.max_risk_per_trade
        self._sizer = PositionSizer()
        from strategies.base import SignalCooldown

        self._cooldown = SignalCooldown(gap=int(getattr(cfg, "signal_cooldown_cycles", 4)))

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

        # Build ATR and rolling-vol series per symbol for fill model + dynamic stops
        atr_series: dict[str, np.ndarray] = {}
        ann_vol_series: dict[str, np.ndarray] = {}
        ann_factor = np.sqrt(365 * 24)  # hourly crypto annualisation (matches GARCHVolModel)
        for sym, df in self.data.items():
            df = df.reset_index()
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"] - df["close"].shift()).abs(),
            ], axis=1).max(axis=1)
            atr_series[sym] = tr.rolling(14).mean().bfill().values
            log_ret = np.log(df["close"] / df["close"].shift(1))
            ann_vol_series[sym] = (log_ret.rolling(20).std() * ann_factor).bfill().values

        equity = self.initial_equity
        equity_ts: list[tuple] = []
        positions: list[SimPosition] = []
        pending_orders: list[Order] = []
        closed_trades: list[dict] = []
        total_signals = 0

        try:
            from tqdm import tqdm
            bar_iter = tqdm(
                range(self.lookback, self._n_bars),
                desc="Backtesting",
                unit="bar",
                dynamic_ncols=True,
            )
        except ImportError:
            bar_iter = range(self.lookback, self._n_bars)

        for i in bar_iter:
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
                    vol = float(ann_vol_series.get(sym, [0.5] * (i + 1))[i] or 0.5)
                    exit_plan = self._sizer.exit_plan(realised_vol=vol, regime=self._estimate_regime(store, sym))
                    self._apply_fill_to_positions(
                        fill, positions,
                        stop_loss_pct=exit_plan.stop_loss_pct,
                        take_profit_pct=exit_plan.take_profit_pct,
                    )

            # Remove completed orders
            pending_orders = [o for o in pending_orders if not o.is_done]

            # ---------------------------------------------------------
            # 3. Exit existing positions (stop-loss / take-profit)
            #
            # Stops/TPs are checked INTRABAR against high/low (close-only
            # checks understate drawdowns), and settled synchronously through
            # the fill model at the triggered level (with fees + slippage).
            # NOTE: exits must NOT be queued as pending orders — the old code
            # did that after already removing the position, and the next-bar
            # fill re-opened a phantom REVERSED position, corrupting every
            # historical backtest.
            # ---------------------------------------------------------
            for pos in list(positions):
                sym_row = self.data[pos.symbol].reset_index().iloc[i]
                bar_open = float(sym_row["open"])
                bar_high = float(sym_row["high"])
                bar_low = float(sym_row["low"])

                if pos.side == "LONG":
                    stop_price = pos.entry_price * (1.0 - pos.stop_loss_pct)
                    tp_price = pos.entry_price * (1.0 + pos.take_profit_pct)
                    stop_hit = bar_low <= stop_price
                    tp_hit = bar_high >= tp_price
                    # Pessimistic: when both trigger in one bar, assume the stop.
                    if stop_hit:
                        exit_price = min(bar_open, stop_price)  # gap-through fills worse
                    elif tp_hit:
                        exit_price = max(bar_open, tp_price)    # gap-through fills better
                    else:
                        continue
                else:  # SHORT
                    stop_price = pos.entry_price * (1.0 + pos.stop_loss_pct)
                    tp_price = pos.entry_price * (1.0 - pos.take_profit_pct)
                    stop_hit = bar_high >= stop_price
                    tp_hit = bar_low <= tp_price
                    if stop_hit:
                        exit_price = max(bar_open, stop_price)
                    elif tp_hit:
                        exit_price = min(bar_open, tp_price)
                    else:
                        continue

                fee = exit_price * pos.quantity * self.fill_model.taker_fee
                if pos.side == "LONG":
                    equity += exit_price * pos.quantity - fee   # sell proceeds
                else:
                    equity -= exit_price * pos.quantity + fee   # buy-back cost

                pnl_abs = pos.pnl(exit_price) - fee
                closed_trades.append({
                    "symbol": pos.symbol,
                    "strategy": pos.strategy,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "quantity": pos.quantity,
                    "pnl_pct": pos.pnl_pct(exit_price),
                    "pnl_abs": pnl_abs,
                    "holding_bars": i - pos.entry_bar,
                    "bar_index": i,
                    "exit_reason": "stop" if stop_hit else "take_profit",
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
                        # Same turnover throttle the live orchestrator applies,
                        # so backtests measure the throttled behavior.
                        sigs = self._cooldown.filter(sigs, i)
                        total_signals += len(sigs)
                        for sig in sigs:
                            sym_vol = float(ann_vol_series.get(sig.symbol, [0.5] * (i + 1))[min(i, len(ann_vol_series.get(sig.symbol, [0.5])) - 1)] or 0.5)
                            order = self._signal_to_order(
                                sig, equity, i, atr_series,
                                realised_vol=sym_vol, regime=regime,
                            )
                            if order:
                                pending_orders.append(order)
                    except Exception as exc:
                        log.debug("Strategy %s error at bar %d: %s", strategy.name, i, exc)

            # ---------------------------------------------------------
            # 5. Mark-to-market all open positions
            #
            # `equity` is CASH (entries debit/credit full notional via
            # fill.net_cost), so marking must add position VALUE (±qty*price),
            # not just PnL — the old +pnl-only marking made the equity curve
            # fake-drop by the full notional on every entry.
            # ---------------------------------------------------------
            open_value = sum(
                (1.0 if pos.side == "LONG" else -1.0)
                * pos.quantity
                * float(self.data[pos.symbol].reset_index().iloc[i]["close"])
                for pos in positions
                if pos.symbol in self.data
            )
            total_equity = equity + open_value

            if bar_ts is not None:
                equity_ts.append((bar_ts, total_equity))

            if i % 500 == 0:
                log.info("  bar %d/%d  equity=%.2f  positions=%d  pending=%d",
                         i, self._n_bars, total_equity, len(positions), len(pending_orders))
                if hasattr(bar_iter, "set_postfix"):
                    bar_iter.set_postfix(equity=f"${total_equity:,.0f}", pos=len(positions))

        # ---------------------------------------------------------
        # 6. Force-close any remaining positions at last bar
        # ---------------------------------------------------------
        last_i = self._n_bars - 1
        for pos in positions:
            price = float(self.data[pos.symbol].reset_index().iloc[last_i]["close"])
            pnl_pct = pos.pnl_pct(price)
            fee = price * pos.quantity * self.fill_model.taker_fee
            closed_trades.append({
                "symbol": pos.symbol, "strategy": pos.strategy,
                "side": pos.side, "entry_price": pos.entry_price,
                "exit_price": price, "quantity": pos.quantity,
                "pnl_pct": pnl_pct, "pnl_abs": pos.pnl(price) - fee,
                "holding_bars": last_i - pos.entry_bar, "bar_index": last_i,
                "exit_reason": "end_of_data",
            })
            # Cash settlement consistent with the entry-side notional debit.
            if pos.side == "LONG":
                equity += price * pos.quantity - fee
            else:
                equity -= price * pos.quantity + fee

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
        self, sig, equity: float, bar_index: int, atr_series: dict,
        realised_vol: float = 0.5, regime: str = "ranging",
    ) -> Order | None:
        """Convert a live Signal to a backtester Order with realistic sizing."""
        price = sig.price or float(self.data.get(sig.symbol, self.data[self._anchor_sym])
                                   .reset_index().iloc[bar_index]["close"])
        if price <= 0:
            return None

        # Cost-aware edge floor — parity with risk/position_sizer.size_signal:
        # skip trades whose profit target can't clear a multiple of the
        # roundtrip cost at this engine's OWN fee schedule.
        plan = self._sizer.exit_plan(realised_vol=realised_vol, regime=regime)
        roundtrip = 2.0 * self.fill_model.taker_fee + 0.0010
        if plan.take_profit_pct < float(getattr(cfg, "cost_edge_multiple", 3.0)) * roundtrip:
            return None

        # Size: risk-based
        notional = equity * self.max_risk_per_trade * sig.confidence
        notional = max(notional, cfg.min_notional)
        quantity = notional / price

        if sig.price > 0 and not cfg.maker_only:
            return make_limit_order(sig.symbol, sig.side, quantity, sig.price, sig.strategy, bar_index)
        return make_market_order(sig.symbol, sig.side, quantity, sig.strategy, bar_index)

    def _apply_fill_to_positions(
        self,
        fill: Fill,
        positions: list[SimPosition],
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ) -> None:
        """Open or close a position based on a fill."""
        existing = next((p for p in positions if p.symbol == fill.order.symbol), None)
        sl = stop_loss_pct or self.stop_loss_pct
        tp = take_profit_pct or self.take_profit_pct

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
                    stop_loss_pct=sl,
                    take_profit_pct=tp,
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
                    stop_loss_pct=sl,
                    take_profit_pct=tp,
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
        """
        Attribute metrics per strategy from that strategy's OWN trades.

        Each strategy gets an equity series built from its cumulative trade
        PnL (indexed by exit-bar timestamp) — passing the shared PORTFOLIO
        curve to every strategy (the old behavior) contaminated per-strategy
        Sharpe/return/drawdown with everyone else's trades.
        """
        if trades_df.empty or "strategy" not in trades_df.columns:
            return {}
        result = {}
        for strat in trades_df["strategy"].unique():
            strat_trades = trades_df[trades_df["strategy"] == strat].sort_values("bar_index")
            rets = strat_trades["pnl_pct"].tolist()
            holds = strat_trades["holding_bars"].tolist()

            strat_eq = eq_series
            try:
                # eq_series row 0 corresponds to bar `lookback`, not bar 0.
                bar_positions = (strat_trades["bar_index"].astype(int) - self.lookback).clip(0, len(eq_series) - 1)
                timestamps = eq_series.index[bar_positions.to_numpy()]
                equity_values = self.initial_equity + strat_trades["pnl_abs"].cumsum().to_numpy()
                own = pd.Series(equity_values, index=timestamps, name="equity")
                # Prepend the starting point so returns/drawdowns are anchored.
                start_point = pd.Series([self.initial_equity], index=[eq_series.index[0]], name="equity")
                own = pd.concat([start_point, own])
                own = own[~own.index.duplicated(keep="last")]
                if len(own) >= 2:
                    strat_eq = own
            except Exception:
                pass  # fall back to portfolio curve rather than dropping the strategy

            result[strat] = compute_metrics(strat_eq, rets, holds)
        return result
