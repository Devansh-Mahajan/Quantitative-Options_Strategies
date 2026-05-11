"""
Main orchestrator: ties exchange, ML models, strategies, risk, and execution together.
Runs a 60-second async trading cycle with parallel WebSocket market data.
"""

from __future__ import annotations
import asyncio
import logging
import time
from collections import defaultdict
from typing import Any

import numpy as np

from bot.config import cfg
from bot import state, notifications
from bot.logger import setup_logging
from exchange import get_client
from exchange.streams import MarketDataStore
from ml.regime_hmm import RegimeHMM, load_or_init_hmm
from ml.price_lstm import PricePredictorTrainer, load_or_init_predictor
from ml.rl_allocator import RLAllocator, load_or_init_allocator
from ml.vol_model import GARCHVolModel, IVSurface
from ml.features import compute_features
from risk.portfolio_risk import PortfolioRiskEngine
from risk.position_sizer import PositionSizer, SizingContext
from risk.risk_guard import RiskGuard
from execution.order_manager import OrderManager
from execution.position_manager import PositionManager
from strategies.registry import build_registry
from strategies.base import Signal
from core.live_allocation import build_live_allocation_snapshot, market_session_open_now, write_live_allocation
from core.trade_decision_tape import record_trade_decision

log = logging.getLogger("bot.orchestrator")

# How often each sub-task runs (seconds)
CYCLE_INTERVAL = 60
RISK_CHECK_INTERVAL = 30
POSITION_REFRESH_INTERVAL = 30
FUNDING_REFRESH_INTERVAL = 300    # 5 min
OI_REFRESH_INTERVAL = 300


ALPACA_CRYPTO_SPOT_STRATEGIES = {
    "momentum",
    "mean_reversion",
    "order_flow",
    "breakout",
    "tsmom",
    "knn_predictor",
    "pivot_sr",
    "hp_trend",
    "microstructure_pressure",
    "pullback_confluence",
    "liquidation_cascade",
    "contrarian_oi",
    "rma_strategy",
}


class Orchestrator:
    def __init__(
        self,
        *,
        runtime_symbols: list[str] | None = None,
        model_symbols: list[str] | None = None,
        primary_regime_symbol: str | None = None,
        book_name: str = "primary",
        long_only_spot: bool = False,
        strategy_whitelist: set[str] | None = None,
    ) -> None:
        self.client = get_client()
        self.store = MarketDataStore()
        self.stream_manager: StreamManager | None = None
        self.book_name = str(book_name or "primary")
        self.runtime_symbols = self._normalize_symbols(runtime_symbols or cfg.live_runtime_symbols)
        self.model_symbols = self._normalize_symbols(model_symbols or cfg.live_model_symbols)
        self.primary_regime_symbol = str(
            (primary_regime_symbol or cfg.live_primary_regime_symbol or (self.model_symbols[0] if self.model_symbols else "SPY"))
        ).upper()
        self.long_only_spot = bool(long_only_spot and cfg.is_alpaca)
        self.strategy_whitelist = {str(name).strip() for name in (strategy_whitelist or set()) if str(name).strip()}
        self._latest_live_allocation: dict[str, Any] = {}

        # ML models — keyed by primary symbol for regime/price prediction
        self.regime_hmm: dict[str, RegimeHMM] = {}
        self.predictors: dict[str, PricePredictorTrainer] = {}
        self.vol_models: dict[str, GARCHVolModel] = {}
        self.iv_surfaces: dict[str, IVSurface] = {}    # per underlying (BTC, ETH)
        self.rl_allocator: RLAllocator | None = None

        # Store references on store for strategies to access
        self.store.iv_surfaces = self.iv_surfaces       # type: ignore[attr-defined]
        self.store.options_chains = {}                  # type: ignore[attr-defined]

        self.strategies = build_registry()
        if self.strategy_whitelist:
            self.strategies = [strategy for strategy in self.strategies if strategy.name in self.strategy_whitelist]
        self.risk_engine = PortfolioRiskEngine()
        self.sizer = PositionSizer()
        self.guard = RiskGuard()
        self.order_manager: OrderManager | None = None
        self.position_manager: PositionManager | None = None

        self._start_equity: float = 0.0
        self._last_funding_refresh = 0.0
        self._last_oi_refresh = 0.0
        self._cycle_count = 0

    @staticmethod
    def _normalize_symbols(symbols: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            normalized = str(symbol or "").strip().upper()
            if not normalized or normalized in seen:
                continue
            cleaned.append(normalized)
            seen.add(normalized)
        return cleaned

    def _position_symbol_filter(self, symbol: str) -> bool:
        normalized = str(symbol or "").strip().upper()
        if not normalized:
            return False
        if self.book_name == "crypto":
            return cfg.is_crypto_symbol(normalized)
        return True

    # ------------------------------------------------------------------ #
    # Startup
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        cfg.validate()
        setup_logging(cfg.log_dir, cfg.log_level)
        state.init_db()

        broker_info = f"alpaca(paper={cfg.alpaca_paper})" if cfg.is_alpaca else f"binance(testnet={cfg.testnet})"
        log.info(
            "Starting Quant Bot — broker=%s book=%s long_only_spot=%s runtime_symbols=%s",
            broker_info,
            self.book_name,
            self.long_only_spot,
            self.runtime_symbols,
        )
        await notifications.alert(
            f"Bot starting — broker={broker_info}, strategies={[s.name for s in self.strategies]}",
            title="Bot Startup",
        )

        await self.client.start()
        self.order_manager = OrderManager(self.client)
        self.position_manager = PositionManager(self.client, symbol_filter=self._position_symbol_filter)

        # Warm up stream manager (Binance or Alpaca depending on BROKER)
        all_symbols = self.runtime_symbols
        self.stream_manager = self.client.create_stream_manager(self.store)
        await self.stream_manager.start(all_symbols, intervals=["1m", "5m", "15m", "1h", "4h"])

        # Load ML models
        await self._load_ml_models()

        # Seed initial equity
        balance = await self.client.get_futures_balance()
        account_snapshot = await self.client.get_account_snapshot() if hasattr(self.client, "get_account_snapshot") else {}
        self._start_equity = float(
            account_snapshot.get("equity", 0.0)
            or balance.get("USDT", cfg.initial_capital)
            or balance.get("USD", cfg.initial_capital)
        )
        self.guard.maybe_reset_daily(self._start_equity)

        log.info("Startup complete — initial equity=%.2f", self._start_equity)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        await self.start()

        tasks = [
            asyncio.create_task(self._trading_loop(), name="trading_loop"),
            asyncio.create_task(self._risk_monitor_loop(), name="risk_monitor"),
            asyncio.create_task(self._funding_refresh_loop(), name="funding_refresh"),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            log.info("Orchestrator shutting down...")
        finally:
            if self.stream_manager:
                await self.stream_manager.stop()
            await self.client.close()
            log.info("Orchestrator stopped")

    async def _sync_live_allocation(self, equity: float) -> None:
        if not cfg.is_alpaca or not hasattr(self.client, "get_all_positions_raw"):
            return
        try:
            positions = await self.client.get_all_positions_raw()
            snapshot = build_live_allocation_snapshot(
                total_equity=equity,
                positions=positions,
                market_open=market_session_open_now(),
            )
            self._latest_live_allocation = write_live_allocation(snapshot)
        except Exception as exc:
            log.debug("Live allocation sync skipped: %s", exc)

    async def _trading_loop(self) -> None:
        while True:
            t0 = time.monotonic()
            try:
                await self._trading_cycle()
            except Exception as exc:
                log.exception("Trading cycle error: %s", exc)
                await notifications.alert(f"Cycle error: {exc}", level="ERROR")
            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0, CYCLE_INTERVAL - elapsed))

    async def _risk_monitor_loop(self) -> None:
        while True:
            try:
                await self._run_risk_checks()
            except Exception as exc:
                log.warning("Risk monitor error: %s", exc)
            await asyncio.sleep(RISK_CHECK_INTERVAL)

    async def _funding_refresh_loop(self) -> None:
        while True:
            try:
                await self._refresh_funding_and_oi()
            except Exception as exc:
                log.debug("Funding refresh error: %s", exc)
            await asyncio.sleep(FUNDING_REFRESH_INTERVAL)

    # ------------------------------------------------------------------ #
    # Trading cycle
    # ------------------------------------------------------------------ #

    async def _trading_cycle(self) -> None:
        self._cycle_count += 1
        log.info("=== Cycle %d ===", self._cycle_count)

        # --- 1. Refresh positions ---
        await self.position_manager.refresh()

        # --- 2. Get current equity ---
        balance = await self.client.get_futures_balance()
        account_snapshot = await self.client.get_account_snapshot() if hasattr(self.client, "get_account_snapshot") else {}
        equity = float(
            account_snapshot.get("equity", 0)
            or balance.get("USDT", self._start_equity)
            or balance.get("USD", self._start_equity)
        )
        await self._sync_live_allocation(equity)
        state.record_equity(equity, self.guard.current_drawdown if hasattr(self.guard, 'current_drawdown') else 0)

        # --- 3. Daily reset ---
        self.guard.maybe_reset_daily(equity)

        # --- 4. Check exits ---
        await self._handle_exits()

        # --- 5. Monitor pending orders ---
        await self.order_manager.monitor_and_reprice()

        # --- 6. Determine regime ---
        regime = self._get_regime()
        log.info("Market regime: %s", regime)

        # --- 7. ML price predictions ---
        predictions = await self._get_predictions()

        # --- 8. Collect strategy signals ---
        raw_signals = self._collect_signals(regime, predictions)
        log.info("Raw signals: %d", len(raw_signals))
        for sig in raw_signals:
            record_trade_decision(
                status="SUGGESTED",
                strategy=sig.strategy,
                symbol=sig.symbol,
                action=f"{sig.side.lower()}_{sig.market}",
                confidence=sig.confidence,
                reason="Strategy generated raw signal.",
                details={"market": sig.market, "quantity": sig.quantity, "price": sig.price, "regime": regime, "meta": sig.meta},
            )

        # --- 9. Apply RL allocator weights ---
        signals = self._apply_rl_weights(raw_signals, equity, regime)

        # --- 10. Size positions ---
        sized = self._size_signals(signals, equity, regime)

        # --- 11. Risk guard filter ---
        peak_equity = state.get_peak_equity() or equity
        gross_exp = self.position_manager.gross_notional() / (equity + 1e-10)
        hhi = self.position_manager.hhi(equity)

        approved = self.guard.filter_signals(
            sized,
            current_equity=equity,
            peak_equity=peak_equity,
            start_equity=self.guard.daily_start_equity,
            gross_exposure=gross_exp,
            hhi=hhi,
            regime=regime,
        )
        log.info("Approved signals after risk filter: %d", len(approved))
        approved_keys = {(s.symbol, s.market, s.side, s.strategy, round(float(s.quantity or 0), 10)) for s in approved}
        for sig in sized:
            key = (sig.symbol, sig.market, sig.side, sig.strategy, round(float(sig.quantity or 0), 10))
            if key not in approved_keys:
                record_trade_decision(
                    status="GATED",
                    strategy=sig.strategy,
                    symbol=sig.symbol,
                    action=f"{sig.side.lower()}_{sig.market}",
                    confidence=sig.confidence,
                    reason="Risk guard filtered the sized signal before execution.",
                    details={"market": sig.market, "quantity": sig.quantity, "price": sig.price, "regime": regime, "meta": sig.meta},
                )

        # --- 12. Execute orders ---
        for sig in approved:
            vol_model = self.vol_models.get(sig.symbol) or self.vol_models.get(self.primary_regime_symbol) or GARCHVolModel()
            leverage = self.sizer.safe_leverage(
                vol_model.current_vol,
                regime,
            )
            try:
                await self.order_manager.execute(sig, leverage=leverage)
                record_trade_decision(
                    status="EXECUTED",
                    strategy=sig.strategy,
                    symbol=sig.symbol,
                    action=f"{sig.side.lower()}_{sig.market}",
                    confidence=sig.confidence,
                    reason="Order manager accepted approved signal for execution.",
                    details={"market": sig.market, "quantity": sig.quantity, "price": sig.price, "leverage": leverage, "regime": regime, "meta": sig.meta},
                )
            except Exception as exc:
                record_trade_decision(
                    status="REJECTED",
                    strategy=sig.strategy,
                    symbol=sig.symbol,
                    action=f"{sig.side.lower()}_{sig.market}",
                    confidence=sig.confidence,
                    reason=f"Order manager execution failed: {exc}",
                    details={"market": sig.market, "quantity": sig.quantity, "price": sig.price, "leverage": leverage, "regime": regime, "meta": sig.meta},
                )
                raise

        # --- 13. Log snapshot ---
        self._log_snapshot(equity, regime, len(approved))

    # ------------------------------------------------------------------ #
    # Risk checks (parallel loop)
    # ------------------------------------------------------------------ #

    async def _run_risk_checks(self) -> None:
        await self.position_manager.refresh(force=True)
        balance = await self.client.get_futures_balance()
        equity = float(balance.get("USDT", self._start_equity))

        snapshot = self.risk_engine.compute_snapshot(
            self.position_manager.position_list_for_risk(),
            equity,
        )

        if snapshot.current_drawdown >= cfg.max_drawdown:
            msg = f"DRAWDOWN ALERT: {snapshot.current_drawdown:.1%} >= {cfg.max_drawdown:.1%}"
            await notifications.alert(msg, title="Risk Alert", level="CRITICAL")
            self.guard.halt(msg)

        if snapshot.gross_exposure > cfg.max_leverage:
            await notifications.alert(
                f"Leverage alert: {snapshot.gross_exposure:.1f}x > {cfg.max_leverage}x",
                level="WARNING",
            )

    # ------------------------------------------------------------------ #
    # Exit management
    # ------------------------------------------------------------------ #

    async def _handle_exits(self) -> None:
        for pos in self.position_manager.all_positions():
            vol_model = self.vol_models.get(pos.symbol) or self.vol_models.get(self.primary_regime_symbol) or GARCHVolModel()
            regime = self._get_regime(pos.symbol)
            plan = self.sizer.exit_plan(
                realised_vol=vol_model.current_vol or 0.5,
                regime=regime,
                confidence=0.55,
            )
            exit_reason = ""
            if pos.pnl_pct <= -plan.stop_loss_pct:
                exit_reason = f"dynamic_stop_loss:{plan.stop_loss_pct:.2%}"
            elif pos.pnl_pct >= plan.take_profit_pct:
                exit_reason = f"dynamic_take_profit:{plan.take_profit_pct:.2%}"
            if not exit_reason:
                continue

            close_side = "SELL" if pos.side == "LONG" else "BUY"
            exit_sig = Signal(
                symbol=pos.symbol,
                market=pos.market,
                side=close_side,
                quantity=pos.quantity,
                price=0,     # market order for exits
                confidence=1.0,
                strategy="exit_manager",
                meta={"exit_reason": exit_reason, "exit_plan": plan.as_dict()},
            )
            prepared_exit = self._prepare_signal_for_execution(exit_sig, regime=regime, stage="exit")
            if prepared_exit is None:
                continue
            await self.order_manager.execute(prepared_exit)
            log.info(
                "Exiting %s %s pnl=%.2f%% reason=%s stop=%.2f%% take=%.2f%% regime=%s",
                pos.symbol,
                pos.side,
                pos.pnl_pct * 100,
                exit_reason,
                plan.stop_loss_pct * 100,
                plan.take_profit_pct * 100,
                regime,
            )

    # ------------------------------------------------------------------ #
    # ML helpers
    # ------------------------------------------------------------------ #

    def _get_regime(self, anchor_symbol: str | None = None) -> str:
        anchors: list[str] = []
        if anchor_symbol:
            anchors.append(anchor_symbol)
        anchors.extend([self.primary_regime_symbol, *self.model_symbols])

        seen: set[str] = set()
        for symbol in anchors:
            if symbol in seen:
                continue
            seen.add(symbol)
            hmm = self.regime_hmm.get(symbol)
            if hmm is None:
                continue
            df = self.store.get_history_df(symbol, "1h")
            if len(df) < 30:
                continue
            regime = hmm.predict_regime(df)
            if regime and regime != "unknown":
                return regime
        return "ranging"

    async def _get_predictions(self) -> dict[str, dict]:
        results: dict[str, dict] = {}
        for symbol in self.model_symbols:
            predictor = self.predictors.get(symbol)
            if predictor is None:
                continue
            df = self.store.get_history_df(symbol, "1h")
            if len(df) < cfg.lstm_lookback + 5:
                continue
            try:
                from ml.features import compute_feature_matrix
                X = compute_feature_matrix(
                    df,
                    funding_rate=self.store.funding.get(symbol, 0.0),
                    open_interest=self.store.open_interest.get(symbol, 0.0),
                )
                results[symbol] = predictor.predict(X)
            except Exception as exc:
                log.debug("Prediction error %s: %s", symbol, exc)
        return results

    def _collect_signals(self, regime: str, predictions: dict) -> list[Signal]:
        signals: list[Signal] = []
        for strategy in self.strategies:
            try:
                sigs = strategy.generate_signals(self.store, regime, predictions)
                for signal in sigs:
                    normalized = self._normalize_signal_for_book(signal)
                    if normalized is not None:
                        signals.append(normalized)
            except Exception as exc:
                log.warning("Strategy %s error: %s", strategy.name, exc)
        return signals

    def _normalize_signal_for_book(self, signal: Signal) -> Signal | None:
        symbol = str(signal.symbol or "").upper()
        if self.book_name == "crypto":
            if not cfg.is_crypto_symbol(symbol):
                return None
            if signal.market not in {"futures", "spot"}:
                return None
            if self.long_only_spot:
                return signal._replace(market="spot")
        return signal

    def _book_budget_remaining(self) -> float | None:
        if self.book_name != "crypto":
            return None
        books = self._latest_live_allocation.get("books") or {}
        book = books.get("crypto") or {}
        try:
            return max(0.0, float(book.get("remaining_risk_budget", 0.0)))
        except (TypeError, ValueError):
            return None

    def _position_for_symbol(self, symbol: str):
        normalized = str(symbol or "").upper()
        return (
            self.position_manager.position(normalized, market="spot")
            or self.position_manager.position(normalized, market="futures")
            or self.position_manager.position(normalized, market="options")
        )

    def _prepare_signal_for_execution(self, signal: Signal, *, regime: str, stage: str) -> Signal | None:
        if not self.long_only_spot or not cfg.is_crypto_symbol(signal.symbol):
            return signal

        meta = dict(signal.meta or {})
        if signal.side == "BUY":
            return signal._replace(market="spot", meta=meta)

        position = self._position_for_symbol(signal.symbol)
        held_qty = float(getattr(position, "quantity", 0.0) or 0.0)
        if held_qty <= 0:
            record_trade_decision(
                status="GATED",
                strategy=signal.strategy,
                symbol=signal.symbol,
                action=f"{signal.side.lower()}_spot",
                confidence=signal.confidence,
                reason="Spot-safe Alpaca crypto mode blocks naked short sells.",
                details={"market": signal.market, "stage": stage, "regime": regime, "meta": meta},
            )
            return None

        reduced_qty = min(float(signal.quantity or 0.0), held_qty)
        if reduced_qty <= 0:
            return None
        meta["reduce_only"] = True
        meta["held_quantity"] = held_qty
        return signal._replace(market="spot", quantity=reduced_qty, meta=meta)

    def _reference_price(self, symbol: str, fallback: float = 0.0) -> float:
        price = float(fallback or 0.0)
        if price > 0:
            return price
        book = self.store.book.get(str(symbol or "").upper())
        bid = float(getattr(book, "bid", 0.0) or 0.0)
        ask = float(getattr(book, "ask", 0.0) or 0.0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        for interval in ("1m", "5m", "15m", "1h"):
            closes = self.store.get_closes(str(symbol or "").upper(), interval, 1)
            if closes:
                close = float(closes[-1] or 0.0)
                if close > 0:
                    return close
        return 0.0

    def _apply_rl_weights(self, signals: list[Signal], equity: float, regime: str) -> list[Signal]:
        """Scale signal confidences by RL-derived strategy weights."""
        if self.rl_allocator is None:
            return signals

        strategy_names = list({s.name for s in self.strategies})
        regime_map = {"bull": 0, "bear": 1, "ranging": 2, "volatile": 3}
        regime_idx = regime_map.get(regime, 2)

        # Build a flat obs vector for RL (portfolio state)
        pos_count = len(self.position_manager.all_positions())
        dummy_obs = np.zeros(10 + len(strategy_names) + 5, dtype=np.float32)
        dummy_obs[regime_idx] = 1.0
        dummy_obs[4] = equity / (self._start_equity + 1e-10)
        dummy_obs[5] = pos_count / 20.0

        weights = self.rl_allocator.get_weights(dummy_obs)
        weight_map = {name: float(w) for name, w in zip(strategy_names, weights)}

        scaled = []
        for sig in signals:
            w = weight_map.get(sig.strategy, 1.0 / len(strategy_names))
            scaled.append(sig._replace(confidence=sig.confidence * w * len(strategy_names)))
        return scaled

    def _size_signals(self, signals: list[Signal], equity: float, regime: str) -> list[Signal]:
        sized = []
        positions = self.position_manager.all_positions()
        gross_exposure = self.position_manager.gross_notional()
        peak_equity = state.get_peak_equity() or equity
        daily_start_equity = self.guard.daily_start_equity or equity
        open_positions = len(positions)
        remaining_book_budget = self._book_budget_remaining()

        for sig in signals:
            vol_model = self.vol_models.get(sig.symbol) or self.vol_models.get(self.primary_regime_symbol) or GARCHVolModel()
            reference_price = self._reference_price(sig.symbol, sig.price)
            meta = dict(sig.meta or {})
            symbol_exposure = sum(p.notional for p in positions if p.symbol == sig.symbol)
            decision = self.sizer.size_signal(
                SizingContext(
                    signal_confidence=sig.confidence,
                    equity=equity,
                    price=reference_price,
                    realised_vol=vol_model.current_vol or 0.5,
                    regime=regime,
                    side=sig.side,
                    market=sig.market,
                    strategy=sig.strategy,
                    current_gross_exposure=gross_exposure,
                    symbol_exposure=symbol_exposure,
                    open_positions=open_positions,
                    daily_start_equity=daily_start_equity,
                    peak_equity=peak_equity,
                    stop_loss_pct=meta.get("stop_loss_pct"),
                    take_profit_pct=meta.get("take_profit_pct"),
                    liquidity_score=meta.get("liquidity_score", 1.0),
                    execution_quality=meta.get("execution_quality", 1.0),
                )
            )
            if decision.quantity > 0:
                meta["sizing"] = decision.as_dict()
                meta["reference_price"] = reference_price
                meta["stop_loss_pct"] = decision.stop_loss_pct
                meta["take_profit_pct"] = decision.take_profit_pct
                prepared = self._prepare_signal_for_execution(
                    sig._replace(quantity=decision.quantity, meta=meta),
                    regime=regime,
                    stage="sizing",
                )
                if prepared is None:
                    continue
                projected_notional = float(decision.notional)
                if decision.quantity > 0 and prepared.quantity != decision.quantity:
                    projected_notional *= prepared.quantity / decision.quantity
                if remaining_book_budget is not None and prepared.side == "BUY":
                    if remaining_book_budget <= 0:
                        record_trade_decision(
                            status="GATED",
                            strategy=prepared.strategy,
                            symbol=prepared.symbol,
                            action=f"{prepared.side.lower()}_{prepared.market}",
                            confidence=prepared.confidence,
                            reason="Live crypto allocation budget is fully consumed.",
                            details={"market": prepared.market, "regime": regime, "meta": prepared.meta},
                        )
                        continue
                    if projected_notional > remaining_book_budget:
                        scale = remaining_book_budget / max(projected_notional, 1e-9)
                        capped_qty = prepared.quantity * scale
                        effective_price = max(float(prepared.price or 0.0), projected_notional / max(prepared.quantity, 1e-9))
                        if capped_qty * effective_price < cfg.min_notional:
                            record_trade_decision(
                                status="GATED",
                                strategy=prepared.strategy,
                                symbol=prepared.symbol,
                                action=f"{prepared.side.lower()}_{prepared.market}",
                                confidence=prepared.confidence,
                                reason="Live crypto allocation cap left less than the minimum executable notional.",
                                details={"market": prepared.market, "regime": regime, "remaining_book_budget": remaining_book_budget, "meta": prepared.meta},
                            )
                            continue
                        meta = dict(prepared.meta or {})
                        meta["allocation_capped"] = True
                        meta["allocation_remaining_before"] = remaining_book_budget
                        prepared = prepared._replace(quantity=capped_qty, meta=meta)
                        projected_notional = remaining_book_budget
                sized.append(prepared)
                if remaining_book_budget is not None and prepared.side == "BUY":
                    remaining_book_budget = max(0.0, remaining_book_budget - projected_notional)
                log.info(
                    "Sized %s %s strategy=%s notional=%.2f qty=%.8f risk=%.2f stop=%.2f%% take=%.2f%% bind=%s",
                    prepared.symbol,
                    prepared.side,
                    prepared.strategy,
                    projected_notional,
                    prepared.quantity,
                    decision.risk_dollars,
                    decision.stop_loss_pct * 100,
                    decision.take_profit_pct * 100,
                    decision.binding_constraint,
                )
            else:
                log.info(
                    "Sizing blocked %s %s strategy=%s reason=%s confidence=%.3f",
                    sig.symbol,
                    sig.side,
                    sig.strategy,
                    decision.reason,
                    sig.confidence,
                )
                record_trade_decision(
                    status="REJECTED",
                    strategy=sig.strategy,
                    symbol=sig.symbol,
                    action=f"{sig.side.lower()}_{sig.market}",
                    confidence=sig.confidence,
                    reason=f"Position sizing blocked signal: {decision.reason}",
                    details={"market": sig.market, "price": sig.price, "regime": regime, "sizing": decision.as_dict(), "meta": meta},
                )
        return sized

    # ------------------------------------------------------------------ #
    # Funding / OI refresh
    # ------------------------------------------------------------------ #

    async def _refresh_funding_and_oi(self) -> None:
        for symbol in (self.runtime_symbols if self.book_name == "crypto" else cfg.crypto_symbols):
            try:
                self.store.funding[symbol] = await self.client.get_funding_rate(symbol)
                self.store.open_interest[symbol] = await self.client.get_open_interest(symbol)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # ML model loading
    # ------------------------------------------------------------------ #

    async def _load_ml_models(self) -> None:
        log.info("Loading ML models...")
        for symbol in self.model_symbols:
            self.regime_hmm[symbol] = load_or_init_hmm(symbol)
            from ml.features import FEATURE_NAMES
            self.predictors[symbol] = load_or_init_predictor(symbol, n_features=len(FEATURE_NAMES))
            self.vol_models[symbol] = GARCHVolModel()

        n_strats = len(self.strategies)
        self.rl_allocator = load_or_init_allocator(n_strats)

        for underlying in cfg.options_underlying:
            self.iv_surfaces[underlying] = IVSurface()

        # Train GARCH on recent candles
        for symbol in list(self.vol_models.keys()):
            df = self.store.get_history_df(symbol, "1h")
            if len(df) >= 30:
                import pandas as pd
                closes = pd.Series([c for c in self.store.get_closes(symbol, "1h", 200)])
                log_rets = closes.pct_change().dropna()
                if len(log_rets) >= 30:
                    self.vol_models[symbol].fit(log_rets)

        log.info("ML models loaded")

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def _log_snapshot(self, equity: float, regime: str, approved_count: int) -> None:
        pos_count = len(self.position_manager.all_positions())
        upnl = self.position_manager.total_unrealised_pnl()
        log.info(
            "Snapshot | equity=%.2f | regime=%s | positions=%d | upnl=%.2f | signals_approved=%d",
            equity, regime, pos_count, upnl, approved_count,
        )
