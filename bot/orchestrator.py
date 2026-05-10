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
from risk.position_sizer import PositionSizer
from risk.risk_guard import RiskGuard
from execution.order_manager import OrderManager
from execution.position_manager import PositionManager
from strategies.registry import build_registry
from strategies.base import Signal

log = logging.getLogger("bot.orchestrator")

# How often each sub-task runs (seconds)
CYCLE_INTERVAL = 60
RISK_CHECK_INTERVAL = 30
POSITION_REFRESH_INTERVAL = 30
FUNDING_REFRESH_INTERVAL = 300    # 5 min
OI_REFRESH_INTERVAL = 300


class Orchestrator:
    def __init__(self) -> None:
        self.client = get_client()
        self.store = MarketDataStore()
        self.stream_manager: StreamManager | None = None

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
        self.risk_engine = PortfolioRiskEngine()
        self.sizer = PositionSizer()
        self.guard = RiskGuard()
        self.order_manager: OrderManager | None = None
        self.position_manager: PositionManager | None = None

        self._start_equity: float = 0.0
        self._last_funding_refresh = 0.0
        self._last_oi_refresh = 0.0
        self._cycle_count = 0

    # ------------------------------------------------------------------ #
    # Startup
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        cfg.validate()
        setup_logging(cfg.log_dir, cfg.log_level)
        state.init_db()

        broker_info = f"alpaca(paper={cfg.alpaca_paper})" if cfg.is_alpaca else f"binance(testnet={cfg.testnet})"
        log.info("Starting Quant Bot — broker=%s", broker_info)
        await notifications.alert(
            f"Bot starting — broker={broker_info}, strategies={[s.name for s in self.strategies]}",
            title="Bot Startup",
        )

        await self.client.start()
        self.order_manager = OrderManager(self.client)
        self.position_manager = PositionManager(self.client)

        # Warm up stream manager (Binance or Alpaca depending on BROKER)
        all_symbols = cfg.live_runtime_symbols
        self.stream_manager = self.client.create_stream_manager(self.store)
        await self.stream_manager.start(all_symbols, intervals=["1m", "5m", "15m", "1h", "4h"])

        # Load ML models
        await self._load_ml_models()

        # Seed initial equity
        balance = await self.client.get_futures_balance()
        self._start_equity = float(balance.get("USDT", cfg.initial_capital))
        self.guard.maybe_reset_daily(self._start_equity)

        log.info("Startup complete — initial equity=%.2f USDT", self._start_equity)

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

        # --- 12. Execute orders ---
        for sig in approved:
            vol_model = self.vol_models.get(sig.symbol) or self.vol_models.get(cfg.live_primary_regime_symbol) or GARCHVolModel()
            leverage = self.sizer.safe_leverage(
                vol_model.current_vol,
                regime,
            )
            await self.order_manager.execute(sig, leverage=leverage)

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
        exiting = self.position_manager.positions_needing_exit(
            stop_loss_pct=0.05, take_profit_pct=0.10
        )
        for pos in exiting:
            close_side = "SELL" if pos.side == "LONG" else "BUY"
            exit_sig = Signal(
                symbol=pos.symbol,
                market=pos.market,
                side=close_side,
                quantity=pos.quantity,
                price=0,     # market order for exits
                confidence=1.0,
                strategy="exit_manager",
            )
            await self.order_manager.execute(exit_sig)
            log.info("Exiting %s %s pnl=%.2f%%", pos.symbol, pos.side, pos.pnl_pct * 100)

    # ------------------------------------------------------------------ #
    # ML helpers
    # ------------------------------------------------------------------ #

    def _get_regime(self, anchor_symbol: str | None = None) -> str:
        anchors: list[str] = []
        if anchor_symbol:
            anchors.append(anchor_symbol)
        anchors.extend([cfg.live_primary_regime_symbol, *cfg.live_model_symbols])

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
        for symbol in cfg.live_model_symbols:
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
                signals.extend(sigs)
            except Exception as exc:
                log.warning("Strategy %s error: %s", strategy.name, exc)
        return signals

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
        for sig in signals:
            vol_model = self.vol_models.get(sig.symbol) or self.vol_models.get(cfg.live_primary_regime_symbol) or GARCHVolModel()
            notional, qty = self.sizer.size_from_signal(
                signal_confidence=sig.confidence,
                equity=equity,
                price=sig.price or 1.0,
                realised_vol=vol_model.current_vol or 0.5,
                regime=regime,
            )
            if qty > 0:
                sized.append(sig._replace(quantity=qty))
        return sized

    # ------------------------------------------------------------------ #
    # Funding / OI refresh
    # ------------------------------------------------------------------ #

    async def _refresh_funding_and_oi(self) -> None:
        for symbol in cfg.crypto_symbols:
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
        for symbol in cfg.live_model_symbols:
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
