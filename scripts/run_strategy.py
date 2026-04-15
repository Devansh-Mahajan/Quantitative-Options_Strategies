import re
import os
from pathlib import Path
from datetime import datetime

from core.runtime_env import apply_accelerator_policy

os.environ.update(apply_accelerator_policy(os.environ.copy())[0])

from core.broker_client import BrokerClient
from core.execution import sell_puts, sell_calls, buy_straddles, sell_iron_condors, buy_tail_hedge, deploy_asymmetric_bets
from core.equity_overlay import rebalance_equity_overlay
from core.state_manager import update_state, calculate_risk
from config.credentials import ALPACA_API_KEY, ALPACA_SECRET_KEY, IS_PAPER
from logging.strategy_logger import StrategyLogger
from logging.logger_setup import setup_logger
from core.cli_args import parse_args
import core.state_manager as state_manager_module

from config.params import (
    RISK_ALLOCATION,
    MAX_SPREADS_PER_SYMBOL,
    PROFIT_TARGET,
    STOP_LOSS,
    MAX_RISK_PER_SPREAD,
    MIN_SIGNAL_CONFIDENCE,
    LOW_CONFIDENCE_RISK_MULTIPLIER,
    MAX_NEW_TRADES_PER_CYCLE,
    ENABLE_PAIRS_TRADING,
    PAIR_MAX_SIGNALS,
    PAIR_ENTRY_ZSCORE,
    PAIR_LOOKBACK_DAYS,
    PAIR_MIN_CORRELATION,
    PAIR_MIN_CONFIDENCE,
    PLATINUM_MODE,
    TARGET_DAILY_RETURN_GOAL,
    MAX_KELLY_FRACTION,
)
# --- Dashboard and Alert Tools ---
from core.manager import manage_open_spreads, get_portfolio_greeks, sweep_idle_cash, calculate_dynamic_risk
from core.notifications import send_alert
from core.sentiment import get_vix_level
from core.movement_predictor import aggregate_movement_signals
from core.ml_alpha import live_alpha_signal_map
from core.greeks_targeting import derive_portfolio_greek_targets

# 🧠 THE AI INFERENCE ENGINES
from core.regime_detection import get_brain_prediction # Macro market mood
from scripts.mega_screener import get_mega_brain_targets
from core.market_intelligence import estimate_institutional_flow
from core.pairs_trading import generate_pairs_trading_signals
from core.portfolio_optimizer import recommend_deployment_fraction, estimate_pair_overlay_confidence
from core.adaptive_recalibration import AdaptiveRecalibrationEngine
from core.runtime_calibration import load_runtime_calibration
from core.resource_profile import load_resource_profile
from core.signal_fusion import empty_ai_targets, route_strategy_candidates
from core.strategy_regime import synthesize_live_controls
from core.system_preflight import DEFAULT_STATE_PATH, run_preflight
from core.system_telemetry import DEFAULT_RISK_SNAPSHOT_PATH, write_risk_snapshot
from core.terminal_ui import ProgressTracker


def _validate_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise ValueError(f"Invalid date '{date_str}'. Expected YYYY-MM-DD.") from exc


def _show_portfolio_history(client, logger, start_date, end_date, timeframe):
    history = client.get_portfolio_history(date_start=start_date, date_end=end_date, timeframe=timeframe)
    timestamps = list(getattr(history, "timestamp", []) or [])
    equities = list(getattr(history, "equity", []) or [])
    if not timestamps or not equities:
        logger.warning("No portfolio history returned for the requested range.")
        return

    rows = list(zip(timestamps, equities))
    start_equity = float(rows[0][1])
    end_equity = float(rows[-1][1])
    pnl = end_equity - start_equity
    pnl_pct = (pnl / start_equity * 100.0) if start_equity else 0.0

    logger.info(
        "Portfolio history (%s to %s, %s): %d points | Start $%.2f | End $%.2f | P/L %+.2f (%+.2f%%)",
        start_date or "default",
        end_date or "default",
        timeframe,
        len(rows),
        start_equity,
        end_equity,
        pnl,
        pnl_pct,
    )

    logger.info("Recent equity points:")
    for ts, eq in rows[-10:]:
        logger.info("  %s -> $%.2f", ts, float(eq))


def _runtime_policy_has_live_controls(runtime_calibration) -> bool:
    return bool(
        runtime_calibration.selected_profile
        and runtime_calibration.current_market_state
        and runtime_calibration.strategy_weights
        and runtime_calibration.bucket_thresholds
        and runtime_calibration.bucket_cap_multipliers
    )


def _apply_synthesized_runtime_controls(runtime_calibration, controls: dict[str, object], brain_strategy: str) -> None:
    runtime_calibration.current_regime = str(brain_strategy)
    runtime_calibration.current_market_state = str(controls.get("market_state") or "transition")
    runtime_calibration.selected_profile = str(controls.get("selected_profile") or "all_weather")
    runtime_calibration.strategy_weights = dict(controls.get("strategy_weights") or {})
    runtime_calibration.bucket_thresholds = dict(controls.get("bucket_thresholds") or {})
    runtime_calibration.bucket_cap_multipliers = dict(controls.get("bucket_cap_multipliers") or {})
    runtime_calibration.dynamic_top_k = int(controls.get("dynamic_top_k") or 0) or runtime_calibration.dynamic_top_k
    runtime_calibration.predictor_universe_cap = int(controls.get("predictor_universe_cap") or 0) or runtime_calibration.predictor_universe_cap
    runtime_calibration.mega_confidence_threshold = float(controls.get("mega_confidence_threshold") or 0.0) or runtime_calibration.mega_confidence_threshold
    runtime_calibration.min_signal_confidence = float(controls.get("min_signal_confidence") or 0.0) or runtime_calibration.min_signal_confidence
    runtime_calibration.max_symbol_weight = float(controls.get("max_symbol_weight") or 0.0) or runtime_calibration.max_symbol_weight
    runtime_calibration.theta_enabled = bool(controls.get("theta_enabled", True))
    runtime_calibration.vega_enabled = bool(controls.get("vega_enabled", True))
    runtime_calibration.directional_enabled = bool(controls.get("directional_enabled", True))
    runtime_calibration.min_vix_for_directional_credit = float(
        controls.get("min_vix_for_directional_credit", runtime_calibration.min_vix_for_directional_credit)
    )
    runtime_calibration.max_vix_for_short_premium = float(
        controls.get("max_vix_for_short_premium", runtime_calibration.max_vix_for_short_premium)
    )
    runtime_calibration.risk_multiplier = float(controls.get("risk_bias", runtime_calibration.risk_multiplier))
    runtime_calibration.deployment_multiplier = float(controls.get("deployment_bias", runtime_calibration.deployment_multiplier))
    runtime_calibration.trade_intensity_multiplier = float(
        controls.get("trade_intensity_bias", runtime_calibration.trade_intensity_multiplier)
    )
    note = str(controls.get("control_source") or "synthetic_live_policy")
    if note not in runtime_calibration.notes:
        runtime_calibration.notes.append(note)
    profile_note = f"profile:{runtime_calibration.selected_profile}"
    if profile_note not in runtime_calibration.notes:
        runtime_calibration.notes.append(profile_note)

def main():
    args = parse_args()
    resource_profile = load_resource_profile(Path(__file__).parent.parent)
    strat_logger = StrategyLogger(enabled=args.strat_log)
    logger = setup_logger(level=args.log_level, to_file=args.log_to_file)
    progress = ProgressTracker(
        logger=logger,
        label="RUN",
        total_steps=8,
        enabled=not args.no_progress_ui,
    )
    strat_logger.set_fresh_start(args.fresh_start)
    progress.advance("Bootstrapping runtime")

    if not args.skip_preflight:
        def _preflight_callback(percent: int, message: str, detail: str | None = None) -> None:
            progress.substep(percent, 100, message, detail=detail)

        preflight_result = run_preflight(
            state_path=DEFAULT_STATE_PATH,
            max_age_seconds=args.preflight_max_age_seconds,
            progress_callback=_preflight_callback if not args.no_progress_ui else None,
        )
        if not preflight_result.ok:
            logger.critical("Preflight blocked run-strategy: %s", preflight_result.summary)
            for issue in preflight_result.issues:
                if issue.severity == "error":
                    logger.critical("  %s: %s", issue.check, issue.detail)
            raise SystemExit(2)
        if not preflight_result.skipped:
            logger.info("✅ %s", preflight_result.summary)

    runtime_calibration = load_runtime_calibration(include_live_policy=not args.disable_runtime_regime_policy)
    if runtime_calibration.notes:
        logger.info(
            "🗂️ Loaded runtime calibration artifacts: %s",
            ", ".join(runtime_calibration.notes),
        )
    logger.info(
        "🖥️ Resource profile | cpu=%d mem=%.1fGB backtest_workers=%d rf_jobs=%d model_parallelism=%d",
        resource_profile.cpu_count,
        resource_profile.memory_gb,
        resource_profile.backtest_workers,
        resource_profile.research_rf_jobs,
        resource_profile.model_parallelism,
    )
    progress.advance("Loading calibration and account context")
    runtime_min_signal_confidence = max(
        float(MIN_SIGNAL_CONFIDENCE),
        float(runtime_calibration.min_signal_confidence or 0.0),
    )
    if args.min_signal_confidence_override is not None:
        runtime_min_signal_confidence = max(runtime_min_signal_confidence, float(args.min_signal_confidence_override))
    router_top_k = max(1, int(args.router_top_k))
    if runtime_calibration.dynamic_top_k:
        router_top_k = max(1, min(router_top_k, int(runtime_calibration.dynamic_top_k)))
    mega_confidence_threshold = float(
        runtime_calibration.mega_confidence_threshold
        if runtime_calibration.mega_confidence_threshold is not None
        else args.mega_confidence_threshold
    )
    predictor_cap = max(
        1,
        int(
            runtime_calibration.predictor_universe_cap
            if runtime_calibration.predictor_universe_cap is not None
            else args.predictor_universe_cap
        ),
    )
    if args.min_vix_for_directional_credit is not None:
        runtime_calibration.min_vix_for_directional_credit = float(args.min_vix_for_directional_credit)
    if args.max_vix_for_short_premium is not None:
        runtime_calibration.max_vix_for_short_premium = float(args.max_vix_for_short_premium)

    symbols_file = Path(__file__).parent.parent / "config" / "symbol_list.txt"
    weekend_symbols_file = Path(__file__).parent.parent / "config" / "volatile_symbols.txt"
    chosen_file = weekend_symbols_file if weekend_symbols_file.exists() else symbols_file
    with open(chosen_file, 'r') as file:
        SYMBOLS = [line.strip() for line in file.readlines() if line.strip()]

    client = BrokerClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, paper=IS_PAPER)
    try:
        execution_reconciliation = client.reconcile_recent_fills(lookback_hours=36.0, max_orders=128)
        if execution_reconciliation.get("checked_orders"):
            logger.info(
                "📒 Execution reconciliation | checked=%d | updated=%d | broker_fill_updates=%d | fill_rate=%s | adaptive_reprice=%s",
                int(execution_reconciliation.get("checked_orders") or 0),
                int(execution_reconciliation.get("reconciled_records") or 0),
                int(execution_reconciliation.get("broker_fill_updates") or 0),
                execution_reconciliation.get("fill_rate"),
                execution_reconciliation.get("adaptive_reprice_factor"),
            )
    except Exception as exc:
        logger.debug("Execution reconciliation skipped: %s", exc)

    history_start = _validate_date(args.history_start)
    history_end = _validate_date(args.history_end)
    if history_start and history_end and history_start > history_end:
        raise ValueError("--history-start must be on or before --history-end.")

    if history_start or history_end or args.history_only:
        _show_portfolio_history(client, logger, history_start, history_end, args.history_timeframe)
        if args.history_only:
            return

    # --- Initialize holding_status BEFORE the if/else block ---
    holding_status = []

    if args.fresh_start:
        logger.info("Fresh Start: Canceling pending orders and liquidating positions.")
        client.trade_client.cancel_orders() 
        client.liquidate_all_positions()
    else:
        logger.info("Step 1: Managing active positions for TP/SL/Expiry...")
        holding_status = manage_open_spreads(client, profit_target=PROFIT_TARGET, stop_loss=STOP_LOSS)
        if holding_status is None: 
            holding_status = [] # Safety catch
    progress.advance("Managing current portfolio and exposures")

    positions = client.get_positions()
    strat_logger.add_current_positions(positions)

    current_risk = calculate_risk(positions)
    states = update_state(positions)
    strat_logger.add_state_dict(states)

    # Handle Assigned Shares (Wheel Fallback)
    for symbol, state in states.items():
        if state["type"] == "long_shares":
            sell_calls(client, symbol, purchase_price=state["price"], stock_qty=state["qty"], max_risk_limit=MAX_RISK_PER_SPREAD, strat_logger=strat_logger)

    # --- CONCENTRATION LIMIT (Max Spreads Per Symbol) ---
    active_counts = {}
    for pos in positions:
        match = re.match(r'^([A-Z]{1,6})\d{6}[CP]\d{8}$', pos.symbol)
        if match:
            root = match.group(1)
            active_counts[root] = active_counts.get(root, 0) + 1
            
    allowed_symbols = []
    for sym in SYMBOLS:
        if active_counts.get(sym, 0) < (MAX_SPREADS_PER_SYMBOL * 2):
            if sym not in states:
                allowed_symbols.append(sym)
        else:
            logger.info(f"Skipping {sym}: Maximum allocation ({MAX_SPREADS_PER_SYMBOL} spread) reached.")

    # Hedge-fund flow proxy: prioritize candidates with persistent momentum + volume surprise
    flow_map = estimate_institutional_flow(allowed_symbols) if allowed_symbols else {}
    allowed_symbols = sorted(allowed_symbols, key=lambda s: flow_map.get(s, 0.0), reverse=True)

    account = client.trade_client.get_account()
    total_equity = float(account.portfolio_value)
    
    # ==========================================================
    # --- QUANT CIRCUIT BREAKER & DRAWDOWN GUARD (DAILY P/L) ---
    # ==========================================================
    last_equity = float(account.last_equity) if account.last_equity else total_equity
    daily_pnl_dollars = total_equity - last_equity
    daily_pnl_pct = (daily_pnl_dollars / last_equity) * 100 if last_equity > 0 else 0
    portfolio_risk_guard = client.get_portfolio_risk_snapshot(
        positions=positions,
        account=account,
        write_runtime=True,
        phase="cycle-pretrade",
        order_label="run-strategy-cycle",
    )

    # Grab VIX early so we can scale risk and print it on the dashboard
    current_vix = get_vix_level()
    predictor_universe = allowed_symbols[:predictor_cap] if allowed_symbols else SYMBOLS[:predictor_cap]
    movement_signals = aggregate_movement_signals(predictor_universe, lookback="5y")
    alpha_universe = allowed_symbols[: max(12, min(len(allowed_symbols), predictor_cap * 2))] if allowed_symbols else SYMBOLS[: max(12, predictor_cap)]
    alpha_signals = live_alpha_signal_map(alpha_universe)
    runtime_policy_mode = "weekend_policy" if _runtime_policy_has_live_controls(runtime_calibration) else "synthetic_live_policy"
    progress.advance("Scoring live market signals")

    if portfolio_risk_guard.kill_switch_active:
        hard_reasons = ", ".join(portfolio_risk_guard.hard_kill_reasons) or "portfolio-risk-limit"
        logger.critical(
            "🔴 KILL SWITCH TRIGGERED 🔴 Hard reasons=%s | Daily P/L %+.2f%% ($%+.2f) | CVaR %.2f%% | Stress %.2f%%. Trading halted.",
            hard_reasons,
            daily_pnl_pct,
            daily_pnl_dollars,
            portfolio_risk_guard.cvar_pct_equity * 100.0,
            portfolio_risk_guard.stress_pct_equity * 100.0,
        )
        send_alert(
            f"🚨 **KILL SWITCH TRIGGERED** 🚨\n"
            f"Reasons: {hard_reasons}\n"
            f"Daily P/L: {daily_pnl_pct:.2f}%\n"
            f"Portfolio CVaR: {portfolio_risk_guard.cvar_pct_equity*100.0:.2f}%\n"
            f"Stress Loss: {portfolio_risk_guard.stress_pct_equity*100.0:.2f}%",
            "ERROR",
        )
        buying_power = 0  
        dynamic_max_risk = 0
        port_delta = 0.0
        port_theta = 0.0
        port_vega = 0.0
        greek_targets = derive_portfolio_greek_targets(
            movement_signals=movement_signals,
            equity=total_equity,
            vix_level=current_vix,
        )
        brain_strategy = "KILL_SWITCH"
        brain_confidence = 100.0
        deployment_scale = 0.0
        adaptive_profile = None
    else:
        if portfolio_risk_guard.breaches:
            logger.warning(
                "🛑 Active portfolio risk pressure: %s | VaR %.2f%% | CVaR %.2f%% | Stress %.2f%% | Top weight %.1f%%",
                ", ".join(portfolio_risk_guard.breaches),
                portfolio_risk_guard.var_pct_equity * 100.0,
                portfolio_risk_guard.cvar_pct_equity * 100.0,
                portfolio_risk_guard.stress_pct_equity * 100.0,
                portfolio_risk_guard.max_underlying_weight * 100.0,
            )
        max_risk_allowed = total_equity * RISK_ALLOCATION
        buying_power = max_risk_allowed - current_risk

        # --- VIX-SCALED DYNAMIC RISK ---
        vix_scaled_risk = calculate_dynamic_risk(current_vix)

        if daily_pnl_pct <= -5.0:
            dynamic_max_risk = vix_scaled_risk / 2
            logger.warning(f"🟡 DRAWDOWN GUARD: Daily P/L is {daily_pnl_pct:.2f}%. Halving max risk per trade to ${dynamic_max_risk:.2f}.")
        else:
            dynamic_max_risk = vix_scaled_risk

        # --- BULLETPROOF GREEKS CALCULATION ---
        greeks = get_portfolio_greeks(client, positions)
        port_delta = greeks[0] if len(greeks) > 0 else 0.0
        port_theta = greeks[1] if len(greeks) > 1 else 0.0
        port_vega  = greeks[2] if len(greeks) > 2 else 0.0 

        greek_targets = derive_portfolio_greek_targets(
            movement_signals=movement_signals,
            equity=total_equity,
            vix_level=current_vix,
        )
        
        if port_delta > 10: bias = "Bullish"
        elif port_delta < -10: bias = "Bearish"
        else: bias = "Neutral"

        # --- 🧠 ASK THE MACRO BRAIN (Regime Gatekeeper) ---
        logger.info("🧠 Consulting the Macro Regime Gatekeeper...")
        brain_strategy, brain_confidence, _brain_probs = get_brain_prediction()

        # --- TERMINAL DASHBOARD ---
        logger.info(f"🟢 PORTFOLIO HEALTHY: Daily P/L is {daily_pnl_pct:+.2f}%. VIX: {current_vix:.2f} -> Risk/Trade: ${dynamic_max_risk:.2f}.")
        logger.info(f"📊 PORTFOLIO DELTA: {port_delta:+.2f} ({bias})")
        logger.info(
            "🎯 TARGET GREEKS -> Delta: %+.2f Theta: %+.2f Vega: %+.2f Gamma: %+.2f | Bias: %s (Conf: %.1f%%)",
            greek_targets.target_delta,
            greek_targets.target_theta,
            greek_targets.target_vega,
            greek_targets.target_gamma,
            greek_targets.movement_bias,
            greek_targets.target_confidence * 100.0,
        )
        logger.info(f"⏳ PORTFOLIO THETA: ${port_theta:+.2f} / day")
        logger.info(f"🌊 PORTFOLIO VEGA: ${port_vega:+.2f} per 1% IV change")
        logger.info(f"🔮 MACRO REGIME VERDICT: {brain_strategy} ({brain_confidence:.1f}% Confidence)")
        if alpha_signals:
            strongest_alpha = sorted(alpha_signals.values(), key=lambda item: (-item.alpha_score, item.symbol))[:5]
            logger.info(
                "🧠 CROSS-SECTIONAL ML ALPHA: %s",
                " | ".join(
                    f"{signal.symbol}:{signal.direction}:{signal.percentile:.2f}"
                    for signal in strongest_alpha
                ),
            )

        signal_confidence = greek_targets.target_confidence
        macro_confidence = max(0.0, min(1.0, brain_confidence / 100.0))
        deployment_scale = signal_confidence * macro_confidence
        if current_vix >= 28:
            deployment_scale *= 0.75

        if args.disable_adaptive_recalibration:
            adaptive_profile = None
            logger.info("🧭 Adaptive recalibration disabled for this run.")
        else:
            adaptive_engine = AdaptiveRecalibrationEngine(lookback=args.adaptive_lookback)
            adaptive_profile = adaptive_engine.update(
                daily_return_pct=daily_pnl_pct,
                signal_confidence=signal_confidence,
                macro_confidence=macro_confidence,
                vix_level=current_vix,
            )
            dynamic_max_risk *= adaptive_profile["risk_multiplier"]
            deployment_scale *= adaptive_profile["deployment_multiplier"]
            logger.info(
                "🧭 Adaptive recalibration: regime=%s | risk x%.2f | deploy x%.2f | trade-intensity x%.2f",
                adaptive_profile["regime"],
                adaptive_profile["risk_multiplier"],
                adaptive_profile["deployment_multiplier"],
                adaptive_profile["trade_intensity_multiplier"],
            )

        if _runtime_policy_has_live_controls(runtime_calibration):
            runtime_policy_mode = "weekend_policy"
        else:
            synthesized_controls = synthesize_live_controls(
                macro_strategy=brain_strategy,
                movement_bias=greek_targets.movement_bias,
                signal_confidence=signal_confidence,
                macro_confidence=macro_confidence,
                vix_level=current_vix,
                adaptive_profile=adaptive_profile or {},
            )
            _apply_synthesized_runtime_controls(runtime_calibration, synthesized_controls, brain_strategy)
            runtime_policy_mode = "synthetic_live_policy"
            logger.info(
                "🛠️ Synthesized live policy: %s / %s / %s | predictive=%.2f vix=%.2f",
                runtime_calibration.current_regime or "n/a",
                runtime_calibration.current_market_state or "n/a",
                runtime_calibration.selected_profile or "n/a",
                float(synthesized_controls.get("predictive_score", 0.0)),
                current_vix,
            )

        dynamic_max_risk *= runtime_calibration.risk_multiplier
        deployment_scale *= runtime_calibration.deployment_multiplier
        logger.info(
            "🗺️ Runtime policy (%s): %s / %s / %s | risk x%.2f | deploy x%.2f | trade-intensity x%.2f | conf=%s",
            runtime_policy_mode,
            runtime_calibration.current_regime or "n/a",
            runtime_calibration.current_market_state or "n/a",
            runtime_calibration.selected_profile or "n/a",
            runtime_calibration.risk_multiplier,
            runtime_calibration.deployment_multiplier,
            runtime_calibration.trade_intensity_multiplier,
            f"{runtime_calibration.regime_confidence:.2f}" if runtime_calibration.regime_confidence is not None else "synthetic",
        )
        if runtime_calibration.max_symbol_weight:
            per_symbol_cap = total_equity * float(runtime_calibration.max_symbol_weight)
            dynamic_max_risk = min(dynamic_max_risk, per_symbol_cap)

        pair_overlay_cache = {"signals": []}
        
        # --- DISCORD DASHBOARD ---
        holdings_str = "\n".join([f"• {h}" for h in holding_status]) if holding_status else "No open positions."
        
        # Pick an emoji based on the AI's mood
        brain_emoji = "🟢" if brain_strategy == "THETA_ENGINE" else "🟡" if brain_strategy == "VEGA_SNIPER" else "🔴"
        
        dashboard_msg = (
            f"**Equity:** ${total_equity:.2f} (P/L: {daily_pnl_pct:+.2f}%)\n"
            f"**Buying Power:** ${buying_power:.2f}\n"
            f"**VIX Level:** {current_vix:.2f} (Risk Cap: ${dynamic_max_risk:.0f})\n"
            f"**Delta:** {port_delta:+.2f} ({bias})\n"
            f"**Target Delta Bias:** {greek_targets.movement_bias} ({greek_targets.target_delta:+.2f})\n"
            f"**Theta (Daily Rent):** ${port_theta:+.2f}\n"
            f"**Vega (Vol Risk):** ${port_vega:+.2f}\n"
            f"**🧠 Macro Regime:** {brain_emoji} **{brain_strategy}** ({brain_confidence:.1f}%)\n\n"
            f"**📦 CURRENT HOLDINGS:**\n{holdings_str}"
        )
        send_alert(dashboard_msg, "INFO")
    # ==========================================================

    risk_snapshot = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "mode": "manage-only" if args.manage_only else "full",
        "daily_pnl_pct": round(float(daily_pnl_pct), 4),
        "daily_pnl_dollars": round(float(daily_pnl_dollars), 2),
        "total_equity": round(float(total_equity), 2),
        "buying_power_budget": round(float(buying_power), 2),
        "vix": round(float(current_vix), 4),
        "portfolio_delta": round(float(port_delta), 4),
        "portfolio_theta": round(float(port_theta), 4),
        "portfolio_vega": round(float(port_vega), 4),
        "target_delta": round(float(greek_targets.target_delta), 4),
        "target_theta": round(float(greek_targets.target_theta), 4),
        "target_vega": round(float(greek_targets.target_vega), 4),
        "movement_bias": greek_targets.movement_bias,
        "macro_regime": brain_strategy,
        "macro_confidence": round(float(brain_confidence), 4),
        "runtime_regime_policy": runtime_calibration.current_regime,
        "runtime_market_state": runtime_calibration.current_market_state,
        "runtime_profile": runtime_calibration.selected_profile,
        "runtime_policy_mode": runtime_policy_mode,
        "open_positions": len(positions),
        "allowed_symbols": len(allowed_symbols),
        "risk_allocation_in_use": round(float(current_risk), 2),
        "portfolio_risk_engine": portfolio_risk_guard.to_dict(),
        "resource_profile": resource_profile.to_dict(),
    }
    write_risk_snapshot(DEFAULT_RISK_SNAPSHOT_PATH, risk_snapshot)

    strat_logger.set_buying_power(buying_power)
    strat_logger.set_allowed_symbols(allowed_symbols)

    # ==========================================================
    # --- MANAGE-ONLY GUARD & HUNTING ROUTINE ---
    # ==========================================================
    if not args.manage_only:
        if buying_power >= 50:
            if signal_confidence < runtime_min_signal_confidence:
                dynamic_max_risk *= LOW_CONFIDENCE_RISK_MULTIPLIER
                buying_power *= LOW_CONFIDENCE_RISK_MULTIPLIER
                logger.warning(
                    "🧩 LOW-CONFIDENCE REGIME: signal confidence %.2f below %.2f. Cutting risk and fresh deployment by %.0f%%.",
                    signal_confidence,
                    runtime_min_signal_confidence,
                    (1.0 - LOW_CONFIDENCE_RISK_MULTIPLIER) * 100.0,
                )

            logger.info(f"Step 2: Hunting for new setups with ${buying_power:.2f} BP (Total Equity: ${total_equity:.2f})...")
            progress.advance("Routing candidate trades across strategy buckets")
            
            # --- 0A. THE DOOMSDAY PROTOCOL ---
            try:
                buy_tail_hedge(client, total_equity, positions, dynamic_max_risk, port_delta) 
            except TypeError:
                logger.error("⚠️ buy_tail_hedge in execution.py needs to be updated to accept port_delta!")

            # =======================================================
            # --- 1. RTX 5090 MEGA BRAIN INTELLIGENCE GATHERING ---
            # =======================================================
            logger.info("🧠 Awakening the Mega Brain: scanning the trained universe for live strategy priors...")

            ai_targets = empty_ai_targets()
            try:
                ai_targets = get_mega_brain_targets(confidence_threshold=mega_confidence_threshold)
            except Exception as exc:
                logger.error("Mega Brain target scan failed; continuing with non-neural signals only: %s", exc)

            if ENABLE_PAIRS_TRADING:
                pair_overlay = generate_pairs_trading_signals(
                    allowed_symbols=allowed_symbols,
                    max_signals=PAIR_MAX_SIGNALS,
                    entry_zscore=PAIR_ENTRY_ZSCORE,
                    lookback_days=PAIR_LOOKBACK_DAYS,
                    min_pair_corr=PAIR_MIN_CORRELATION,
                    min_confidence=PAIR_MIN_CONFIDENCE,
                )
                pair_overlay_cache = pair_overlay

            routing_plan = route_strategy_candidates(
                allowed_symbols=allowed_symbols,
                ai_targets=ai_targets,
                movement_signals=movement_signals,
                alpha_signals=alpha_signals,
                flow_map=flow_map,
                pair_overlay=pair_overlay_cache,
                greek_targets=greek_targets,
                macro_strategy=brain_strategy,
                macro_confidence=macro_confidence,
                top_k=router_top_k,
                strategy_weight_overrides=runtime_calibration.strategy_weights,
                score_threshold_overrides=runtime_calibration.bucket_thresholds,
            )
            deployment_scale *= routing_plan.deployment_multiplier
            theta_candidates = routing_plan.theta_candidates
            vega_candidates = routing_plan.vega_candidates
            bull_candidates = routing_plan.bull_candidates
            bear_candidates = routing_plan.bear_candidates

            if not runtime_calibration.theta_enabled:
                theta_candidates = []
            if not runtime_calibration.vega_enabled:
                vega_candidates = []
            if not runtime_calibration.directional_enabled:
                bull_candidates = []
                bear_candidates = []

            strat_logger.add_model_routing(
                {
                    "mega_confidence_threshold": mega_confidence_threshold,
                    "predictor_universe_cap": predictor_cap,
                    "alpha_signal_count": len(alpha_signals),
                    "router_top_k": router_top_k,
                    "min_signal_confidence": runtime_min_signal_confidence,
                    "runtime_regime_policy": runtime_calibration.current_regime,
                    "runtime_market_state": runtime_calibration.current_market_state,
                    "runtime_profile": runtime_calibration.selected_profile,
                    "runtime_policy_mode": runtime_policy_mode,
                    "strategy_weights": runtime_calibration.strategy_weights,
                    "bucket_thresholds": runtime_calibration.bucket_thresholds,
                    "bucket_cap_multipliers": runtime_calibration.bucket_cap_multipliers,
                    "consensus_score": routing_plan.consensus_score,
                    "deployment_multiplier": routing_plan.deployment_multiplier,
                    "diagnostics": routing_plan.diagnostics,
                }
            )
            logger.info(
                "🧬 Model fusion | consensus=%.2f deploy x%.2f | Theta=%d Vega=%d Bull=%d Bear=%d",
                routing_plan.consensus_score,
                routing_plan.deployment_multiplier,
                len(theta_candidates),
                len(vega_candidates),
                len(bull_candidates),
                len(bear_candidates),
            )

            if PLATINUM_MODE:
                pair_confidence = estimate_pair_overlay_confidence(pair_overlay_cache.get("signals", []))
                deploy_fraction = recommend_deployment_fraction(
                    signal_confidence=signal_confidence,
                    macro_confidence=macro_confidence,
                    pair_confidence=pair_confidence,
                    vix_level=current_vix,
                    target_daily_return=TARGET_DAILY_RETURN_GOAL,
                    max_kelly_fraction=MAX_KELLY_FRACTION,
                )
                deployment_scale *= deploy_fraction
                dynamic_max_risk *= deploy_fraction
                logger.info(
                    "🧮 Platinum sizing active | deploy_fraction=%.2f pair_conf=%.2f risk/trade=$%.2f",
                    deploy_fraction,
                    pair_confidence,
                    dynamic_max_risk,
                )

            throttle_n = max(
                1,
                min(
                    MAX_NEW_TRADES_PER_CYCLE,
                    int(round(MAX_NEW_TRADES_PER_CYCLE * max(0.25, deployment_scale)))
                ),
            )
            if adaptive_profile:
                throttle_n = max(
                    1,
                    min(
                        MAX_NEW_TRADES_PER_CYCLE,
                        int(round(throttle_n * adaptive_profile["trade_intensity_multiplier"])),
                    ),
                )
            throttle_n = max(
                1,
                min(
                    MAX_NEW_TRADES_PER_CYCLE,
                    int(round(throttle_n * runtime_calibration.trade_intensity_multiplier)),
                ),
            )
            bucket_caps = {}
            for bucket in ("THETA", "VEGA", "BULL", "BEAR"):
                multiplier = float(runtime_calibration.bucket_cap_multipliers.get(bucket, 1.0) or 1.0)
                bucket_caps[bucket] = max(
                    1,
                    min(
                        MAX_NEW_TRADES_PER_CYCLE,
                        int(round(throttle_n * multiplier)),
                    ),
                )
            theta_candidates = theta_candidates[: bucket_caps["THETA"]]
            vega_candidates = vega_candidates[: bucket_caps["VEGA"]]
            bull_candidates = bull_candidates[: bucket_caps["BULL"]]
            bear_candidates = bear_candidates[: bucket_caps["BEAR"]]
            
            logger.info(f"📊 NEURAL TARGETS ALIGNED WITH RISK: Theta(Condors): {len(theta_candidates)} | Vega(Straddles): {len(vega_candidates)} | Bull(Puts): {len(bull_candidates)} | Bear(Calls): {len(bear_candidates)}")

            # --- 0B. ASYMMETRIC / CONVEXITY BETS (The Lottery Tickets) ---
            # Feed the most stable/range-bound (THETA) candidates to asymmetric generator
            deploy_asymmetric_bets(client, theta_candidates, total_equity, positions)

            # --- 2A. DEPLOY VEGA ENGINE (Long Straddles) ---
            if greek_targets.target_vega > 0 and runtime_calibration.vega_enabled:
                boosted_vega = [sym for sym in ai_targets.get('VEGA', []) if sym in allowed_symbols]
                vega_candidates = list(dict.fromkeys(vega_candidates + boosted_vega))
                vega_candidates = vega_candidates[: max(throttle_n, 3)]

            if vega_candidates:
                logger.info(f"Step 2A: Launching Vega Sniper on {len(vega_candidates)} AI targets.")
                buying_power = buy_straddles(
                    client=client, 
                    symbols_list=vega_candidates, 
                    buying_power=buying_power, 
                    max_risk_limit=dynamic_max_risk, 
                    state_manager=state_manager_module, 
                    strat_logger=strat_logger
                )
            else:
                logger.info("Step 2A: No Vega (Straddle) opportunities met the AI confidence threshold.")

            # --- 2B. DEPLOY THETA ENGINE (Iron Condors) ---
            if theta_candidates and buying_power >= 100:
                # 🛑 MACRO GATEKEEPER CHECK FOR CREDIT STRATEGIES 🛑
                if brain_strategy == "TAIL_HEDGE":
                    logger.warning("🚨 MACRO AI SAYS TAIL_HEDGE: Market crash probability high. Iron Condors DEACTIVATED.")
                elif brain_strategy == "VEGA_SNIPER":
                    logger.warning("🟡 MACRO AI SAYS VEGA_SNIPER: Volatility expanding. Iron Condors DEACTIVATED to prevent blowouts.")
                elif current_vix > runtime_calibration.max_vix_for_short_premium:
                    logger.warning(
                        "🛑 Regime policy capped short premium at VIX %.2f. Current VIX %.2f. Iron Condors DEACTIVATED.",
                        runtime_calibration.max_vix_for_short_premium,
                        current_vix,
                    )
                else:
                    logger.info(f"Step 2B: Launching AI Theta Engine (Iron Condors) on {len(theta_candidates)} targets.")
                    buying_power = sell_iron_condors(
                        client=client, 
                        symbols_list=theta_candidates, 
                        buying_power=buying_power, 
                        max_risk_limit=dynamic_max_risk, 
                        strat_logger=strat_logger
                    )
            else:
                logger.info("Step 2B: Insufficient BP or no Theta (Condor) opportunities found.")

            # --- 2C. DEPLOY DIRECTIONAL EDGE (Bull / Bear Credit Spreads) ---
            if buying_power >= 50:
                min_vix_for_directional_credit = float(runtime_calibration.min_vix_for_directional_credit)
                if greek_targets.movement_bias == "bullish":
                    bear_candidates = bear_candidates[: max(1, len(bear_candidates) // 4)]
                elif greek_targets.movement_bias == "bearish":
                    bull_candidates = bull_candidates[: max(1, len(bull_candidates) // 4)]

                if brain_strategy == "TAIL_HEDGE" or brain_strategy == "VEGA_SNIPER":
                    logger.warning("🟡 MACRO AI GATE: Market is too volatile for directional short premium. Skipping Put/Call Spreads.")
                elif not runtime_calibration.directional_enabled:
                    logger.info("🧯 Runtime regime policy disabled directional short premium for this market state.")
                elif current_vix > runtime_calibration.max_vix_for_short_premium:
                    logger.info(
                        "🧯 Runtime regime policy capped short premium at VIX %.2f. Holding fire on directional spreads.",
                        runtime_calibration.max_vix_for_short_premium,
                    )
                elif current_vix >= min_vix_for_directional_credit:
                    half_bp = buying_power / 2.0
                    
                    if bull_candidates and half_bp >= 50:
                        logger.info(f"🧠 Step 2C: Deploying AI Bullish Edge (Put Credit Spreads) on {len(bull_candidates)} tickers.")
                        sell_puts(client, bull_candidates, half_bp, dynamic_max_risk, strat_logger)
                        
                    if bear_candidates and half_bp >= 50:
                        logger.info(f"🧠 Step 2D: Deploying AI Bearish Edge (Call Credit Spreads) on {len(bear_candidates)} tickers.")
                        sell_calls(client, bear_candidates, purchase_price=None, stock_qty=0, buying_power=half_bp, max_risk_limit=dynamic_max_risk, strat_logger=strat_logger)
                else:
                    logger.info(
                        "Step 2C: VIX is too low (%.2f < %.2f) for directional Credit Spreads. Remaining BP ($%.2f) held in cash.",
                        current_vix,
                        min_vix_for_directional_credit,
                        buying_power,
                    )
            elif buying_power < 50:
                logger.info(f"Remaining BP (${buying_power:.2f}) exhausted. Skipping Directional Bets.")
                
        else:
            if buying_power == 0 and dynamic_max_risk == 0:
                pass 
            else:
                logger.info(f"Remaining BP (${buying_power:.2f}) is too low for new ${dynamic_max_risk} spreads.")
        progress.advance("Deploying options books")
    else:
        logger.info("Manage-Only mode activated. Skipping Step 2 (Hunting for new Spreads).")
        progress.advance("Managing positions without new entries")
    # ==========================================================

    if daily_pnl_pct > -10.0:
        try:
            refreshed_positions = client.get_positions()
            live_buying_power = float(client.trade_client.get_account().buying_power)
            overlay_buying_power = min(max(0.0, buying_power), max(0.0, live_buying_power))
            overlay_buying_power, overlay_actions = rebalance_equity_overlay(
                client=client,
                positions=refreshed_positions,
                movement_signals=movement_signals,
                alpha_signals=alpha_signals,
                flow_map=flow_map,
                total_equity=total_equity,
                buying_power=overlay_buying_power,
                deployment_scale=deployment_scale,
                current_vix=current_vix,
                movement_bias=greek_targets.movement_bias,
                runtime_calibration=runtime_calibration,
                current_port_delta=port_delta,
                target_port_delta=greek_targets.target_delta,
                allow_new_entries=not args.manage_only,
            )
            if overlay_actions:
                logger.info("📈 Direct equity overlay actions: %s", " | ".join(overlay_actions))
        except Exception as exc:
            logger.error("Direct equity overlay rebalance failed: %s", exc)

    # Sweep only after all entries/trims for this cycle so live cash stays available for fills.
    sweep_idle_cash(client, total_equity)
    progress.advance("Rebalancing stock overlay, delta hedge, and cash sweep")

    strat_logger.save()
    progress.advance("Persisting strategy artifacts")

if __name__ == "__main__":
    main()
