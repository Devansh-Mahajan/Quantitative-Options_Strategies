"""
Standalone parallel risk monitor — runs as a separate process.
Polls positions and equity every 60s; sends alerts; writes to shared SQLite state.
Designed to run alongside run_bot.py via systemd or cron.

Usage:
    python -m scripts.risk_monitor
    # or:
    risk-monitor
"""

from __future__ import annotations
import asyncio
import logging
import sys
import time

from bot.config import cfg
from bot.logger import setup_logging
from bot import state, notifications

setup_logging(cfg.log_dir, cfg.log_level)
log = logging.getLogger("scripts.risk_monitor")

POLL_INTERVAL = 60   # seconds


async def _monitor_loop() -> None:
    from exchange import get_client
    from execution.position_manager import PositionManager
    from risk.portfolio_risk import PortfolioRiskEngine

    client = get_client()
    await client.start()

    pos_manager = PositionManager(client)
    risk_engine = PortfolioRiskEngine()
    peak_equity = state.get_peak_equity()

    log.info("Risk monitor started — polling every %ds", POLL_INTERVAL)

    while True:
        try:
            await pos_manager.refresh(force=True)
            balance = await client.get_futures_balance()
            equity = float(balance.get("USDT", 0))

            if equity > peak_equity:
                peak_equity = equity

            snapshot = risk_engine.compute_snapshot(
                pos_manager.position_list_for_risk(),
                equity,
            )
            state.record_equity(equity, snapshot.current_drawdown)

            log.info(
                "Risk | equity=%.2f | dd=%.2f%% | var99=%.2f%% | gross_exp=%.1fx | positions=%d",
                equity,
                snapshot.current_drawdown * 100,
                snapshot.var_99 * 100,
                snapshot.gross_exposure,
                len(pos_manager.all_positions()),
            )

            # --- Alert thresholds ---
            if snapshot.current_drawdown >= cfg.max_drawdown * 0.80:
                await notifications.alert(
                    f"Drawdown warning: {snapshot.current_drawdown:.1%} "
                    f"(limit={cfg.max_drawdown:.1%})",
                    title="Drawdown Alert",
                    level="WARNING",
                )

            if snapshot.var_99 > 0.10:
                await notifications.alert(
                    f"VaR(99%) = {snapshot.var_99:.1%} — portfolio risk elevated",
                    title="VaR Alert",
                    level="WARNING",
                )

            if snapshot.gross_exposure > cfg.max_leverage * 0.85:
                await notifications.alert(
                    f"Leverage near limit: {snapshot.gross_exposure:.1f}x / {cfg.max_leverage}x",
                    title="Leverage Alert",
                    level="WARNING",
                )

            daily_pnl = state.get_daily_pnl()
            daily_ret = daily_pnl / (equity + 1e-10)
            if daily_ret <= -cfg.daily_loss_limit * 0.80:
                await notifications.alert(
                    f"Daily loss warning: {daily_ret:.1%} (limit={-cfg.daily_loss_limit:.1%})",
                    title="Daily Loss",
                    level="WARNING",
                )

        except Exception as exc:
            log.error("Risk monitor error: %s", exc)

        await asyncio.sleep(POLL_INTERVAL)


def main() -> None:
    try:
        asyncio.run(_monitor_loop())
    except KeyboardInterrupt:
        log.info("Risk monitor stopped")


if __name__ == "__main__":
    main()
