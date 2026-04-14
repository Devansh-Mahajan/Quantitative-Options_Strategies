import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Liquidate all positions before running",
    )

    parser.add_argument(
        "--strat-log",
        action="store_true",
        help="Enable strategy JSON logging",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level for consol/file logs",
    )

    parser.add_argument(
        "--log-to-file",
        action="store_true",
        help="Write logs to file instead of just printing to stdout",
    )
    parser.add_argument(
        "--manage-only",
        action="store_true",
        help="Only manage open positions, do not open new trades.",
    )

    parser.add_argument(
        "--history-start",
        type=str,
        help="Portfolio history start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--history-end",
        type=str,
        help="Portfolio history end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--history-timeframe",
        type=str,
        default="1D",
        choices=["1Min", "5Min", "15Min", "1H", "1D"],
        help="Resolution for portfolio history pull.",
    )
    parser.add_argument(
        "--history-only",
        action="store_true",
        help="Only fetch and print the portfolio history range.",
    )
    parser.add_argument(
        "--disable-adaptive-recalibration",
        action="store_true",
        help="Disable online adaptive risk/deployment tuning for this run.",
    )
    parser.add_argument(
        "--adaptive-lookback",
        type=int,
        default=30,
        help="Rolling window (days) used by the adaptive calibration profile.",
    )
    parser.add_argument(
        "--mega-confidence-threshold",
        type=float,
        default=75.0,
        help="Minimum Mega Brain confidence required before a symbol is treated as a primary candidate.",
    )
    parser.add_argument(
        "--predictor-universe-cap",
        type=int,
        default=20,
        help="Maximum number of ranked symbols to feed through the movement predictor each cycle.",
    )
    parser.add_argument(
        "--router-top-k",
        type=int,
        default=12,
        help="How many fused candidates per strategy bucket to retain before deployment throttling.",
    )
    parser.add_argument(
        "--min-signal-confidence-override",
        type=float,
        help="Optional manual floor for signal confidence gating after runtime policy is loaded.",
    )
    parser.add_argument(
        "--min-vix-for-directional-credit",
        type=float,
        help="Optional live override for the minimum VIX required before directional short premium is allowed.",
    )
    parser.add_argument(
        "--max-vix-for-short-premium",
        type=float,
        help="Optional live override for the maximum VIX allowed for short-premium deployment.",
    )
    parser.add_argument(
        "--disable-runtime-regime-policy",
        action="store_true",
        help="Ignore weekend market_regime_policy live controls for this run.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Bypass the cached compile/import/config validation gate for this manual run.",
    )
    parser.add_argument(
        "--preflight-max-age-seconds",
        type=int,
        default=300,
        help="Reuse a successful preflight result for this many seconds before validating again.",
    )
    parser.add_argument(
        "--no-progress-ui",
        action="store_true",
        help="Disable clean percentage-based progress lines in the terminal.",
    )

    return parser.parse_args()
