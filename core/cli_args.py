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

    return parser.parse_args()
