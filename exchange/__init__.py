"""
Exchange factory — returns the right client based on BROKER in .env.

Usage (everywhere in the codebase):
    from exchange import get_client
    client = get_client()
    await client.start()
"""

from __future__ import annotations
from bot.config import cfg


def get_client():
    """Return BinanceClient or AlpacaClient depending on BROKER env var."""
    if cfg.is_alpaca:
        from exchange.alpaca_client import AlpacaClient
        return AlpacaClient()
    else:
        from exchange.client import BinanceClient
        return BinanceClient()
