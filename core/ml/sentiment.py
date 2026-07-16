import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import yfinance as yf
logger = logging.getLogger(f"strategy.{__name__}")
from config.params import DELTA_MIN, DELTA_MAX, YIELD_MIN, YIELD_MAX, OPEN_INTEREST_MIN, SCORE_MIN


def get_vix_level():
    """
    Fetches the current live CBOE Volatility Index (^VIX) using Yahoo Finance.
    """
    try:
        vix_ticker = yf.Ticker("^VIX")
        # Get the most recent price
        vix_data = vix_ticker.history(period="1d")
        if not vix_data.empty:
            current_vix = round(vix_data['Close'].iloc[-1], 2)
            logger.info(f"Market Volatility (VIX): {current_vix}")
            return current_vix
        else:
            logger.warning("VIX data empty, defaulting to 15.0")
            return 15.0
    except Exception as e:
        logger.error(f"Failed to fetch VIX: {e}")
        return 15.0 # Failsafe so the bot doesn't crash
def get_dynamic_yield():
    """Calculates the required yield based on current VIX."""
    vix = get_vix_level()
    
    if vix < 15.0:
        adjusted_yield = YIELD_MIN
    elif 15.0 <= vix < 20.0:
        adjusted_yield = YIELD_MIN + 0.02
    elif 20.0 <= vix < 30.0:
        adjusted_yield = YIELD_MIN + 0.05
    else:
        adjusted_yield = YIELD_MIN + 0.10
        
    logger.info(f"VIX is {vix}. Dynamic YIELD_MIN set to {adjusted_yield}")
    return adjusted_yield
def get_market_sentiment(client):
    tz_ny = ZoneInfo("America/New_York")
    tz_utc = ZoneInfo("UTC")
    
    now_ny = datetime.now(tz_ny)
    market_open_ny = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    
    if now_ny < market_open_ny:
        return "neutral"
        
    now_utc = now_ny.astimezone(tz_utc)
    market_open_utc = market_open_ny.astimezone(tz_utc)
    
    # NEW: Subtract 16 minutes to bypass Alpaca's Free Tier SIP Data restriction
    delayed_utc = now_utc - timedelta(minutes=16)
        
    req = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame.Minute,
        start=market_open_utc,
        end=delayed_utc
    )
    
    try:
        bars = client.stock_client.get_stock_bars(req)
        if not bars or "SPY" not in bars.data or bars.df.empty:
            return "neutral"
            
        df = bars.df
        open_price = df.iloc[0]['open']
        current_price = df.iloc[-1]['close']
        price_change_pct = (current_price - open_price) / open_price
        
        if price_change_pct > 0.001:
            logger.info(f"Market Sentiment: BULLISH (SPY up {price_change_pct*100:.2f}%)")
            return "bullish"
        elif price_change_pct < -0.001:
            logger.info(f"Market Sentiment: BEARISH (SPY down {price_change_pct*100:.2f}%)")
            return "bearish"
        else:
            logger.info("Market Sentiment: NEUTRAL")
            return "neutral"
            
    except Exception as e:
        logger.error(f"Failed to fetch SPY sentiment: {e}")
        return "neutral"