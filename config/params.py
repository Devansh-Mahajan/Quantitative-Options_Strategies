RISK_ALLOCATION = 0.50
DELTA_MIN = 0.12 
DELTA_MAX = 0.30
YIELD_MIN = 0.05
YIELD_MAX = 1.00
EXPIRATION_MIN = 35
EXPIRATION_MAX = 55
OPEN_INTEREST_MIN = 100
SCORE_MIN = 0.05

# --- ADVANCED RISK FILTERS ---
MAX_BID_ASK_SPREAD = 1.50   # Hard cap in dollars
MAX_RELATIVE_SPREAD = 0.15  # Dynamic cap (15% of option value)
MAX_SPREADS_PER_SYMBOL = 1
MAX_NEW_TRADES_PER_CYCLE = 8

# 0.98 means we ask for 98% of the mid-price (giving the market maker 2% incentive to fill us).
SLIPPAGE_ALLOWANCE = 0.0  
# SLIPPAGE_ALLOWANCE = 1.0: Purely dynamic (trusts the bot 100%).

# SLIPPAGE_ALLOWANCE = 0.90: Makes the bot 10% more aggressive to get fills.

# SLIPPAGE_ALLOWANCE = 1.10: Makes the bot 10% greedier (demands better prices).

AVOID_EARNINGS = True
MAX_RISK_PER_SPREAD = 1000.0  
PROFIT_TARGET = 0.50  
STOP_LOSS = 3.0

# --- FUND DASHBOARD SETTINGS ---

# 1. Discord Alerts (Create a free Discord server, go to Channel Settings -> Integrations -> Webhooks, and paste the URL here)
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1483972575238426717/J5yYPUhtiwOzs9bXTFwOypf-0SfqBvBK-ogerhd21Vfx7vrsR6ftnP5w1-Atd383Q36C" # e.g., "https://discord.com/api/webhooks/12345/abcde"

# 2. Idle Cash Sweep
SWEEP_TICKER = "SGOV"  # SGOV = 0-3 Month Treasury Bill ETF (~5.3% Yield)
TARGET_CASH_BUFFER = 1000.0 # Always keep at least $1,000 in pure cash for daily settlements

# 3. Dynamic VIX Risk Scaling
MAX_RISK_BASE = 1000.0 # Your standard risk when VIX is normal

# 4. Signal Quality Guardrails
# Avoid forcing full deployment when the directional models disagree or are weak.
MIN_SIGNAL_CONFIDENCE = 0.18
LOW_CONFIDENCE_RISK_MULTIPLIER = 0.50
