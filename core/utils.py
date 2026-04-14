import re
import pytz
from datetime import date, datetime, timezone


OPTION_SYMBOL_PATTERN = re.compile(r"^([A-Za-z]+)(\d{6})([PC])(\d{8})$")

def parse_option_symbol(symbol):
    """
    Parses OCC-style option symbol.

    Example:
        'AAPL250516P00207500' -> ('AAPL', 'P', 207.5)
    """
    parsed = try_parse_option_symbol(symbol)
    if parsed is None:
        raise ValueError(f"Invalid option symbol format: {symbol}")
    return parsed


def try_parse_option_symbol(symbol):
    """Parse an OCC option symbol and return ``None`` when the symbol is not an option."""
    normalized = str(symbol or "").strip().upper()
    match = OPTION_SYMBOL_PATTERN.match(normalized)
    if match is None:
        return None

    underlying = match.group(1)
    option_type = match.group(3)
    strike_raw = match.group(4)
    strike_price = int(strike_raw) / 1000.0
    return underlying, option_type, strike_price


def is_option_symbol(symbol) -> bool:
    return try_parse_option_symbol(symbol) is not None


def get_option_expiry_date(symbol) -> date:
    normalized = str(symbol or "").strip().upper()
    match = OPTION_SYMBOL_PATTERN.match(normalized)
    if match is None:
        raise ValueError(f"Invalid option symbol format: {symbol}")
    return datetime.strptime(match.group(2), "%y%m%d").date()


def get_option_days_to_expiry(symbol, reference_date: date | None = None) -> int:
    expiry = get_option_expiry_date(symbol)
    ref = reference_date or datetime.now(timezone.utc).date()
    return (expiry - ref).days

def get_ny_timestamp():
    ny_tz = pytz.timezone("America/New_York")
    ny_time = datetime.now(ny_tz)
    return ny_time.isoformat()
