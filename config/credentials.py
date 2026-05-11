from dotenv import load_dotenv
import os

load_dotenv(override=True)  # Load from .env file in root

def _env(*keys: str, default: str = "") -> str:
    """Read env values defensively across legacy and current key names."""
    for key in keys:
        raw = os.getenv(key)
        if raw is None:
            continue
        value = raw.strip()
        if "#" in value:
            value = value.split("#", 1)[0].strip()
        return value.strip("\"'")
    return default


ALPACA_API_KEY = _env("ALPACA_API_KEY")
ALPACA_SECRET_KEY = _env("ALPACA_SECRET_KEY", "ALPACA_API_SECRET", "APCA_API_SECRET_KEY")
IS_PAPER = _env("IS_PAPER", "ALPACA_PAPER", default="true").lower() in {"1", "true", "yes"}
