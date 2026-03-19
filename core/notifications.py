import requests
import logging
from config.params import DISCORD_WEBHOOK_URL

logger = logging.getLogger(f"strategy.{__name__}")

def send_alert(message, level="INFO"):
    """Sends a formatted embed message to your Discord channel and pings everyone."""
    if not DISCORD_WEBHOOK_URL:
        return
        
    colors = {
        "INFO": 3447003,      
        "SUCCESS": 3066993,   
        "WARNING": 16776960,  
        "ERROR": 15158332,    
        "ALERT": 10181046     
    }
    
    data = {
        "content": "@everyone",  # <-- THIS TRIGGERS THE PUSH NOTIFICATION
        "embeds": [{
            "description": message,
            "color": colors.get(level, 3447003)
        }]
    }
    
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=data, timeout=5)
    except Exception as e:
        logger.error(f"Failed to send Discord alert: {e}")