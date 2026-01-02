"""
Configuration for Alpaca trading bot.
Loads credentials from environment variables and manages paper/live mode.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add nick-scalp to path for strategy import
# In production (Railway), strategy files will be copied to the repo
NICK_SCALP_PATH = os.getenv("NICK_SCALP_PATH", "/Users/youssefeid/nick-scalp")
if Path(NICK_SCALP_PATH).exists():
    sys.path.insert(0, NICK_SCALP_PATH)

# Data directory for persistent storage (Railway volume mount point)
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
TRADES_DIR = DATA_DIR / "trades"


class Config:
    """Alpaca API configuration."""

    API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    API_SECRET: str = os.getenv("ALPACA_API_SECRET", "")
    PAPER: bool = os.getenv("ALPACA_PAPER", "true").lower() == "true"

    # API URLs
    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"
    DATA_URL = "https://data.alpaca.markets"

    @classmethod
    def validate(cls) -> bool:
        """Validate that required credentials are present."""
        if not cls.API_KEY or not cls.API_SECRET:
            raise ValueError(
                "Missing API credentials. Set ALPACA_API_KEY and ALPACA_API_SECRET "
                "in your .env file."
            )
        return True

    @classmethod
    def get_base_url(cls) -> str:
        """Get the appropriate API base URL based on trading mode."""
        return cls.PAPER_URL if cls.PAPER else cls.LIVE_URL

    @classmethod
    def print_mode(cls) -> None:
        """Print current trading mode with warning for live trading."""
        if cls.PAPER:
            print("📝 Mode: PAPER TRADING (simulation)")
        else:
            print("⚠️  Mode: LIVE TRADING (real money)")
            print("⚠️  WARNING: All trades will use real funds!")


config = Config()
