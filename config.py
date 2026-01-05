"""
Configuration for IB Gateway trading bot.
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
    """IB Gateway configuration."""

    # IB Gateway connection settings
    HOST: str = os.getenv("IB_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("IB_PORT", "4002"))  # 4002=paper, 4001=live
    CLIENT_ID: int = int(os.getenv("IB_CLIENT_ID", "1"))
    PAPER: bool = os.getenv("IB_PAPER", "true").lower() == "true"

    # Timeout for connection
    TIMEOUT: int = int(os.getenv("IB_TIMEOUT", "20"))

    @classmethod
    def validate(cls) -> bool:
        """Validate that connection settings are valid."""
        # Validate port matches paper/live mode
        if cls.PAPER and cls.PORT == 4001:
            print("Warning: PAPER mode enabled but using live port 4001")
        if not cls.PAPER and cls.PORT == 4002:
            print("Warning: LIVE mode enabled but using paper port 4002")
        return True

    @classmethod
    def print_mode(cls) -> None:
        """Print current trading mode with warning for live trading."""
        if cls.PAPER:
            print("Mode: PAPER TRADING (simulation)")
            print(f"Connecting to IB Gateway at {cls.HOST}:{cls.PORT}")
        else:
            print("Mode: LIVE TRADING (real money)")
            print("WARNING: All trades will use real funds!")
            print(f"Connecting to IB Gateway at {cls.HOST}:{cls.PORT}")


config = Config()
