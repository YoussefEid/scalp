"""
Alpaca client initialization.
Provides configured clients for trading, historical data, and real-time streaming.
"""
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed

from config import config


class AlpacaClient:
    """
    Wrapper for all Alpaca API clients.

    Provides:
    - trading: TradingClient for orders, positions, account
    - data: StockHistoricalDataClient for historical bars/quotes
    - stream: StockDataStream for real-time data
    """

    def __init__(self):
        """Initialize all Alpaca clients with credentials from config."""
        config.validate()

        # Trading client - handles orders, positions, account
        self.trading = TradingClient(
            api_key=config.API_KEY,
            secret_key=config.API_SECRET,
            paper=config.PAPER,
        )

        # Historical data client - handles bars, quotes, trades
        self.data = StockHistoricalDataClient(
            api_key=config.API_KEY,
            secret_key=config.API_SECRET,
        )

        # Real-time streaming client - handles live bar/quote/trade subscriptions
        # Use IEX feed for free tier, SIP for paid subscription
        self.stream = StockDataStream(
            api_key=config.API_KEY,
            secret_key=config.API_SECRET,
            feed=DataFeed.IEX,  # Change to DataFeed.SIP if you have a paid data subscription
        )


# Singleton instance
_client = None


def get_client() -> AlpacaClient:
    """Get or create the Alpaca client singleton."""
    global _client
    if _client is None:
        _client = AlpacaClient()
    return _client
