"""
IB Gateway client initialization.
Provides a configured IB connection for trading and market data.
"""
import asyncio
from typing import Optional, Dict
from ib_async import IB, Stock, Contract

from config import config


class IBClient:
    """
    Wrapper for IB Gateway connection.

    Provides:
    - ib: IB instance for all trading and data operations
    - Contract caching for efficiency
    - Connection management
    """

    def __init__(self):
        """Initialize IB client (not connected yet)."""
        self.ib = IB()
        self._contracts: Dict[str, Contract] = {}
        self._connected = False

    def connect(self) -> bool:
        """
        Connect to IB Gateway.

        Returns:
            True if connection successful, False otherwise.
        """
        if self._connected and self.ib.isConnected():
            return True

        config.validate()

        try:
            self.ib.connect(
                host=config.HOST,
                port=config.PORT,
                clientId=config.CLIENT_ID,
                timeout=config.TIMEOUT,
            )
            self._connected = True
            print(f"Connected to IB Gateway (clientId={config.CLIENT_ID})")
            return True
        except Exception as e:
            print(f"Failed to connect to IB Gateway: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from IB Gateway."""
        if self.ib.isConnected():
            self.ib.disconnect()
        self._connected = False
        print("Disconnected from IB Gateway")

    def ensure_connected(self) -> bool:
        """Ensure connection is active, reconnect if needed."""
        if not self.ib.isConnected():
            self._connected = False
            return self.connect()
        return True

    def get_contract(self, symbol: str) -> Contract:
        """
        Get or create a Stock contract for a symbol.

        IB uses Contract objects instead of simple ticker strings.
        This method caches contracts for efficiency.

        Args:
            symbol: Stock symbol (e.g., "AAPL")

        Returns:
            Qualified Stock contract
        """
        if symbol in self._contracts:
            return self._contracts[symbol]

        # Create and qualify the contract
        contract = Stock(symbol, "SMART", "USD")

        # Qualify to get full contract details
        try:
            qualified = self.ib.qualifyContracts(contract)
            if qualified:
                self._contracts[symbol] = qualified[0]
                return qualified[0]
        except Exception as e:
            print(f"Warning: Could not qualify contract for {symbol}: {e}")

        # Fall back to unqualified contract
        self._contracts[symbol] = contract
        return contract

    def sleep(self, seconds: float = 0):
        """
        Allow IB message processing.

        Call this periodically to process incoming messages.
        """
        self.ib.sleep(seconds)


# Singleton instance
_client: Optional[IBClient] = None


def get_client() -> IBClient:
    """Get or create the IB client singleton."""
    global _client
    if _client is None:
        _client = IBClient()
    return _client
