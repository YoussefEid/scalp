"""
Market data operations for Alpaca.
Handles historical 1-minute bars and real-time streaming.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Callable, Optional, List, Dict
import pandas as pd

from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from client import get_client


class MarketData:
    """Market data operations for the scalping strategy."""

    def __init__(self):
        self._client = get_client().data
        self._stream = get_client().stream
        self._bar_handlers: Dict[str, Callable] = {}

    # ==================== Quotes ====================

    def get_latest_quote(self, symbol: str):
        """
        Get latest quote for a symbol.

        Returns Quote with: ask_price, bid_price, ask_size, bid_size, timestamp
        """
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = self._client.get_stock_latest_quote(request)
        return quotes[symbol]

    def get_latest_quotes(self, symbols: List[str]) -> dict:
        """Get latest quotes for multiple symbols."""
        request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
        return self._client.get_stock_latest_quote(request)

    # ==================== Historical Bars ====================

    def get_bars_1min(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get historical 1-minute OHLCV bars.

        Args:
            symbol: Stock symbol
            start: Start datetime (default: 1 day ago)
            end: End datetime (default: now)
            limit: Max number of bars

        Returns:
            DataFrame with columns: open, high, low, close, volume, vwap, timestamp
        """
        if start is None:
            start = datetime.now() - timedelta(days=1)
        if end is None:
            end = datetime.now()

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            limit=limit,
        )
        bars = self._client.get_stock_bars(request)
        return bars.df

    def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.Minute,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get historical bars with custom timeframe.

        Args:
            symbol: Stock symbol
            timeframe: TimeFrame.Minute, Hour, Day, Week, Month
            start: Start datetime
            end: End datetime
            limit: Max number of bars

        Returns:
            DataFrame with OHLCV data
        """
        if start is None:
            start = datetime.now() - timedelta(days=1)
        if end is None:
            end = datetime.now()

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=limit,
        )
        bars = self._client.get_stock_bars(request)
        return bars.df

    # ==================== Real-time Streaming ====================

    def subscribe_bars(
        self,
        symbols: List[str],
        handler: Callable,
    ) -> None:
        """
        Subscribe to real-time 1-minute bars.

        Args:
            symbols: List of symbols to subscribe to
            handler: Callback function that receives Bar objects
                     Bar has: symbol, open, high, low, close, volume, timestamp

        Example:
            def on_bar(bar):
                print(f"{bar.symbol}: {bar.close}")

            data.subscribe_bars(["AAPL", "MSFT"], on_bar)
            data.run_stream()  # Blocking call
        """
        for symbol in symbols:
            self._bar_handlers[symbol] = handler

        self._stream.subscribe_bars(handler, *symbols)

    def unsubscribe_bars(self, symbols: List[str]) -> None:
        """Unsubscribe from real-time bars for symbols."""
        self._stream.unsubscribe_bars(*symbols)
        for symbol in symbols:
            self._bar_handlers.pop(symbol, None)

    def run_stream(self) -> None:
        """
        Start the real-time data stream (blocking).
        Call this after setting up subscriptions.
        """
        self._stream.run()

    async def run_stream_async(self) -> None:
        """Start the real-time data stream (async)."""
        await self._stream._run_forever()

    def stop_stream(self) -> None:
        """Stop the real-time data stream."""
        self._stream.stop()


# Convenience instance
market_data = MarketData()
