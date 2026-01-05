"""
Market data operations for IB Gateway.
Handles historical bars and real-time quotes.
"""
import math
from datetime import datetime, timedelta
from typing import Callable, Optional, List, Dict
from dataclasses import dataclass
import pandas as pd

from client import get_client


@dataclass
class Quote:
    """Quote wrapper to match Alpaca's Quote interface."""
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    timestamp: datetime


class TimeFrame:
    """TimeFrame constants matching Alpaca's interface."""
    Minute = "1 min"
    Hour = "1 hour"
    Day = "1 day"
    Week = "1 week"
    Month = "1 month"


class MarketData:
    """Market data operations for the scalping strategy."""

    def __init__(self):
        self._client = get_client()
        self._bar_handlers: Dict[str, Callable] = {}
        self._subscriptions: Dict[str, any] = {}

    def _ensure_connected(self):
        """Ensure IB connection is active."""
        self._client.ensure_connected()

    # ==================== Quotes ====================

    def get_latest_quote(self, symbol: str) -> Quote:
        """
        Get latest quote for a symbol.

        Uses snapshot mode to get a one-time quote without streaming.
        See: https://interactivebrokers.github.io/tws-api/md_request.html

        Returns Quote with: ask_price, bid_price, ask_size, bid_size, timestamp
        """
        self._ensure_connected()
        ib = self._client.ib

        contract = self._client.get_contract(symbol)

        # Use snapshot mode (4th param = True) for one-off quotes
        # This avoids pacing issues from repeated stream start/stop
        ticker = ib.reqMktData(contract, "", False, True)

        # Wait for snapshot data (IB can take up to 11s, but usually faster)
        # Poll until we get valid data or timeout
        max_wait = 5.0  # seconds
        wait_interval = 0.1
        elapsed = 0.0

        while elapsed < max_wait:
            ib.sleep(wait_interval)
            elapsed += wait_interval

            # Check if we have valid bid/ask
            has_bid = ticker.bid and not math.isnan(ticker.bid) and ticker.bid > 0
            has_ask = ticker.ask and not math.isnan(ticker.ask) and ticker.ask > 0

            if has_bid and has_ask:
                break

        # Get bid/ask (handle NaN values)
        bid_price = ticker.bid if ticker.bid and not math.isnan(ticker.bid) and ticker.bid > 0 else 0.0
        ask_price = ticker.ask if ticker.ask and not math.isnan(ticker.ask) and ticker.ask > 0 else 0.0
        bid_size = int(ticker.bidSize) if ticker.bidSize and not math.isnan(ticker.bidSize) else 0
        ask_size = int(ticker.askSize) if ticker.askSize and not math.isnan(ticker.askSize) else 0

        # Snapshot mode auto-cancels, but ensure cleanup
        try:
            ib.cancelMktData(contract)
        except:
            pass  # May already be cancelled

        return Quote(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            timestamp=datetime.now(),
        )

    def get_latest_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Get latest quotes for multiple symbols."""
        quotes = {}
        for symbol in symbols:
            quotes[symbol] = self.get_latest_quote(symbol)
        return quotes

    # ==================== Historical Bars ====================

    def _timeframe_to_ib(self, timeframe: str) -> str:
        """Convert timeframe to IB bar size setting."""
        mapping = {
            "1 min": "1 min",
            "1 hour": "1 hour",
            "1 day": "1 day",
            "1 week": "1 week",
            "1 month": "1 month",
        }
        return mapping.get(timeframe, "1 min")

    def _calculate_duration(self, start: datetime, end: datetime, timeframe: str) -> str:
        """Calculate IB duration string based on date range."""
        delta = end - start

        if timeframe in ["1 day", "1 week", "1 month"]:
            days = delta.days + 1
            if days <= 365:
                return f"{days} D"
            else:
                return f"{(days // 365) + 1} Y"
        else:
            # For intraday, use seconds
            seconds = int(delta.total_seconds())
            if seconds <= 86400:
                return f"{seconds} S"
            else:
                days = (seconds // 86400) + 1
                return f"{min(days, 30)} D"  # IB limits intraday to ~30 days

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
        return self.get_bars(symbol, TimeFrame.Minute, start, end, limit)

    def get_bars(
        self,
        symbol: str,
        timeframe: str = TimeFrame.Minute,
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
        self._ensure_connected()
        ib = self._client.ib

        if start is None:
            start = datetime.now() - timedelta(days=1)
        if end is None:
            end = datetime.now()

        contract = self._client.get_contract(symbol)
        bar_size = self._timeframe_to_ib(timeframe)
        duration = self._calculate_duration(start, end, timeframe)

        # Request historical data
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True,  # Regular trading hours only
            formatDate=1,
        )

        if not bars:
            return pd.DataFrame()

        # Convert to DataFrame
        data = []
        for bar in bars:
            data.append({
                "timestamp": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "vwap": bar.average if hasattr(bar, 'average') else (bar.open + bar.close) / 2,
            })

        df = pd.DataFrame(data)

        if not df.empty:
            df.set_index("timestamp", inplace=True)

            # Filter by start date (handle timezone-aware index)
            if df.index.tz is not None:
                # Convert naive datetime to timezone-aware
                import pytz
                start_tz = pytz.timezone(str(df.index.tz)).localize(start)
                df = df[df.index >= start_tz]
            else:
                df = df[df.index >= start]

            # Apply limit
            if limit:
                df = df.tail(limit)

        return df

    # ==================== Real-time Streaming ====================

    def subscribe_bars(
        self,
        symbols: List[str],
        handler: Callable,
    ) -> None:
        """
        Subscribe to real-time 5-second bars (IB's minimum).

        Note: IB provides 5-second real-time bars, not 1-minute.
        For 1-minute bars, you would aggregate 5-second bars.

        Args:
            symbols: List of symbols to subscribe to
            handler: Callback function that receives Bar objects
                     Bar has: symbol, open, high, low, close, volume, timestamp

        Example:
            def on_bar(bar):
                print(f"{bar.symbol}: {bar.close}")

            data.subscribe_bars(["AAPL", "MSFT"], on_bar)
        """
        self._ensure_connected()
        ib = self._client.ib

        for symbol in symbols:
            self._bar_handlers[symbol] = handler
            contract = self._client.get_contract(symbol)

            # Request real-time 5-second bars
            bars = ib.reqRealTimeBars(
                contract,
                barSize=5,  # 5-second bars (IB minimum)
                whatToShow="TRADES",
                useRTH=True,
            )

            self._subscriptions[symbol] = bars

            # Set up callback
            bars.updateEvent += lambda bars, symbol=symbol: self._on_bar_update(bars, symbol)

    def _on_bar_update(self, bars, symbol: str):
        """Handle incoming bar updates."""
        if symbol in self._bar_handlers and bars:
            bar = bars[-1]  # Get latest bar
            # Create a simple bar object
            bar_data = type('Bar', (), {
                'symbol': symbol,
                'open': bar.open_,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'timestamp': bar.time,
            })()
            self._bar_handlers[symbol](bar_data)

    def unsubscribe_bars(self, symbols: List[str]) -> None:
        """Unsubscribe from real-time bars for symbols."""
        self._ensure_connected()
        ib = self._client.ib

        for symbol in symbols:
            if symbol in self._subscriptions:
                ib.cancelRealTimeBars(self._subscriptions[symbol])
                del self._subscriptions[symbol]
            self._bar_handlers.pop(symbol, None)

    def run_stream(self) -> None:
        """
        Run the event loop for streaming data.

        This is a blocking call that processes IB events.
        """
        self._ensure_connected()
        ib = self._client.ib

        # Run the IB event loop
        ib.run()

    async def run_stream_async(self) -> None:
        """Run the event loop asynchronously."""
        self._ensure_connected()
        ib = self._client.ib
        await ib.runAsync()

    def stop_stream(self) -> None:
        """Stop all streaming subscriptions."""
        self._ensure_connected()
        ib = self._client.ib

        # Cancel all subscriptions
        for symbol in list(self._subscriptions.keys()):
            self.unsubscribe_bars([symbol])

        # Disconnect from IB event loop
        ib.disconnect()


# Convenience instance
market_data = MarketData()
