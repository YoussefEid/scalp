"""
Alpha Vantage API for fetching historical daily data.
Used as fallback when IB Gateway doesn't return data.
"""
import requests
import pandas as pd
from datetime import datetime
from typing import Optional

from config import config


class AlphaVantageData:
    """Fetches daily bar data from Alpha Vantage API."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.ALPHA_VANTAGE_API_KEY

    def get_daily_bars(self, symbol: str, outputsize: str = "compact") -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLCV bars from Alpha Vantage.

        Args:
            symbol: Stock ticker symbol
            outputsize: 'compact' (last 100 days) or 'full' (20+ years)

        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index is datetime
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if "Error Message" in data:
                print(f"Alpha Vantage API Error: {data['Error Message']}")
                return None
            if "Note" in data:
                print(f"Alpha Vantage rate limit: {data['Note']}")
                return None

            if "Time Series (Daily)" not in data:
                print(f"No daily data found for {symbol} in Alpha Vantage")
                return None

            # Parse the time series data
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Rename columns (Alpha Vantage uses "1. open", "2. high", etc.)
            df = df.rename(columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume",
            })

            df = df.astype({
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": int,
            })

            return df

        except requests.exceptions.RequestException as e:
            print(f"Alpha Vantage request failed: {e}")
            return None
        except Exception as e:
            print(f"Alpha Vantage error: {e}")
            return None

    def get_previous_close(self, symbol: str) -> Optional[float]:
        """Get the previous day's closing price."""
        daily_data = self.get_daily_bars(symbol, outputsize="compact")
        if daily_data is None or len(daily_data) < 2:
            return None

        # Second to last row is previous day's data
        return float(daily_data.iloc[-2]["close"])

    def calculate_quote_width(self, symbol: str, lookback_days: int = 1) -> Optional[float]:
        """
        Calculate quote width percentage from daily high-low range.

        Args:
            symbol: Stock ticker symbol
            lookback_days: Number of days to look back

        Returns:
            Median daily range as percentage, or None if failed
        """
        daily_data = self.get_daily_bars(symbol, outputsize="compact")
        if daily_data is None or len(daily_data) < lookback_days:
            return None

        # Get the last N trading days
        recent_days = daily_data.tail(lookback_days)

        # Calculate daily range as percentage: (high - low) / close * 100
        recent_days = recent_days.copy()
        recent_days["range_pct"] = (
            (recent_days["high"] - recent_days["low"]) / recent_days["close"] * 100
        )

        return float(recent_days["range_pct"].median())


# Convenience instance
alpha_vantage = AlphaVantageData()
