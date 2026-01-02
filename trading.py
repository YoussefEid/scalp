"""
Trading operations for Alpaca.
Handles orders, positions, and account management.
"""
from typing import Optional
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

from client import get_client


class Trading:
    """Trading operations for the scalping strategy."""

    def __init__(self):
        self._client = get_client().trading

    # ==================== Account ====================

    def get_account(self):
        """
        Get account information.

        Returns Account with: buying_power, cash, equity, portfolio_value, status
        """
        return self._client.get_account()

    # ==================== Positions ====================

    def get_positions(self) -> list:
        """Get all open positions."""
        return self._client.get_all_positions()

    def get_position(self, symbol: str):
        """Get position for a specific symbol."""
        return self._client.get_open_position(symbol)

    def close_position(self, symbol: str):
        """
        Close entire position for a symbol.
        Useful for quick exits in scalping.
        """
        return self._client.close_position(symbol)

    def close_all_positions(self) -> list:
        """
        Close all open positions.
        Emergency exit function.
        """
        return self._client.close_all_positions(cancel_orders=True)

    # ==================== Orders ====================

    def place_market_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ):
        """
        Place a market order for immediate execution.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            qty: Number of shares (supports fractional)
            side: OrderSide.BUY or OrderSide.SELL
            time_in_force: Order duration (default: DAY)

        Returns:
            Order object with id, status, filled_qty, etc.
        """
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=time_in_force,
        )
        return self._client.submit_order(request)

    def place_limit_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        limit_price: float,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ):
        """
        Place a limit order at specified price.

        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: OrderSide.BUY or OrderSide.SELL
            limit_price: Price limit
            time_in_force: Order duration (default: DAY)

        Returns:
            Order object
        """
        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            limit_price=limit_price,
            time_in_force=time_in_force,
        )
        return self._client.submit_order(request)

    def buy(self, symbol: str, qty: float):
        """Shorthand for market buy order."""
        return self.place_market_order(symbol, qty, OrderSide.BUY)

    def sell(self, symbol: str, qty: float):
        """Shorthand for market sell order."""
        return self.place_market_order(symbol, qty, OrderSide.SELL)

    def get_orders(
        self,
        status: QueryOrderStatus = QueryOrderStatus.OPEN,
    ) -> list:
        """
        Get orders by status.

        Args:
            status: OPEN, CLOSED, or ALL
        """
        request = GetOrdersRequest(status=status)
        return self._client.get_orders(filter=request)

    def get_order(self, order_id: str):
        """Get a specific order by ID."""
        return self._client.get_order_by_id(order_id)

    def cancel_order(self, order_id: str) -> None:
        """Cancel a specific order."""
        self._client.cancel_order_by_id(order_id)

    def cancel_all_orders(self) -> list:
        """Cancel all open orders."""
        return self._client.cancel_orders()


# Convenience instance
trading = Trading()
