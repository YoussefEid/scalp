"""
Trading operations for IB Gateway.
Handles orders, positions, and account management.
"""
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum
from ib_async import LimitOrder, MarketOrder, Trade as IBTrade

from client import get_client


class OrderSide(Enum):
    """Order side enum matching Alpaca's interface."""
    BUY = "BUY"
    SELL = "SELL"


class TimeInForce(Enum):
    """Time in force enum."""
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"


@dataclass
class OrderStatus:
    """Order status wrapper."""
    value: str


@dataclass
class Order:
    """Order wrapper to match Alpaca's Order interface."""
    id: str
    symbol: str
    side: str
    qty: float
    filled_qty: float
    filled_avg_price: Optional[float]
    status: OrderStatus
    limit_price: Optional[float] = None


@dataclass
class Position:
    """Position wrapper to match Alpaca's Position interface."""
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    unrealized_pl: float
    market_value: float


@dataclass
class Account:
    """Account wrapper to match Alpaca's Account interface."""
    status: str
    buying_power: float
    cash: float
    equity: float
    portfolio_value: float


class Trading:
    """Trading operations for the scalping strategy."""

    def __init__(self):
        self._client = get_client()
        self._trades: dict = {}  # order_id -> IBTrade

    def _ensure_connected(self):
        """Ensure IB connection is active."""
        self._client.ensure_connected()

    # ==================== Account ====================

    def get_account(self) -> Account:
        """
        Get account information.

        Returns Account with: buying_power, cash, equity, portfolio_value, status
        """
        self._ensure_connected()
        ib = self._client.ib

        account_values = ib.accountValues()

        # Extract relevant values
        values = {}
        for av in account_values:
            if av.currency == "USD":
                try:
                    values[av.tag] = float(av.value) if av.value else 0.0
                except ValueError:
                    pass  # Skip non-numeric values

        return Account(
            status="ACTIVE",
            buying_power=values.get("BuyingPower", 0.0),
            cash=values.get("CashBalance", values.get("TotalCashBalance", 0.0)),
            equity=values.get("NetLiquidation", 0.0),
            portfolio_value=values.get("GrossPositionValue", 0.0),
        )

    # ==================== Positions ====================

    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        self._ensure_connected()
        ib = self._client.ib

        positions = []
        for pos in ib.positions():
            contract = pos.contract
            qty = pos.position
            avg_cost = pos.avgCost

            # Get current price
            ticker = ib.reqMktData(contract, "", False, False)
            ib.sleep(0.5)  # Wait for data
            current_price = ticker.marketPrice() if ticker.marketPrice() else avg_cost

            # Cancel the market data subscription
            ib.cancelMktData(contract)

            market_value = qty * current_price
            unrealized_pl = (current_price - avg_cost) * qty

            positions.append(Position(
                symbol=contract.symbol,
                qty=qty,
                avg_entry_price=avg_cost,
                current_price=current_price,
                unrealized_pl=unrealized_pl,
                market_value=market_value,
            ))

        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    def close_position(self, symbol: str) -> Optional[Order]:
        """
        Close entire position for a symbol.
        Useful for quick exits in scalping.
        """
        position = self.get_position(symbol)
        if not position or position.qty == 0:
            return None

        if position.qty > 0:
            return self.place_market_order(symbol, abs(position.qty), OrderSide.SELL)
        else:
            return self.place_market_order(symbol, abs(position.qty), OrderSide.BUY)

    def close_all_positions(self) -> List[Order]:
        """
        Close all open positions.
        Emergency exit function.
        """
        orders = []
        for position in self.get_positions():
            order = self.close_position(position.symbol)
            if order:
                orders.append(order)
        return orders

    # ==================== Orders ====================

    def place_market_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> Order:
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
        self._ensure_connected()
        ib = self._client.ib

        contract = self._client.get_contract(symbol)
        action = "BUY" if side == OrderSide.BUY else "SELL"

        order = MarketOrder(action, qty)
        order.tif = time_in_force.value

        trade = ib.placeOrder(contract, order)
        ib.sleep(0.1)  # Allow order to be acknowledged

        order_id = str(trade.order.orderId)
        self._trades[order_id] = trade

        return self._trade_to_order(trade, symbol)

    def place_limit_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        limit_price: float,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> Order:
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
        self._ensure_connected()
        ib = self._client.ib

        contract = self._client.get_contract(symbol)
        action = "BUY" if side == OrderSide.BUY else "SELL"

        order = LimitOrder(action, qty, limit_price)
        order.tif = time_in_force.value

        trade = ib.placeOrder(contract, order)
        ib.sleep(0.1)  # Allow order to be acknowledged

        order_id = str(trade.order.orderId)
        self._trades[order_id] = trade

        return self._trade_to_order(trade, symbol)

    def _trade_to_order(self, trade: IBTrade, symbol: str) -> Order:
        """Convert IB Trade to Order wrapper."""
        order = trade.order
        order_status = trade.orderStatus

        # Map IB status to simplified status
        # Full list: https://interactivebrokers.github.io/tws-api/order_submission.html
        status_map = {
            "ApiPending": "pending_new",
            "PendingSubmit": "pending_new",
            "PendingCancel": "pending_cancel",
            "PreSubmitted": "accepted",
            "Submitted": "accepted",
            "ApiCancelled": "canceled",
            "Cancelled": "canceled",
            "Filled": "filled",
            "Inactive": "rejected",
        }
        status = status_map.get(order_status.status, order_status.status.lower())

        return Order(
            id=str(order.orderId),
            symbol=symbol,
            side=order.action.lower(),
            qty=order.totalQuantity,
            filled_qty=order_status.filled,
            filled_avg_price=order_status.avgFillPrice if order_status.avgFillPrice else None,
            status=OrderStatus(value=status),
            limit_price=order.lmtPrice if hasattr(order, 'lmtPrice') else None,
        )

    def buy(self, symbol: str, qty: float) -> Order:
        """Shorthand for market buy order."""
        return self.place_market_order(symbol, qty, OrderSide.BUY)

    def sell(self, symbol: str, qty: float) -> Order:
        """Shorthand for market sell order."""
        return self.place_market_order(symbol, qty, OrderSide.SELL)

    def get_orders(self, status: str = "open") -> List[Order]:
        """
        Get orders by status.

        Args:
            status: "open", "closed", or "all"
        """
        self._ensure_connected()
        ib = self._client.ib

        orders = []
        for trade in ib.trades():
            order_status = trade.orderStatus.status
            is_open = order_status in ["PendingSubmit", "PreSubmitted", "Submitted"]

            if status == "open" and not is_open:
                continue
            if status == "closed" and is_open:
                continue

            symbol = trade.contract.symbol if trade.contract else "UNKNOWN"
            orders.append(self._trade_to_order(trade, symbol))

        return orders

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get a specific order by ID."""
        self._ensure_connected()
        ib = self._client.ib

        # First check our cache
        if order_id in self._trades:
            trade = self._trades[order_id]
            symbol = trade.contract.symbol if trade.contract else "UNKNOWN"
            return self._trade_to_order(trade, symbol)

        # Search in IB trades
        for trade in ib.trades():
            if str(trade.order.orderId) == order_id:
                symbol = trade.contract.symbol if trade.contract else "UNKNOWN"
                self._trades[order_id] = trade
                return self._trade_to_order(trade, symbol)

        return None

    def cancel_order(self, order_id: str) -> None:
        """Cancel a specific order."""
        self._ensure_connected()
        ib = self._client.ib

        # Find the trade
        trade = self._trades.get(order_id)
        if not trade:
            for t in ib.trades():
                if str(t.order.orderId) == order_id:
                    trade = t
                    break

        if trade:
            ib.cancelOrder(trade.order)
            ib.sleep(0.1)  # Allow cancel to process

    def cancel_all_orders(self) -> List[str]:
        """Cancel all open orders."""
        self._ensure_connected()
        ib = self._client.ib

        canceled_ids = []
        ib.reqGlobalCancel()
        ib.sleep(0.5)  # Allow cancels to process

        for trade in ib.trades():
            canceled_ids.append(str(trade.order.orderId))

        return canceled_ids


# Convenience instance
trading = Trading()
