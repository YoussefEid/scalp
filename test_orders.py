"""
Test script for placing buy/sell limit orders in paper trading account.
Run this before market hours to verify order placement works correctly.

Usage:
    python test_orders.py [symbol]

Example:
    python test_orders.py AAPL
"""
import sys
from client import get_client
from trading import trading, OrderSide
from data import market_data


def test_order_placement(symbol: str = "AAPL"):
    """Test placing and canceling buy/sell limit orders."""
    print("=" * 50)
    print("Order Placement Test - Paper Trading")
    print("=" * 50)
    print()

    # Connect to IB Gateway
    print("Connecting to IB Gateway...")
    client = get_client()
    if not client.connect():
        print("Failed to connect to IB Gateway!")
        print("\nMake sure:")
        print("  1. IB Gateway is running")
        print("  2. API connections are enabled")
        print("  3. Port 4002 (paper) or 4001 (live) is open")
        return False

    print("Connected successfully!")
    print()

    # Try to get a quote
    print(f"Getting quote for {symbol}...")
    try:
        quote = market_data.get_latest_quote(symbol)
        print(f"  Bid: ${quote.bid_price} x {quote.bid_size}")
        print(f"  Ask: ${quote.ask_price} x {quote.ask_size}")

        # Use quote prices if available
        if quote.bid_price > 0 and quote.ask_price > 0:
            buy_price = round(quote.bid_price * 0.95, 2)  # 5% below bid
            sell_price = round(quote.ask_price * 1.05, 2)  # 5% above ask
        else:
            print("  Quote prices unavailable, using fallback prices")
            buy_price = 200.00
            sell_price = 260.00
    except Exception as e:
        print(f"  Could not get quote: {e}")
        print("  Using fallback prices")
        buy_price = 200.00
        sell_price = 260.00

    print()
    print(f"Test prices (won't fill):")
    print(f"  Buy limit:  ${buy_price} (below market)")
    print(f"  Sell limit: ${sell_price} (above market)")
    print()

    # Place buy limit order
    print("--- Placing BUY limit order ---")
    buy_order = None
    try:
        buy_order = trading.place_limit_order(symbol, 1, OrderSide.BUY, buy_price)
        print(f"  ✓ Order ID: {buy_order.id}")
        print(f"  ✓ Status: {buy_order.status.value}")
        print(f"  ✓ Side: {buy_order.side}")
        print(f"  ✓ Qty: {buy_order.qty}")
        print(f"  ✓ Limit Price: ${buy_order.limit_price}")
    except Exception as e:
        print(f"  ✗ ERROR placing buy order: {e}")

    print()

    # Place sell limit order
    print("--- Placing SELL limit order ---")
    sell_order = None
    try:
        sell_order = trading.place_limit_order(symbol, 1, OrderSide.SELL, sell_price)
        print(f"  ✓ Order ID: {sell_order.id}")
        print(f"  ✓ Status: {sell_order.status.value}")
        print(f"  ✓ Side: {sell_order.side}")
        print(f"  ✓ Qty: {sell_order.qty}")
        print(f"  ✓ Limit Price: ${sell_order.limit_price}")
    except Exception as e:
        print(f"  ✗ ERROR placing sell order: {e}")

    print()

    # Check open orders
    print("--- Open Orders ---")
    open_orders = trading.get_orders("open")
    if open_orders:
        for order in open_orders:
            limit_str = f"@ ${order.limit_price}" if order.limit_price else ""
            print(f"  {order.id}: {order.side.upper()} {order.qty} {order.symbol} {limit_str} - {order.status.value}")
    else:
        print("  No open orders found")

    print()

    # Cancel test orders
    print("--- Canceling test orders ---")
    if buy_order:
        try:
            trading.cancel_order(buy_order.id)
            print(f"  ✓ Canceled buy order {buy_order.id}")
        except Exception as e:
            print(f"  ✗ Error canceling buy: {e}")

    if sell_order:
        try:
            trading.cancel_order(sell_order.id)
            print(f"  ✓ Canceled sell order {sell_order.id}")
        except Exception as e:
            print(f"  ✗ Error canceling sell: {e}")

    # Wait for cancels to process
    client.sleep(1)

    # Verify orders are canceled
    print()
    print("--- Verifying cancellation ---")
    open_orders = trading.get_orders("open")

    # Filter to just our test orders
    test_order_ids = []
    if buy_order:
        test_order_ids.append(buy_order.id)
    if sell_order:
        test_order_ids.append(sell_order.id)

    remaining = [o for o in open_orders if o.id in test_order_ids]

    if not remaining:
        print("  ✓ All test orders canceled successfully!")
    else:
        print(f"  ⚠ Still have {len(remaining)} test orders:")
        for order in remaining:
            print(f"    {order.id}: {order.status.value}")

    print()
    print("=" * 50)
    print("Order placement test complete!")
    print("=" * 50)

    client.disconnect()
    return True


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    test_order_placement(symbol)
