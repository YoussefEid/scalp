"""
Entry point for IB Gateway trading bot.
Tests connection and demonstrates basic functionality.
"""
import argparse
from datetime import datetime, timedelta

from config import config
from trading import trading
from data import market_data, TimeFrame
from client import get_client


def test_connection():
    """Test connection to IB Gateway and display account info."""
    print("=" * 50)
    print("IB Gateway Trading Bot - Connection Test")
    print("=" * 50)

    # Connect to IB Gateway
    client = get_client()
    if not client.connect():
        print("Failed to connect to IB Gateway!")
        print("\nMake sure:")
        print("  1. IB Gateway is running")
        print("  2. API connections are enabled in IB Gateway settings")
        print("  3. Port matches (4002 for paper, 4001 for live)")
        return False

    # Show trading mode
    config.print_mode()
    print()

    # Get account info
    print("--- Account Info ---")
    account = trading.get_account()
    print(f"Status: {account.status}")
    print(f"Buying Power: ${account.buying_power:,.2f}")
    print(f"Cash: ${account.cash:,.2f}")
    print(f"Portfolio Value: ${account.portfolio_value:,.2f}")
    print(f"Equity: ${account.equity:,.2f}")
    print()

    # Get positions
    print("--- Current Positions ---")
    positions = trading.get_positions()
    if positions:
        for pos in positions:
            pl = pos.unrealized_pl
            pl_sign = "+" if pl >= 0 else ""
            print(f"{pos.symbol}: {pos.qty} shares @ ${pos.avg_entry_price:.2f}")
            print(f"  Current: ${pos.current_price:.2f} | P/L: {pl_sign}${pl:.2f}")
    else:
        print("No open positions")
    print()

    return True


def test_market_data():
    """Test market data retrieval."""
    symbol = "AAPL"

    print("--- Market Data Test ---")

    # Latest quote
    print(f"\nLatest quote for {symbol}:")
    quote = market_data.get_latest_quote(symbol)
    print(f"  Bid: ${quote.bid_price} x {quote.bid_size}")
    print(f"  Ask: ${quote.ask_price} x {quote.ask_size}")

    # Historical 1-min bars
    print(f"\nLast 5 one-minute bars for {symbol}:")
    end = datetime.now()
    start = end - timedelta(hours=1)
    bars = market_data.get_bars_1min(symbol, start=start, end=end, limit=5)

    if not bars.empty:
        # Reset index for cleaner display
        bars_display = bars.reset_index()
        for _, row in bars_display.iterrows():
            ts = row["timestamp"]
            print(f"  {ts}: O={row['open']:.2f} H={row['high']:.2f} "
                  f"L={row['low']:.2f} C={row['close']:.2f} V={int(row['volume'])}")
    else:
        print("  No bars available (market may be closed)")
    print()


def test_order():
    """Test placing a limit order (immediately canceled)."""
    symbol = "AAPL"

    print("--- Order Test ---")
    print(f"Testing limit order for {symbol}...")

    # Get current quote
    quote = market_data.get_latest_quote(symbol)
    bid = quote.bid_price

    # Place a limit buy order far below market (won't fill)
    test_price = round(bid * 0.95, 2)  # 5% below bid
    print(f"Placing limit BUY @ ${test_price:.2f} (won't fill - 5% below bid)")

    from trading import OrderSide
    order = trading.place_limit_order(symbol, 1, OrderSide.BUY, test_price)
    print(f"Order ID: {order.id}")
    print(f"Status: {order.status.value}")

    # Cancel the order
    print("Canceling order...")
    trading.cancel_order(order.id)
    get_client().sleep(0.5)

    # Verify canceled
    updated_order = trading.get_order(order.id)
    if updated_order:
        print(f"Final status: {updated_order.status.value}")
    else:
        print("Order canceled successfully")
    print()


def run_live_trader(args):
    """Run the live trading strategy."""
    from live_trader import LiveTrader

    trader = LiveTrader(
        ticker=args.ticker,
        lookback=args.lookback,
        multiplier=args.multiplier,
        trade_size=args.trade_size,
        enable_stop_loss=args.enable_stop_loss,
        consecutive_limit=args.consecutive_limit,
        enable_regime_filter=args.enable_regime_filter,
        regime_threshold=args.regime_threshold,
        regime_lookback=args.regime_lookback,
        flatten_on_stop=not args.no_flatten_on_stop,
        enable_gap_filter=args.enable_gap_filter,
        gap_up_threshold=args.gap_up_threshold,
    )

    trader.start()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="IB Gateway Trading Bot")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Test command (default behavior)
    test_parser = subparsers.add_parser("test", help="Test connection to IB Gateway")

    # Live trading command
    live_parser = subparsers.add_parser("live", help="Run live trading strategy")
    live_parser.add_argument(
        "--ticker", "-t",
        type=str,
        required=True,
        help="Stock symbol to trade (e.g., AAPL)"
    )
    live_parser.add_argument(
        "--lookback",
        type=int,
        default=1,
        help="Days to look back for quote width calculation (default: 1)"
    )
    live_parser.add_argument(
        "--multiplier",
        type=float,
        default=0.4,
        help="Quote width multiplier (default: 0.4)"
    )
    live_parser.add_argument(
        "--trade-size",
        type=int,
        default=100,
        help="Number of shares per trade (default: 100)"
    )
    live_parser.add_argument(
        "--enable-stop-loss",
        action="store_true",
        help="Enable consecutive trade stop loss"
    )
    live_parser.add_argument(
        "--consecutive-limit",
        type=int,
        default=10,
        help="Stop after N consecutive same-direction trades (default: 10)"
    )
    live_parser.add_argument(
        "--enable-regime-filter",
        action="store_true",
        help="Skip high volatility days"
    )
    live_parser.add_argument(
        "--regime-threshold",
        type=float,
        default=1.5,
        help="Skip if avg daily range > threshold %% (default: 1.5)"
    )
    live_parser.add_argument(
        "--regime-lookback",
        type=int,
        default=5,
        help="Days to look back for regime filter (default: 5)"
    )
    live_parser.add_argument(
        "--no-flatten-on-stop",
        action="store_true",
        help="Don't flatten position when stop loss triggers"
    )
    live_parser.add_argument(
        "--enable-gap-filter",
        action="store_true",
        help="Skip trading on days with large gap up"
    )
    live_parser.add_argument(
        "--gap-up-threshold",
        type=float,
        default=1.0,
        help="Skip trading if gap up > this %% (default: 1.0)"
    )

    args = parser.parse_args()

    if args.command == "live":
        run_live_trader(args)
    else:
        # Default: run connection test
        if test_connection():
            test_market_data()
            test_order()
            print("=" * 50)
            print("Connection test complete!")
            print("=" * 50)

        # Disconnect
        get_client().disconnect()


if __name__ == "__main__":
    main()
