"""
Entry point for Alpaca trading bot.
Tests connection and demonstrates basic functionality.
"""
import argparse
from datetime import datetime, timedelta

from config import config
from trading import trading
from data import market_data


def test_connection():
    """Test connection to Alpaca and display account info."""
    print("=" * 50)
    print("Alpaca Trading Bot - Connection Test")
    print("=" * 50)

    # Show trading mode
    config.print_mode()
    print()

    # Get account info
    print("--- Account Info ---")
    account = trading.get_account()
    print(f"Status: {account.status}")
    print(f"Buying Power: ${float(account.buying_power):,.2f}")
    print(f"Cash: ${float(account.cash):,.2f}")
    print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print(f"Equity: ${float(account.equity):,.2f}")
    print()

    # Get positions
    print("--- Current Positions ---")
    positions = trading.get_positions()
    if positions:
        for pos in positions:
            pl = float(pos.unrealized_pl)
            pl_sign = "+" if pl >= 0 else ""
            print(f"{pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f}")
            print(f"  Current: ${float(pos.current_price):.2f} | P/L: {pl_sign}${pl:.2f}")
    else:
        print("No open positions")
    print()


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


def test_streaming():
    """Test real-time bar streaming (runs for 30 seconds)."""
    print("--- Real-time Streaming Test ---")
    print("Subscribing to AAPL 1-minute bars...")
    print("(Will run for 30 seconds or until market closes)")
    print()

    bar_count = 0

    def on_bar(bar):
        nonlocal bar_count
        bar_count += 1
        print(f"[{bar.timestamp}] {bar.symbol}: "
              f"O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} "
              f"C={bar.close:.2f} V={bar.volume}")

    market_data.subscribe_bars(["AAPL"], on_bar)

    try:
        # Run stream (blocking) - press Ctrl+C to stop
        market_data.run_stream()
    except KeyboardInterrupt:
        print(f"\nStopped streaming. Received {bar_count} bars.")
        market_data.stop_stream()


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
    )

    trader.start()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Alpaca Trading Bot")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Test command (default behavior)
    test_parser = subparsers.add_parser("test", help="Test connection to Alpaca")

    # Live trading command
    live_parser = subparsers.add_parser("live", help="Run live trading strategy")
    live_parser.add_argument(
        "--ticker", "-t",
        type=str,
        required=True,
        help="Stock symbol to trade (e.g., DE)"
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

    args = parser.parse_args()

    if args.command == "live":
        run_live_trader(args)
    else:
        # Default: run connection test
        test_connection()
        test_market_data()
        print("=" * 50)
        print("Connection test complete!")
        print("=" * 50)


if __name__ == "__main__":
    main()
