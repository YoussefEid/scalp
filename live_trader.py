"""
Live trading module for the mean reversion scalping strategy.
Connects the nick-scalp strategy to IB Gateway for paper/live trading.

Uses pre-submitted limit orders for low-latency execution:
1. At market open, submit BOTH resting buy and sell limit orders
2. Monitor for fills via polling
3. On fill, retreat levels and resubmit the filled side
"""
import json
import signal
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict
import pytz

from data import market_data, TimeFrame
from config import config, TRADES_DIR
from trading import trading, OrderSide
from client import get_client

# Import strategy from nick-scalp
from strategy import MeanReversionScalper, Side, Trade


ET = pytz.timezone("America/New_York")


class LiveTrader:
    """
    Live trading wrapper for the MeanReversionScalper strategy.

    Uses simultaneous buy AND sell limit orders (IB allows this!):
    - Both orders sit on the book at calculated levels
    - When one fills, we retreat and resubmit that side
    - Much simpler than Alpaca's single-order limitation
    """

    def __init__(
        self,
        ticker: str,
        lookback: int = 1,
        multiplier: float = 0.4,
        trade_size: int = 100,
        enable_stop_loss: bool = False,
        consecutive_limit: int = 7,
        enable_regime_filter: bool = False,
        regime_threshold: float = 1.5,
        regime_lookback: int = 5,
        flatten_on_stop: bool = True,
        enable_gap_filter: bool = False,
        gap_up_threshold: float = 1.0,
    ):
        """
        Initialize the live trader.

        Args:
            ticker: Stock symbol to trade
            lookback: Days to look back for quote width calculation
            multiplier: Quote width multiplier
            trade_size: Number of shares per trade
            enable_stop_loss: Enable consecutive trade stop loss
            consecutive_limit: Stop after N consecutive same-direction trades
            enable_regime_filter: Skip high volatility days
            regime_threshold: Skip if avg daily range > threshold %
            regime_lookback: Days to look back for regime filter
            flatten_on_stop: Flatten position when stop loss triggers
            enable_gap_filter: Skip trading on days with large gap up
            gap_up_threshold: Skip trading if gap up > this % (default: 1.0)
        """
        self.ticker = ticker
        self.lookback = lookback
        self.multiplier = multiplier
        self.trade_size = trade_size
        self.enable_stop_loss = enable_stop_loss
        self.consecutive_limit = consecutive_limit
        self.enable_regime_filter = enable_regime_filter
        self.regime_threshold = regime_threshold
        self.regime_lookback = regime_lookback
        self.flatten_on_stop = flatten_on_stop
        self.enable_gap_filter = enable_gap_filter
        self.gap_up_threshold = gap_up_threshold

        self.strategy: Optional[MeanReversionScalper] = None
        self.quote_width_pct: float = 0.0
        self.zero_point: float = 0.0
        self.running: bool = False
        self.trading_active: bool = False  # False when stop loss triggered
        self.bars_processed: int = 0
        self.today_str: str = ""

        # Order tracking - IB allows BOTH orders simultaneously!
        self.active_buy_order_id: Optional[str] = None
        self.active_sell_order_id: Optional[str] = None
        self.active_buy_price: float = 0.0
        self.active_sell_price: float = 0.0

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n[{self._now()}] Shutdown signal received, stopping...")
        self.stop()

    def _now(self) -> str:
        """Get current time in ET as string."""
        return datetime.now(ET).strftime("%H:%M:%S")

    def _log(self, msg: str):
        """Log a message with timestamp."""
        print(f"[{self._now()}] {msg}")

    def _is_market_day(self) -> bool:
        """Check if today is a market day (simple weekday check)."""
        return datetime.now(ET).weekday() < 5

    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now(ET)
        return 9 <= now.hour < 16 and now.weekday() < 5

    def _wait_for_premarket(self):
        """Wait until 9:25 AM ET on a market day."""
        while self.running:
            now = datetime.now(ET)

            if not self._is_market_day():
                self._log("Not a market day. Sleeping until tomorrow...")
                tomorrow = (now + timedelta(days=1)).replace(
                    hour=9, minute=0, second=0, microsecond=0
                )
                sleep_seconds = (tomorrow - now).total_seconds()
                time.sleep(min(sleep_seconds, 3600))
                continue

            if now.hour >= 16:
                self._log("Market closed. Sleeping until tomorrow...")
                tomorrow = (now + timedelta(days=1)).replace(
                    hour=9, minute=0, second=0, microsecond=0
                )
                sleep_seconds = (tomorrow - now).total_seconds()
                time.sleep(min(sleep_seconds, 3600))
                continue

            if now.hour == 9 and now.minute >= 25:
                self._log("Pre-market time reached!")
                return

            if now.hour > 9:
                self._log("Already past 9:25 AM, starting immediately")
                return

            target = now.replace(hour=9, minute=25, second=0, microsecond=0)
            sleep_seconds = (target - now).total_seconds()

            if sleep_seconds > 60:
                self._log(f"Waiting for pre-market... ({int(sleep_seconds/60)} minutes)")
                time.sleep(60)
            else:
                time.sleep(max(1, sleep_seconds))

    def _calculate_quote_width(self) -> float:
        """Calculate quote width from daily high-low range."""
        self._log(f"Calculating quote width from last {self.lookback} day(s) daily range...")

        end = datetime.now(ET)
        start = end - timedelta(days=self.lookback + 10)  # Extra days to ensure we get enough data

        try:
            # Use DAILY bars
            bars = market_data.get_bars(
                self.ticker,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )

            if bars.empty:
                self._log("Warning: No historical daily bars found, using default 0.4%")
                return 0.4

            bars_reset = bars.reset_index()

            # Calculate daily range as percentage: (high - low) / close * 100
            bars_reset["range_pct"] = (
                (bars_reset["high"] - bars_reset["low"]) / bars_reset["close"] * 100
            )

            # Get the most recent N days (lookback)
            recent_bars = bars_reset.tail(self.lookback)

            if len(recent_bars) == 0:
                self._log("Warning: No recent daily bars, using default 0.4%")
                return 0.4

            median_range = recent_bars["range_pct"].median()
            self._log(f"Median daily range: {median_range:.4f}%")

            # If median range is 0 or very small, use default
            if median_range < 0.1:
                self._log("Warning: Daily range too small, using default 0.4%")
                return 0.4

            quote_width = median_range * self.multiplier
            self._log(f"Quote width (with {self.multiplier}x multiplier): {quote_width:.4f}%")

            return quote_width

        except Exception as e:
            self._log(f"Error calculating quote width: {e}")
            return 0.4

    def _check_regime_filter(self) -> bool:
        """Check if today passes the regime filter."""
        if not self.enable_regime_filter:
            return True

        self._log(f"Checking regime filter (threshold: {self.regime_threshold}%)...")

        end = datetime.now(ET)
        start = end - timedelta(days=self.regime_lookback + 10)

        try:
            bars = market_data.get_bars(
                self.ticker,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )

            if bars.empty or len(bars) < self.regime_lookback:
                self._log("Not enough daily data for regime filter, allowing trading")
                return True

            bars_reset = bars.reset_index().tail(self.regime_lookback)
            bars_reset["range_pct"] = (
                (bars_reset["high"] - bars_reset["low"]) / bars_reset["close"] * 100
            )
            avg_range = bars_reset["range_pct"].mean()

            self._log(f"Avg daily range (last {self.regime_lookback} days): {avg_range:.2f}%")

            if avg_range > self.regime_threshold:
                self._log(f"Regime filter BLOCKED: {avg_range:.2f}% > {self.regime_threshold}%")
                return False

            self._log("Regime filter PASSED")
            return True

        except Exception as e:
            self._log(f"Error checking regime filter: {e}")
            return True

    def _check_gap_filter(self, opening_price: float) -> bool:
        """
        Check if today passes the gap filter.

        Skip trading if the stock gaps UP more than the threshold percentage
        from the previous day's close. Gap downs are allowed (mean reversion
        tends to work better on gap downs).

        Args:
            opening_price: Today's opening price

        Returns:
            True if trading allowed (no gap up or gap < threshold), False if gap up exceeds threshold
        """
        if not self.enable_gap_filter:
            return True

        self._log(f"Checking gap filter (threshold: {self.gap_up_threshold}%)...")

        end = datetime.now(ET)
        start = end - timedelta(days=5)  # Get a few days to ensure we have yesterday's close

        try:
            bars = market_data.get_bars(
                self.ticker,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )

            if bars.empty or len(bars) < 2:
                self._log("Not enough daily data for gap filter, allowing trading")
                return True

            # Get previous day's close (second to last bar)
            bars_reset = bars.reset_index()
            prev_close = float(bars_reset.iloc[-2]["close"])

            # Calculate gap percentage (positive = gap up, negative = gap down)
            gap_pct = (opening_price - prev_close) / prev_close * 100

            self._log(f"Previous close: ${prev_close:.2f}, Opening: ${opening_price:.2f}")
            self._log(f"Gap: {gap_pct:+.2f}%")

            # Only filter gap UPs, not gap downs
            if gap_pct > self.gap_up_threshold:
                self._log(f"Gap filter BLOCKED: gap up {gap_pct:.2f}% > {self.gap_up_threshold}%")
                return False

            self._log("Gap filter PASSED")
            return True

        except Exception as e:
            self._log(f"Error checking gap filter: {e}")
            return True

    def _get_opening_price(self) -> float:
        """Get the opening price for today."""
        self._log("Waiting for opening price...")

        while self.running:
            now = datetime.now(ET)
            if now.hour == 9 and now.minute >= 30:
                break
            if now.hour > 9:
                break
            time.sleep(1)

        today = datetime.now(ET).replace(hour=9, minute=30, second=0, microsecond=0)

        for attempt in range(10):
            try:
                bars = market_data.get_bars_1min(
                    self.ticker,
                    start=today,
                    end=today + timedelta(minutes=5),
                    limit=1,
                )

                if not bars.empty:
                    opening = float(bars.iloc[0]["open"])
                    self._log(f"Opening price: ${opening:.2f}")
                    return opening

            except Exception as e:
                self._log(f"Attempt {attempt+1}: Error getting opening price: {e}")

            time.sleep(5)

        self._log("Using latest quote as fallback for opening price")
        quote = market_data.get_latest_quote(self.ticker)
        mid = (float(quote.ask_price) + float(quote.bid_price)) / 2
        self._log(f"Mid price: ${mid:.2f}")
        return mid

    def _init_strategy(self):
        """Initialize the strategy with calculated parameters."""
        self._log("Initializing strategy...")

        self.strategy = MeanReversionScalper(
            zero_point=self.zero_point,
            quote_width_pct=self.quote_width_pct,
            trade_size=self.trade_size,
            quote_width_multiplier=1.0,
            enable_stop_loss=self.enable_stop_loss,
            consecutive_trade_limit=self.consecutive_limit,
            flatten_on_stop=self.flatten_on_stop,
        )

        self._log(f"Strategy initialized:")
        self._log(f"  Zero point: ${self.zero_point:.2f}")
        self._log(f"  Quote width: {self.quote_width_pct:.4f}%")
        self._log(f"  Quote width ($): ${self.strategy.state.quote_width:.4f}")
        self._log(f"  Trade size: {self.trade_size} shares")
        self._log(f"  Stop loss: {self.enable_stop_loss} (limit: {self.consecutive_limit})")

    # ==================== Order Management ====================

    def _submit_both_orders(self):
        """Submit BOTH buy and sell limit orders simultaneously.

        This is the key advantage of IB over Alpaca - we can have both
        sides resting on the book at the same time!
        """
        if not self.trading_active or not self.strategy:
            return

        bid, ask = self.strategy.state.get_quotes()
        bid_price = round(bid.price, 2)
        ask_price = round(ask.price, 2)

        # Submit buy if not active
        if not self.active_buy_order_id:
            self._submit_limit_buy(bid_price)

        # Submit sell if not active
        if not self.active_sell_order_id:
            self._submit_limit_sell(ask_price)

    def _check_and_update_orders(self):
        """Check for fills and update order prices if levels changed."""
        if not self.trading_active or not self.strategy:
            return

        # Check market hours
        now = datetime.now(ET)
        if now.hour == 15 and now.minute >= 55:
            self._log("Near market close, stopping new orders")
            return
        if now.hour >= 16:
            return

        # Get current strategy levels
        bid, ask = self.strategy.state.get_quotes()
        bid_price = round(bid.price, 2)
        ask_price = round(ask.price, 2)

        # Check buy order
        if self.active_buy_order_id:
            self._check_buy_order(bid_price)

        # Check sell order
        if self.active_sell_order_id:
            self._check_sell_order(ask_price)

        # Resubmit any missing orders
        self._submit_both_orders()

    def _check_buy_order(self, current_bid_level: float):
        """Check buy order status and handle fills/updates."""
        try:
            order = trading.get_order(self.active_buy_order_id)
            if not order:
                self.active_buy_order_id = None
                self.active_buy_price = 0.0
                return

            status = order.status.value
            filled_qty = order.filled_qty
            total_qty = order.qty

            # Check for partial fill (filled_qty > 0 but status not "filled")
            if filled_qty > 0 and status != "filled":
                self._log(f"⚠️ PARTIAL FILL detected on BUY: {filled_qty}/{total_qty} shares @ ${order.filled_avg_price:.2f}")
                self._log(f"   Order status: {status}, remaining: {total_qty - filled_qty} shares")

            if status == "filled":
                fill_price = float(order.filled_avg_price) if order.filled_avg_price else current_bid_level
                self._log(f"BUY FILLED @ ${fill_price:.2f} ({filled_qty} shares)")
                self._handle_buy_fill(fill_price)
                self.active_buy_order_id = None
                self.active_buy_price = 0.0
                return

            if status in ["canceled", "expired", "rejected"]:
                # Log if there was a partial fill before cancel
                if filled_qty > 0:
                    self._log(f"⚠️ Buy order {status} with PARTIAL FILL: {filled_qty}/{total_qty} shares @ ${order.filled_avg_price:.2f}")
                else:
                    self._log(f"Buy order {status}")
                self.active_buy_order_id = None
                self.active_buy_price = 0.0
                return

            # Update price if level changed significantly
            if abs(current_bid_level - self.active_buy_price) > 0.01:
                # Warn if canceling an order with partial fills
                if filled_qty > 0:
                    self._log(f"⚠️ Canceling buy order with PARTIAL FILL: {filled_qty}/{total_qty} shares already filled")
                self._log(f"Updating buy: ${self.active_buy_price:.2f} -> ${current_bid_level:.2f}")
                trading.cancel_order(self.active_buy_order_id)
                get_client().sleep(0.2)  # Wait for cancel
                self.active_buy_order_id = None
                self.active_buy_price = 0.0
                self._submit_limit_buy(current_bid_level)

        except Exception as e:
            self._log(f"Error checking buy order: {e}")
            self.active_buy_order_id = None
            self.active_buy_price = 0.0

    def _check_sell_order(self, current_ask_level: float):
        """Check sell order status and handle fills/updates."""
        try:
            order = trading.get_order(self.active_sell_order_id)
            if not order:
                self.active_sell_order_id = None
                self.active_sell_price = 0.0
                return

            status = order.status.value
            filled_qty = order.filled_qty
            total_qty = order.qty

            # Check for partial fill (filled_qty > 0 but status not "filled")
            if filled_qty > 0 and status != "filled":
                self._log(f"⚠️ PARTIAL FILL detected on SELL: {filled_qty}/{total_qty} shares @ ${order.filled_avg_price:.2f}")
                self._log(f"   Order status: {status}, remaining: {total_qty - filled_qty} shares")

            if status == "filled":
                fill_price = float(order.filled_avg_price) if order.filled_avg_price else current_ask_level
                self._log(f"SELL FILLED @ ${fill_price:.2f} ({filled_qty} shares)")
                self._handle_sell_fill(fill_price)
                self.active_sell_order_id = None
                self.active_sell_price = 0.0
                return

            if status in ["canceled", "expired", "rejected"]:
                # Log if there was a partial fill before cancel
                if filled_qty > 0:
                    self._log(f"⚠️ Sell order {status} with PARTIAL FILL: {filled_qty}/{total_qty} shares @ ${order.filled_avg_price:.2f}")
                else:
                    self._log(f"Sell order {status}")
                self.active_sell_order_id = None
                self.active_sell_price = 0.0
                return

            # Update price if level changed significantly
            if abs(current_ask_level - self.active_sell_price) > 0.01:
                # Warn if canceling an order with partial fills
                if filled_qty > 0:
                    self._log(f"⚠️ Canceling sell order with PARTIAL FILL: {filled_qty}/{total_qty} shares already filled")
                self._log(f"Updating sell: ${self.active_sell_price:.2f} -> ${current_ask_level:.2f}")
                trading.cancel_order(self.active_sell_order_id)
                get_client().sleep(0.2)  # Wait for cancel
                self.active_sell_order_id = None
                self.active_sell_price = 0.0
                self._submit_limit_sell(current_ask_level)

        except Exception as e:
            self._log(f"Error checking sell order: {e}")
            self.active_sell_order_id = None
            self.active_sell_price = 0.0

    def _submit_limit_buy(self, price: float):
        """Submit a limit buy order."""
        try:
            order = trading.place_limit_order(
                self.ticker, self.trade_size, OrderSide.BUY, price
            )
            self.active_buy_order_id = str(order.id)
            self.active_buy_price = price
            self._log(f"Limit BUY submitted: {order.id} @ ${price:.2f}")
        except Exception as e:
            self._log(f"Error submitting limit buy: {e}")

    def _submit_limit_sell(self, price: float):
        """Submit a limit sell order."""
        try:
            order = trading.place_limit_order(
                self.ticker, self.trade_size, OrderSide.SELL, price
            )
            self.active_sell_order_id = str(order.id)
            self.active_sell_price = price
            self._log(f"Limit SELL submitted: {order.id} @ ${price:.2f}")
        except Exception as e:
            self._log(f"Error submitting limit sell: {e}")

    def _handle_buy_fill(self, fill_price: float):
        """Handle a filled buy order."""
        timestamp = datetime.now(ET)
        trade = self.strategy._execute_buy(
            timestamp, fill_price, self.trade_size,
            reason=f"Limit order filled @ ${fill_price:.2f}"
        )
        self.strategy._retreat_after_buy()
        self._log_trade_fill(trade, self.active_buy_order_id or "unknown")

        # Check stop loss
        if self.strategy._check_stop_loss():
            self._log("STOP LOSS TRIGGERED!")
            self.trading_active = False
            self._cancel_all_orders()
            if self.flatten_on_stop and self.strategy.state.position != 0:
                self._flatten_position_market()
            return

        # After retreat, BOTH quote levels changed - cancel and resubmit the sell order
        # The buy order is already filled, so we just need to update the sell
        if self.active_sell_order_id:
            self._log("Retreat: canceling sell to resubmit at new level")
            try:
                trading.cancel_order(self.active_sell_order_id)
                get_client().sleep(0.2)
            except Exception as e:
                self._log(f"Error canceling sell for retreat: {e}")
            self.active_sell_order_id = None
            self.active_sell_price = 0.0

    def _handle_sell_fill(self, fill_price: float):
        """Handle a filled sell order."""
        timestamp = datetime.now(ET)
        trade = self.strategy._execute_sell(
            timestamp, fill_price, self.trade_size,
            reason=f"Limit order filled @ ${fill_price:.2f}"
        )
        self.strategy._retreat_after_sell()
        self._log_trade_fill(trade, self.active_sell_order_id or "unknown")

        # Check stop loss
        if self.strategy._check_stop_loss():
            self._log("STOP LOSS TRIGGERED!")
            self.trading_active = False
            self._cancel_all_orders()
            if self.flatten_on_stop and self.strategy.state.position != 0:
                self._flatten_position_market()
            return

        # After retreat, BOTH quote levels changed - cancel and resubmit the buy order
        # The sell order is already filled, so we just need to update the buy
        if self.active_buy_order_id:
            self._log("Retreat: canceling buy to resubmit at new level")
            try:
                trading.cancel_order(self.active_buy_order_id)
                get_client().sleep(0.2)
            except Exception as e:
                self._log(f"Error canceling buy for retreat: {e}")
            self.active_buy_order_id = None
            self.active_buy_price = 0.0

    def _cancel_all_orders(self):
        """Cancel all active orders."""
        try:
            if self.active_buy_order_id:
                trading.cancel_order(self.active_buy_order_id)
                self._log("Canceled buy order")
                self.active_buy_order_id = None
        except Exception as e:
            self._log(f"Error canceling buy order: {e}")

        try:
            if self.active_sell_order_id:
                trading.cancel_order(self.active_sell_order_id)
                self._log("Canceled sell order")
                self.active_sell_order_id = None
        except Exception as e:
            self._log(f"Error canceling sell order: {e}")

    def _flatten_position_market(self):
        """Flatten position using market order."""
        if not self.strategy or self.strategy.state.position == 0:
            return

        position = self.strategy.state.position
        self._log(f"Flattening position: {position} shares")

        try:
            if position > 0:
                order = trading.sell(self.ticker, abs(position))
            else:
                order = trading.buy(self.ticker, abs(position))

            self._log(f"Flatten order submitted: {order.id}")
        except Exception as e:
            self._log(f"Error flattening position: {e}")

    # ==================== Trade Logging ====================

    def _log_trade_fill(self, trade: Trade, order_id: str):
        """Log trade to persistent storage."""
        try:
            TRADES_DIR.mkdir(parents=True, exist_ok=True)
            log_file = TRADES_DIR / f"{self.today_str}_{self.ticker}.json"

            trades = []
            if log_file.exists():
                with open(log_file, "r") as f:
                    trades = json.load(f)

            trade_entry = {
                "timestamp": str(trade.timestamp),
                "side": trade.side.value,
                "price": trade.price,
                "quantity": trade.quantity,
                "position_after": trade.position_after,
                "realized_pnl": trade.realized_pnl,
                "reason": trade.reason,
                "order_id": order_id,
                "execution_type": "limit_fill",
            }
            trades.append(trade_entry)

            with open(log_file, "w") as f:
                json.dump(trades, f, indent=2)

        except Exception as e:
            self._log(f"Error logging trade: {e}")

    def _save_daily_summary(self, final_price: float):
        """Save end-of-day summary."""
        if not self.strategy:
            return

        try:
            TRADES_DIR.mkdir(parents=True, exist_ok=True)

            summary = self.strategy.get_summary(final_price)
            summary["ticker"] = self.ticker
            summary["date"] = self.today_str
            summary["execution_mode"] = "simultaneous_limit_orders"
            summary["broker"] = "IB_Gateway"

            summary_file = TRADES_DIR / f"{self.today_str}_{self.ticker}_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

            self._log(f"Daily summary saved to {summary_file}")

        except Exception as e:
            self._log(f"Error saving summary: {e}")

    # ==================== Main Loop ====================

    def start(self):
        """Start the live trading session."""
        self._log("=" * 50)
        self._log(f"Starting Live Trader for {self.ticker}")
        self._log("Execution mode: SIMULTANEOUS limit orders (IB Gateway)")
        self._log("=" * 50)

        # Connect to IB Gateway
        client = get_client()
        if not client.connect():
            self._log("Failed to connect to IB Gateway. Exiting.")
            return

        config.print_mode()

        self.running = True

        while self.running:
            try:
                self._wait_for_premarket()

                if not self.running:
                    break

                self.today_str = datetime.now(ET).strftime("%Y-%m-%d")
                self._log(f"Starting trading day: {self.today_str}")

                if not self._check_regime_filter():
                    self._log("Skipping today due to regime filter")
                    self._wait_for_market_close()
                    continue

                self.quote_width_pct = self._calculate_quote_width()
                self.zero_point = self._get_opening_price()

                # Check gap filter (needs opening price)
                if not self._check_gap_filter(self.zero_point):
                    self._log("Skipping today due to gap filter")
                    self._wait_for_market_close()
                    continue

                self._init_strategy()

                # Start trading
                self.trading_active = True

                # Submit initial orders on both sides
                self._submit_both_orders()

                # Main loop: monitor and update orders
                self._run_trading_loop()

                # End of day
                self._end_of_day()

            except Exception as e:
                self._log(f"Error in main loop: {e}")
                time.sleep(60)

    def _run_trading_loop(self):
        """Main trading loop - monitors orders and updates them."""
        last_status_log = time.time()
        status_interval = 60  # Log status every minute

        while self.running and self._is_market_open():
            try:
                now = time.time()

                # Periodic status log
                if now - last_status_log > status_interval:
                    self._log_status()
                    last_status_log = now

                # Check and update orders
                if self.trading_active:
                    self._check_and_update_orders()

                # Allow IB message processing
                get_client().sleep(0.1)

            except Exception as e:
                self._log(f"Error in trading loop: {e}")
                time.sleep(5)

    def _log_status(self):
        """Log current status."""
        if not self.strategy:
            return

        try:
            quote = market_data.get_latest_quote(self.ticker)
            mid = (float(quote.ask_price) + float(quote.bid_price)) / 2
            pnl = self.strategy.get_total_pnl(mid)
            pos = self.strategy.state.position
            trades = self.strategy.get_trade_count()

            self._log(
                f"Status | Price: ${mid:.2f} | Position: {pos} | "
                f"Trades: {trades} | P&L: ${pnl:.2f} | "
                f"Buy: {'Active' if self.active_buy_order_id else 'None'} | "
                f"Sell: {'Active' if self.active_sell_order_id else 'None'}"
            )
        except Exception as e:
            self._log(f"Error logging status: {e}")

    def _wait_for_market_close(self):
        """Wait until market closes."""
        while self.running:
            now = datetime.now(ET)
            if now.hour >= 16:
                return
            time.sleep(60)

    def _end_of_day(self):
        """End of day cleanup."""
        self._log("=" * 50)
        self._log("End of Day")
        self._log("=" * 50)

        # Cancel any remaining orders
        self._cancel_all_orders()

        if self.strategy:
            try:
                quote = market_data.get_latest_quote(self.ticker)
                final_price = (float(quote.ask_price) + float(quote.bid_price)) / 2
            except:
                final_price = self.zero_point

            summary = self.strategy.get_summary(final_price)
            self._log(f"Total trades: {summary['total_trades']}")
            self._log(f"Final position: {summary['final_position']}")
            self._log(f"Realized P&L: ${summary['realized_pnl']:.2f}")
            self._log(f"Unrealized P&L: ${summary['unrealized_pnl']:.2f}")
            self._log(f"Total P&L: ${summary['total_pnl']:.2f}")

            self._save_daily_summary(final_price)

        self.trading_active = False
        self._log("=" * 50)

    def stop(self):
        """Stop the live trader."""
        self._log("Stopping live trader...")
        self.running = False
        self.trading_active = False

        self._cancel_all_orders()
        self._end_of_day()

        # Disconnect from IB
        get_client().disconnect()
