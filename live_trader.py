"""
Live trading module for the mean reversion scalping strategy.
Connects the nick-scalp strategy to Alpaca for paper/live trading.

Uses pre-submitted limit orders for low-latency execution:
1. At market open, submit resting limit orders at bid/ask levels
2. Monitor for fills via Alpaca's trade updates stream
3. On fill, cancel opposite order, retreat levels, resubmit
"""
import json
import signal
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict
import pytz

import pandas as pd
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.stream import TradingStream

from config import config, TRADES_DIR
from trading import trading
from data import market_data

# Import strategy from nick-scalp
from strategy import MeanReversionScalper, Side, Trade


ET = pytz.timezone("America/New_York")


class LiveTrader:
    """
    Live trading wrapper for the MeanReversionScalper strategy.

    Uses pre-submitted limit orders for low-latency execution:
    - Resting limit orders sit on the book at calculated levels
    - When filled, we update strategy state and resubmit at retreated levels
    - Orders are managed via Alpaca's trade updates WebSocket stream
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

        self.strategy: Optional[MeanReversionScalper] = None
        self.quote_width_pct: float = 0.0
        self.zero_point: float = 0.0
        self.running: bool = False
        self.trading_active: bool = False  # False when stop loss triggered
        self.bars_processed: int = 0
        self.today_str: str = ""

        # Order tracking
        self.active_buy_order_id: Optional[str] = None
        self.active_sell_order_id: Optional[str] = None
        self.active_buy_price: float = 0.0
        self.active_sell_price: float = 0.0
        self.pending_orders: Dict[str, dict] = {}  # order_id -> order info

        # Trade updates stream
        self.trade_stream: Optional[TradingStream] = None
        self.stream_thread: Optional[threading.Thread] = None

        # Lock for thread-safe order management
        self.order_lock = threading.Lock()

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
        """Check if today is a market day using Alpaca's calendar."""
        try:
            clock = trading._client.get_clock()
            return clock.is_open or (
                clock.next_open.date() == datetime.now(ET).date()
            )
        except Exception as e:
            self._log(f"Error checking market day: {e}")
            return datetime.now(ET).weekday() < 5

    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        try:
            clock = trading._client.get_clock()
            return clock.is_open
        except:
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
            # Use DAILY bars, not 1-minute bars
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

    def _start_trade_stream(self):
        """Start the trade updates WebSocket stream."""
        self._log("Starting trade updates stream...")

        self.trade_stream = TradingStream(
            api_key=config.API_KEY,
            secret_key=config.API_SECRET,
            paper=config.PAPER,
        )

        @self.trade_stream.subscribe_trade_updates
        async def on_trade_update(data):
            self._handle_trade_update(data)

        def run_stream():
            try:
                self.trade_stream.run()
            except Exception as e:
                self._log(f"Trade stream error: {e}")

        self.stream_thread = threading.Thread(target=run_stream, daemon=True)
        self.stream_thread.start()
        self._log("Trade updates stream started")

    def _stop_trade_stream(self):
        """Stop the trade updates stream."""
        if self.trade_stream:
            try:
                self.trade_stream.stop()
            except:
                pass

    def _handle_trade_update(self, data):
        """Handle trade update events from Alpaca."""
        try:
            event = data.event
            order = data.order

            order_id = str(order.id)
            symbol = order.symbol
            side = order.side
            filled_qty = float(order.filled_qty) if order.filled_qty else 0
            filled_avg_price = float(order.filled_avg_price) if order.filled_avg_price else 0

            # Only process our ticker
            if symbol != self.ticker:
                return

            self._log(f"Trade update: {event} | Order {order_id[:8]}... | {side} | Filled: {filled_qty}")

            if event == "fill":
                self._handle_fill(order_id, side, filled_qty, filled_avg_price)
            elif event == "partial_fill":
                self._log(f"Partial fill: {filled_qty} @ ${filled_avg_price:.2f}")
            elif event in ("canceled", "expired", "rejected"):
                self._handle_order_closed(order_id, event)

        except Exception as e:
            self._log(f"Error handling trade update: {e}")

    def _handle_fill(self, order_id: str, side: str, qty: float, price: float):
        """Handle a filled order."""
        with self.order_lock:
            timestamp = datetime.now(ET)

            # Determine which order was filled
            is_buy = side == "buy"

            self._log(f"FILL: {'BUY' if is_buy else 'SELL'} {int(qty)} @ ${price:.2f}")

            # Update strategy state manually (since we're not using process_bar)
            if self.strategy:
                if is_buy:
                    trade = self.strategy._execute_buy(
                        timestamp, price, int(qty),
                        reason=f"Limit order filled @ ${price:.2f}"
                    )
                    self.strategy._retreat_after_buy()
                    self.active_buy_order_id = None
                else:
                    trade = self.strategy._execute_sell(
                        timestamp, price, int(qty),
                        reason=f"Limit order filled @ ${price:.2f}"
                    )
                    self.strategy._retreat_after_sell()
                    self.active_sell_order_id = None

                # Log the trade
                self._log_trade_fill(trade, order_id)

                # Check stop loss after trade
                if self.strategy._check_stop_loss():
                    self._log("STOP LOSS TRIGGERED!")
                    self.trading_active = False
                    self._cancel_all_orders()

                    if self.flatten_on_stop and self.strategy.state.position != 0:
                        self._flatten_position_market()
                    return

                # Cancel the opposite order and resubmit both at new levels
                self._cancel_opposite_order(is_buy)
                self._submit_resting_orders()

    def _handle_order_closed(self, order_id: str, reason: str):
        """Handle canceled/expired/rejected order."""
        with self.order_lock:
            if order_id == self.active_buy_order_id:
                self.active_buy_order_id = None
                self._log(f"Buy order {reason}")
            elif order_id == self.active_sell_order_id:
                self.active_sell_order_id = None
                self._log(f"Sell order {reason}")

    def _cancel_opposite_order(self, filled_was_buy: bool):
        """Cancel the order on the opposite side after a fill."""
        try:
            if filled_was_buy and self.active_sell_order_id:
                trading.cancel_order(self.active_sell_order_id)
                self._log(f"Canceled sell order {self.active_sell_order_id[:8]}...")
                self.active_sell_order_id = None
            elif not filled_was_buy and self.active_buy_order_id:
                trading.cancel_order(self.active_buy_order_id)
                self._log(f"Canceled buy order {self.active_buy_order_id[:8]}...")
                self.active_buy_order_id = None
        except Exception as e:
            self._log(f"Error canceling opposite order: {e}")

    def _cancel_all_orders(self):
        """Cancel all active orders."""
        with self.order_lock:
            try:
                if self.active_buy_order_id:
                    trading.cancel_order(self.active_buy_order_id)
                    self._log(f"Canceled buy order")
                    self.active_buy_order_id = None
            except Exception as e:
                self._log(f"Error canceling buy order: {e}")

            try:
                if self.active_sell_order_id:
                    trading.cancel_order(self.active_sell_order_id)
                    self._log(f"Canceled sell order")
                    self.active_sell_order_id = None
            except Exception as e:
                self._log(f"Error canceling sell order: {e}")

    def _check_and_execute(self):
        """Manage limit orders with quick switching based on price direction.

        Strategy:
        1. Only one limit order active at a time (Alpaca restriction)
        2. Track price direction - if price moves toward opposite level, switch sides
        3. Submit the side that price is moving TOWARD (likely to fill)
        4. Quick cancel and switch when price reverses direction
        """
        if not self.trading_active or not self.strategy:
            return

        # Check market hours
        now = datetime.now(ET)
        if now.hour == 15 and now.minute >= 55:
            self._log("Near market close, stopping execution")
            return
        if now.hour >= 16:
            return

        with self.order_lock:
            bid, ask = self.strategy.state.get_quotes()

            # Round prices to 2 decimal places
            bid_price = round(bid.price, 2)
            ask_price = round(ask.price, 2)

            # Get current market price
            try:
                quote = market_data.get_latest_quote(self.ticker)
                current_bid = float(quote.bid_price)
                current_ask = float(quote.ask_price)
                current_mid = (current_bid + current_ask) / 2
            except Exception as e:
                self._log(f"Error getting quote: {e}")
                return

            # Track price direction
            if not hasattr(self, '_last_mid_price'):
                self._last_mid_price = current_mid

            price_moving_down = current_mid < self._last_mid_price
            price_moving_up = current_mid > self._last_mid_price
            self._last_mid_price = current_mid

            # Determine which side SHOULD have the order based on price direction
            # If price moving down -> submit BUY (price approaching bid level)
            # If price moving up -> submit SELL (price approaching ask level)
            # If price stable -> use distance to determine

            dist_to_bid = current_mid - bid_price
            dist_to_ask = ask_price - current_mid

            if price_moving_down:
                desired_side = "buy"
            elif price_moving_up:
                desired_side = "sell"
            else:
                # Price stable - pick closer side
                desired_side = "buy" if dist_to_bid <= dist_to_ask else "sell"

            # Check active orders and switch if needed
            if self.active_buy_order_id:
                # Check for fill first
                filled = self._check_buy_order_fill(bid_price)
                if filled:
                    return

                # If we should be on sell side and price moved significantly, switch
                if desired_side == "sell" and price_moving_up:
                    self._cancel_and_switch_to_sell(ask_price)

            elif self.active_sell_order_id:
                # Check for fill first
                filled = self._check_sell_order_fill(ask_price)
                if filled:
                    return

                # If we should be on buy side and price moved significantly, switch
                if desired_side == "buy" and price_moving_down:
                    self._cancel_and_switch_to_buy(bid_price)

            else:
                # No active order - submit based on desired side
                if desired_side == "buy":
                    self._submit_limit_buy(bid_price)
                else:
                    self._submit_limit_sell(ask_price)

    def _check_buy_order_fill(self, current_bid_level: float) -> bool:
        """Check if buy order filled. Returns True if filled."""
        try:
            order = trading.get_order(self.active_buy_order_id)

            if order.status.value == "filled":
                fill_price = float(order.filled_avg_price)
                self._log(f"BUY FILLED @ ${fill_price:.2f}")
                self._handle_buy_fill(fill_price)
                self.active_buy_order_id = None
                self.active_buy_price = 0.0
                return True

            elif order.status.value in ["canceled", "expired", "rejected"]:
                self._log(f"Buy order {order.status.value}")
                self.active_buy_order_id = None
                self.active_buy_price = 0.0

            # Update price if level changed
            elif abs(current_bid_level - self.active_buy_price) > 0.01:
                self._log(f"Updating buy: ${self.active_buy_price:.2f} -> ${current_bid_level:.2f}")
                try:
                    trading.cancel_order(self.active_buy_order_id)
                except Exception:
                    pass
                self.active_buy_order_id = None
                self.active_buy_price = 0.0
                self._submit_limit_buy(current_bid_level)

        except Exception as e:
            self._log(f"Error checking buy order: {e}")
            self.active_buy_order_id = None
            self.active_buy_price = 0.0

        return False

    def _check_sell_order_fill(self, current_ask_level: float) -> bool:
        """Check if sell order filled. Returns True if filled."""
        try:
            order = trading.get_order(self.active_sell_order_id)

            if order.status.value == "filled":
                fill_price = float(order.filled_avg_price)
                self._log(f"SELL FILLED @ ${fill_price:.2f}")
                self._handle_sell_fill(fill_price)
                self.active_sell_order_id = None
                self.active_sell_price = 0.0
                return True

            elif order.status.value in ["canceled", "expired", "rejected"]:
                self._log(f"Sell order {order.status.value}")
                self.active_sell_order_id = None
                self.active_sell_price = 0.0

            # Update price if level changed
            elif abs(current_ask_level - self.active_sell_price) > 0.01:
                self._log(f"Updating sell: ${self.active_sell_price:.2f} -> ${current_ask_level:.2f}")
                try:
                    trading.cancel_order(self.active_sell_order_id)
                except Exception:
                    pass
                self.active_sell_order_id = None
                self.active_sell_price = 0.0
                self._submit_limit_sell(current_ask_level)

        except Exception as e:
            self._log(f"Error checking sell order: {e}")
            self.active_sell_order_id = None
            self.active_sell_price = 0.0

        return False

    def _cancel_and_switch_to_sell(self, ask_price: float):
        """Cancel buy order and switch to sell side."""
        order_id = self.active_buy_order_id
        self.active_buy_order_id = None
        self.active_buy_price = 0.0

        try:
            trading.cancel_order(order_id)
            self._log("Switching: canceling buy...")

            # Wait for cancel to complete (poll up to 3 seconds)
            canceled = False
            for _ in range(30):
                time.sleep(0.1)
                try:
                    order = trading.get_order(order_id)
                    status = order.status.value
                    if status in ["canceled", "expired"]:
                        canceled = True
                        break
                    elif status == "filled":
                        # Order filled while we were canceling - handle the fill
                        self._log("Buy filled during cancel!")
                        fill_price = float(order.filled_avg_price)
                        self._handle_buy_fill(fill_price)
                        return  # Don't switch, we just got filled
                except Exception:
                    canceled = True  # Order may not exist anymore
                    break

            if canceled:
                self._log("Buy canceled, submitting sell")
                self._submit_limit_sell(ask_price)
            else:
                self._log("Warning: Cancel may not have completed")
                # Don't submit sell - wait for next iteration

        except Exception as e:
            self._log(f"Error canceling buy: {e}")

    def _cancel_and_switch_to_buy(self, bid_price: float):
        """Cancel sell order and switch to buy side."""
        order_id = self.active_sell_order_id
        self.active_sell_order_id = None
        self.active_sell_price = 0.0

        try:
            trading.cancel_order(order_id)
            self._log("Switching: canceling sell...")

            # Wait for cancel to complete (poll up to 3 seconds)
            canceled = False
            for _ in range(30):
                time.sleep(0.1)
                try:
                    order = trading.get_order(order_id)
                    status = order.status.value
                    if status in ["canceled", "expired"]:
                        canceled = True
                        break
                    elif status == "filled":
                        # Order filled while we were canceling - handle the fill
                        self._log("Sell filled during cancel!")
                        fill_price = float(order.filled_avg_price)
                        self._handle_sell_fill(fill_price)
                        return  # Don't switch, we just got filled
                except Exception:
                    canceled = True  # Order may not exist anymore
                    break

            if canceled:
                self._log("Sell canceled, submitting buy")
                self._submit_limit_buy(bid_price)
            else:
                self._log("Warning: Cancel may not have completed")
                # Don't submit buy - wait for next iteration

        except Exception as e:
            self._log(f"Error canceling sell: {e}")

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
        self._log_trade_fill(trade, self.active_buy_order_id)

        # Check stop loss
        if self.strategy._check_stop_loss():
            self._log("STOP LOSS TRIGGERED!")
            self.trading_active = False
            if self.flatten_on_stop and self.strategy.state.position != 0:
                self._flatten_position_market()

    def _handle_sell_fill(self, fill_price: float):
        """Handle a filled sell order."""
        timestamp = datetime.now(ET)
        trade = self.strategy._execute_sell(
            timestamp, fill_price, self.trade_size,
            reason=f"Limit order filled @ ${fill_price:.2f}"
        )
        self.strategy._retreat_after_sell()
        self._log_trade_fill(trade, self.active_sell_order_id)

        # Check stop loss
        if self.strategy._check_stop_loss():
            self._log("STOP LOSS TRIGGERED!")
            self.trading_active = False
            if self.flatten_on_stop and self.strategy.state.position != 0:
                self._flatten_position_market()

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
            summary["execution_mode"] = "limit_orders"

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
        self._log("Execution mode: LIMIT orders (one at a time)")
        self._log("=" * 50)

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
                self._init_strategy()

                # Start trading
                self.trading_active = True

                # Main loop: monitor price and execute
                self._run_trading_loop()

                # End of day
                self._end_of_day()

            except Exception as e:
                self._log(f"Error in main loop: {e}")
                time.sleep(60)

    def _run_trading_loop(self):
        """Main trading loop - monitors price and executes market orders."""
        last_status_log = time.time()
        status_interval = 60  # Log status every minute

        while self.running and self._is_market_open():
            try:
                now = time.time()

                # Periodic status log
                if now - last_status_log > status_interval:
                    self._log_status()
                    last_status_log = now

                # Check price and execute if levels are hit
                if self.trading_active:
                    self._check_and_execute()

                # Poll every 100ms for faster response
                time.sleep(0.1)

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

        # Stop trade stream
        self._stop_trade_stream()

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
        self._stop_trade_stream()
        self._end_of_day()
