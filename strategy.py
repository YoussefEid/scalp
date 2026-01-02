"""
Mean Reversion Scalping Strategy Engine

Implements a market-making style scalping strategy that:
1. Places quotes around a "zero point" (opening price)
2. Captures spread when price oscillates
3. Retreats quotes after fills to manage inventory
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime, time, timedelta
from enum import Enum
import pandas as pd


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Trade:
    """Record of a single trade execution."""

    timestamp: datetime
    side: Side
    price: float
    quantity: int
    position_after: int
    realized_pnl: float  # P&L realized on this trade (for closing trades)
    capital_required: float  # Capital required to purchase these shares
    adjusted_zero_point: Optional[float] = None  # Zero point after lambda adjustment
    reason: str = ""  # Explanation for why this trade occurred

    def __str__(self):
        base = (
            f"{self.timestamp} | {self.side.value:4} | "
            f"Price: ${self.price:.2f} | Qty: {self.quantity} | "
            f"Capital: ${self.capital_required:.2f} | "
            f"Position: {self.position_after} | P&L: ${self.realized_pnl:.2f}"
        )
        if self.adjusted_zero_point is not None:
            base += f" | Zero: ${self.adjusted_zero_point:.2f}"
        if self.reason:
            base += f" | {self.reason}"
        return base


@dataclass
class Quote:
    """A bid or ask quote."""

    price: float
    size: int
    side: Side


@dataclass
class Lot:
    """A lot of shares purchased/sold at a specific price (for LIFO tracking)."""

    price: float
    quantity: int  # Positive for long lots, negative for short lots


@dataclass
class StrategyState:
    """Current state of the strategy."""

    zero_point: float  # Opening price / reference price
    quote_width: float  # Width on each side of zero point (in dollars)
    retreat: float  # Amount to retreat after a fill (in dollars)
    trade_size: int  # Number of shares per trade

    # Current position
    position: int = 0  # Positive = long, negative = short
    avg_entry_price: float = 0.0  # Kept for backward compatibility

    # LIFO lot tracking (stack - most recent lot at end)
    long_lots: List[Lot] = field(default_factory=list)  # Stack of long positions
    short_lots: List[Lot] = field(default_factory=list)  # Stack of short positions

    # Quote adjustments from retreats
    bid_adjustment: float = 0.0
    ask_adjustment: float = 0.0

    # P&L tracking
    realized_pnl: float = 0.0
    trades: List[Trade] = field(default_factory=list)

    # Capital tracking for return calculation
    total_position_capital_bars: float = 0.0  # Sum of |position| * price for each bar
    total_bars_processed: int = 0  # Count of bars processed

    # Zero point adjustment (lambda) settings
    enable_zero_point_adjustment: bool = False
    zero_point_decay_factor: float = 0.95
    zero_point_blend_factor: float = 0.05
    original_zero_point: float = 0.0  # Track starting point for reporting

    # Stop loss settings (consecutive trade based)
    enable_stop_loss: bool = False
    consecutive_trade_limit: int = 7  # Stop loss triggers at this many consecutive same-direction trades
    stop_loss_triggered: bool = False  # Flag to stop trading for the day
    flatten_on_stop: bool = False  # Flatten position immediately when stop loss triggers
    position_flattened_on_stop: bool = False  # Flag indicating position was flattened on stop
    consecutive_buy_count: int = 0  # Count of consecutive buy trades
    consecutive_sell_count: int = 0  # Count of consecutive sell trades
    # Captured counts at the moment stop loss triggers (before flatten resets them)
    consecutive_buys_at_stop: int = 0
    consecutive_sells_at_stop: int = 0

    # Trend stop loss settings (distance from open)
    enable_trend_stop: bool = False
    trend_stop_multiplier: float = 15.0  # Pause if price moves > Nx width from open
    trend_stop_triggered: bool = False
    trend_threshold_pct: float = 0.0  # Calculated threshold (width * multiplier)

    # Volatility stop loss settings (bar range vs historical width)
    enable_vol_stop: bool = False
    vol_stop_multiplier: float = 2.0  # Stop if rolling bar range > Nx historical width
    vol_stop_lookback: int = 10  # Number of recent bars to average for rolling range
    vol_stop_triggered: bool = False
    recent_bar_ranges: List[float] = field(default_factory=list)  # Rolling window of bar ranges

    # Spike stop loss settings (single bar exceeds threshold)
    enable_spike_stop: bool = False
    spike_stop_multiplier: float = 3.0  # Stop if any single bar > Nx historical width
    spike_stop_triggered: bool = False

    # EMA flatness filter state (dynamic pause/resume)
    enable_ema_filter: bool = False
    ema_period: int = 20
    ema_flat_threshold: float = 0.02
    ema_value: float = 0.0  # Current EMA value
    ema_initialized: bool = False  # True once we have enough bars
    ema_prices: List[float] = field(default_factory=list)  # Price history for EMA warmup
    ema_is_flat: bool = True  # True when EMA slope is within threshold (trading allowed)
    flatten_on_ema_deviation: bool = False  # Exit position when EMA starts trending
    ema_flattened_position: bool = False  # Track if we've flattened due to EMA deviation

    # Oscillation stop state (resumable pause/resume)
    enable_oscillation_stop: bool = False
    oscillation_stop_consecutive: int = 5
    oscillation_resume_profitable: int = 4
    oscillation_stop_triggered: bool = False  # In "testing" mode after first trigger
    oscillation_stop_permanent: bool = False  # Permanent stop (hit threshold twice)
    profitable_trade_count: int = 0  # Tracks profitable trades since stop triggered

    # Adaptive intraday volatility stop (resumable - builds baseline from today's data)
    enable_adaptive_vol_stop: bool = False
    adaptive_vol_warmup_bars: int = 30  # Build baseline from first N bars
    adaptive_vol_multiplier: float = 2.5  # Pause if rolling vol > Nx baseline
    adaptive_vol_lookback: int = 10  # Rolling window for current volatility
    adaptive_vol_baseline: float = 0.0  # Median bar range from warmup period
    adaptive_vol_bar_ranges: List[float] = field(default_factory=list)  # All bar ranges
    adaptive_vol_triggered: bool = False  # Currently paused due to high volatility

    # Delayed start filter state (only trade after specified time)
    enable_delayed_start: bool = False
    delayed_start_hour: int = 11
    delayed_start_minute: int = 30
    delayed_start_active: bool = True  # True means still waiting, False means trading allowed

    # Breakout filter state (skip trading if trend detected in lookback window)
    enable_breakout_filter: bool = False
    breakout_lookback_bars: int = 120  # 2 hours of 1-min bars
    breakout_threshold_pct: float = 2.0  # Skip if price moved > this % from window start
    bar_history: List[dict] = field(default_factory=list)  # Rolling window of bars
    breakout_detected: bool = False  # Currently paused due to breakout

    # Rapid buildup stop state (time-window based consecutive trade detection with flatten)
    enable_rapid_buildup_stop: bool = False
    rapid_buildup_window_minutes: int = 60
    rapid_buildup_consecutive: int = 10
    rapid_buildup_triggered: bool = False

    def get_current_bid(self) -> float:
        """Calculate current bid price."""
        return self.zero_point - self.quote_width + self.bid_adjustment

    def get_current_ask(self) -> float:
        """Calculate current ask price."""
        return self.zero_point + self.quote_width + self.ask_adjustment

    def get_quotes(self) -> Tuple[Quote, Quote]:
        """Get current bid and ask quotes."""
        bid = Quote(
            price=self.get_current_bid(), size=self.trade_size, side=Side.BUY
        )
        ask = Quote(
            price=self.get_current_ask(), size=self.trade_size, side=Side.SELL
        )
        return bid, ask


class MeanReversionScalper:
    """
    Mean reversion scalping strategy.

    The strategy maintains quotes around a zero point (opening price).
    When filled, it retreats the quotes to capture spread on the next fill.
    """

    def __init__(
        self,
        zero_point: float,
        quote_width_pct: float,
        trade_size: int = 100,
        quote_width_multiplier: float = 1.0,
        enable_zero_point_adjustment: bool = False,
        zero_point_decay_factor: float = 0.95,
        zero_point_blend_factor: float = 0.05,
        enable_stop_loss: bool = False,
        consecutive_trade_limit: int = 7,
        flatten_on_stop: bool = False,
        enable_trend_stop: bool = False,
        trend_stop_multiplier: float = 15.0,
        enable_vol_stop: bool = False,
        vol_stop_multiplier: float = 2.0,
        vol_stop_lookback: int = 10,
        enable_spike_stop: bool = False,
        spike_stop_multiplier: float = 3.0,
        enable_ema_filter: bool = False,
        ema_period: int = 20,
        ema_flat_threshold: float = 0.02,
        flatten_on_ema_deviation: bool = False,
        enable_oscillation_stop: bool = False,
        oscillation_stop_consecutive: int = 5,
        oscillation_resume_profitable: int = 4,
        enable_adaptive_vol_stop: bool = False,
        adaptive_vol_warmup_bars: int = 30,
        adaptive_vol_multiplier: float = 2.5,
        adaptive_vol_lookback: int = 10,
        enable_delayed_start: bool = False,
        delayed_start_hour: int = 11,
        delayed_start_minute: int = 30,
        enable_breakout_filter: bool = False,
        breakout_lookback_bars: int = 120,
        breakout_threshold_pct: float = 2.0,
        enable_rapid_buildup_stop: bool = False,
        rapid_buildup_window_minutes: int = 60,
        rapid_buildup_consecutive: int = 10,
    ):
        """
        Initialize the scalping strategy.

        Args:
            zero_point: The reference price (typically opening price)
            quote_width_pct: Total width as percentage (e.g., 0.4 means 0.4% total, 0.2% each side)
            trade_size: Number of shares per trade
            quote_width_multiplier: Multiplier to adjust quote width
            enable_zero_point_adjustment: Enable dynamic zero point adjustment (lambda)
            zero_point_decay_factor: Weight for current zero point in adjustment formula
            zero_point_blend_factor: Weight for trade price in adjustment formula
            enable_stop_loss: Enable daily stop loss based on consecutive trades
            consecutive_trade_limit: Stop loss triggers at this many consecutive same-direction trades
            flatten_on_stop: Flatten position immediately when stop loss triggers
            enable_trend_stop: Enable trend stop loss (pause if price moves too far from open)
            trend_stop_multiplier: Trend stop triggers when price moves > Nx width from open
            enable_vol_stop: Enable volatility stop loss (pause if current bar ranges exceed historical)
            vol_stop_multiplier: Stop when rolling bar range > Nx historical width
            vol_stop_lookback: Number of recent bars to average for rolling range comparison
            enable_spike_stop: Enable spike stop loss (pause if any single bar exceeds threshold)
            spike_stop_multiplier: Stop when any single bar > Nx historical width
            enable_ema_filter: Enable EMA flatness filter (dynamic pause/resume trading)
            ema_period: Number of bars for EMA calculation
            ema_flat_threshold: Max absolute slope % to consider EMA "flat" (allow trading)
            flatten_on_ema_deviation: Exit position immediately when EMA starts trending
            enable_oscillation_stop: Enable resumable oscillation stop
            oscillation_stop_consecutive: Stop after N consecutive same-direction trades
            oscillation_resume_profitable: Resume after N profitable trades
            enable_adaptive_vol_stop: Enable adaptive intraday volatility stop (resumable)
            adaptive_vol_warmup_bars: Number of bars to build baseline from
            adaptive_vol_multiplier: Pause if rolling vol > Nx baseline
            adaptive_vol_lookback: Rolling window for current volatility comparison
            enable_delayed_start: Only start trading after specified time (11:30am default)
            delayed_start_hour: Hour to start trading (24-hour format)
            delayed_start_minute: Minute to start trading
            enable_breakout_filter: Skip trading if breakout detected in lookback window
            breakout_lookback_bars: Number of bars to look back for breakout detection
            breakout_threshold_pct: Skip trading if price moved > this % from window start
            enable_rapid_buildup_stop: Stop and flatten when N consecutive same-side trades in time window
            rapid_buildup_window_minutes: Rolling time window in minutes (default: 60)
            rapid_buildup_consecutive: Number of consecutive same-side trades to trigger (default: 10)
        """
        # Calculate quote width in dollars (half of total width for each side)
        quote_width_dollars = (zero_point * quote_width_pct / 100) / 2
        quote_width_dollars *= quote_width_multiplier

        # Retreat is 1/2 of quote width
        retreat_dollars = quote_width_dollars / 2

        # Calculate trend stop threshold (based on intraday volatility width)
        # quote_width_pct is the intraday volatility width percentage
        trend_threshold_pct = quote_width_pct * trend_stop_multiplier

        self.state = StrategyState(
            zero_point=zero_point,
            quote_width=quote_width_dollars,
            retreat=retreat_dollars,
            trade_size=trade_size,
            enable_zero_point_adjustment=enable_zero_point_adjustment,
            zero_point_decay_factor=zero_point_decay_factor,
            zero_point_blend_factor=zero_point_blend_factor,
            original_zero_point=zero_point,
            enable_stop_loss=enable_stop_loss,
            consecutive_trade_limit=consecutive_trade_limit,
            flatten_on_stop=flatten_on_stop,
            enable_trend_stop=enable_trend_stop,
            trend_stop_multiplier=trend_stop_multiplier,
            trend_threshold_pct=trend_threshold_pct,
            enable_vol_stop=enable_vol_stop,
            vol_stop_multiplier=vol_stop_multiplier,
            vol_stop_lookback=vol_stop_lookback,
            enable_spike_stop=enable_spike_stop,
            spike_stop_multiplier=spike_stop_multiplier,
            enable_ema_filter=enable_ema_filter,
            ema_period=ema_period,
            ema_flat_threshold=ema_flat_threshold,
            flatten_on_ema_deviation=flatten_on_ema_deviation,
            enable_oscillation_stop=enable_oscillation_stop,
            oscillation_stop_consecutive=oscillation_stop_consecutive,
            oscillation_resume_profitable=oscillation_resume_profitable,
            enable_adaptive_vol_stop=enable_adaptive_vol_stop,
            adaptive_vol_warmup_bars=adaptive_vol_warmup_bars,
            adaptive_vol_multiplier=adaptive_vol_multiplier,
            adaptive_vol_lookback=adaptive_vol_lookback,
            enable_delayed_start=enable_delayed_start,
            delayed_start_hour=delayed_start_hour,
            delayed_start_minute=delayed_start_minute,
            enable_breakout_filter=enable_breakout_filter,
            breakout_lookback_bars=breakout_lookback_bars,
            breakout_threshold_pct=breakout_threshold_pct,
            enable_rapid_buildup_stop=enable_rapid_buildup_stop,
            rapid_buildup_window_minutes=rapid_buildup_window_minutes,
            rapid_buildup_consecutive=rapid_buildup_consecutive,
        )

    def _adjust_zero_point(self, trade_price: float) -> Optional[float]:
        """
        Adjust zero point after a trade using the lambda formula.

        Formula: NEW_ZERO = (CURRENT_ZERO * decay) + (blend * TRADE_PRICE)

        Args:
            trade_price: The price at which the trade was executed

        Returns:
            The new zero point if adjustment is enabled, None otherwise
        """
        if not self.state.enable_zero_point_adjustment:
            return None

        new_zero = (
            self.state.zero_point * self.state.zero_point_decay_factor
        ) + (self.state.zero_point_blend_factor * trade_price)

        self.state.zero_point = new_zero
        return new_zero

    def _execute_buy(self, timestamp: datetime, price: float, quantity: int, reason: str = "") -> Trade:
        """Execute a buy order."""
        # Calculate capital required to purchase these shares
        capital_required = price * quantity

        # Calculate realized P&L if closing a short position (LIFO)
        realized_pnl = 0.0
        remaining_qty = quantity

        if self.state.position < 0:
            # Closing short position using LIFO
            while remaining_qty > 0 and self.state.short_lots:
                lot = self.state.short_lots[-1]  # Get most recent short lot
                shares_to_close = min(remaining_qty, abs(lot.quantity))

                # Short P&L: (entry price - exit price) * shares
                realized_pnl += (lot.price - price) * shares_to_close

                remaining_qty -= shares_to_close
                lot.quantity += shares_to_close  # Make less negative

                if lot.quantity == 0:
                    self.state.short_lots.pop()

        # If we have remaining quantity after closing shorts, open long position
        if remaining_qty > 0:
            self.state.long_lots.append(Lot(price=price, quantity=remaining_qty))

        # Update position
        self.state.position += quantity
        self.state.realized_pnl += realized_pnl

        # Adjust zero point if lambda is enabled
        adjusted_zero = self._adjust_zero_point(price)

        # Update consecutive trade counters
        self.state.consecutive_buy_count += 1
        self.state.consecutive_sell_count = 0  # Reset sell counter

        trade = Trade(
            timestamp=timestamp,
            side=Side.BUY,
            price=price,
            quantity=quantity,
            position_after=self.state.position,
            realized_pnl=realized_pnl,
            capital_required=capital_required,
            adjusted_zero_point=adjusted_zero,
            reason=reason,
        )
        self.state.trades.append(trade)

        # Track profitable trades for oscillation resume
        if self.state.oscillation_stop_triggered and trade.realized_pnl > 0:
            self.state.profitable_trade_count += 1

        return trade

    def _execute_sell(self, timestamp: datetime, price: float, quantity: int, reason: str = "") -> Trade:
        """Execute a sell order."""
        # Calculate capital required (proceeds from sale)
        capital_required = price * quantity

        # Calculate realized P&L if closing a long position (LIFO)
        realized_pnl = 0.0
        remaining_qty = quantity

        if self.state.position > 0:
            # Closing long position using LIFO
            while remaining_qty > 0 and self.state.long_lots:
                lot = self.state.long_lots[-1]  # Get most recent long lot
                shares_to_close = min(remaining_qty, lot.quantity)

                # Long P&L: (exit price - entry price) * shares
                realized_pnl += (price - lot.price) * shares_to_close

                remaining_qty -= shares_to_close
                lot.quantity -= shares_to_close

                if lot.quantity == 0:
                    self.state.long_lots.pop()

        # If we have remaining quantity after closing longs, open short position
        if remaining_qty > 0:
            self.state.short_lots.append(Lot(price=price, quantity=-remaining_qty))

        # Update position
        self.state.position -= quantity
        self.state.realized_pnl += realized_pnl

        # Adjust zero point if lambda is enabled
        adjusted_zero = self._adjust_zero_point(price)

        # Update consecutive trade counters
        self.state.consecutive_sell_count += 1
        self.state.consecutive_buy_count = 0  # Reset buy counter

        trade = Trade(
            timestamp=timestamp,
            side=Side.SELL,
            price=price,
            quantity=quantity,
            position_after=self.state.position,
            realized_pnl=realized_pnl,
            capital_required=capital_required,
            adjusted_zero_point=adjusted_zero,
            reason=reason,
        )
        self.state.trades.append(trade)

        # Track profitable trades for oscillation resume
        if self.state.oscillation_stop_triggered and trade.realized_pnl > 0:
            self.state.profitable_trade_count += 1

        return trade

    def _retreat_after_buy(self):
        """Adjust quotes after a buy fill (retreat downward)."""
        self.state.bid_adjustment -= self.state.retreat
        self.state.ask_adjustment -= self.state.retreat

    def _retreat_after_sell(self):
        """Adjust quotes after a sell fill (retreat upward)."""
        self.state.bid_adjustment += self.state.retreat
        self.state.ask_adjustment += self.state.retreat

    def _check_stop_loss(self) -> bool:
        """
        Check if stop loss has been triggered based on consecutive trades.

        Stop loss triggers when there are N or more consecutive trades in the
        same direction (all buys or all sells), indicating trending behavior
        rather than mean reversion.

        Returns:
            True if stop loss is triggered, False otherwise
        """
        if not self.state.enable_stop_loss:
            return False

        if self.state.stop_loss_triggered:
            return True  # Already triggered

        # Check for consecutive trades in same direction
        if (self.state.consecutive_buy_count >= self.state.consecutive_trade_limit or
                self.state.consecutive_sell_count >= self.state.consecutive_trade_limit):
            self.state.stop_loss_triggered = True
            # Capture the counts at the moment of triggering (before flatten resets them)
            self.state.consecutive_buys_at_stop = self.state.consecutive_buy_count
            self.state.consecutive_sells_at_stop = self.state.consecutive_sell_count
            return True

        return False

    def _check_trend_stop(self, current_price: float) -> bool:
        """
        Check if price has moved too far from open, indicating a trend day.

        Trend stop triggers when the distance from opening price exceeds
        a threshold based on the intraday volatility width. This helps
        avoid trading on trending days where mean reversion is unlikely.

        Args:
            current_price: The current price to check against opening price

        Returns:
            True if trend stop is triggered, False otherwise
        """
        if not self.state.enable_trend_stop:
            return False

        if self.state.trend_stop_triggered:
            return True  # Already triggered

        # Calculate distance from open as percentage
        distance_pct = abs(current_price - self.state.original_zero_point) / self.state.original_zero_point * 100

        if distance_pct > self.state.trend_threshold_pct:
            self.state.trend_stop_triggered = True
            return True

        return False

    def _check_vol_stop(self, bar_range: float) -> bool:
        """
        Check if current bar volatility exceeds historical volatility threshold.

        Tracks a rolling window of recent bar ranges and compares the average
        to the expected bar range (quote_width). If current volatility is
        significantly higher than historical, triggers a stop.

        The quote_width represents the historical median bar range when using
        intraday volatility mode, so we compare: avg_recent_range > N * quote_width

        Args:
            bar_range: The high-low range of the current bar

        Returns:
            True if vol stop is triggered, False otherwise
        """
        if not self.state.enable_vol_stop:
            return False

        if self.state.vol_stop_triggered:
            return True  # Already triggered

        # Add current bar range to rolling window
        self.state.recent_bar_ranges.append(bar_range)

        # Keep only the most recent N bars
        if len(self.state.recent_bar_ranges) > self.state.vol_stop_lookback:
            self.state.recent_bar_ranges.pop(0)

        # Need at least a few bars before checking (avoid false triggers on first bars)
        if len(self.state.recent_bar_ranges) < min(3, self.state.vol_stop_lookback):
            return False

        # Calculate average of recent bar ranges
        avg_recent_range = sum(self.state.recent_bar_ranges) / len(self.state.recent_bar_ranges)

        # Compare to historical expected range (quote_width * 2 gives full width)
        # quote_width is half the expected price range, so full range = 2 * quote_width
        expected_range = self.state.quote_width * 2

        # Trigger if average recent range exceeds threshold
        if avg_recent_range > self.state.vol_stop_multiplier * expected_range:
            self.state.vol_stop_triggered = True
            return True

        return False

    def _check_spike_stop(self, bar_range: float) -> bool:
        """
        Check if a single bar's range exceeds the spike threshold.

        Unlike vol stop which averages over multiple bars, spike stop triggers
        immediately when ANY single bar exceeds the threshold. This provides
        faster reaction to sudden volatility spikes.

        Args:
            bar_range: The high-low range of the current bar

        Returns:
            True if spike stop is triggered, False otherwise
        """
        if not self.state.enable_spike_stop:
            return False

        if self.state.spike_stop_triggered:
            return True  # Already triggered

        # Compare single bar to historical expected range
        # quote_width is half the expected price range, so full range = 2 * quote_width
        expected_range = self.state.quote_width * 2

        # Trigger immediately if this single bar exceeds threshold
        if bar_range > self.state.spike_stop_multiplier * expected_range:
            self.state.spike_stop_triggered = True
            return True

        return False

    def _check_adaptive_vol_stop(self, bar_range: float) -> bool:
        """
        Adaptive intraday volatility stop (RESUMABLE).

        Builds a baseline from the first N bars of the day, then pauses trading
        when rolling volatility exceeds the baseline by a multiplier. Unlike
        permanent stops, this can resume when volatility normalizes.

        1. During warmup (first N bars): Collect bar ranges, allow trading
        2. After warmup: Calculate baseline as median of warmup ranges
        3. Compare rolling recent volatility vs baseline
        4. Pause if recent vol exceeds multiplier * baseline
        5. Resume if volatility drops back below threshold

        Args:
            bar_range: The high-low range of the current bar

        Returns:
            True if trading should be skipped for this bar, False otherwise
        """
        if not self.state.enable_adaptive_vol_stop:
            return False

        # Add current bar range to history
        self.state.adaptive_vol_bar_ranges.append(bar_range)

        # During warmup: just collect data, allow trading
        if len(self.state.adaptive_vol_bar_ranges) <= self.state.adaptive_vol_warmup_bars:
            return False

        # Calculate baseline (only once, after warmup)
        if self.state.adaptive_vol_baseline == 0.0:
            warmup_ranges = self.state.adaptive_vol_bar_ranges[:self.state.adaptive_vol_warmup_bars]
            sorted_ranges = sorted(warmup_ranges)
            self.state.adaptive_vol_baseline = sorted_ranges[len(sorted_ranges) // 2]  # median

        # Get recent rolling average
        recent_ranges = self.state.adaptive_vol_bar_ranges[-self.state.adaptive_vol_lookback:]
        avg_recent = sum(recent_ranges) / len(recent_ranges)

        # Check threshold (resumable - can toggle on/off each bar)
        exceeds_threshold = avg_recent > self.state.adaptive_vol_multiplier * self.state.adaptive_vol_baseline
        self.state.adaptive_vol_triggered = exceeds_threshold

        return exceeds_threshold

    def _check_delayed_start_filter(self, timestamp: datetime) -> bool:
        """
        Check if current time has passed the delayed start time.

        Returns:
            True if trading allowed (past start time), False if before start time
        """
        if not self.state.enable_delayed_start:
            return True

        # Extract time from timestamp (handle pandas Timestamp and datetime)
        if hasattr(timestamp, 'time'):
            bar_time = timestamp.time()
        else:
            bar_time = timestamp.to_pydatetime().time()

        start_time = time(self.state.delayed_start_hour, self.state.delayed_start_minute)

        if bar_time >= start_time:
            self.state.delayed_start_active = False
            return True

        # Still before start time
        return False

    def _check_breakout_filter(self, bar: pd.Series, timestamp: datetime) -> bool:
        """
        Track bars and check if there's been a breakout in the lookback window.

        A breakout is detected if:
        1. Price has moved > threshold % from the earliest bar in the lookback window
        2. The overall window range exceeds threshold × 1.5

        This filter is resumable - if volatility subsides, trading can resume.

        Args:
            bar: Current bar data with open, high, low, close
            timestamp: Current bar timestamp

        Returns:
            True if trading allowed (no breakout), False if breakout detected
        """
        if not self.state.enable_breakout_filter:
            return True

        # Always track bars (even before delayed start)
        self.state.bar_history.append({
            'timestamp': timestamp,
            'open': bar['open'],
            'high': bar['high'],
            'low': bar['low'],
            'close': bar['close']
        })

        # Trim to lookback window
        while len(self.state.bar_history) > self.state.breakout_lookback_bars:
            self.state.bar_history.pop(0)

        # Need at least some history to check
        if len(self.state.bar_history) < 10:
            return True  # Not enough data yet

        # Get reference price (open of oldest bar in window)
        reference_price = self.state.bar_history[0]['open']
        current_price = bar['close']

        # Check 1: Price displacement from window start
        move_pct = abs(current_price - reference_price) / reference_price * 100

        if move_pct > self.state.breakout_threshold_pct:
            self.state.breakout_detected = True
            return False  # Breakout detected, skip trading

        # Check 2: Overall window range (catches volatile whipsaws)
        window_high = max(b['high'] for b in self.state.bar_history)
        window_low = min(b['low'] for b in self.state.bar_history)
        window_range_pct = (window_high - window_low) / reference_price * 100

        if window_range_pct > self.state.breakout_threshold_pct * 1.5:
            self.state.breakout_detected = True
            return False  # High volatility detected, skip trading

        # No breakout - trading allowed
        self.state.breakout_detected = False
        return True

    def _check_oscillation_stop(self, current_price: float, timestamp) -> Tuple[bool, Optional[Trade]]:
        """
        Check oscillation stop (resumable).

        Triggers when consecutive same-direction trades reach threshold.
        When triggered:
        1. Flatten position immediately
        2. Reset consecutive counters (fresh start)
        3. Continue trading normally
        4. Track profitable trades (trades with positive realized P&L)
        5. Resume fully (reset profitable count) after N profitable trades
        6. If consecutive threshold is hit AGAIN before resume, permanently stop for day

        This allows the strategy to "test the waters" after a momentum spike.
        If the market starts oscillating again (profitable trades), we continue.
        If momentum continues (hits consecutive threshold again), we stop for the day.

        Args:
            current_price: Current price for potential flatten
            timestamp: Current bar timestamp

        Returns:
            Tuple of (should_skip_trading, optional_flatten_trade)
        """
        if not self.state.enable_oscillation_stop:
            return False, None

        # Permanent stop check - if we've hit the threshold twice, stop for day
        if self.state.oscillation_stop_permanent:
            return True, None

        # If in "testing" mode (triggered but not resumed), check conditions
        if self.state.oscillation_stop_triggered:
            # Check if we've accumulated enough profitable trades to fully resume
            if self.state.profitable_trade_count >= self.state.oscillation_resume_profitable:
                # Full resume - reset everything
                self.state.oscillation_stop_triggered = False
                self.state.profitable_trade_count = 0
                self.state.consecutive_buy_count = 0
                self.state.consecutive_sell_count = 0
                return False, None  # Allow full trading

            # Check if consecutive threshold hit AGAIN - permanent stop
            if (self.state.consecutive_buy_count >= self.state.oscillation_stop_consecutive or
                    self.state.consecutive_sell_count >= self.state.oscillation_stop_consecutive):
                self.state.oscillation_stop_permanent = True
                # Flatten position before permanent stop
                if self.state.position != 0:
                    position_type = "long" if self.state.position > 0 else "short"
                    reason = f"Oscillation stop (permanent) flatten at ${current_price:.2f} (closing {abs(self.state.position)} {position_type})"
                    flatten_trade = self.flatten_position(timestamp, current_price, reason=reason)
                    return True, flatten_trade
                return True, None

            # In testing mode, allow trading to continue
            return False, None

        # Check for first trigger
        if (self.state.consecutive_buy_count >= self.state.oscillation_stop_consecutive or
                self.state.consecutive_sell_count >= self.state.oscillation_stop_consecutive):
            self.state.oscillation_stop_triggered = True
            self.state.profitable_trade_count = 0
            # Reset consecutive counters for fresh start
            self.state.consecutive_buy_count = 0
            self.state.consecutive_sell_count = 0

            # Flatten position
            if self.state.position != 0:
                position_type = "long" if self.state.position > 0 else "short"
                reason = f"Oscillation stop flatten at ${current_price:.2f} (closing {abs(self.state.position)} {position_type})"
                flatten_trade = self.flatten_position(timestamp, current_price, reason=reason)
                return False, flatten_trade  # Continue trading with fresh start
            return False, None  # Continue trading

        return False, None

    def _check_rapid_buildup_stop(self, current_price: float, timestamp: datetime) -> Tuple[bool, Optional[Trade]]:
        """
        Check rapid buildup stop (time-window based, always flattens).

        Detects N consecutive same-side trades within a rolling time window.
        When triggered, immediately flattens position and stops trading for the day.

        Args:
            current_price: Current price for potential flatten
            timestamp: Current bar timestamp

        Returns:
            Tuple of (should_skip_trading, optional_flatten_trade)
        """
        if not self.state.enable_rapid_buildup_stop:
            return False, None

        # Already triggered - permanent stop for the day
        if self.state.rapid_buildup_triggered:
            return True, None

        # Need at least N trades to check
        if len(self.state.trades) < self.state.rapid_buildup_consecutive:
            return False, None

        # Count consecutive same-side trades within time window
        window_cutoff = timestamp - timedelta(minutes=self.state.rapid_buildup_window_minutes)

        # Get recent trades within window
        recent_trades = [t for t in self.state.trades if t.timestamp >= window_cutoff]

        if len(recent_trades) < self.state.rapid_buildup_consecutive:
            return False, None

        # Check for consecutive same-side trades (from most recent backwards)
        consecutive_count = 1
        last_side = recent_trades[-1].side

        for i in range(len(recent_trades) - 2, -1, -1):
            if recent_trades[i].side == last_side:
                consecutive_count += 1
            else:
                break

        # Check if threshold reached
        if consecutive_count >= self.state.rapid_buildup_consecutive:
            self.state.rapid_buildup_triggered = True

            # Always flatten position
            flatten_trade = None
            if self.state.position != 0:
                position_type = "long" if self.state.position > 0 else "short"
                side_str = "buys" if last_side == Side.BUY else "sells"
                reason = f"Rapid buildup stop: {consecutive_count} consecutive {side_str} in {self.state.rapid_buildup_window_minutes}min, flatten {abs(self.state.position)} {position_type} @ ${current_price:.2f}"
                flatten_trade = self.flatten_position(timestamp, current_price, reason=reason)

            return True, flatten_trade

        return False, None

    def _update_ema(self, close_price: float) -> bool:
        """
        Update EMA and check flatness.

        This is a dynamic filter that can pause and resume trading.
        Unlike stop losses which are permanent for the day, the EMA filter
        allows trading to resume when the EMA flattens back within threshold.

        The slope is measured as the percentage change from the EMA value
        N bars ago (where N = ema_period), not just the previous bar.
        This captures the actual trend direction over the lookback window.

        Args:
            close_price: The close price of the current bar

        Returns:
            True if EMA is flat (trading allowed), False if too steep (skip trading)
        """
        if not self.state.enable_ema_filter:
            return True  # Filter disabled, always allow trading

        # Warmup period: collect prices for initial SMA
        # NO TRADING during warmup - we need the EMA to initialize before allowing trades
        if not self.state.ema_initialized:
            self.state.ema_prices.append(close_price)
            if len(self.state.ema_prices) >= self.state.ema_period:
                # Initialize EMA with SMA
                self.state.ema_value = sum(self.state.ema_prices) / len(self.state.ema_prices)
                self.state.ema_initialized = True
                # Store initial EMA as the baseline for slope calculation
                self.state.ema_prices = [self.state.ema_value]  # Repurpose as EMA history
            return False  # NO trading during warmup - wait for EMA to initialize

        # Calculate new EMA
        multiplier = 2 / (self.state.ema_period + 1)
        self.state.ema_value = (close_price - self.state.ema_value) * multiplier + self.state.ema_value

        # Store EMA history for slope calculation
        self.state.ema_prices.append(self.state.ema_value)

        # Keep only enough history for slope calculation (ema_period bars)
        if len(self.state.ema_prices) > self.state.ema_period:
            self.state.ema_prices.pop(0)

        # Calculate slope as percentage change over the lookback period
        # Compare current EMA to EMA from ema_period bars ago
        if len(self.state.ema_prices) >= 2:
            oldest_ema = self.state.ema_prices[0]
            slope_pct = (self.state.ema_value - oldest_ema) / oldest_ema * 100
        else:
            slope_pct = 0.0

        # Update flatness state (checked each bar - can resume when flat again)
        self.state.ema_is_flat = abs(slope_pct) <= self.state.ema_flat_threshold

        # Reset the flattened flag when EMA becomes flat again (allow new positions)
        if self.state.ema_is_flat:
            self.state.ema_flattened_position = False

        return self.state.ema_is_flat

    def process_bar(self, bar: pd.Series) -> List[Trade]:
        """
        Process a single OHLCV bar and execute any trades.

        The strategy checks if the bar's price range would have hit our quotes.
        We assume fills happen at our quote prices (limit orders).

        Only ONE trade is allowed per bar. We use the open price relative to
        the zero point to determine which direction price likely moved first:
        - If open > zero_point: price started high, likely went down first (check buy first)
        - If open <= zero_point: price started low, likely went up first (check sell first)

        No new orders are opened in the last minute before market close (15:59+).
        No new orders if stop loss has been triggered.
        No new orders if trend stop has been triggered (price too far from open).
        No new orders if vol stop has been triggered (current volatility exceeds historical).
        No new orders if spike stop has been triggered (single bar exceeds threshold).

        Args:
            bar: OHLCV bar with 'open', 'high', 'low', 'close' and index as timestamp

        Returns:
            List of trades executed during this bar (max 1 trade, or 2 if flatten on stop)
        """
        trades = []
        timestamp = bar.name
        current_price = bar["close"]

        # Check rapid buildup stop FIRST (time-window based, always flattens)
        # This takes precedence over stop_loss since it always flattens
        should_skip, flatten_trade = self._check_rapid_buildup_stop(current_price, timestamp)
        if flatten_trade:
            trades.append(flatten_trade)
        if should_skip:
            return trades

        # Check consecutive trade stop loss before trading
        if self._check_stop_loss():
            # If flatten_on_stop is enabled and we have a position, flatten it
            if (self.state.flatten_on_stop and
                self.state.position != 0 and
                not self.state.position_flattened_on_stop):
                consecutive_count = max(self.state.consecutive_buys_at_stop, self.state.consecutive_sells_at_stop)
                direction = "buys" if self.state.consecutive_buys_at_stop > self.state.consecutive_sells_at_stop else "sells"
                position_type = "long" if self.state.position > 0 else "short"
                reason = f"Stop loss flatten at ${current_price:.2f} ({consecutive_count} consecutive {direction}, closing {abs(self.state.position)} {position_type})"
                flatten_trade = self.flatten_position(timestamp, current_price, reason=reason)
                if flatten_trade:
                    trades.append(flatten_trade)
                    self.state.position_flattened_on_stop = True
            return trades  # Stop loss triggered, no more trading today

        # Check trend stop loss (price distance from open)
        # Use the bar's high/low to detect trend breakout earlier
        if self._check_trend_stop(bar["high"]) or self._check_trend_stop(bar["low"]):
            return trades  # Trend stop triggered, no more trading today

        # Check volatility stop loss (current bar range vs historical)
        bar_range = bar["high"] - bar["low"]
        if self._check_vol_stop(bar_range):
            return trades  # Vol stop triggered, no more trading today

        # Check spike stop loss (single bar exceeds threshold) - checked BEFORE trading
        if self._check_spike_stop(bar_range):
            return trades  # Spike stop triggered, no more trading today

        # Check adaptive volatility stop (resumable - builds baseline from today's data)
        if self._check_adaptive_vol_stop(bar_range):
            return trades  # Adaptive vol stop triggered, skip this bar

        # Update EMA and check flatness filter (dynamic - can resume)
        if not self._update_ema(bar["close"]):
            # EMA not flat - flatten position if enabled and have a position
            if (self.state.flatten_on_ema_deviation and
                self.state.position != 0 and
                not self.state.ema_flattened_position):
                position_type = "long" if self.state.position > 0 else "short"
                reason = f"EMA deviation flatten at ${current_price:.2f} (closing {abs(self.state.position)} {position_type})"
                flatten_trade = self.flatten_position(timestamp, current_price, reason=reason)
                if flatten_trade:
                    trades.append(flatten_trade)
                    self.state.ema_flattened_position = True
            return trades  # EMA not flat, skip this bar (will check again next bar)

        # Check oscillation stop (resumable - can pause and resume)
        should_skip, flatten_trade = self._check_oscillation_stop(current_price, timestamp)
        if flatten_trade:
            trades.append(flatten_trade)
        if should_skip:
            return trades

        # Check delayed start filter (only trade after specified time)
        if not self._check_delayed_start_filter(timestamp):
            # Still track bars for breakout detection even if not trading
            if self.state.enable_breakout_filter:
                self._check_breakout_filter(bar, timestamp)
            return trades  # Before start time, skip trading

        # Check breakout filter (skip if trend detected in lookback window)
        if not self._check_breakout_filter(bar, timestamp):
            return trades  # Breakout detected, skip this bar (resumable)

        # Don't open new orders in the last minute before market close (15:59 or later)
        bar_time = timestamp.time() if hasattr(timestamp, 'time') else timestamp.to_pydatetime().time()
        if bar_time.hour == 15 and bar_time.minute >= 59:
            return trades
        if bar_time.hour >= 16:
            return trades

        open_price = bar["open"]
        low = bar["low"]
        high = bar["high"]

        bid, ask = self.state.get_quotes()

        # Helper to build trade reason
        def build_buy_reason(price: float) -> str:
            retreat_info = f", retreat=${self.state.bid_adjustment:.2f}" if self.state.bid_adjustment != 0 else ""
            base = f"Bid hit at ${price:.2f} (zero=${self.state.zero_point:.2f}, width=${self.state.quote_width:.2f}{retreat_info})"
            # Check if this will close a short position
            if self.state.position < 0:
                shares_closing = min(bid.size, abs(self.state.position))
                return f"{base} - closing {shares_closing} short"
            return base

        def build_sell_reason(price: float) -> str:
            retreat_info = f", retreat=${self.state.ask_adjustment:.2f}" if self.state.ask_adjustment != 0 else ""
            base = f"Ask hit at ${price:.2f} (zero=${self.state.zero_point:.2f}, width=${self.state.quote_width:.2f}{retreat_info})"
            # Check if this will close a long position
            if self.state.position > 0:
                shares_closing = min(ask.size, self.state.position)
                return f"{base} - closing {shares_closing} long"
            return base

        # Determine which side to check first based on open price relative to zero point
        # If open is above zero point, price likely dipped down first (check buy first)
        # If open is at/below zero point, price likely went up first (check sell first)
        check_buy_first = open_price > self.state.zero_point

        if check_buy_first:
            # Check for buy fill first (price went down to our bid)
            if low <= bid.price:
                reason = build_buy_reason(bid.price)
                trade = self._execute_buy(timestamp, bid.price, bid.size, reason=reason)
                trades.append(trade)
                self._retreat_after_buy()
            # Only check sell if no buy occurred
            elif high >= ask.price:
                reason = build_sell_reason(ask.price)
                trade = self._execute_sell(timestamp, ask.price, ask.size, reason=reason)
                trades.append(trade)
                self._retreat_after_sell()
        else:
            # Check for sell fill first (price went up to our ask)
            if high >= ask.price:
                reason = build_sell_reason(ask.price)
                trade = self._execute_sell(timestamp, ask.price, ask.size, reason=reason)
                trades.append(trade)
                self._retreat_after_sell()
            # Only check buy if no sell occurred
            elif low <= bid.price:
                reason = build_buy_reason(bid.price)
                trade = self._execute_buy(timestamp, bid.price, bid.size, reason=reason)
                trades.append(trade)
                self._retreat_after_buy()

        # Track capital for return calculation
        self.state.total_position_capital_bars += abs(self.state.position) * current_price
        self.state.total_bars_processed += 1

        return trades

    def flatten_position(self, timestamp: datetime, price: float, reason: str = "") -> Optional[Trade]:
        """
        Flatten the position at a given price (for end of day).

        Args:
            timestamp: Time of the flatten
            price: Price to flatten at (typically closing price)
            reason: Explanation for why the position is being flattened

        Returns:
            Trade if position was flattened, None if already flat
        """
        if self.state.position == 0:
            return None

        # Build default reason if not provided
        if not reason:
            position_type = "long" if self.state.position > 0 else "short"
            reason = f"Flatten at ${price:.2f} (closing {abs(self.state.position)} {position_type})"

        if self.state.position > 0:
            # Close long position
            return self._execute_sell(timestamp, price, abs(self.state.position), reason=reason)
        else:
            # Close short position
            return self._execute_buy(timestamp, price, abs(self.state.position), reason=reason)

    def set_initial_position(self, position: int, lots: List[Lot], realized_pnl: float = 0.0):
        """
        Initialize strategy with a carried-over position from previous day.

        Used for overnight position holding to carry position and P&L forward.

        Args:
            position: The position to carry forward (positive = long, negative = short)
            lots: The LIFO lots associated with the position
            realized_pnl: Cumulative realized P&L to carry forward (includes overnight gaps)
        """
        self.state.position = position
        if position > 0:
            self.state.long_lots = lots.copy()
            self.state.short_lots = []
        elif position < 0:
            self.state.short_lots = lots.copy()
            self.state.long_lots = []
        self.state.realized_pnl = realized_pnl

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L based on current price using LIFO lots."""
        if self.state.position == 0:
            return 0.0

        unrealized = 0.0
        if self.state.position > 0:
            # Sum up unrealized P&L from all long lots
            for lot in self.state.long_lots:
                unrealized += (current_price - lot.price) * lot.quantity
        else:
            # Sum up unrealized P&L from all short lots
            for lot in self.state.short_lots:
                unrealized += (lot.price - current_price) * abs(lot.quantity)

        return unrealized

    def get_total_pnl(self, current_price: float) -> float:
        """Calculate total P&L (realized + unrealized)."""
        return self.state.realized_pnl + self.get_unrealized_pnl(current_price)

    def get_trade_count(self) -> int:
        """Get the number of trades executed."""
        return len(self.state.trades)

    def get_avg_capital_required(self) -> float:
        """Get average capital required (time-weighted average of |position| * price)."""
        if self.state.total_bars_processed == 0:
            return 0.0
        return self.state.total_position_capital_bars / self.state.total_bars_processed

    def get_trade_log(self) -> List[dict]:
        """Get a detailed trade log as a list of dictionaries."""
        log = []
        for trade in self.state.trades:
            entry = {
                "timestamp": trade.timestamp,
                "side": trade.side.value,
                "price": trade.price,
                "quantity": trade.quantity,
                "capital_required": trade.capital_required,
                "position_after": trade.position_after,
                "position_value": abs(trade.position_after) * trade.price,
                "realized_pnl": trade.realized_pnl,
                "reason": trade.reason,
            }
            if trade.adjusted_zero_point is not None:
                entry["adjusted_zero_point"] = trade.adjusted_zero_point
            log.append(entry)
        return log

    def get_summary(self, final_price: float) -> dict:
        """Get a summary of the strategy performance."""
        summary = {
            "total_trades": self.get_trade_count(),
            "final_position": self.state.position,
            "realized_pnl": self.state.realized_pnl,
            "unrealized_pnl": self.get_unrealized_pnl(final_price),
            "total_pnl": self.get_total_pnl(final_price),
            "zero_point": self.state.zero_point,
            "quote_width": self.state.quote_width,
            "retreat": self.state.retreat,
            "avg_capital_required": self.get_avg_capital_required(),
            "bars_processed": self.state.total_bars_processed,
        }
        if self.state.enable_zero_point_adjustment:
            summary["original_zero_point"] = self.state.original_zero_point
            summary["lambda_enabled"] = True
            summary["lambda_decay"] = self.state.zero_point_decay_factor
            summary["lambda_blend"] = self.state.zero_point_blend_factor
        if self.state.enable_stop_loss:
            summary["stop_loss_enabled"] = True
            summary["consecutive_trade_limit"] = self.state.consecutive_trade_limit
            summary["stop_loss_triggered"] = self.state.stop_loss_triggered
            summary["flatten_on_stop"] = self.state.flatten_on_stop
            if self.state.stop_loss_triggered:
                # Use captured counts from when stop loss triggered (before flatten reset them)
                summary["consecutive_buys_at_stop"] = self.state.consecutive_buys_at_stop
                summary["consecutive_sells_at_stop"] = self.state.consecutive_sells_at_stop
                summary["position_flattened_on_stop"] = self.state.position_flattened_on_stop
        if self.state.enable_trend_stop:
            summary["trend_stop_enabled"] = True
            summary["trend_stop_multiplier"] = self.state.trend_stop_multiplier
            summary["trend_threshold_pct"] = self.state.trend_threshold_pct
            summary["trend_stop_triggered"] = self.state.trend_stop_triggered
        if self.state.enable_vol_stop:
            summary["vol_stop_enabled"] = True
            summary["vol_stop_multiplier"] = self.state.vol_stop_multiplier
            summary["vol_stop_lookback"] = self.state.vol_stop_lookback
            summary["vol_stop_triggered"] = self.state.vol_stop_triggered
        if self.state.enable_spike_stop:
            summary["spike_stop_enabled"] = True
            summary["spike_stop_multiplier"] = self.state.spike_stop_multiplier
            summary["spike_stop_triggered"] = self.state.spike_stop_triggered
        if self.state.enable_ema_filter:
            summary["ema_filter_enabled"] = True
            summary["ema_period"] = self.state.ema_period
            summary["ema_flat_threshold"] = self.state.ema_flat_threshold
            summary["ema_is_flat"] = self.state.ema_is_flat
            summary["final_ema_value"] = self.state.ema_value
            summary["flatten_on_ema_deviation"] = self.state.flatten_on_ema_deviation
        if self.state.enable_oscillation_stop:
            summary["oscillation_stop_enabled"] = True
            summary["oscillation_stop_consecutive"] = self.state.oscillation_stop_consecutive
            summary["oscillation_resume_profitable"] = self.state.oscillation_resume_profitable
            summary["oscillation_stop_triggered"] = self.state.oscillation_stop_triggered
            summary["oscillation_stop_permanent"] = self.state.oscillation_stop_permanent
            summary["profitable_trade_count"] = self.state.profitable_trade_count
        if self.state.enable_adaptive_vol_stop:
            summary["adaptive_vol_stop_enabled"] = True
            summary["adaptive_vol_warmup_bars"] = self.state.adaptive_vol_warmup_bars
            summary["adaptive_vol_multiplier"] = self.state.adaptive_vol_multiplier
            summary["adaptive_vol_lookback"] = self.state.adaptive_vol_lookback
            summary["adaptive_vol_baseline"] = self.state.adaptive_vol_baseline
            summary["adaptive_vol_triggered"] = self.state.adaptive_vol_triggered
        if self.state.enable_delayed_start:
            summary["delayed_start_enabled"] = True
            summary["delayed_start_time"] = f"{self.state.delayed_start_hour:02d}:{self.state.delayed_start_minute:02d}"
            summary["delayed_start_active"] = self.state.delayed_start_active
        if self.state.enable_breakout_filter:
            summary["breakout_filter_enabled"] = True
            summary["breakout_lookback_bars"] = self.state.breakout_lookback_bars
            summary["breakout_threshold_pct"] = self.state.breakout_threshold_pct
            summary["breakout_detected"] = self.state.breakout_detected
        if self.state.enable_rapid_buildup_stop:
            summary["rapid_buildup_stop_enabled"] = True
            summary["rapid_buildup_window_minutes"] = self.state.rapid_buildup_window_minutes
            summary["rapid_buildup_consecutive"] = self.state.rapid_buildup_consecutive
            summary["rapid_buildup_triggered"] = self.state.rapid_buildup_triggered
        return summary
