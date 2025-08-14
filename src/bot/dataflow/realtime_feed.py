"""
Real-time Data Feed with WebSocket Support
Phase 2.5 - Day 3

Production-ready data pipeline with WebSocket connections,
automatic reconnection, and data validation.
"""

import asyncio
import json
import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from queue import Empty, Queue
from typing import Any

import numpy as np
import pytz
import websocket

from ..database.database_manager import get_db_manager
from ..database.models import SystemMetric

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Available data sources"""

    ALPACA = "alpaca"
    POLYGON = "polygon"
    YAHOO = "yahoo"
    IEX = "iex"
    BINANCE = "binance"


class MarketStatus(Enum):
    """Market status states"""

    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"
    AFTER_HOURS = "after_hours"
    MARKET_CLOSED = "market_closed"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"


@dataclass
class MarketData:
    """Market data container"""

    symbol: str
    timestamp: datetime
    price: Decimal
    bid: Decimal | None = None
    ask: Decimal | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    volume: int | None = None
    source: DataSource | None = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "price": float(self.price),
            "bid": float(self.bid) if self.bid else None,
            "ask": float(self.ask) if self.ask else None,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "volume": self.volume,
            "source": self.source.value if self.source else None,
        }


@dataclass
class DataFeedConfig:
    """Data feed configuration"""

    primary_source: DataSource = DataSource.ALPACA
    fallback_sources: list[DataSource] = field(
        default_factory=lambda: [DataSource.POLYGON, DataSource.YAHOO]
    )
    reconnect_interval: int = 5  # seconds
    max_reconnect_attempts: int = 10
    heartbeat_interval: int = 30  # seconds
    buffer_size: int = 10000
    validate_data: bool = True

    # API endpoints
    alpaca_ws_url: str = "wss://stream.data.alpaca.markets/v2/sip"
    polygon_ws_url: str = "wss://socket.polygon.io/stocks"

    # API keys (loaded from environment)
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    polygon_api_key: str = ""


class MarketCalendar:
    """
    Market hours and holiday handling

    Handles US equity market hours with pre-market and after-hours trading.
    """

    def __init__(self):
        self.tz = pytz.timezone("US/Eastern")
        self.holidays_2025 = [
            datetime(2025, 1, 1),  # New Year's Day
            datetime(2025, 1, 20),  # MLK Day
            datetime(2025, 2, 17),  # Presidents Day
            datetime(2025, 4, 18),  # Good Friday
            datetime(2025, 5, 26),  # Memorial Day
            datetime(2025, 6, 19),  # Juneteenth
            datetime(2025, 7, 4),  # Independence Day
            datetime(2025, 9, 1),  # Labor Day
            datetime(2025, 11, 27),  # Thanksgiving
            datetime(2025, 12, 25),  # Christmas
        ]

        # Market hours in Eastern Time
        self.pre_market_open = time(4, 0)  # 4:00 AM ET
        self.market_open = time(9, 30)  # 9:30 AM ET
        self.market_close = time(16, 0)  # 4:00 PM ET
        self.after_hours_close = time(20, 0)  # 8:00 PM ET

    def get_market_status(self, timestamp: datetime | None = None) -> MarketStatus:
        """Get current market status"""
        if timestamp is None:
            timestamp = datetime.now(self.tz)
        elif timestamp.tzinfo is None:
            timestamp = self.tz.localize(timestamp)
        else:
            timestamp = timestamp.astimezone(self.tz)

        # Check if weekend
        if timestamp.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return MarketStatus.WEEKEND

        # Check if holiday
        date_only = timestamp.date()
        for holiday in self.holidays_2025:
            if holiday.date() == date_only:
                return MarketStatus.HOLIDAY

        # Check time of day
        current_time = timestamp.time()

        if current_time < self.pre_market_open:
            return MarketStatus.MARKET_CLOSED
        elif current_time < self.market_open:
            return MarketStatus.PRE_MARKET
        elif current_time < self.market_close:
            return MarketStatus.MARKET_OPEN
        elif current_time < self.after_hours_close:
            return MarketStatus.AFTER_HOURS
        else:
            return MarketStatus.MARKET_CLOSED

    def is_market_open(self, timestamp: datetime | None = None) -> bool:
        """Check if regular market is open"""
        return self.get_market_status(timestamp) == MarketStatus.MARKET_OPEN

    def is_extended_hours(self, timestamp: datetime | None = None) -> bool:
        """Check if in extended hours (pre-market or after-hours)"""
        status = self.get_market_status(timestamp)
        return status in [MarketStatus.PRE_MARKET, MarketStatus.AFTER_HOURS]

    def next_market_open(self, timestamp: datetime | None = None) -> datetime:
        """Get next market open time"""
        if timestamp is None:
            timestamp = datetime.now(self.tz)
        elif timestamp.tzinfo is None:
            timestamp = self.tz.localize(timestamp)
        else:
            timestamp = timestamp.astimezone(self.tz)

        # Start from next potential trading day
        next_day = timestamp + timedelta(days=1)
        next_day = next_day.replace(hour=9, minute=30, second=0, microsecond=0)

        # Skip weekends and holidays
        while next_day.weekday() >= 5 or next_day.date() in [h.date() for h in self.holidays_2025]:
            next_day += timedelta(days=1)

        return next_day

    def get_trading_hours(self, date: datetime) -> dict[str, datetime]:
        """Get trading hours for a specific date"""
        if date.tzinfo is None:
            date = self.tz.localize(date)
        else:
            date = date.astimezone(self.tz)

        return {
            "pre_market_open": date.replace(hour=4, minute=0, second=0),
            "market_open": date.replace(hour=9, minute=30, second=0),
            "market_close": date.replace(hour=16, minute=0, second=0),
            "after_hours_close": date.replace(hour=20, minute=0, second=0),
        }


class DataValidator:
    """
    Data validation and anomaly detection

    Validates incoming market data for quality and detects anomalies.
    """

    def __init__(self):
        self.price_history: dict[str, deque] = {}
        self.volume_history: dict[str, deque] = {}
        self.max_history = 1000
        self.anomaly_threshold = 5  # Standard deviations

    def validate_market_data(self, data: MarketData) -> tuple[bool, str | None]:
        """
        Validate market data

        Returns:
            (is_valid, error_message)
        """
        # Check required fields
        if not data.symbol or not data.timestamp or data.price is None:
            return False, "Missing required fields"

        # Check price is positive
        if data.price <= 0:
            return False, f"Invalid price: {data.price}"

        # Check bid/ask spread if available
        if data.bid and data.ask:
            if data.bid > data.ask:
                return False, f"Inverted bid/ask: bid={data.bid}, ask={data.ask}"

            spread_pct = float((data.ask - data.bid) / data.price) * 100
            if spread_pct > 5:  # More than 5% spread is suspicious
                return False, f"Excessive spread: {spread_pct:.2f}%"

        # Check for price anomalies
        if self.is_price_anomaly(data.symbol, float(data.price)):
            return False, f"Price anomaly detected for {data.symbol}"

        # Check volume if provided
        if data.volume is not None and data.volume < 0:
            return False, f"Invalid volume: {data.volume}"

        # Check timestamp is not in future
        if data.timestamp > datetime.now(pytz.UTC) + timedelta(seconds=60):
            return False, f"Future timestamp: {data.timestamp}"

        # Update history
        self._update_history(data)

        return True, None

    def is_price_anomaly(self, symbol: str, price: float) -> bool:
        """Check if price is anomalous based on recent history"""
        if symbol not in self.price_history:
            return False

        history = list(self.price_history[symbol])
        if len(history) < 20:  # Need minimum history
            return False

        # Calculate statistics
        mean = np.mean(history)
        std = np.std(history)

        if std == 0:
            return False

        # Check if price is outside threshold
        z_score = abs((price - mean) / std)
        return z_score > self.anomaly_threshold

    def is_volume_spike(self, symbol: str, volume: int) -> bool:
        """Check if volume is unusually high"""
        if symbol not in self.volume_history:
            return False

        history = list(self.volume_history[symbol])
        if len(history) < 20:
            return False

        # Calculate average volume
        avg_volume = np.mean(history)

        # Check if current volume is significantly higher
        return volume > avg_volume * 5

    def _update_history(self, data: MarketData):
        """Update price and volume history"""
        # Update price history
        if data.symbol not in self.price_history:
            self.price_history[data.symbol] = deque(maxlen=self.max_history)
        self.price_history[data.symbol].append(float(data.price))

        # Update volume history
        if data.volume is not None:
            if data.symbol not in self.volume_history:
                self.volume_history[data.symbol] = deque(maxlen=self.max_history)
            self.volume_history[data.symbol].append(data.volume)

    def get_data_quality_score(self, symbol: str) -> float:
        """
        Get data quality score for a symbol (0-1)

        Based on data completeness and consistency.
        """
        if symbol not in self.price_history:
            return 0.0

        history = list(self.price_history[symbol])
        if len(history) < 10:
            return 0.5

        # Calculate metrics
        price_changes = np.diff(history)
        volatility = np.std(price_changes)

        # Check for stuck prices (no changes)
        unique_prices = len(set(history[-20:])) if len(history) >= 20 else len(set(history))
        diversity_score = min(unique_prices / 10, 1.0)

        # Check for gaps
        max_change = np.max(np.abs(price_changes)) if len(price_changes) > 0 else 0
        avg_price = np.mean(history)
        gap_score = 1.0 - min(max_change / avg_price, 1.0) if avg_price > 0 else 0

        # Combine scores
        quality_score = (diversity_score + gap_score) / 2

        return quality_score


class RealtimeDataFeed:
    """
    Production real-time data feed with WebSocket support

    Features:
    - Multiple data source support with automatic failover
    - WebSocket connections with reconnection logic
    - Data validation and anomaly detection
    - Market hours awareness
    - Performance monitoring
    """

    def __init__(self, config: DataFeedConfig | None = None):
        self.config = config or DataFeedConfig()
        self.calendar = MarketCalendar()
        self.validator = DataValidator()

        # WebSocket connections
        self.ws_connections: dict[DataSource, Any] = {}
        self.ws_threads: dict[DataSource, threading.Thread] = {}

        # Data buffers
        self.data_buffer = Queue(maxsize=self.config.buffer_size)
        self.error_buffer = deque(maxlen=1000)

        # Callbacks
        self.data_callbacks: list[Callable[[MarketData], None]] = []
        self.error_callbacks: list[Callable[[str], None]] = []

        # State
        self.is_running = False
        self.subscribed_symbols: set = set()
        self.reconnect_counts: dict[DataSource, int] = {}

        # Metrics
        self.metrics = {
            "messages_received": 0,
            "messages_validated": 0,
            "messages_rejected": 0,
            "reconnections": 0,
            "current_source": None,
        }

        # Database manager for metrics
        self.db_manager = get_db_manager()

        logger.info("RealtimeDataFeed initialized")

    def start(self, symbols: list[str]):
        """Start the data feed with specified symbols"""
        if self.is_running:
            logger.warning("Data feed already running")
            return

        self.subscribed_symbols = set(symbols)
        self.is_running = True

        # Check market status
        market_status = self.calendar.get_market_status()
        logger.info(f"Market status: {market_status.value}")

        if market_status == MarketStatus.WEEKEND:
            logger.warning("Market is closed (weekend). Will connect when market opens.")
            # Schedule connection for next market open
            next_open = self.calendar.next_market_open()
            logger.info(f"Next market open: {next_open}")

        # Start primary data source
        self._connect_data_source(self.config.primary_source)

        # Start data processor thread
        processor_thread = threading.Thread(target=self._process_data_buffer, daemon=True)
        processor_thread.start()

        # Start metrics reporter thread
        metrics_thread = threading.Thread(target=self._report_metrics, daemon=True)
        metrics_thread.start()

        logger.info(f"Data feed started with {len(symbols)} symbols")

    def stop(self):
        """Stop the data feed"""
        if not self.is_running:
            return

        logger.info("Stopping data feed...")
        self.is_running = False

        # Close all WebSocket connections
        for source, ws in self.ws_connections.items():
            try:
                if ws:
                    ws.close()
            except Exception as e:
                logger.error(f"Error closing {source.value} connection: {e}")

        self.ws_connections.clear()
        logger.info("Data feed stopped")

    def _connect_data_source(self, source: DataSource):
        """Connect to a data source via WebSocket"""
        logger.info(f"Connecting to {source.value}...")

        if source == DataSource.ALPACA:
            self._connect_alpaca()
        elif source == DataSource.POLYGON:
            self._connect_polygon()
        else:
            logger.warning(f"WebSocket not implemented for {source.value}")
            self._try_fallback_source(source)

    def _connect_alpaca(self):
        """Connect to Alpaca WebSocket"""
        try:
            ws_url = self.config.alpaca_ws_url

            def on_open(ws):
                logger.info("Alpaca WebSocket connected")
                # Authenticate
                auth_msg = {
                    "action": "auth",
                    "key": self.config.alpaca_api_key,
                    "secret": self.config.alpaca_secret_key,
                }
                ws.send(json.dumps(auth_msg))

                # Subscribe to symbols
                if self.subscribed_symbols:
                    sub_msg = {
                        "action": "subscribe",
                        "trades": list(self.subscribed_symbols),
                        "quotes": list(self.subscribed_symbols),
                    }
                    ws.send(json.dumps(sub_msg))

            def on_message(ws, message):
                self._handle_alpaca_message(json.loads(message))

            def on_error(ws, error):
                logger.error(f"Alpaca WebSocket error: {error}")
                self.error_buffer.append(f"Alpaca: {error}")

            def on_close(ws, close_status_code, close_msg):
                logger.warning(f"Alpaca WebSocket closed: {close_msg}")
                if self.is_running:
                    self._handle_disconnection(DataSource.ALPACA)

            # Create WebSocket connection
            ws = websocket.WebSocketApp(
                ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close
            )

            self.ws_connections[DataSource.ALPACA] = ws

            # Run in separate thread
            ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
            ws_thread.start()
            self.ws_threads[DataSource.ALPACA] = ws_thread

            self.metrics["current_source"] = DataSource.ALPACA

        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            self._try_fallback_source(DataSource.ALPACA)

    def _connect_polygon(self):
        """Connect to Polygon WebSocket"""
        try:
            ws_url = f"{self.config.polygon_ws_url}?apikey={self.config.polygon_api_key}"

            def on_open(ws):
                logger.info("Polygon WebSocket connected")
                # Subscribe to symbols
                for symbol in self.subscribed_symbols:
                    sub_msg = {"action": "subscribe", "params": f"T.{symbol},Q.{symbol}"}
                    ws.send(json.dumps(sub_msg))

            def on_message(ws, message):
                self._handle_polygon_message(json.loads(message))

            def on_error(ws, error):
                logger.error(f"Polygon WebSocket error: {error}")
                self.error_buffer.append(f"Polygon: {error}")

            def on_close(ws, close_status_code, close_msg):
                logger.warning(f"Polygon WebSocket closed: {close_msg}")
                if self.is_running:
                    self._handle_disconnection(DataSource.POLYGON)

            # Create WebSocket connection
            ws = websocket.WebSocketApp(
                ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close
            )

            self.ws_connections[DataSource.POLYGON] = ws

            # Run in separate thread
            ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
            ws_thread.start()
            self.ws_threads[DataSource.POLYGON] = ws_thread

            self.metrics["current_source"] = DataSource.POLYGON

        except Exception as e:
            logger.error(f"Failed to connect to Polygon: {e}")
            self._try_fallback_source(DataSource.POLYGON)

    def _handle_alpaca_message(self, message: dict):
        """Process Alpaca WebSocket message"""
        try:
            self.metrics["messages_received"] += 1

            # Parse message type
            msg_type = message.get("T")

            if msg_type == "t":  # Trade
                data = MarketData(
                    symbol=message["S"],
                    timestamp=datetime.fromisoformat(message["t"]),
                    price=Decimal(str(message["p"])),
                    volume=message.get("s"),
                    source=DataSource.ALPACA,
                )
                self._process_market_data(data)

            elif msg_type == "q":  # Quote
                data = MarketData(
                    symbol=message["S"],
                    timestamp=datetime.fromisoformat(message["t"]),
                    price=Decimal(str((message["bp"] + message["ap"]) / 2)),
                    bid=Decimal(str(message["bp"])),
                    ask=Decimal(str(message["ap"])),
                    bid_size=message.get("bs"),
                    ask_size=message.get("as"),
                    source=DataSource.ALPACA,
                )
                self._process_market_data(data)

        except Exception as e:
            logger.error(f"Error processing Alpaca message: {e}")

    def _handle_polygon_message(self, message: dict):
        """Process Polygon WebSocket message"""
        try:
            self.metrics["messages_received"] += 1

            # Parse message type
            msg_type = message.get("ev")

            if msg_type == "T":  # Trade
                data = MarketData(
                    symbol=message["sym"],
                    timestamp=datetime.fromtimestamp(message["t"] / 1000),
                    price=Decimal(str(message["p"])),
                    volume=message.get("s"),
                    source=DataSource.POLYGON,
                )
                self._process_market_data(data)

            elif msg_type == "Q":  # Quote
                data = MarketData(
                    symbol=message["sym"],
                    timestamp=datetime.fromtimestamp(message["t"] / 1000),
                    price=Decimal(str((message["bp"] + message["ap"]) / 2)),
                    bid=Decimal(str(message["bp"])),
                    ask=Decimal(str(message["ap"])),
                    bid_size=message.get("bs"),
                    ask_size=message.get("as"),
                    source=DataSource.POLYGON,
                )
                self._process_market_data(data)

        except Exception as e:
            logger.error(f"Error processing Polygon message: {e}")

    def _process_market_data(self, data: MarketData):
        """Process and validate market data"""
        # Validate data
        if self.config.validate_data:
            is_valid, error_msg = self.validator.validate_market_data(data)

            if not is_valid:
                self.metrics["messages_rejected"] += 1
                logger.warning(f"Invalid data rejected: {error_msg}")
                return

            self.metrics["messages_validated"] += 1

        # Add to buffer
        try:
            self.data_buffer.put_nowait(data)
        except:
            logger.warning("Data buffer full, dropping oldest data")
            try:
                self.data_buffer.get_nowait()
                self.data_buffer.put_nowait(data)
            except:
                pass

    def _process_data_buffer(self):
        """Process data from buffer and call callbacks"""
        while self.is_running:
            try:
                data = self.data_buffer.get(timeout=1)

                # Call registered callbacks
                for callback in self.data_callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"Error in data callback: {e}")

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing data buffer: {e}")

    def _handle_disconnection(self, source: DataSource):
        """Handle WebSocket disconnection with reconnection logic"""
        if not self.is_running:
            return

        # Update reconnection count
        if source not in self.reconnect_counts:
            self.reconnect_counts[source] = 0
        self.reconnect_counts[source] += 1

        self.metrics["reconnections"] += 1

        # Check if max reconnections reached
        if self.reconnect_counts[source] >= self.config.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts reached for {source.value}")
            self._try_fallback_source(source)
            return

        # Wait before reconnecting
        wait_time = self.config.reconnect_interval * self.reconnect_counts[source]
        logger.info(f"Reconnecting to {source.value} in {wait_time} seconds...")

        threading.Timer(wait_time, lambda: self._connect_data_source(source)).start()

    def _try_fallback_source(self, failed_source: DataSource):
        """Try connecting to fallback data source"""
        for source in self.config.fallback_sources:
            if source != failed_source and source not in self.ws_connections:
                logger.info(f"Trying fallback source: {source.value}")
                self._connect_data_source(source)
                return

        logger.error("All data sources failed!")

    def _report_metrics(self):
        """Report metrics to database periodically"""
        while self.is_running:
            try:
                # Wait 60 seconds
                for _ in range(60):
                    if not self.is_running:
                        return
                    asyncio.sleep(1)

                # Store metrics in database
                self.db_manager.create(
                    SystemMetric,
                    timestamp=datetime.utcnow(),
                    metric_name="datafeed_messages_received",
                    metric_value=Decimal(str(self.metrics["messages_received"])),
                    component="realtime_feed",
                )

                self.db_manager.create(
                    SystemMetric,
                    timestamp=datetime.utcnow(),
                    metric_name="datafeed_messages_validated",
                    metric_value=Decimal(str(self.metrics["messages_validated"])),
                    component="realtime_feed",
                )

                logger.info(f"Data feed metrics: {self.metrics}")

            except Exception as e:
                logger.error(f"Error reporting metrics: {e}")

    def register_data_callback(self, callback: Callable[[MarketData], None]):
        """Register callback for market data"""
        self.data_callbacks.append(callback)

    def register_error_callback(self, callback: Callable[[str], None]):
        """Register callback for errors"""
        self.error_callbacks.append(callback)

    def get_data_quality(self, symbol: str) -> float:
        """Get data quality score for a symbol"""
        return self.validator.get_data_quality_score(symbol)

    def get_market_status(self) -> MarketStatus:
        """Get current market status"""
        return self.calendar.get_market_status()

    def get_metrics(self) -> dict:
        """Get current metrics"""
        return self.metrics.copy()
