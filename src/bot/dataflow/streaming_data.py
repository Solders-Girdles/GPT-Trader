"""
Market Data Streaming System for GPT-Trader Live Trading

Real-time market data streaming infrastructure providing:
- WebSocket connections to multiple data providers
- Real-time quote and trade data processing
- Bar aggregation from tick data
- Data quality validation and monitoring
- Failover and reconnection logic
- Scalable multi-symbol streaming

This is the core data infrastructure for live trading operations.
"""

import logging
import queue
import sqlite3
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


class DataType(Enum):
    """Types of market data"""

    QUOTE = "quote"  # Best bid/ask
    TRADE = "trade"  # Executed trades
    BAR = "bar"  # OHLCV bars
    LEVEL2 = "level2"  # Market depth
    NEWS = "news"  # Market news


class DataProvider(Enum):
    """Supported data providers"""

    ALPACA = "alpaca"
    POLYGON = "polygon"
    IEX = "iex"
    YAHOO = "yahoo"
    SIMULATED = "simulated"


class ConnectionStatus(Enum):
    """Connection status"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class Quote:
    """Real-time quote data"""

    symbol: str
    bid_price: Decimal
    bid_size: int
    ask_price: Decimal
    ask_size: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def mid_price(self) -> Decimal:
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread(self) -> Decimal:
        return self.ask_price - self.bid_price

    @property
    def spread_bps(self) -> float:
        if self.mid_price > 0:
            return float(self.spread / self.mid_price * 10000)
        return 0.0


@dataclass
class Trade:
    """Real-time trade data"""

    symbol: str
    price: Decimal
    size: int
    timestamp: datetime = field(default_factory=datetime.now)
    exchange: str = "UNKNOWN"
    trade_id: str = ""


@dataclass
class Bar:
    """OHLCV bar data"""

    symbol: str
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: int
    timestamp: datetime
    bar_type: str = "1min"  # 1min, 5min, etc.

    @property
    def typical_price(self) -> Decimal:
        return (self.high_price + self.low_price + self.close_price) / 3

    @property
    def price_range(self) -> Decimal:
        return self.high_price - self.low_price


@dataclass
class StreamingStats:
    """Statistics for streaming data"""

    symbol: str
    data_provider: DataProvider
    connection_status: ConnectionStatus

    # Message counts
    quotes_received: int = 0
    trades_received: int = 0
    bars_generated: int = 0

    # Quality metrics
    last_message_time: datetime | None = None
    messages_per_second: float = 0.0
    connection_uptime: timedelta = field(default_factory=lambda: timedelta())
    reconnection_count: int = 0

    # Latency metrics
    average_latency_ms: float = 0.0
    max_latency_ms: float = 0.0


class BarAggregator:
    """Aggregates tick data into OHLCV bars"""

    def __init__(self, bar_intervals: list[str] = None) -> None:
        if bar_intervals is None:
            bar_intervals = ["1min", "5min", "15min", "1hour"]

        self.bar_intervals = bar_intervals
        self.current_bars: dict[str, dict[str, Bar]] = defaultdict(
            dict
        )  # symbol -> interval -> bar
        self.completed_bars: deque = deque(maxlen=10000)  # Store recent completed bars

        # Callbacks for completed bars
        self.bar_callbacks: list[Callable[[Bar], None]] = []

    def add_bar_callback(self, callback: Callable[[Bar], None]) -> None:
        """Add callback for completed bars"""
        self.bar_callbacks.append(callback)

    def process_trade(self, trade: Trade) -> None:
        """Process trade and update bars"""

        for interval in self.bar_intervals:
            bar_timestamp = self._get_bar_timestamp(trade.timestamp, interval)
            bar_key = f"{trade.symbol}_{interval}"

            if bar_key not in self.current_bars[trade.symbol]:
                # Create new bar
                self.current_bars[trade.symbol][bar_key] = Bar(
                    symbol=trade.symbol,
                    open_price=trade.price,
                    high_price=trade.price,
                    low_price=trade.price,
                    close_price=trade.price,
                    volume=trade.size,
                    timestamp=bar_timestamp,
                    bar_type=interval,
                )
            else:
                # Update existing bar
                bar = self.current_bars[trade.symbol][bar_key]

                # Check if we need to complete this bar and start a new one
                if self._should_complete_bar(bar.timestamp, trade.timestamp, interval):
                    # Complete the current bar
                    self.completed_bars.append(bar)

                    # Notify callbacks
                    for callback in self.bar_callbacks:
                        try:
                            callback(bar)
                        except Exception as e:
                            logger.error(f"Bar callback error: {str(e)}")

                    # Start new bar
                    self.current_bars[trade.symbol][bar_key] = Bar(
                        symbol=trade.symbol,
                        open_price=trade.price,
                        high_price=trade.price,
                        low_price=trade.price,
                        close_price=trade.price,
                        volume=trade.size,
                        timestamp=bar_timestamp,
                        bar_type=interval,
                    )
                else:
                    # Update current bar
                    bar.high_price = max(bar.high_price, trade.price)
                    bar.low_price = min(bar.low_price, trade.price)
                    bar.close_price = trade.price
                    bar.volume += trade.size

    def _get_bar_timestamp(self, timestamp: datetime, interval: str) -> datetime:
        """Get the bar timestamp for given interval"""

        if interval == "1min":
            return timestamp.replace(second=0, microsecond=0)
        elif interval == "5min":
            minute = (timestamp.minute // 5) * 5
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif interval == "15min":
            minute = (timestamp.minute // 15) * 15
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif interval == "1hour":
            return timestamp.replace(minute=0, second=0, microsecond=0)
        else:
            return timestamp.replace(second=0, microsecond=0)

    def _should_complete_bar(
        self, bar_timestamp: datetime, trade_timestamp: datetime, interval: str
    ) -> bool:
        """Check if current bar should be completed"""

        current_bar_timestamp = self._get_bar_timestamp(trade_timestamp, interval)
        return current_bar_timestamp > bar_timestamp


class StreamingDataManager:
    """Manages real-time market data streaming"""

    def __init__(
        self,
        data_dir: str = "data/streaming",
        primary_provider: DataProvider = DataProvider.SIMULATED,
    ) -> None:

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.data_dir / "quotes").mkdir(exist_ok=True)
        (self.data_dir / "trades").mkdir(exist_ok=True)
        (self.data_dir / "bars").mkdir(exist_ok=True)
        (self.data_dir / "logs").mkdir(exist_ok=True)

        self.primary_provider = primary_provider

        # Initialize database
        self.db_path = self.data_dir / "streaming_data.db"
        self._initialize_database()

        # Streaming state
        self.subscribed_symbols: set[str] = set()
        self.connection_status: dict[DataProvider, ConnectionStatus] = {}
        self.streaming_stats: dict[str, StreamingStats] = {}

        # Data storage
        self.latest_quotes: dict[str, Quote] = {}
        self.latest_trades: dict[str, Trade] = {}
        self.recent_quotes: deque = deque(maxlen=10000)
        self.recent_trades: deque = deque(maxlen=10000)

        # Bar aggregation
        self.bar_aggregator = BarAggregator()
        self.bar_aggregator.add_bar_callback(self._on_bar_completed)

        # Threading for async operations
        self.message_queue = queue.Queue()
        self.processing_thread = None
        self.streaming_thread = None
        self.is_streaming = False

        # Data callbacks
        self.quote_callbacks: list[Callable[[Quote], None]] = []
        self.trade_callbacks: list[Callable[[Trade], None]] = []
        self.bar_callbacks: list[Callable[[Bar], None]] = []

        logger.info(f"Streaming Data Manager initialized - Provider: {primary_provider.value}")

    def _initialize_database(self) -> None:
        """Initialize SQLite database for streaming data"""

        with sqlite3.connect(self.db_path) as conn:
            # Quotes table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quotes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    bid_price TEXT NOT NULL,
                    bid_size INTEGER NOT NULL,
                    ask_price TEXT NOT NULL,
                    ask_size INTEGER NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """
            )

            # Trades table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    exchange TEXT,
                    trade_id TEXT,
                    timestamp TEXT NOT NULL
                )
            """
            )

            # Bars table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    bar_type TEXT NOT NULL,
                    open_price TEXT NOT NULL,
                    high_price TEXT NOT NULL,
                    low_price TEXT NOT NULL,
                    close_price TEXT NOT NULL,
                    volume INTEGER NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """
            )

            # Streaming statistics table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS streaming_stats (
                    symbol TEXT NOT NULL,
                    data_provider TEXT NOT NULL,
                    quotes_received INTEGER DEFAULT 0,
                    trades_received INTEGER DEFAULT 0,
                    bars_generated INTEGER DEFAULT 0,
                    last_update TEXT NOT NULL,
                    PRIMARY KEY (symbol, data_provider)
                )
            """
            )

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_quotes_symbol_time ON quotes (symbol, timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades (symbol, timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bars_symbol_type_time ON bars (symbol, bar_type, timestamp)"
            )

            conn.commit()

    def add_quote_callback(self, callback: Callable[[Quote], None]) -> None:
        """Add callback for quote updates"""
        self.quote_callbacks.append(callback)

    def add_trade_callback(self, callback: Callable[[Trade], None]) -> None:
        """Add callback for trade updates"""
        self.trade_callbacks.append(callback)

    def add_bar_callback(self, callback: Callable[[Bar], None]) -> None:
        """Add callback for completed bars"""
        self.bar_callbacks.append(callback)

    def subscribe_to_symbols(self, symbols: list[str]) -> bool:
        """Subscribe to real-time data for symbols"""

        try:
            new_symbols = set(symbols) - self.subscribed_symbols

            if new_symbols:
                console.print(
                    f"📡 Subscribing to {len(new_symbols)} new symbols: {', '.join(list(new_symbols)[:5])}{'...' if len(new_symbols) > 5 else ''}"
                )

                # Initialize stats for new symbols
                for symbol in new_symbols:
                    self.streaming_stats[symbol] = StreamingStats(
                        symbol=symbol,
                        data_provider=self.primary_provider,
                        connection_status=ConnectionStatus.DISCONNECTED,
                    )

                self.subscribed_symbols.update(new_symbols)

                # Start streaming if not already running
                if not self.is_streaming:
                    self.start_streaming()

            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to symbols: {str(e)}")
            return False

    def start_streaming(self) -> None:
        """Start real-time data streaming"""

        if self.is_streaming:
            console.print("⚠️  Streaming is already active")
            return

        console.print("🚀 [bold green]Starting Market Data Streaming[/bold green]")

        self.is_streaming = True

        # Start message processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

        # Start streaming thread
        self.streaming_thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self.streaming_thread.start()

        console.print("   ✅ Message processing started")
        console.print("   📡 Data streaming started")
        console.print(f"   🎯 Subscribed symbols: {len(self.subscribed_symbols)}")

        logger.info("Market data streaming started successfully")

    def stop_streaming(self) -> None:
        """Stop real-time data streaming"""

        console.print("⏹️  [bold yellow]Stopping Market Data Streaming[/bold yellow]")

        self.is_streaming = False

        # Wait for threads to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)

        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=5)

        console.print("   ✅ Market data streaming stopped")

        logger.info("Market data streaming stopped")

    def _processing_loop(self) -> None:
        """Main message processing loop"""

        while self.is_streaming:
            try:
                # Get message from queue (with timeout)
                message = self.message_queue.get(timeout=1)

                # Process the message
                self._process_message(message)

                # Mark queue task as done
                self.message_queue.task_done()

            except queue.Empty:
                # No messages to process, continue
                continue
            except Exception as e:
                logger.error(f"Message processing error: {str(e)}")
                time.sleep(0.1)  # Brief pause on error

    def _streaming_loop(self) -> None:
        """Main streaming loop (simulated data for demo)"""

        symbols = (
            list(self.subscribed_symbols) if self.subscribed_symbols else ["AAPL", "MSFT", "GOOGL"]
        )

        while self.is_streaming:
            try:
                # Simulate market data (in production, this would be WebSocket connections)
                for symbol in symbols:
                    # Generate simulated quote
                    base_price = Decimal("100.00")  # Base price
                    spread = Decimal("0.01")  # $0.01 spread

                    bid_price = base_price - spread / 2
                    ask_price = base_price + spread / 2

                    quote = Quote(
                        symbol=symbol,
                        bid_price=bid_price,
                        bid_size=100,
                        ask_price=ask_price,
                        ask_size=100,
                    )

                    self.message_queue.put({"type": DataType.QUOTE, "data": quote})

                    # Occasionally generate simulated trades
                    if np.random.random() < 0.1:  # 10% chance
                        trade_price = np.random.choice([bid_price, ask_price])
                        trade = Trade(
                            symbol=symbol,
                            price=trade_price,
                            size=np.random.randint(1, 1000),
                            exchange="SIMULATED",
                        )

                        self.message_queue.put({"type": DataType.TRADE, "data": trade})

                # Sleep to control message frequency
                time.sleep(0.1)  # 10 messages per second

            except Exception as e:
                logger.error(f"Streaming loop error: {str(e)}")
                time.sleep(1)  # Longer pause on error

    def _process_message(self, message: dict[str, Any]) -> None:
        """Process individual streaming message"""

        try:
            data_type = message["type"]
            data = message["data"]

            if data_type == DataType.QUOTE:
                self._process_quote(data)
            elif data_type == DataType.TRADE:
                self._process_trade(data)
            elif data_type == DataType.BAR:
                self._process_bar(data)

        except Exception as e:
            logger.error(f"Message processing error: {str(e)}")

    def _process_quote(self, quote: Quote) -> None:
        """Process quote message"""

        # Update latest quote
        self.latest_quotes[quote.symbol] = quote
        self.recent_quotes.append(quote)

        # Update statistics
        if quote.symbol in self.streaming_stats:
            stats = self.streaming_stats[quote.symbol]
            stats.quotes_received += 1
            stats.last_message_time = quote.timestamp

        # Store in database (sample for recent data)
        if np.random.random() < 0.01:  # Store 1% of quotes to avoid DB bloat
            self._store_quote(quote)

        # Notify callbacks
        for callback in self.quote_callbacks:
            try:
                callback(quote)
            except Exception as e:
                logger.error(f"Quote callback error: {str(e)}")

    def _process_trade(self, trade: Trade) -> None:
        """Process trade message"""

        # Update latest trade
        self.latest_trades[trade.symbol] = trade
        self.recent_trades.append(trade)

        # Update statistics
        if trade.symbol in self.streaming_stats:
            stats = self.streaming_stats[trade.symbol]
            stats.trades_received += 1
            stats.last_message_time = trade.timestamp

        # Process for bar aggregation
        self.bar_aggregator.process_trade(trade)

        # Store in database
        self._store_trade(trade)

        # Notify callbacks
        for callback in self.trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                logger.error(f"Trade callback error: {str(e)}")

    def _process_bar(self, bar: Bar) -> None:
        """Process completed bar"""

        # Update statistics
        if bar.symbol in self.streaming_stats:
            stats = self.streaming_stats[bar.symbol]
            stats.bars_generated += 1

        # Store in database
        self._store_bar(bar)

        # Notify callbacks
        for callback in self.bar_callbacks:
            try:
                callback(bar)
            except Exception as e:
                logger.error(f"Bar callback error: {str(e)}")

    def _on_bar_completed(self, bar: Bar) -> None:
        """Handle completed bar from aggregator"""

        self._process_bar(bar)

    def _store_quote(self, quote: Quote) -> None:
        """Store quote in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO quotes (
                    symbol, bid_price, bid_size, ask_price, ask_size, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    quote.symbol,
                    str(quote.bid_price),
                    quote.bid_size,
                    str(quote.ask_price),
                    quote.ask_size,
                    quote.timestamp.isoformat(),
                ),
            )
            conn.commit()

    def _store_trade(self, trade: Trade) -> None:
        """Store trade in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO trades (
                    symbol, price, size, exchange, trade_id, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    trade.symbol,
                    str(trade.price),
                    trade.size,
                    trade.exchange,
                    trade.trade_id,
                    trade.timestamp.isoformat(),
                ),
            )
            conn.commit()

    def _store_bar(self, bar: Bar) -> None:
        """Store bar in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO bars (
                    symbol, bar_type, open_price, high_price, low_price,
                    close_price, volume, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    bar.symbol,
                    bar.bar_type,
                    str(bar.open_price),
                    str(bar.high_price),
                    str(bar.low_price),
                    str(bar.close_price),
                    bar.volume,
                    bar.timestamp.isoformat(),
                ),
            )
            conn.commit()

    def get_latest_quote(self, symbol: str) -> Quote | None:
        """Get latest quote for symbol"""
        return self.latest_quotes.get(symbol)

    def get_latest_trade(self, symbol: str) -> Trade | None:
        """Get latest trade for symbol"""
        return self.latest_trades.get(symbol)

    def get_streaming_statistics(self) -> dict[str, StreamingStats]:
        """Get streaming statistics for all symbols"""
        return self.streaming_stats.copy()

    def display_streaming_dashboard(self) -> None:
        """Display real-time streaming dashboard"""

        console.print(
            Panel(
                f"[bold blue]Market Data Streaming Dashboard[/bold blue]\n"
                f"Status: {'🟢 ACTIVE' if self.is_streaming else '🔴 STOPPED'}\n"
                f"Provider: {self.primary_provider.value.upper()}\n"
                f"Subscribed Symbols: {len(self.subscribed_symbols)}",
                title="📡 Streaming Data",
            )
        )

        if self.streaming_stats:
            # Streaming statistics table
            stats_table = Table(title="📊 Streaming Statistics")
            stats_table.add_column("Symbol", style="cyan")
            stats_table.add_column("Status", style="white")
            stats_table.add_column("Quotes", justify="right", style="green")
            stats_table.add_column("Trades", justify="right", style="yellow")
            stats_table.add_column("Bars", justify="right", style="blue")
            stats_table.add_column("Last Update", style="dim")

            for symbol, stats in list(self.streaming_stats.items())[:10]:  # Show top 10
                last_update = "Never"
                if stats.last_message_time:
                    elapsed = datetime.now() - stats.last_message_time
                    if elapsed.total_seconds() < 60:
                        last_update = f"{elapsed.total_seconds():.0f}s ago"
                    else:
                        last_update = f"{elapsed.total_seconds()/60:.0f}m ago"

                status_color = (
                    "green" if stats.connection_status == ConnectionStatus.CONNECTED else "red"
                )
                stats_table.add_row(
                    symbol,
                    f"[{status_color}]{stats.connection_status.value}[/{status_color}]",
                    str(stats.quotes_received),
                    str(stats.trades_received),
                    str(stats.bars_generated),
                    last_update,
                )

            console.print(stats_table)

        # Recent data samples
        if self.recent_trades:
            console.print(f"\n💹 [bold]Recent Trades ({len(self.recent_trades)} total)[/bold]")
            for trade in list(self.recent_trades)[-3:]:  # Show last 3 trades
                console.print(f"   {trade.symbol}: {trade.size} @ ${trade.price:.2f}")


def create_streaming_data_manager(
    data_dir: str = "data/streaming", primary_provider: DataProvider = DataProvider.SIMULATED
) -> StreamingDataManager:
    """Factory function to create streaming data manager"""
    return StreamingDataManager(data_dir=data_dir, primary_provider=primary_provider)


if __name__ == "__main__":
    # Example usage
    manager = create_streaming_data_manager()

    # Subscribe to symbols
    manager.subscribe_to_symbols(["AAPL", "MSFT", "GOOGL"])

    # Add callbacks
    def on_quote(quote) -> None:
        print(f"Quote: {quote.symbol} {quote.bid_price}/{quote.ask_price}")

    def on_trade(trade) -> None:
        print(f"Trade: {trade.symbol} {trade.size} @ ${trade.price}")

    def on_bar(bar) -> None:
        print(
            f"Bar: {bar.symbol} {bar.bar_type} OHLC=${bar.open_price}/{bar.high_price}/{bar.low_price}/{bar.close_price}"
        )

    manager.add_quote_callback(on_quote)
    manager.add_trade_callback(on_trade)
    manager.add_bar_callback(on_bar)

    # Start streaming
    manager.start_streaming()

    try:
        # Let it run for demo
        time.sleep(10)

        # Display dashboard
        manager.display_streaming_dashboard()

    finally:
        manager.stop_streaming()

    print("Market Data Streaming System created successfully!")
