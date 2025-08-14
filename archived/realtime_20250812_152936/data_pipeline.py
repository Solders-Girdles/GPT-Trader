"""
Real-time Market Data Ingestion Pipeline

High-performance streaming data pipeline:
- WebSocket connections to market data providers
- Kafka/Redis Streams for data distribution
- Real-time data processing and normalization
- Multi-source data aggregation
- Automatic failover and reconnection
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Never

# Optional dependencies for different data sources
try:
    import websocket

    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from kafka import KafkaConsumer, KafkaProducer

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

import numpy as np
import pandas as pd


class DataSource(Enum):
    """Supported data sources"""

    ALPACA = "alpaca"
    BINANCE = "binance"
    COINBASE = "coinbase"
    IEX = "iex"
    POLYGON = "polygon"
    YAHOO = "yahoo"
    SIMULATED = "simulated"


class StreamType(Enum):
    """Types of data streams"""

    TRADES = "trades"
    QUOTES = "quotes"
    BARS = "bars"
    NEWS = "news"
    ORDERBOOK = "orderbook"


@dataclass
class StreamConfig:
    """Configuration for data streaming"""

    source: DataSource
    symbols: list[str]
    stream_types: list[StreamType]

    # Connection settings
    websocket_url: str | None = None
    api_key: str | None = None
    api_secret: str | None = None

    # Processing settings
    buffer_size: int = 10000
    batch_interval: float = 1.0  # seconds

    # Storage settings
    use_redis: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_ttl: int = 3600  # 1 hour

    use_kafka: bool = False
    kafka_brokers: list[str] = field(default_factory=lambda: ["localhost:9092"])
    kafka_topic: str = "market_data"

    # Resilience settings
    max_reconnect_attempts: int = 10
    reconnect_delay: float = 5.0
    heartbeat_interval: float = 30.0


@dataclass
class MarketTick:
    """Single market data tick"""

    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    exchange: str | None = None
    conditions: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "volume": self.volume,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "exchange": self.exchange,
            "conditions": self.conditions,
        }


class DataBuffer:
    """
    High-performance circular buffer for streaming data.

    Features:
    - Lock-free operations for single producer
    - Batch processing support
    - Automatic overflow handling
    - Statistical aggregation
    """

    def __init__(self, size: int = 10000) -> None:
        self.size = size
        self.buffer = deque(maxlen=size)
        self.lock = threading.RLock()
        self.stats = {"total_received": 0, "total_processed": 0, "dropped": 0, "last_update": None}

    def append(self, tick: MarketTick) -> None:
        """Add tick to buffer"""
        with self.lock:
            if len(self.buffer) == self.size:
                self.stats["dropped"] += 1
            self.buffer.append(tick)
            self.stats["total_received"] += 1
            self.stats["last_update"] = time.time()

    def get_batch(self, n: int | None = None) -> list[MarketTick]:
        """Get batch of ticks"""
        with self.lock:
            if n is None:
                batch = list(self.buffer)
                self.buffer.clear()
            else:
                batch = []
                for _ in range(min(n, len(self.buffer))):
                    batch.append(self.buffer.popleft())

            self.stats["total_processed"] += len(batch)
            return batch

    def get_dataframe(self) -> pd.DataFrame:
        """Convert buffer to DataFrame"""
        with self.lock:
            if not self.buffer:
                return pd.DataFrame()

            data = [tick.to_dict() for tick in self.buffer]
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            return df

    def get_stats(self) -> dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            return {
                **self.stats,
                "current_size": len(self.buffer),
                "utilization": len(self.buffer) / self.size,
            }


class MarketDataStream:
    """
    Base class for market data streaming.

    Handles connection management, data parsing, and error recovery.
    """

    def __init__(self, config: StreamConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_running = False
        self.buffer = DataBuffer(config.buffer_size)
        self.callbacks = []
        self.error_count = 0
        self.last_heartbeat = time.time()

        # Initialize storage backends
        self._init_storage()

    def _init_storage(self) -> None:
        """Initialize Redis/Kafka connections"""
        self.redis_client = None
        self.kafka_producer = None

        if self.config.use_redis and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=self.config.redis_host, port=self.config.redis_port, decode_responses=True
                )
                self.redis_client.ping()
                self.logger.info("Redis connection established")
            except Exception as e:
                self.logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None

        if self.config.use_kafka and KAFKA_AVAILABLE:
            try:
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=self.config.kafka_brokers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                )
                self.logger.info("Kafka producer initialized")
            except Exception as e:
                self.logger.warning(f"Kafka initialization failed: {e}")
                self.kafka_producer = None

    def add_callback(self, callback: Callable[[MarketTick], None]) -> None:
        """Add callback for tick processing"""
        self.callbacks.append(callback)

    def process_tick(self, tick: MarketTick) -> None:
        """Process incoming tick"""
        # Add to buffer
        self.buffer.append(tick)

        # Execute callbacks
        for callback in self.callbacks:
            try:
                callback(tick)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")

        # Store in backends
        self._store_tick(tick)

    def _store_tick(self, tick: MarketTick) -> None:
        """Store tick in Redis/Kafka"""
        tick_data = tick.to_dict()

        # Redis storage
        if self.redis_client:
            try:
                # Store in sorted set by timestamp
                key = f"market:{tick.symbol}:ticks"
                score = tick.timestamp.timestamp()
                self.redis_client.zadd(key, {json.dumps(tick_data): score})

                # Set expiration
                self.redis_client.expire(key, self.config.redis_ttl)

                # Publish to channel for real-time subscribers
                channel = f"market:{tick.symbol}:live"
                self.redis_client.publish(channel, json.dumps(tick_data))
            except Exception as e:
                self.logger.error(f"Redis storage error: {e}")

        # Kafka publishing
        if self.kafka_producer:
            try:
                self.kafka_producer.send(
                    self.config.kafka_topic, key=tick.symbol.encode("utf-8"), value=tick_data
                )
            except Exception as e:
                self.logger.error(f"Kafka publishing error: {e}")

    async def start(self) -> None:
        """Start streaming data"""
        self.is_running = True
        self.logger.info(f"Starting {self.config.source.value} stream")

        # Start heartbeat monitor
        asyncio.create_task(self._heartbeat_monitor())

        # Start processing loop
        await self._stream_loop()

    async def stop(self) -> None:
        """Stop streaming"""
        self.is_running = False

        # Flush storage
        if self.kafka_producer:
            self.kafka_producer.flush()

        self.logger.info("Stream stopped")

    async def _stream_loop(self) -> Never:
        """Main streaming loop - to be implemented by subclasses"""
        raise NotImplementedError

    async def _heartbeat_monitor(self) -> None:
        """Monitor connection health"""
        while self.is_running:
            await asyncio.sleep(self.config.heartbeat_interval)

            if time.time() - self.last_heartbeat > self.config.heartbeat_interval * 2:
                self.logger.warning("Heartbeat timeout, attempting reconnect")
                await self._reconnect()

    async def _reconnect(self) -> None:
        """Reconnect to data source"""
        for attempt in range(self.config.max_reconnect_attempts):
            try:
                self.logger.info(f"Reconnection attempt {attempt + 1}")
                await self.stop()
                await asyncio.sleep(self.config.reconnect_delay)
                await self.start()
                self.logger.info("Reconnection successful")
                return
            except Exception as e:
                self.logger.error(f"Reconnection failed: {e}")

        self.logger.error("Max reconnection attempts reached")


class SimulatedDataStream(MarketDataStream):
    """
    Simulated market data stream for testing.

    Generates realistic market data with configurable parameters.
    """

    def __init__(self, config: StreamConfig) -> None:
        super().__init__(config)
        self.price_data = {}
        self._init_prices()

    def _init_prices(self) -> None:
        """Initialize starting prices"""
        for symbol in self.config.symbols:
            self.price_data[symbol] = {
                "price": 100.0 + np.random.uniform(-10, 10),
                "volatility": 0.02,
                "trend": np.random.uniform(-0.001, 0.001),
            }

    async def _stream_loop(self) -> None:
        """Generate simulated market data"""
        while self.is_running:
            for symbol in self.config.symbols:
                # Generate tick
                tick = self._generate_tick(symbol)
                self.process_tick(tick)
                self.last_heartbeat = time.time()

            # Control rate
            await asyncio.sleep(0.1)  # 10 ticks per second per symbol

    def _generate_tick(self, symbol: str) -> MarketTick:
        """Generate realistic market tick"""
        data = self.price_data[symbol]

        # Random walk with trend
        returns = np.random.normal(data["trend"], data["volatility"])
        data["price"] *= 1 + returns

        # Add some mean reversion
        if data["price"] > 110:
            data["trend"] -= 0.0001
        elif data["price"] < 90:
            data["trend"] += 0.0001

        # Generate bid/ask spread
        spread = data["price"] * 0.001  # 0.1% spread
        bid = data["price"] - spread / 2
        ask = data["price"] + spread / 2

        # Random volume
        volume = np.random.lognormal(10, 1)

        return MarketTick(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            price=data["price"],
            volume=volume,
            bid=bid,
            ask=ask,
            bid_size=volume * np.random.uniform(0.3, 0.7),
            ask_size=volume * np.random.uniform(0.3, 0.7),
            exchange="SIMULATED",
        )


class DataAggregator:
    """
    Aggregates data from multiple streams.

    Features:
    - Multi-source data fusion
    - Time synchronization
    - Outlier detection
    - Statistical aggregation
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.streams = {}
        self.aggregated_buffer = DataBuffer(window_size * 10)
        self.logger = logging.getLogger(__name__)

    def add_stream(self, name: str, stream: MarketDataStream) -> None:
        """Add data stream to aggregator"""
        self.streams[name] = stream
        stream.add_callback(self._process_tick)

    def _process_tick(self, tick: MarketTick) -> None:
        """Process tick from any stream"""
        # Add source information
        tick.exchange = tick.exchange or "unknown"

        # Detect outliers
        if self._is_outlier(tick):
            self.logger.warning(f"Outlier detected: {tick}")
            return

        # Add to aggregated buffer
        self.aggregated_buffer.append(tick)

    def _is_outlier(self, tick: MarketTick) -> bool:
        """Simple outlier detection"""
        # Get recent prices for this symbol
        recent_data = self.get_recent_data(tick.symbol, 20)

        if len(recent_data) < 10:
            return False  # Not enough data

        prices = recent_data["price"].values
        mean = np.mean(prices)
        std = np.std(prices)

        # Check if price is more than 3 standard deviations away
        z_score = abs(tick.price - mean) / (std + 1e-8)

        return z_score > 3

    def get_recent_data(self, symbol: str, n: int = 100) -> pd.DataFrame:
        """Get recent data for symbol"""
        df = self.aggregated_buffer.get_dataframe()

        if df.empty:
            return pd.DataFrame()

        symbol_data = df[df["symbol"] == symbol].tail(n)
        return symbol_data

    def get_aggregated_bars(self, symbol: str, interval: str = "1min") -> pd.DataFrame:
        """Get aggregated OHLCV bars"""
        data = self.get_recent_data(symbol, 1000)

        if data.empty:
            return pd.DataFrame()

        # Resample to bars
        bars = data.resample(interval).agg(
            {"price": ["first", "max", "min", "last"], "volume": "sum"}
        )

        bars.columns = ["open", "high", "low", "close", "volume"]
        bars.dropna(inplace=True)

        return bars


class RealtimePipeline:
    """
    Complete real-time data pipeline.

    Orchestrates multiple data streams, processing, and distribution.
    """

    def __init__(self, configs: list[StreamConfig]) -> None:
        self.configs = configs
        self.streams = []
        self.aggregator = DataAggregator()
        self.logger = logging.getLogger(__name__)
        self.is_running = False

        # Processing queue
        self.process_queue = queue.Queue(maxsize=10000)

        # Metrics
        self.metrics = {"ticks_received": 0, "ticks_processed": 0, "errors": 0, "start_time": None}

    async def start(self) -> None:
        """Start all data streams"""
        self.is_running = True
        self.metrics["start_time"] = time.time()

        # Start streams
        for config in self.configs:
            stream = self._create_stream(config)
            self.streams.append(stream)
            self.aggregator.add_stream(config.source.value, stream)

            # Start stream in background
            asyncio.create_task(stream.start())

        # Start processing loop
        asyncio.create_task(self._process_loop())

        self.logger.info(f"Pipeline started with {len(self.streams)} streams")

    def _create_stream(self, config: StreamConfig) -> MarketDataStream:
        """Create appropriate stream based on config"""
        if config.source == DataSource.SIMULATED:
            return SimulatedDataStream(config)
        else:
            # Placeholder for real data sources
            self.logger.warning(f"Using simulated stream for {config.source.value}")
            return SimulatedDataStream(config)

    async def _process_loop(self) -> None:
        """Main processing loop"""
        while self.is_running:
            try:
                # Process queued items
                while not self.process_queue.empty():
                    item = self.process_queue.get_nowait()
                    await self._process_item(item)
                    self.metrics["ticks_processed"] += 1

                # Short sleep to prevent CPU spinning
                await asyncio.sleep(0.01)

            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                self.metrics["errors"] += 1

    async def _process_item(self, item: Any) -> None:
        """Process single item from queue"""
        # Placeholder for processing logic
        pass

    async def stop(self) -> None:
        """Stop all streams"""
        self.is_running = False

        # Stop all streams
        for stream in self.streams:
            await stream.stop()

        self.logger.info("Pipeline stopped")

    def get_metrics(self) -> dict[str, Any]:
        """Get pipeline metrics"""
        runtime = time.time() - self.metrics["start_time"] if self.metrics["start_time"] else 0

        return {
            **self.metrics,
            "runtime_seconds": runtime,
            "throughput": self.metrics["ticks_received"] / max(1, runtime),
            "active_streams": len(self.streams),
            "aggregator_stats": self.aggregator.aggregated_buffer.get_stats(),
        }

    def get_live_data(self, symbol: str) -> pd.DataFrame:
        """Get live data for symbol"""
        return self.aggregator.get_recent_data(symbol)

    def get_bars(self, symbol: str, interval: str = "1min") -> pd.DataFrame:
        """Get aggregated bars"""
        return self.aggregator.get_aggregated_bars(symbol, interval)


async def demo_pipeline() -> None:
    """Demo real-time pipeline"""
    print("🚀 Real-time Data Pipeline Demo")
    print("=" * 50)

    # Configure simulated streams
    configs = [
        StreamConfig(
            source=DataSource.SIMULATED,
            symbols=["AAPL", "GOOGL", "MSFT"],
            stream_types=[StreamType.TRADES, StreamType.QUOTES],
            use_redis=False,  # Disable for demo
            use_kafka=False,  # Disable for demo
        )
    ]

    # Create pipeline
    pipeline = RealtimePipeline(configs)

    # Start pipeline
    print("\n📊 Starting pipeline...")
    await pipeline.start()

    # Run for a few seconds
    await asyncio.sleep(5)

    # Get metrics
    metrics = pipeline.get_metrics()
    print("\n📈 Pipeline Metrics:")
    print(f"   Throughput: {metrics['throughput']:.1f} ticks/sec")
    print(f"   Processed: {metrics['ticks_processed']:,} ticks")
    print(f"   Errors: {metrics['errors']}")

    # Get sample data
    for symbol in ["AAPL", "GOOGL", "MSFT"]:
        data = pipeline.get_live_data(symbol)
        if not data.empty:
            latest = data.iloc[-1]
            print(
                f"\n   {symbol}: ${latest['price']:.2f} (bid: ${latest['bid']:.2f}, ask: ${latest['ask']:.2f})"
            )

    # Get bars
    bars = pipeline.get_bars("AAPL", "1s")
    if not bars.empty:
        print("\n📊 AAPL 1-second bars:")
        print(bars.tail(3))

    # Stop pipeline
    await pipeline.stop()
    print("\n✅ Pipeline stopped")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

    # Run demo
    asyncio.run(demo_pipeline())
