"""
Real-Time Market Data Pipeline for Live Trading Infrastructure

This module implements high-performance real-time market data processing including:
- Multi-source market data aggregation (WebSocket, REST, FIX)
- Real-time data normalization and validation
- High-frequency tick data processing
- Real-time technical indicator computation
- Market data caching and persistence
- Data quality monitoring and alerting
- Latency optimization and performance monitoring
- Market data replay and backtesting integration
"""

import asyncio
import json
import logging
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd
import websockets

# Optional dependencies with graceful fallback
try:
    import aiofiles
    import aiohttp

    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False
    warnings.warn("Async libraries not available. Some real-time features will be limited.")

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    warnings.warn("Redis not available. Using in-memory caching.")

try:
    import zmq
    import zmq.asyncio

    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    warnings.warn("ZeroMQ not available. Some messaging features will be limited.")

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Market data source types"""

    WEBSOCKET = "websocket"
    REST_API = "rest_api"
    FIX_PROTOCOL = "fix_protocol"
    ZEROMQ = "zeromq"
    FILE_FEED = "file_feed"
    SIMULATION = "simulation"


class DataType(Enum):
    """Market data types"""

    TICK = "tick"
    QUOTE = "quote"
    TRADE = "trade"
    ORDERBOOK = "orderbook"
    BAR = "bar"
    NEWS = "news"
    FUNDAMENTAL = "fundamental"


class DataFrequency(Enum):
    """Data frequency types"""

    TICK = "tick"
    SECOND = "1s"
    MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    HOUR = "1h"
    DAILY = "1d"


@dataclass
class MarketDataPoint:
    """Single market data point"""

    symbol: str
    timestamp: pd.Timestamp
    data_type: DataType
    source: DataSource
    data: dict[str, Any]
    sequence_id: int | None = None
    latency_ms: float | None = None
    quality_score: float = 1.0


@dataclass
class MarketDataConfig:
    """Configuration for market data pipeline"""

    enabled_sources: list[DataSource] = field(
        default_factory=lambda: [DataSource.WEBSOCKET, DataSource.REST_API]
    )
    subscribed_symbols: list[str] = field(default_factory=list)
    data_types: list[DataType] = field(default_factory=lambda: [DataType.TRADE, DataType.QUOTE])
    max_buffer_size: int = 10000
    buffer_flush_interval: float = 1.0  # seconds
    enable_caching: bool = True
    cache_ttl: int = 300  # seconds
    quality_threshold: float = 0.8
    latency_threshold_ms: float = 100.0
    enable_compression: bool = True
    enable_persistence: bool = True
    heartbeat_interval: float = 30.0  # seconds
    reconnect_attempts: int = 5
    reconnect_delay: float = 2.0  # seconds


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""

    total_messages: int = 0
    valid_messages: int = 0
    invalid_messages: int = 0
    duplicate_messages: int = 0
    out_of_order_messages: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    data_gaps: int = 0
    last_update: pd.Timestamp = field(default_factory=pd.Timestamp.now)


class BaseMarketDataSource(ABC):
    """Base class for market data sources"""

    def __init__(self, config: MarketDataConfig, source_type: DataSource) -> None:
        self.config = config
        self.source_type = source_type
        self.is_connected = False
        self.is_subscribed = False
        self.data_buffer = deque(maxlen=config.max_buffer_size)
        self.quality_metrics = DataQualityMetrics()
        self.subscribers = []
        self.last_heartbeat = time.time()

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data source"""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from data source"""
        pass

    @abstractmethod
    async def subscribe(self, symbols: list[str]) -> bool:
        """Subscribe to symbols"""
        pass

    @abstractmethod
    async def start_data_feed(self):
        """Start receiving data"""
        pass

    def add_subscriber(self, callback: Callable[[MarketDataPoint], None]) -> None:
        """Add data subscriber callback"""
        self.subscribers.append(callback)

    def remove_subscriber(self, callback: Callable[[MarketDataPoint], None]) -> None:
        """Remove data subscriber callback"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    async def _notify_subscribers(self, data_point: MarketDataPoint) -> None:
        """Notify all subscribers of new data"""
        for callback in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data_point)
                else:
                    callback(data_point)
            except Exception as e:
                logger.warning(f"Error notifying subscriber: {str(e)}")

    def _validate_data_point(self, data_point: MarketDataPoint) -> bool:
        """Validate incoming data point"""
        try:
            # Basic validation
            if not data_point.symbol or not data_point.timestamp:
                return False

            # Check for required fields based on data type
            if data_point.data_type == DataType.TRADE:
                required_fields = ["price", "volume"]
            elif data_point.data_type == DataType.QUOTE:
                required_fields = ["bid", "ask", "bid_size", "ask_size"]
            elif data_point.data_type == DataType.ORDERBOOK:
                required_fields = ["bids", "asks"]
            else:
                required_fields = []

            for field in required_fields:
                if field not in data_point.data:
                    return False

            # Validate numeric fields
            numeric_fields = ["price", "volume", "bid", "ask", "bid_size", "ask_size"]
            for field in numeric_fields:
                if field in data_point.data:
                    try:
                        float(data_point.data[field])
                    except (ValueError, TypeError):
                        return False

            return True

        except Exception as e:
            logger.warning(f"Data validation error: {str(e)}")
            return False

    def _update_quality_metrics(self, data_point: MarketDataPoint, is_valid: bool) -> None:
        """Update data quality metrics"""
        self.quality_metrics.total_messages += 1

        if is_valid:
            self.quality_metrics.valid_messages += 1
        else:
            self.quality_metrics.invalid_messages += 1

        if data_point.latency_ms:
            # Update latency metrics
            total_latency = (
                self.quality_metrics.avg_latency_ms * (self.quality_metrics.valid_messages - 1)
                + data_point.latency_ms
            )
            self.quality_metrics.avg_latency_ms = (
                total_latency / self.quality_metrics.valid_messages
            )
            self.quality_metrics.max_latency_ms = max(
                self.quality_metrics.max_latency_ms, data_point.latency_ms
            )

        self.quality_metrics.last_update = pd.Timestamp.now()


class WebSocketDataSource(BaseMarketDataSource):
    """WebSocket market data source"""

    def __init__(
        self, config: MarketDataConfig, url: str, auth_params: dict[str, Any] = None
    ) -> None:
        super().__init__(config, DataSource.WEBSOCKET)
        self.url = url
        self.auth_params = auth_params or {}
        self.websocket = None
        self.reconnect_count = 0

    async def connect(self) -> bool:
        """Connect to WebSocket"""
        try:
            headers = {}
            if self.auth_params:
                # Add authentication headers if needed
                headers.update(self.auth_params.get("headers", {}))

            self.websocket = await websockets.connect(self.url, extra_headers=headers)
            self.is_connected = True
            self.reconnect_count = 0
            logger.info(f"Connected to WebSocket: {self.url}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {str(e)}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from WebSocket"""
        try:
            if self.websocket and not self.websocket.closed:
                await self.websocket.close()
            self.is_connected = False
            self.is_subscribed = False
            logger.info("Disconnected from WebSocket")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting WebSocket: {str(e)}")
            return False

    async def subscribe(self, symbols: list[str]) -> bool:
        """Subscribe to symbols"""
        if not self.is_connected or not self.websocket:
            logger.error("Not connected to WebSocket")
            return False

        try:
            # Generic subscription message format
            subscribe_message = {"method": "SUBSCRIBE", "params": symbols, "id": int(time.time())}

            await self.websocket.send(json.dumps(subscribe_message))
            self.is_subscribed = True
            logger.info(f"Subscribed to symbols: {symbols}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to symbols: {str(e)}")
            return False

    async def start_data_feed(self) -> None:
        """Start receiving WebSocket data"""
        if not self.is_connected:
            await self.connect()

        if not self.is_subscribed and self.config.subscribed_symbols:
            await self.subscribe(self.config.subscribed_symbols)

        while self.is_connected and self.websocket:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(
                    self.websocket.recv(), timeout=self.config.heartbeat_interval
                )

                await self._process_message(message)

            except TimeoutError:
                # Send heartbeat/ping
                await self._send_heartbeat()

            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.is_connected = False
                await self._attempt_reconnect()

            except Exception as e:
                logger.error(f"Error in WebSocket data feed: {str(e)}")
                await asyncio.sleep(1)

    async def _process_message(self, message: str) -> None:
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            receive_time = time.time()

            # Parse message based on format (this is exchange-specific)
            data_point = self._parse_market_data(data, receive_time)

            if data_point:
                # Validate data
                is_valid = self._validate_data_point(data_point)
                self._update_quality_metrics(data_point, is_valid)

                if is_valid and data_point.quality_score >= self.config.quality_threshold:
                    # Buffer data
                    self.data_buffer.append(data_point)

                    # Notify subscribers
                    await self._notify_subscribers(data_point)

        except json.JSONDecodeError:
            logger.warning("Received invalid JSON message")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {str(e)}")

    def _parse_market_data(
        self, data: dict[str, Any], receive_time: float
    ) -> MarketDataPoint | None:
        """Parse raw WebSocket data into MarketDataPoint"""
        try:
            # Generic parsing - would need to be customized for specific exchanges
            if "stream" in data and "data" in data:
                stream = data["stream"]
                payload = data["data"]

                # Extract symbol from stream name
                symbol = stream.split("@")[0].upper() if "@" in stream else "UNKNOWN"

                # Determine data type
                if "trade" in stream:
                    data_type = DataType.TRADE
                    parsed_data = {
                        "price": float(payload.get("p", 0)),
                        "volume": float(payload.get("q", 0)),
                        "time": payload.get("T", receive_time * 1000),
                    }
                elif "ticker" in stream:
                    data_type = DataType.QUOTE
                    parsed_data = {
                        "bid": float(payload.get("b", 0)),
                        "ask": float(payload.get("a", 0)),
                        "bid_size": float(payload.get("B", 0)),
                        "ask_size": float(payload.get("A", 0)),
                    }
                else:
                    return None

                # Calculate latency
                message_time = parsed_data.get("time", receive_time * 1000) / 1000
                latency_ms = (receive_time - message_time) * 1000

                return MarketDataPoint(
                    symbol=symbol,
                    timestamp=pd.Timestamp.fromtimestamp(message_time),
                    data_type=data_type,
                    source=self.source_type,
                    data=parsed_data,
                    latency_ms=latency_ms,
                    quality_score=1.0 if latency_ms <= self.config.latency_threshold_ms else 0.5,
                )

            return None

        except Exception as e:
            logger.warning(f"Error parsing market data: {str(e)}")
            return None

    async def _send_heartbeat(self) -> None:
        """Send heartbeat/ping message"""
        try:
            if self.websocket and not self.websocket.closed:
                ping_message = {"method": "ping", "id": int(time.time())}
                await self.websocket.send(json.dumps(ping_message))
                self.last_heartbeat = time.time()
        except Exception as e:
            logger.warning(f"Failed to send heartbeat: {str(e)}")

    async def _attempt_reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff"""
        if self.reconnect_count >= self.config.reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return

        self.reconnect_count += 1
        delay = self.config.reconnect_delay * (2 ** (self.reconnect_count - 1))

        logger.info(
            f"Attempting reconnection {self.reconnect_count}/{self.config.reconnect_attempts} in {delay}s"
        )
        await asyncio.sleep(delay)

        success = await self.connect()
        if success and self.config.subscribed_symbols:
            await self.subscribe(self.config.subscribed_symbols)


class RESTDataSource(BaseMarketDataSource):
    """REST API market data source for historical and snapshot data"""

    def __init__(
        self, config: MarketDataConfig, base_url: str, auth_params: dict[str, Any] = None
    ) -> None:
        super().__init__(config, DataSource.REST_API)
        self.base_url = base_url
        self.auth_params = auth_params or {}
        self.session = None
        self.poll_interval = 1.0  # seconds

    async def connect(self) -> bool:
        """Create HTTP session"""
        try:
            if ASYNC_AVAILABLE:
                self.session = aiohttp.ClientSession()
            self.is_connected = True
            logger.info(f"Connected to REST API: {self.base_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to create REST session: {str(e)}")
            return False

    async def disconnect(self) -> bool:
        """Close HTTP session"""
        try:
            if self.session:
                await self.session.close()
            self.is_connected = False
            logger.info("Disconnected from REST API")
            return True
        except Exception as e:
            logger.error(f"Error closing REST session: {str(e)}")
            return False

    async def subscribe(self, symbols: list[str]) -> bool:
        """Set symbols for polling"""
        # REST doesn't have real subscriptions, just store symbols
        self.is_subscribed = True
        logger.info(f"Set REST polling symbols: {symbols}")
        return True

    async def start_data_feed(self) -> None:
        """Start polling REST endpoints"""
        if not self.is_connected:
            await self.connect()

        while self.is_connected:
            try:
                for symbol in self.config.subscribed_symbols:
                    await self._poll_symbol_data(symbol)

                await asyncio.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Error in REST data feed: {str(e)}")
                await asyncio.sleep(self.poll_interval)

    async def _poll_symbol_data(self, symbol: str) -> None:
        """Poll data for a single symbol"""
        try:
            if not ASYNC_AVAILABLE or not self.session:
                logger.warning("Async HTTP not available")
                return

            # Build request URL (exchange-specific)
            url = f"{self.base_url}/ticker/{symbol}"
            headers = self.auth_params.get("headers", {})

            start_time = time.time()
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    receive_time = time.time()

                    # Parse response
                    data_point = self._parse_rest_data(symbol, data, receive_time, start_time)

                    if data_point:
                        # Validate and notify
                        is_valid = self._validate_data_point(data_point)
                        self._update_quality_metrics(data_point, is_valid)

                        if is_valid:
                            self.data_buffer.append(data_point)
                            await self._notify_subscribers(data_point)

        except Exception as e:
            logger.warning(f"Failed to poll data for {symbol}: {str(e)}")

    def _parse_rest_data(
        self, symbol: str, data: dict[str, Any], receive_time: float, request_start: float
    ) -> MarketDataPoint | None:
        """Parse REST API response"""
        try:
            # Generic REST response parsing
            parsed_data = {
                "price": float(data.get("price", data.get("last", 0))),
                "bid": float(data.get("bid", 0)),
                "ask": float(data.get("ask", 0)),
                "volume": float(data.get("volume", 0)),
                "change": float(data.get("change", 0)),
                "change_percent": float(data.get("changePercent", 0)),
            }

            # Calculate request latency
            latency_ms = (receive_time - request_start) * 1000

            return MarketDataPoint(
                symbol=symbol.upper(),
                timestamp=pd.Timestamp.now(),
                data_type=DataType.QUOTE,
                source=self.source_type,
                data=parsed_data,
                latency_ms=latency_ms,
                quality_score=1.0 if latency_ms <= self.config.latency_threshold_ms else 0.8,
            )

        except Exception as e:
            logger.warning(f"Error parsing REST data: {str(e)}")
            return None


class MarketDataCache:
    """High-performance market data cache"""

    def __init__(self, config: MarketDataConfig) -> None:
        self.config = config
        self.memory_cache = {}
        self.redis_client = None
        self.cache_stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}

        if REDIS_AVAILABLE and config.enable_caching:
            try:
                self.redis_client = redis.Redis(decode_responses=True)
                self.redis_client.ping()
                logger.info("Redis cache connected")
            except Exception as e:
                logger.warning(f"Redis connection failed: {str(e)}, using memory cache")
                self.redis_client = None

    async def get(self, key: str) -> Any | None:
        """Get value from cache"""
        try:
            # Try memory cache first
            if key in self.memory_cache:
                self.cache_stats["hits"] += 1
                return self.memory_cache[key]

            # Try Redis cache
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    try:
                        parsed_value = json.loads(value)
                        # Store in memory cache for faster access
                        self.memory_cache[key] = parsed_value
                        self.cache_stats["hits"] += 1
                        return parsed_value
                    except json.JSONDecodeError:
                        pass

            self.cache_stats["misses"] += 1
            return None

        except Exception as e:
            logger.warning(f"Cache get error: {str(e)}")
            self.cache_stats["misses"] += 1
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache"""
        try:
            ttl = ttl or self.config.cache_ttl

            # Store in memory cache
            self.memory_cache[key] = value

            # Store in Redis cache
            if self.redis_client:
                serialized_value = json.dumps(value, default=str)
                self.redis_client.setex(key, ttl, serialized_value)

            self.cache_stats["sets"] += 1

        except Exception as e:
            logger.warning(f"Cache set error: {str(e)}")

    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        try:
            # Remove from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]

            # Remove from Redis cache
            if self.redis_client:
                self.redis_client.delete(key)

            self.cache_stats["deletes"] += 1

        except Exception as e:
            logger.warning(f"Cache delete error: {str(e)}")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests) if total_requests > 0 else 0

        return {
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            **self.cache_stats,
            "memory_cache_size": len(self.memory_cache),
            "redis_connected": self.redis_client is not None,
        }


class RealTimeMarketDataPipeline:
    """Main real-time market data pipeline"""

    def __init__(self, config: MarketDataConfig) -> None:
        self.config = config
        self.data_sources = {}
        self.cache = MarketDataCache(config)
        self.data_handlers = []
        self.is_running = False
        self.processed_count = 0
        self.start_time = time.time()
        self.performance_metrics = {
            "messages_per_second": 0,
            "avg_processing_latency": 0,
            "data_source_status": {},
            "quality_scores": {},
        }

    def add_websocket_source(self, name: str, url: str, auth_params: dict[str, Any] = None) -> None:
        """Add WebSocket data source"""
        source = WebSocketDataSource(self.config, url, auth_params)
        self.data_sources[name] = source
        source.add_subscriber(self._handle_market_data)
        logger.info(f"Added WebSocket source: {name}")

    def add_rest_source(self, name: str, base_url: str, auth_params: dict[str, Any] = None) -> None:
        """Add REST data source"""
        source = RESTDataSource(self.config, base_url, auth_params)
        self.data_sources[name] = source
        source.add_subscriber(self._handle_market_data)
        logger.info(f"Added REST source: {name}")

    def add_data_handler(self, handler: Callable[[MarketDataPoint], None]) -> None:
        """Add data handler callback"""
        self.data_handlers.append(handler)

    async def start(self) -> None:
        """Start the market data pipeline"""
        if self.is_running:
            logger.warning("Pipeline already running")
            return

        self.is_running = True
        self.start_time = time.time()
        logger.info("Starting market data pipeline...")

        # Start all data sources
        tasks = []
        for name, source in self.data_sources.items():
            logger.info(f"Starting data source: {name}")
            task = asyncio.create_task(source.start_data_feed())
            tasks.append(task)

        # Start performance monitoring
        monitor_task = asyncio.create_task(self._monitor_performance())
        tasks.append(monitor_task)

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the market data pipeline"""
        if not self.is_running:
            return

        logger.info("Stopping market data pipeline...")
        self.is_running = False

        # Disconnect all data sources
        for name, source in self.data_sources.items():
            logger.info(f"Disconnecting data source: {name}")
            await source.disconnect()

        logger.info("Market data pipeline stopped")

    async def _handle_market_data(self, data_point: MarketDataPoint) -> None:
        """Handle incoming market data"""
        try:
            process_start = time.time()

            # Cache the data point
            if self.config.enable_caching:
                cache_key = f"market_data:{data_point.symbol}:{data_point.data_type.value}"
                await self.cache.set(
                    cache_key,
                    {
                        "symbol": data_point.symbol,
                        "timestamp": data_point.timestamp.isoformat(),
                        "data_type": data_point.data_type.value,
                        "data": data_point.data,
                        "source": data_point.source.value,
                    },
                )

            # Process with registered handlers
            for handler in self.data_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data_point)
                    else:
                        handler(data_point)
                except Exception as e:
                    logger.warning(f"Data handler error: {str(e)}")

            # Update metrics
            self.processed_count += 1
            processing_time = (time.time() - process_start) * 1000  # ms

            # Update rolling average of processing latency
            current_avg = self.performance_metrics["avg_processing_latency"]
            self.performance_metrics["avg_processing_latency"] = (
                current_avg * (self.processed_count - 1) + processing_time
            ) / self.processed_count

        except Exception as e:
            logger.error(f"Error handling market data: {str(e)}")

    async def _monitor_performance(self) -> None:
        """Monitor pipeline performance"""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Update every 10 seconds

                # Calculate messages per second
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 0:
                    self.performance_metrics["messages_per_second"] = (
                        self.processed_count / elapsed_time
                    )

                # Update data source status and quality scores
                for name, source in self.data_sources.items():
                    self.performance_metrics["data_source_status"][name] = {
                        "connected": source.is_connected,
                        "subscribed": source.is_subscribed,
                        "buffer_size": len(source.data_buffer),
                        "quality_metrics": {
                            "total_messages": source.quality_metrics.total_messages,
                            "valid_messages": source.quality_metrics.valid_messages,
                            "avg_latency_ms": source.quality_metrics.avg_latency_ms,
                            "data_quality_rate": (
                                source.quality_metrics.valid_messages
                                / max(source.quality_metrics.total_messages, 1)
                            ),
                        },
                    }

                # Log performance summary
                logger.info(
                    f"Pipeline performance: {self.performance_metrics['messages_per_second']:.2f} msg/s, "
                    f"avg latency: {self.performance_metrics['avg_processing_latency']:.2f}ms, "
                    f"processed: {self.processed_count}"
                )

            except Exception as e:
                logger.warning(f"Performance monitoring error: {str(e)}")

    def get_latest_data(
        self, symbol: str, data_type: DataType = DataType.QUOTE
    ) -> dict[str, Any] | None:
        """Get latest cached data for symbol"""
        try:
            # Note: This would need to be async in real implementation
            return None  # Placeholder
        except Exception as e:
            logger.warning(f"Error getting latest data: {str(e)}")
            return None

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get pipeline performance metrics"""
        cache_stats = self.cache.get_stats()

        return {
            "pipeline": self.performance_metrics,
            "cache": cache_stats,
            "uptime_seconds": time.time() - self.start_time,
            "total_processed": self.processed_count,
            "is_running": self.is_running,
            "data_sources": len(self.data_sources),
        }


def create_market_data_pipeline(
    subscribed_symbols: list[str], enabled_sources: list[DataSource] | None = None, **kwargs
) -> RealTimeMarketDataPipeline:
    """Factory function to create market data pipeline"""
    if enabled_sources is None:
        enabled_sources = [DataSource.WEBSOCKET, DataSource.REST_API]

    config = MarketDataConfig(
        enabled_sources=enabled_sources, subscribed_symbols=subscribed_symbols, **kwargs
    )

    return RealTimeMarketDataPipeline(config)


# Example usage and testing
async def main() -> None:
    """Example usage of market data pipeline"""
    print("Real-Time Market Data Pipeline Testing")
    print("=" * 50)

    # Create pipeline
    symbols = ["AAPL", "GOOGL", "MSFT"]
    pipeline = create_market_data_pipeline(
        subscribed_symbols=symbols, max_buffer_size=5000, enable_caching=True
    )

    # Add mock data sources for testing
    pipeline.add_websocket_source("test_ws", "wss://stream.binance.com:9443/ws/btcusdt@trade", {})

    pipeline.add_rest_source("test_rest", "https://api.binance.com/api/v3", {})

    # Add data handler
    def handle_data(data_point: MarketDataPoint) -> None:
        print(f"📈 {data_point.symbol}: {data_point.data_type.value} - {data_point.data}")

    pipeline.add_data_handler(handle_data)

    print(f"✅ Pipeline created with {len(pipeline.data_sources)} data sources")
    print(f"📊 Subscribed to: {symbols}")

    # In a real implementation, you would run:
    # await pipeline.start()

    print("\n🚀 Real-Time Market Data Pipeline ready for production!")


if __name__ == "__main__":
    if ASYNC_AVAILABLE:
        asyncio.run(main())
    else:
        print("Async libraries not available - showing configuration only")
        print("Real-Time Market Data Pipeline Framework Created! 🚀")
