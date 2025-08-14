"""
Event-Driven Architecture for Real-Time Trading Infrastructure

This module implements a sophisticated event-driven system that coordinates all live trading components:
- Event definition and routing system
- Real-time message passing and pub/sub patterns
- Event sourcing for audit trails and replay capability
- Complex event processing (CEP) for pattern detection
- Event-driven workflow orchestration
- High-throughput event streaming with backpressure handling
- Event persistence and recovery mechanisms
- Distributed event processing capabilities
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

# Optional dependencies with graceful fallback
try:
    import zmq
    import zmq.asyncio

    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

try:
    import redis
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types in the trading system"""

    # Market Data Events
    MARKET_DATA_TICK = "market_data_tick"
    MARKET_DATA_QUOTE = "market_data_quote"
    MARKET_DATA_TRADE = "market_data_trade"
    MARKET_DATA_ORDERBOOK = "market_data_orderbook"
    MARKET_DATA_CONNECTION = "market_data_connection"

    # Order Management Events
    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACCEPTED = "order_accepted"
    ORDER_REJECTED = "order_rejected"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_CANCELED = "order_canceled"
    ORDER_REPLACED = "order_replaced"
    ORDER_EXPIRED = "order_expired"

    # Portfolio Events
    POSITION_UPDATED = "position_updated"
    PORTFOLIO_REBALANCED = "portfolio_rebalanced"
    RISK_LIMIT_BREACHED = "risk_limit_breached"
    PNL_UPDATED = "pnl_updated"

    # Risk Management Events
    RISK_ALERT = "risk_alert"
    RISK_LIMIT_WARNING = "risk_limit_warning"
    DRAWDOWN_ALERT = "drawdown_alert"
    VOLATILITY_SPIKE = "volatility_spike"

    # Strategy Events
    STRATEGY_SIGNAL = "strategy_signal"
    STRATEGY_STARTED = "strategy_started"
    STRATEGY_STOPPED = "strategy_stopped"
    STRATEGY_ERROR = "strategy_error"

    # System Events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    HEARTBEAT = "heartbeat"
    CONNECTION_LOST = "connection_lost"
    CONNECTION_RESTORED = "connection_restored"
    ERROR_OCCURRED = "error_occurred"

    # Performance Events
    LATENCY_WARNING = "latency_warning"
    THROUGHPUT_ALERT = "throughput_alert"
    MEMORY_WARNING = "memory_warning"

    # Custom Events
    CUSTOM_EVENT = "custom_event"


class EventPriority(Enum):
    """Event priority levels"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class EventSource(Enum):
    """Event sources"""

    MARKET_DATA_PIPELINE = "market_data_pipeline"
    ORDER_MANAGER = "order_manager"
    RISK_MONITOR = "risk_monitor"
    PORTFOLIO_MANAGER = "portfolio_manager"
    STRATEGY_ENGINE = "strategy_engine"
    PERFORMANCE_TRACKER = "performance_tracker"
    SYSTEM = "system"
    EXTERNAL_API = "external_api"
    USER_INPUT = "user_input"


@dataclass
class Event:
    """Base event class"""

    event_id: str
    event_type: EventType
    source: EventSource
    timestamp: pd.Timestamp
    data: dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: str | None = None
    parent_event_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())


@dataclass
class EventPattern:
    """Complex event pattern for CEP"""

    pattern_id: str
    name: str
    event_types: list[EventType]
    time_window_seconds: float
    conditions: list[Callable[[list[Event]], bool]]
    action: Callable[[list[Event]], None]
    is_active: bool = True


@dataclass
class EventRoute:
    """Event routing configuration"""

    event_type: EventType
    handlers: list[str]  # Handler names
    filters: list[Callable[[Event], bool]] = field(default_factory=list)
    is_async: bool = True
    max_retries: int = 3
    timeout_seconds: float = 5.0


@dataclass
class EventStreamConfig:
    """Configuration for event streaming"""

    max_buffer_size: int = 10000
    batch_size: int = 100
    flush_interval_ms: int = 100
    enable_persistence: bool = True
    enable_compression: bool = True
    retention_hours: int = 24
    distributed_processing: bool = False
    redis_url: str | None = None
    zmq_port: int | None = None


class EventHandler(ABC):
    """Base class for event handlers"""

    def __init__(self, handler_id: str) -> None:
        self.handler_id = handler_id
        self.is_active = True
        self.processed_count = 0
        self.error_count = 0

    @abstractmethod
    async def handle_event(self, event: Event) -> bool:
        """Handle an event"""
        pass

    def get_supported_events(self) -> list[EventType]:
        """Get list of supported event types"""
        return []

    def get_metrics(self) -> dict[str, Any]:
        """Get handler performance metrics"""
        return {
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "is_active": self.is_active,
            "error_rate": self.error_count / max(self.processed_count, 1),
        }


class EventStore:
    """Event sourcing store for persistence and replay"""

    def __init__(self, config: EventStreamConfig) -> None:
        self.config = config
        self.events = deque(maxlen=config.max_buffer_size)
        self.event_index = {}  # event_id -> event
        self.type_index = defaultdict(list)  # event_type -> [event_ids]
        self.correlation_index = defaultdict(list)  # correlation_id -> [event_ids]
        self.redis_client = None
        self.lock = threading.RLock()

        # Initialize Redis if available and enabled
        if REDIS_AVAILABLE and config.enable_persistence and config.redis_url:
            try:
                self.redis_client = redis.from_url(config.redis_url)
                logger.info("Event store connected to Redis")
            except Exception as e:
                logger.warning(f"Redis connection failed: {str(e)}")

    async def store_event(self, event: Event) -> None:
        """Store an event"""
        try:
            with self.lock:
                # Store in memory
                self.events.append(event)
                self.event_index[event.event_id] = event
                self.type_index[event.event_type].append(event.event_id)
                if event.correlation_id:
                    self.correlation_index[event.correlation_id].append(event.event_id)

            # Store in Redis if available
            if self.redis_client and self.config.enable_persistence:
                event_data = {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "source": event.source.value,
                    "timestamp": event.timestamp.isoformat(),
                    "data": event.data,
                    "priority": event.priority.value,
                    "correlation_id": event.correlation_id,
                    "parent_event_id": event.parent_event_id,
                    "metadata": event.metadata,
                }

                # Store with TTL
                await self.redis_client.setex(
                    f"event:{event.event_id}",
                    self.config.retention_hours * 3600,
                    json.dumps(event_data, default=str),
                )

                # Store in type index
                await self.redis_client.lpush(
                    f"events_by_type:{event.event_type.value}", event.event_id
                )
                await self.redis_client.expire(
                    f"events_by_type:{event.event_type.value}", self.config.retention_hours * 3600
                )

        except Exception as e:
            logger.error(f"Failed to store event: {str(e)}")

    def get_event(self, event_id: str) -> Event | None:
        """Get event by ID"""
        with self.lock:
            return self.event_index.get(event_id)

    def get_events_by_type(self, event_type: EventType, limit: int = 100) -> list[Event]:
        """Get events by type"""
        with self.lock:
            event_ids = self.type_index.get(event_type, [])
            return [self.event_index[eid] for eid in event_ids[-limit:] if eid in self.event_index]

    def get_events_by_correlation(self, correlation_id: str) -> list[Event]:
        """Get events by correlation ID"""
        with self.lock:
            event_ids = self.correlation_index.get(correlation_id, [])
            return [self.event_index[eid] for eid in event_ids if eid in self.event_index]

    async def replay_events(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        event_types: list[EventType] | None = None,
    ) -> AsyncIterator[Event]:
        """Replay events within time range"""
        with self.lock:
            events = list(self.events)

        for event in events:
            if start_time <= event.timestamp <= end_time:
                if event_types is None or event.event_type in event_types:
                    yield event

    def get_stats(self) -> dict[str, Any]:
        """Get event store statistics"""
        with self.lock:
            return {
                "total_events": len(self.events),
                "events_by_type": {et.value: len(events) for et, events in self.type_index.items()},
                "unique_correlations": len(self.correlation_index),
                "oldest_event": self.events[0].timestamp.isoformat() if self.events else None,
                "newest_event": self.events[-1].timestamp.isoformat() if self.events else None,
                "redis_connected": self.redis_client is not None,
            }


class ComplexEventProcessor:
    """Complex Event Processing engine for pattern detection"""

    def __init__(self) -> None:
        self.patterns = {}
        self.pattern_state = defaultdict(list)  # pattern_id -> [recent_events]
        self.pattern_matches = []
        self.lock = threading.RLock()

    def add_pattern(self, pattern: EventPattern) -> None:
        """Add event pattern for detection"""
        self.patterns[pattern.pattern_id] = pattern
        logger.info(f"Added event pattern: {pattern.name}")

    def remove_pattern(self, pattern_id: str) -> None:
        """Remove event pattern"""
        if pattern_id in self.patterns:
            del self.patterns[pattern_id]
            if pattern_id in self.pattern_state:
                del self.pattern_state[pattern_id]
            logger.info(f"Removed event pattern: {pattern_id}")

    async def process_event(self, event: Event) -> None:
        """Process event against all patterns"""
        current_time = time.time()

        with self.lock:
            for pattern_id, pattern in self.patterns.items():
                if not pattern.is_active:
                    continue

                if event.event_type in pattern.event_types:
                    # Add event to pattern state
                    self.pattern_state[pattern_id].append(event)

                    # Clean old events outside time window
                    cutoff_time = current_time - pattern.time_window_seconds
                    self.pattern_state[pattern_id] = [
                        e
                        for e in self.pattern_state[pattern_id]
                        if e.timestamp.timestamp() >= cutoff_time
                    ]

                    # Check if pattern conditions are met
                    recent_events = self.pattern_state[pattern_id]
                    if len(recent_events) >= len(pattern.event_types):
                        try:
                            if all(condition(recent_events) for condition in pattern.conditions):
                                # Pattern matched - execute action
                                await self._execute_pattern_action(pattern, recent_events)

                                # Clear pattern state to avoid duplicate matches
                                self.pattern_state[pattern_id] = []

                        except Exception as e:
                            logger.error(f"Error in pattern {pattern.name}: {str(e)}")

    async def _execute_pattern_action(
        self, pattern: EventPattern, matching_events: list[Event]
    ) -> None:
        """Execute action for matched pattern"""
        try:
            match_info = {
                "pattern_id": pattern.pattern_id,
                "pattern_name": pattern.name,
                "matched_events": [e.event_id for e in matching_events],
                "match_time": pd.Timestamp.now(),
                "event_span": (
                    matching_events[-1].timestamp - matching_events[0].timestamp
                ).total_seconds(),
            }

            self.pattern_matches.append(match_info)

            # Execute pattern action
            if asyncio.iscoroutinefunction(pattern.action):
                await pattern.action(matching_events)
            else:
                pattern.action(matching_events)

            logger.info(f"Pattern matched: {pattern.name} with {len(matching_events)} events")

        except Exception as e:
            logger.error(f"Failed to execute pattern action for {pattern.name}: {str(e)}")

    def get_pattern_stats(self) -> dict[str, Any]:
        """Get CEP statistics"""
        with self.lock:
            return {
                "active_patterns": len([p for p in self.patterns.values() if p.is_active]),
                "total_patterns": len(self.patterns),
                "pattern_matches": len(self.pattern_matches),
                "pattern_state_size": {
                    pid: len(events) for pid, events in self.pattern_state.items()
                },
                "recent_matches": self.pattern_matches[-10:] if self.pattern_matches else [],
            }


class EventBus:
    """High-performance event bus with pub/sub capabilities"""

    def __init__(self, config: EventStreamConfig) -> None:
        self.config = config
        self.handlers = {}  # handler_id -> EventHandler
        self.routes = {}  # event_type -> EventRoute
        self.subscriptions = defaultdict(set)  # event_type -> {handler_ids}
        self.event_store = EventStore(config)
        self.complex_event_processor = ComplexEventProcessor()

        # Performance metrics
        self.metrics = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "avg_latency_ms": 0.0,
            "throughput_per_second": 0.0,
            "queue_size": 0,
        }

        # Event processing queue
        self.event_queue = asyncio.Queue(maxsize=config.max_buffer_size)
        self.processing_tasks = []
        self.is_running = False

        # ZeroMQ for distributed processing
        self.zmq_context = None
        self.zmq_publisher = None
        self.zmq_subscriber = None

        if ZMQ_AVAILABLE and config.distributed_processing and config.zmq_port:
            self.zmq_context = zmq.asyncio.Context()

    def register_handler(self, handler: EventHandler) -> None:
        """Register event handler"""
        self.handlers[handler.handler_id] = handler

        # Auto-subscribe to supported events
        for event_type in handler.get_supported_events():
            self.subscriptions[event_type].add(handler.handler_id)

        logger.info(f"Registered event handler: {handler.handler_id}")

    def unregister_handler(self, handler_id: str) -> None:
        """Unregister event handler"""
        if handler_id in self.handlers:
            del self.handlers[handler_id]

            # Remove from subscriptions
            for _event_type, handler_ids in self.subscriptions.items():
                handler_ids.discard(handler_id)

            logger.info(f"Unregistered event handler: {handler_id}")

    def add_route(self, route: EventRoute) -> None:
        """Add event routing rule"""
        self.routes[route.event_type] = route
        logger.info(f"Added route for {route.event_type.value}")

    def subscribe(self, handler_id: str, event_type: EventType) -> None:
        """Subscribe handler to event type"""
        if handler_id in self.handlers:
            self.subscriptions[event_type].add(handler_id)
            logger.debug(f"Handler {handler_id} subscribed to {event_type.value}")

    def unsubscribe(self, handler_id: str, event_type: EventType) -> None:
        """Unsubscribe handler from event type"""
        self.subscriptions[event_type].discard(handler_id)
        logger.debug(f"Handler {handler_id} unsubscribed from {event_type.value}")

    async def publish(self, event: Event) -> None:
        """Publish event to the bus"""
        try:
            # Store event
            await self.event_store.store_event(event)

            # Process with CEP
            await self.complex_event_processor.process_event(event)

            # Add to processing queue
            if not self.event_queue.full():
                await self.event_queue.put(event)
                self.metrics["events_published"] += 1
            else:
                logger.warning("Event queue full, dropping event")
                self.metrics["events_failed"] += 1

            # Publish to distributed subscribers if enabled
            if self.zmq_publisher:
                await self._publish_to_zmq(event)

        except Exception as e:
            logger.error(f"Failed to publish event: {str(e)}")
            self.metrics["events_failed"] += 1

    async def _publish_to_zmq(self, event: Event) -> None:
        """Publish event via ZeroMQ"""
        try:
            event_data = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "source": event.source.value,
                "timestamp": event.timestamp.isoformat(),
                "data": event.data,
                "priority": event.priority.value,
                "correlation_id": event.correlation_id,
            }

            await self.zmq_publisher.send_multipart(
                [event.event_type.value.encode(), json.dumps(event_data, default=str).encode()]
            )

        except Exception as e:
            logger.warning(f"Failed to publish to ZMQ: {str(e)}")

    async def start(self, num_workers: int = 4) -> None:
        """Start the event bus"""
        if self.is_running:
            logger.warning("Event bus already running")
            return

        self.is_running = True
        logger.info("Starting event bus...")

        # Start event processing workers
        for i in range(num_workers):
            task = asyncio.create_task(self._event_processor_worker(f"worker_{i}"))
            self.processing_tasks.append(task)

        # Start metrics collection
        asyncio.create_task(self._metrics_collector())

        # Initialize ZeroMQ if configured
        if self.zmq_context and self.config.zmq_port:
            await self._setup_zmq()

        logger.info(f"Event bus started with {num_workers} workers")

    async def stop(self) -> None:
        """Stop the event bus"""
        if not self.is_running:
            return

        logger.info("Stopping event bus...")
        self.is_running = False

        # Cancel processing tasks
        for task in self.processing_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)

        # Close ZeroMQ
        if self.zmq_context:
            self.zmq_context.term()

        logger.info("Event bus stopped")

    async def _event_processor_worker(self, worker_id: str) -> None:
        """Event processing worker"""
        logger.info(f"Event processor {worker_id} started")

        while self.is_running:
            try:
                # Get event from queue with timeout
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                except TimeoutError:
                    continue

                start_time = time.time()

                # Route event to handlers
                await self._route_event(event)

                # Update metrics
                processing_time = (time.time() - start_time) * 1000  # ms
                self.metrics["events_processed"] += 1

                # Update average latency
                current_avg = self.metrics["avg_latency_ms"]
                processed_count = self.metrics["events_processed"]
                self.metrics["avg_latency_ms"] = (
                    current_avg * (processed_count - 1) + processing_time
                ) / processed_count

                self.event_queue.task_done()

            except Exception as e:
                logger.error(f"Event processor {worker_id} error: {str(e)}")
                self.metrics["events_failed"] += 1
                await asyncio.sleep(0.1)

        logger.info(f"Event processor {worker_id} stopped")

    async def _route_event(self, event: Event) -> None:
        """Route event to appropriate handlers"""
        try:
            # Get subscribers for this event type
            handler_ids = self.subscriptions.get(event.event_type, set())

            # Apply routing rules if configured
            route = self.routes.get(event.event_type)
            if route:
                # Apply filters
                if route.filters:
                    for filter_func in route.filters:
                        if not filter_func(event):
                            return  # Event filtered out

                # Override handlers from route
                if route.handlers:
                    handler_ids = set(route.handlers) & set(self.handlers.keys())

            # Process event with handlers
            tasks = []
            for handler_id in handler_ids:
                handler = self.handlers.get(handler_id)
                if handler and handler.is_active:
                    if route and route.is_async:
                        tasks.append(self._handle_event_async(handler, event, route))
                    else:
                        tasks.append(self._handle_event_sync(handler, event))

            # Execute handlers
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Event routing error: {str(e)}")

    async def _handle_event_async(
        self, handler: EventHandler, event: Event, route: EventRoute
    ) -> None:
        """Handle event asynchronously with retries"""
        for attempt in range(route.max_retries + 1):
            try:
                success = await asyncio.wait_for(
                    handler.handle_event(event), timeout=route.timeout_seconds
                )

                if success:
                    handler.processed_count += 1
                    return
                else:
                    handler.error_count += 1
                    if attempt == route.max_retries:
                        logger.warning(
                            f"Handler {handler.handler_id} failed to process event after {route.max_retries} retries"
                        )

            except TimeoutError:
                handler.error_count += 1
                logger.warning(f"Handler {handler.handler_id} timed out processing event")
                if attempt < route.max_retries:
                    await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff

            except Exception as e:
                handler.error_count += 1
                logger.error(f"Handler {handler.handler_id} error: {str(e)}")
                if attempt < route.max_retries:
                    await asyncio.sleep(0.1 * (attempt + 1))

    async def _handle_event_sync(self, handler: EventHandler, event: Event) -> None:
        """Handle event synchronously"""
        try:
            success = await handler.handle_event(event)
            if success:
                handler.processed_count += 1
            else:
                handler.error_count += 1

        except Exception as e:
            handler.error_count += 1
            logger.error(f"Handler {handler.handler_id} error: {str(e)}")

    async def _metrics_collector(self) -> None:
        """Collect performance metrics"""
        last_processed = 0

        while self.is_running:
            try:
                await asyncio.sleep(10)  # Update every 10 seconds

                # Calculate throughput
                current_processed = self.metrics["events_processed"]
                self.metrics["throughput_per_second"] = (current_processed - last_processed) / 10
                last_processed = current_processed

                # Update queue size
                self.metrics["queue_size"] = self.event_queue.qsize()

                # Log metrics periodically
                if (
                    self.metrics["events_processed"] % 1000 == 0
                    and self.metrics["events_processed"] > 0
                ):
                    logger.info(f"Event bus metrics: {self.metrics}")

            except Exception as e:
                logger.warning(f"Metrics collection error: {str(e)}")

    async def _setup_zmq(self) -> None:
        """Setup ZeroMQ for distributed processing"""
        try:
            # Publisher socket
            self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
            self.zmq_publisher.bind(f"tcp://*:{self.config.zmq_port}")

            # Subscriber socket
            self.zmq_subscriber = self.zmq_context.socket(zmq.SUB)
            self.zmq_subscriber.connect(f"tcp://localhost:{self.config.zmq_port + 1}")
            self.zmq_subscriber.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all

            # Start subscriber task
            asyncio.create_task(self._zmq_subscriber_worker())

            logger.info(
                f"ZeroMQ setup complete on ports {self.config.zmq_port}-{self.config.zmq_port + 1}"
            )

        except Exception as e:
            logger.error(f"ZeroMQ setup failed: {str(e)}")

    async def _zmq_subscriber_worker(self) -> None:
        """ZeroMQ subscriber worker"""
        while self.is_running:
            try:
                # Receive message with timeout
                try:
                    topic, message = await asyncio.wait_for(
                        self.zmq_subscriber.recv_multipart(), timeout=1.0
                    )

                    # Parse and process event
                    event_data = json.loads(message.decode())

                    # Reconstruct event
                    event = Event(
                        event_id=event_data["event_id"],
                        event_type=EventType(event_data["event_type"]),
                        source=EventSource(event_data["source"]),
                        timestamp=pd.Timestamp.fromisoformat(event_data["timestamp"]),
                        data=event_data["data"],
                        priority=EventPriority(event_data["priority"]),
                        correlation_id=event_data.get("correlation_id"),
                    )

                    # Route to local handlers
                    await self._route_event(event)

                except TimeoutError:
                    continue

            except Exception as e:
                logger.warning(f"ZMQ subscriber error: {str(e)}")
                await asyncio.sleep(1)

    def get_metrics(self) -> dict[str, Any]:
        """Get event bus performance metrics"""
        return {
            "bus_metrics": self.metrics.copy(),
            "handler_metrics": {
                hid: handler.get_metrics() for hid, handler in self.handlers.items()
            },
            "event_store_stats": self.event_store.get_stats(),
            "cep_stats": self.complex_event_processor.get_pattern_stats(),
            "subscriptions": {
                et.value: len(handlers) for et, handlers in self.subscriptions.items()
            },
            "routes": len(self.routes),
            "queue_size": self.event_queue.qsize(),
            "is_running": self.is_running,
        }


# Example Event Handlers
class MarketDataEventHandler(EventHandler):
    """Handler for market data events"""

    def __init__(self) -> None:
        super().__init__("market_data_handler")

    def get_supported_events(self) -> list[EventType]:
        return [
            EventType.MARKET_DATA_TICK,
            EventType.MARKET_DATA_QUOTE,
            EventType.MARKET_DATA_TRADE,
        ]

    async def handle_event(self, event: Event) -> bool:
        """Handle market data event"""
        try:
            if event.event_type == EventType.MARKET_DATA_TICK:
                # Process tick data
                symbol = event.data.get("symbol")
                price = event.data.get("price")
                logger.debug(f"Processed tick for {symbol}: ${price}")

            elif event.event_type == EventType.MARKET_DATA_QUOTE:
                # Process quote data
                symbol = event.data.get("symbol")
                bid = event.data.get("bid")
                ask = event.data.get("ask")
                logger.debug(f"Processed quote for {symbol}: {bid}/{ask}")

            elif event.event_type == EventType.MARKET_DATA_TRADE:
                # Process trade data
                symbol = event.data.get("symbol")
                volume = event.data.get("volume")
                logger.debug(f"Processed trade for {symbol}: {volume} shares")

            return True

        except Exception as e:
            logger.error(f"Market data handler error: {str(e)}")
            return False


class OrderEventHandler(EventHandler):
    """Handler for order management events"""

    def __init__(self) -> None:
        super().__init__("order_event_handler")

    def get_supported_events(self) -> list[EventType]:
        return [EventType.ORDER_SUBMITTED, EventType.ORDER_FILLED, EventType.ORDER_CANCELED]

    async def handle_event(self, event: Event) -> bool:
        """Handle order event"""
        try:
            order_id = event.data.get("order_id", "unknown")

            if event.event_type == EventType.ORDER_SUBMITTED:
                logger.info(f"Order submitted: {order_id}")

            elif event.event_type == EventType.ORDER_FILLED:
                fill_qty = event.data.get("quantity", 0)
                fill_price = event.data.get("price", 0)
                logger.info(f"Order filled: {order_id} - {fill_qty} @ ${fill_price}")

            elif event.event_type == EventType.ORDER_CANCELED:
                logger.info(f"Order canceled: {order_id}")

            return True

        except Exception as e:
            logger.error(f"Order handler error: {str(e)}")
            return False


class RiskEventHandler(EventHandler):
    """Handler for risk management events"""

    def __init__(self) -> None:
        super().__init__("risk_event_handler")

    def get_supported_events(self) -> list[EventType]:
        return [EventType.RISK_ALERT, EventType.RISK_LIMIT_BREACHED, EventType.DRAWDOWN_ALERT]

    async def handle_event(self, event: Event) -> bool:
        """Handle risk event"""
        try:
            if event.event_type == EventType.RISK_ALERT:
                risk_type = event.data.get("risk_type", "unknown")
                risk_level = event.data.get("risk_level", 0)
                logger.warning(f"Risk alert: {risk_type} at level {risk_level}")

            elif event.event_type == EventType.RISK_LIMIT_BREACHED:
                limit_type = event.data.get("limit_type", "unknown")
                current_value = event.data.get("current_value", 0)
                limit_value = event.data.get("limit_value", 0)
                logger.critical(
                    f"Risk limit breached: {limit_type} = {current_value} (limit: {limit_value})"
                )

            elif event.event_type == EventType.DRAWDOWN_ALERT:
                drawdown_pct = event.data.get("drawdown_percent", 0)
                logger.warning(f"Drawdown alert: {drawdown_pct:.2%}")

            return True

        except Exception as e:
            logger.error(f"Risk handler error: {str(e)}")
            return False


def create_event_driven_system(
    max_buffer_size: int = 10000,
    enable_persistence: bool = True,
    enable_distributed: bool = False,
    **kwargs,
) -> EventBus:
    """Factory function to create event-driven system"""
    config = EventStreamConfig(
        max_buffer_size=max_buffer_size,
        enable_persistence=enable_persistence,
        distributed_processing=enable_distributed,
        **kwargs,
    )

    return EventBus(config)


# Example usage and testing
async def main() -> None:
    """Example usage of event-driven architecture"""
    print("Event-Driven Architecture for Live Trading Testing")
    print("=" * 55)

    # Create event bus
    event_bus = create_event_driven_system(
        max_buffer_size=5000, enable_persistence=True, batch_size=50
    )

    # Register handlers
    market_data_handler = MarketDataEventHandler()
    order_handler = OrderEventHandler()
    risk_handler = RiskEventHandler()

    event_bus.register_handler(market_data_handler)
    event_bus.register_handler(order_handler)
    event_bus.register_handler(risk_handler)

    # Add complex event pattern
    def detect_high_volume_pattern(events):
        """Detect high volume trading pattern"""
        total_volume = sum(e.data.get("volume", 0) for e in events)
        return total_volume > 100000

    def handle_high_volume_alert(events) -> None:
        """Handle high volume alert"""
        total_volume = sum(e.data.get("volume", 0) for e in events)
        print(f"🚨 High volume detected: {total_volume:,} shares")

    high_volume_pattern = EventPattern(
        pattern_id="high_volume_alert",
        name="High Volume Trading Alert",
        event_types=[EventType.MARKET_DATA_TRADE],
        time_window_seconds=60.0,
        conditions=[detect_high_volume_pattern],
        action=handle_high_volume_alert,
    )

    event_bus.complex_event_processor.add_pattern(high_volume_pattern)

    print("✅ Event bus configured with handlers and patterns")

    # Start event bus
    await event_bus.start(num_workers=2)
    print("✅ Event bus started")

    try:
        # Simulate some events
        print("\n📤 Publishing test events...")

        # Market data events
        for i in range(5):
            market_event = Event(
                event_id="",
                event_type=EventType.MARKET_DATA_TRADE,
                source=EventSource.MARKET_DATA_PIPELINE,
                timestamp=pd.Timestamp.now(),
                data={
                    "symbol": "AAPL",
                    "price": 150.0 + i,
                    "volume": 25000 + i * 10000,
                    "timestamp": time.time(),
                },
            )
            await event_bus.publish(market_event)

        # Order events
        order_event = Event(
            event_id="",
            event_type=EventType.ORDER_SUBMITTED,
            source=EventSource.ORDER_MANAGER,
            timestamp=pd.Timestamp.now(),
            data={"order_id": "ORD_123", "symbol": "AAPL", "quantity": 100, "side": "BUY"},
        )
        await event_bus.publish(order_event)

        fill_event = Event(
            event_id="",
            event_type=EventType.ORDER_FILLED,
            source=EventSource.ORDER_MANAGER,
            timestamp=pd.Timestamp.now(),
            data={"order_id": "ORD_123", "quantity": 100, "price": 151.50},
            correlation_id=order_event.correlation_id,
        )
        await event_bus.publish(fill_event)

        # Risk event
        risk_event = Event(
            event_id="",
            event_type=EventType.RISK_ALERT,
            source=EventSource.RISK_MONITOR,
            timestamp=pd.Timestamp.now(),
            data={"risk_type": "position_concentration", "risk_level": 0.85, "symbol": "AAPL"},
            priority=EventPriority.HIGH,
        )
        await event_bus.publish(risk_event)

        # Wait for processing
        await asyncio.sleep(2)

        # Get metrics
        metrics = event_bus.get_metrics()
        print("\n📊 Event Bus Metrics:")
        print(f"   Events published: {metrics['bus_metrics']['events_published']}")
        print(f"   Events processed: {metrics['bus_metrics']['events_processed']}")
        print(f"   Average latency: {metrics['bus_metrics']['avg_latency_ms']:.2f}ms")
        print(f"   Throughput: {metrics['bus_metrics']['throughput_per_second']:.1f} events/sec")
        print(f"   Active handlers: {len(metrics['handler_metrics'])}")
        print(f"   CEP patterns: {metrics['cep_stats']['total_patterns']}")

        # Show event store stats
        store_stats = metrics["event_store_stats"]
        print("\n📚 Event Store Stats:")
        print(f"   Total events stored: {store_stats['total_events']}")
        print(f"   Events by type: {store_stats['events_by_type']}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")

    finally:
        await event_bus.stop()
        print("🛑 Event bus stopped")

    print("\n🚀 Event-Driven Architecture ready for production!")


if __name__ == "__main__":
    asyncio.run(main())
