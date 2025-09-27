"""
Event system for component communication.

This provides a publish-subscribe mechanism for loose coupling between components.
Components can emit events and subscribe to events from other components without
direct dependencies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
import uuid


class EventType(Enum):
    """Types of system events."""
    # Data events
    DATA_RECEIVED = "data_received"
    DATA_ERROR = "data_error"
    
    # Signal events
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_CANCELLED = "signal_cancelled"
    
    # Order events
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIAL = "order_partial"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    
    # Trade events
    TRADE_EXECUTED = "trade_executed"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    
    # Risk events
    RISK_LIMIT_BREACH = "risk_limit_breach"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    TAKE_PROFIT_TRIGGERED = "take_profit_triggered"
    MARGIN_CALL = "margin_call"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    COMPONENT_ERROR = "component_error"
    HEARTBEAT = "heartbeat"


@dataclass
class Event:
    """Base event class."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.HEARTBEAT
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'data': self.data,
            'metadata': self.metadata
        }


@dataclass
class DataEvent(Event):
    """Market data event."""
    def __init__(self, symbol: str, data: Any, source: str = "data_provider"):
        super().__init__(
            event_type=EventType.DATA_RECEIVED,
            source=source,
            data=data,
            metadata={'symbol': symbol}
        )


@dataclass
class SignalEvent(Event):
    """Trading signal event."""
    def __init__(self, signal: Any, source: str):
        super().__init__(
            event_type=EventType.SIGNAL_GENERATED,
            source=source,
            data=signal
        )


@dataclass
class OrderEvent(Event):
    """Order event."""
    def __init__(self, order: Any, event_type: EventType, source: str = "executor"):
        super().__init__(
            event_type=event_type,
            source=source,
            data=order
        )


@dataclass
class TradeEvent(Event):
    """Trade execution event."""
    def __init__(self, trade: Any, source: str = "executor"):
        super().__init__(
            event_type=EventType.TRADE_EXECUTED,
            source=source,
            data=trade
        )


@dataclass
class RiskEvent(Event):
    """Risk management event."""
    def __init__(self, risk_type: str, details: Dict[str, Any], source: str = "risk_manager"):
        super().__init__(
            event_type=EventType.RISK_LIMIT_BREACH,
            source=source,
            data=details,
            metadata={'risk_type': risk_type}
        )


class EventBus:
    """
    Central event bus for component communication.
    
    Components can publish events and subscribe to specific event types.
    This enables loose coupling between components.
    """
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history: int = 1000
    
    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
            except ValueError:
                pass  # Callback not in list
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
        """
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Notify subscribers
        if event.event_type in self._subscribers:
            for callback in self._subscribers[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    # Log error but don't stop event propagation
                    error_event = Event(
                        event_type=EventType.COMPONENT_ERROR,
                        source="event_bus",
                        data=str(e),
                        metadata={'original_event': event.event_id}
                    )
                    # Avoid infinite recursion
                    if event.event_type != EventType.COMPONENT_ERROR:
                        self.publish(error_event)
    
    def get_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_type: Filter by event type (None for all)
            limit: Maximum number of events to return
            
        Returns:
            List of events
        """
        history = self._event_history
        if event_type:
            history = [e for e in history if e.event_type == event_type]
        return history[-limit:]
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
    
    def get_subscriber_count(self, event_type: EventType) -> int:
        """Get number of subscribers for an event type."""
        return len(self._subscribers.get(event_type, []))


# Global event bus instance
_event_bus = EventBus()


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    return _event_bus