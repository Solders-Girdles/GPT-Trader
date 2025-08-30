"""
Core interfaces and contracts for GPT-Trader V2.

This module defines the fundamental interfaces that all components must implement.
This ensures clean separation of concerns and allows components to be developed
and tested independently while guaranteeing they'll work together.
"""

from .interfaces import (
    IDataProvider,
    IStrategy, 
    IRiskManager,
    IPortfolioAllocator,
    IExecutor,
    IAnalytics,
    IBacktester,
    Component,
    ComponentConfig
)

from .types import (
    MarketData,
    Signal,
    Position,
    Order,
    Trade,
    Portfolio,
    RiskMetrics,
    PerformanceMetrics
)

from .events import (
    Event,
    EventType,
    EventBus,
    DataEvent,
    SignalEvent,
    OrderEvent,
    TradeEvent,
    RiskEvent,
    get_event_bus
)

from .registry import (
    ComponentRegistry,
    get_component,
    register_component,
    list_components,
    get_registry
)

__all__ = [
    # Interfaces
    'IDataProvider',
    'IStrategy',
    'IRiskManager', 
    'IPortfolioAllocator',
    'IExecutor',
    'IAnalytics',
    'IBacktester',
    'Component',
    'ComponentConfig',
    
    # Types
    'MarketData',
    'Signal',
    'Position',
    'Order',
    'Trade',
    'Portfolio',
    'RiskMetrics',
    'PerformanceMetrics',
    
    # Events
    'Event',
    'EventType',
    'EventBus',
    'DataEvent',
    'SignalEvent',
    'OrderEvent',
    'TradeEvent',
    'RiskEvent',
    'get_event_bus',
    
    # Registry
    'ComponentRegistry',
    'get_component',
    'register_component',
    'list_components',
    'get_registry'
]