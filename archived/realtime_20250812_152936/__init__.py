"""
Real-time Systems Module

High-performance real-time capabilities:
- Market data streaming pipelines
- WebSocket connections
- Event-driven processing
- Data aggregation and normalization
- Multi-source data fusion
"""

from .data_pipeline import (
    DataAggregator,
    DataBuffer,
    DataSource,
    MarketDataStream,
    MarketTick,
    RealtimePipeline,
    SimulatedDataStream,
    StreamConfig,
    StreamType,
)

__all__ = [
    "DataSource",
    "StreamType",
    "StreamConfig",
    "MarketTick",
    "DataBuffer",
    "MarketDataStream",
    "SimulatedDataStream",
    "DataAggregator",
    "RealtimePipeline",
]
