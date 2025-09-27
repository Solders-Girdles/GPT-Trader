"""
Bot V2 Orchestration Layer

This module provides the core orchestration functionality to connect
all 11 feature slices into a unified trading system.

Components:
- orchestrator.py: Core TradingOrchestrator class
- registry.py: Dynamic slice discovery and management
- adapters.py: Interface adapters for slice standardization
- types.py: Type definitions for orchestration
- config.py: Configuration management
"""

from .orchestrator import TradingOrchestrator, OrchestratorConfig, TradingMode
from .registry import SliceRegistry
from .adapters import AdapterFactory

__all__ = [
    'TradingOrchestrator',
    'OrchestratorConfig', 
    'TradingMode',
    'SliceRegistry',
    'AdapterFactory'
]