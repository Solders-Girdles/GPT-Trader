# Code Organization Best Practices for GPT-Trader

This document provides comprehensive best practices for Python project organization that specifically address the pain points identified in the GPT-Trader codebase analysis. These guidelines focus on practical, actionable recommendations that improve code navigation for both humans and AI agents.

## Table of Contents

1. [Refactoring Workflow: Decompose → Deduplicate → Recompose](#1-refactoring-workflow-decompose--deduplicate--recompose)
2. [Addressing Monolithic Core Files](#2-addressing-monolithic-core-files)
3. [Consistent Module Organization Patterns](#3-consistent-module-organization-patterns)
4. [Appropriate Directory Nesting Levels](#4-appropriate-directory-nesting-levels)
5. [Clear Abstraction Layer Separation](#5-clear-abstraction-layer-separation)
6. [Managing Legacy Code and Deprecation](#6-managing-legacy-code-and-deprecation)
7. [Configuration Organization](#7-configuration-organization)
8. [Dependency Management Patterns](#8-dependency-management-patterns)
9. [Applying Best Practices to GPT-Trader](#9-applying-best-practices-to-gpt-trader)

---

## 1. Refactoring Workflow: Decompose → Deduplicate → Recompose

We are standardising every large refactor around an explicit three-phase loop. Each phase has acceptance gates that must be met before moving forward.

### 1.1 Phase 1 – Decompose
- Split any multi-purpose module into a package of narrowly scoped files, even if doing so introduces temporary duplication.
- Extract compatibility shims immediately so existing import paths keep working.
- Add module-level docstrings that record the original file, the extraction date, and TODOs for follow-up cleanup.
- Gate: the original monolithic file should become a thin façade (or disappear) with all logic living in the new package.

### 1.2 Phase 2 – Deduplicate & Rename
- Once the new package exists, identify copy/pasted blocks introduced during extraction and fold them into shared helpers.
- Align naming with our emerging domain vocabulary (e.g. `manager/`, `coordinator/`, `shared_utils/`).
- Create README-style `__init__.py` files that re-export the public surface with explanatory comments.
- Gate: packages expose a single intent, have no redundant helpers, and tests reference the new names.

### 1.3 Phase 3 – Recompose with Clarity
- Re-introduce higher level orchestrators using the new, smaller building blocks.
- Document dependency directions with short “import ladders” at the top of modules.
- Add focused tests per submodule plus integration coverage for the recomposed workflow.
- Gate: the package has a clear entry point, bounded dependencies, and accompanying tests/specs.

This workflow applies to every subsystem we touch until the codebase is consistently modular. The remaining sections describe the guardrails that should be applied to each phase.

---

## 2. Addressing Monolithic Core Files

### Problem Statement
The GPT-Trader codebase contains monolithic files with excessive responsibilities, most notably the 786-line `PerpsBot` class with a 670-line constructor. This creates maintenance, testing, and cognitive overhead challenges.

### Best Practices

#### 1.1 Single Responsibility Principle (SRP) Enforcement
- **File Size Limits**: Establish maximum file sizes of 300-500 lines per module
- **Class Size Limits**: Target maximum of 200-300 lines per class
- **Method Size Limits**: Keep methods under 50 lines (25 lines ideal)
- **Constructor Complexity**: Limit constructors to 50 lines maximum

```python
# ❌ AVOID: Monolithic class with multiple responsibilities
class PerpsBot:  # 786 lines
    def __init__(self, ...):  # 670 lines
        # Environment parsing
        # Storage setup
        # Dependency creation
        # Configuration validation
        # Symbol normalization
        # Service initialization
        # And more...

# ✅ PREFER: Focused classes with single responsibilities
class BotBuilder:  # 150 lines
    def build_dependencies(self):  # 30 lines
        pass

    def create_coordinators(self):  # 40 lines
        pass

class SymbolNormalizer:  # 80 lines
    def normalize_symbols(self, symbols):  # 25 lines
        pass

class RuntimePathResolver:  # 60 lines
    def resolve_paths(self, config):  # 20 lines
        pass
```

#### 1.2 Extract Method Pattern for Complex Initialization
Break down complex initialization into focused builder classes:

```python
# ✅ PREFER: Builder pattern for complex object construction
class PerpsBotBuilder:
    def __init__(self):
        self._config = None
        self._dependencies = {}
        self._coordinators = {}

    def with_config(self, config: BotConfig) -> 'PerpsBotBuilder':
        self._config = config
        return self

    def with_broker(self, broker: IBrokerage) -> 'PerpsBotBuilder':
        self._dependencies['broker'] = broker
        return self

    def with_risk_manager(self, risk_manager: RiskManager) -> 'PerpsBotBuilder':
        self._dependencies['risk_manager'] = risk_manager
        return self

    def build(self) -> 'PerpsBot':
        # Validate all dependencies are present
        self._validate_dependencies()

        # Create bot instance with clean constructor
        return PerpsBot(
            config=self._config,
            **self._dependencies
        )

    def _validate_dependencies(self):
        required = ['config', 'broker', 'risk_manager']
        missing = [key for key in required if key not in self._dependencies]
        if missing:
            raise ValueError(f"Missing required dependencies: {missing}")
```

#### 1.3 Composition Root Pattern
Implement a composition root that manages object graph construction:

```python
# ✅ PREFER: Composition root for dependency management
class BotCompositionRoot:
    def __init__(self, config: BotConfig):
        self.config = config
        self._service_container = ServiceContainer()

    def create_bot(self) -> 'PerpsBot':
        # Build all dependencies in proper order
        broker = self._create_broker()
        risk_manager = self._create_risk_manager()
        orchestrator = self._create_orchestrator()

        # Create bot with all dependencies injected
        return PerpsBot(
            config=self.config,
            broker=broker,
            risk_manager=risk_manager,
            orchestrator=orchestrator
        )
```

---

## 3. Consistent Module Organization Patterns

### Problem Statement
The GPT-Trader codebase exhibits inconsistent module organization patterns with mixed structural approaches, making navigation difficult for both humans and AI agents.

### Best Practices

#### 2.1 Standardized Directory Structure
Adopt a consistent directory hierarchy based on domain boundaries:

```
src/
├── bot_v2/
│   ├── core/                    # Core business logic and entities
│   │   ├── models/              # Domain models
│   │   ├── services/             # Business services
│   │   └── repositories/         # Data access interfaces
│   ├── infrastructure/            # External concerns
│   │   ├── brokers/             # Broker implementations
│   │   ├── storage/             # Database/file storage
│   │   └── monitoring/           # Metrics and logging
│   ├── application/               # Application services
│   │   ├── coordinators/        # Orchestration logic
│   │   ├── strategies/           # Trading strategies
│   │   └── use_cases/           # Specific use cases
│   └── interfaces/                # Public interfaces
└── shared/                       # Shared utilities
    ├── datetime_helpers.py
    ├── parsing.py
    └── trading_operations.py
```

#### 2.2 Naming Conventions
Establish consistent naming patterns across modules:

```python
# ✅ PREFER: Consistent naming with clear purpose
# Models: Domain entities (PascalCase)
class Order:
class Position:
class Account:

# Services: Business logic (PascalCase + "Service")
class OrderService:
class PositionService:
class RiskManagementService:

# Repositories: Data access (PascalCase + "Repository")
class OrderRepository:
class PositionRepository:

# Coordinators: Orchestration (PascalCase + "Coordinator")
class TradingCoordinator:
class RiskCoordinator:
class ExecutionCoordinator:

# Strategies: Trading logic (PascalCase + "Strategy")
class MomentumStrategy:
class MeanReversionStrategy:
class ArbitrageStrategy:
```

#### 2.3 Module Interface Standards
Define clear interfaces for all major components:

```python
# ✅ PREFER: Clear interfaces for all major components
from abc import ABC, abstractmethod
from typing import Protocol

class IBrokerage(Protocol):
    """Interface for brokerage implementations"""

    @abstractmethod
    def place_order(self, order: Order) -> Order:
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        pass

class IRiskManager(Protocol):
    """Interface for risk management"""

    @abstractmethod
    def evaluate_order(self, order: Order) -> RiskAssessment:
        pass

    @abstractmethod
    def check_limits(self) -> LimitStatus:
        pass

class IStrategy(Protocol):
    """Interface for trading strategies"""

    @abstractmethod
    def generate_signal(self, market_data: MarketData) -> Signal:
        pass
```

---

## 4. Appropriate Directory Nesting Levels

### Problem Statement
The GPT-Trader codebase has deep nesting with inconsistent structure, creating navigation complexity and making it difficult to understand the codebase architecture.

### Best Practices

#### 3.1 Maximum Nesting Depth
- **Limit directory nesting to maximum 4 levels** from source root
- **Prefer horizontal over vertical nesting** when possible
- **Use package namespaces** instead of deep directory structures

```
# ❌ AVOID: Excessive nesting
src/bot_v2/features/live_trade/strategies/perps_baseline_enhanced/signals.py
# Level 6 nesting: src -> bot_v2 -> features -> live_trade -> strategies -> perps_baseline_enhanced -> signals

# ✅ PREFER: Shallow structure with clear boundaries
src/bot_v2/features/
├── live_trade/
│   ├── strategies/
│   │   ├── baseline/
│   │   │   ├── signals.py
│   │   │   ├── state.py
│   │   │   └── strategy.py
│   │   └── enhanced/
│   │       ├── signals.py
│   │       ├── state.py
│   │       └── strategy.py
```

#### 3.2 Domain-Based Organization
Organize by business domain rather than technical layers:

```python
# ✅ PREFER: Domain-based organization
src/bot_v2/domains/
├── trading/
│   ├── execution/
│   │   ├── order_service.py
│   │   ├── position_service.py
│   │   └── risk_service.py
├── risk/
│   ├── limits/
│   │   ├── position_limits.py
│   │   └── portfolio_limits.py
└── analytics/
    ├── performance/
    │   ├── metrics_collector.py
    │   └── reporting.py
```

#### 3.3 Feature-Based Organization
For feature development, organize by capability:

```python
# ✅ PREFER: Feature-based organization
src/bot_v2/features/
├── order_management/
│   ├── order_service.py
│   ├── order_repository.py
│   └── order_models.py
├── risk_management/
│   ├── risk_service.py
│   ├── limit_checker.py
│   └── risk_models.py
└── analytics/
    ├── metrics_service.py
    └── reporting_service.py
```

---

## 5. Clear Abstraction Layer Separation

### Problem Statement
The GPT-Trader codebase has mixed abstraction levels with inconsistent boundaries between domain logic, infrastructure, and application services.

### Best Practices

#### 4.1 Clean Architecture Layers
Implement clear separation between architectural layers:

```python
# ✅ PREFER: Clean architecture with clear layer boundaries
# Domain Layer: Pure business logic, no external dependencies
src/domain/
├── entities/
│   ├── order.py
│   ├── position.py
│   └── account.py
├── services/
│   ├── order_service.py
│   ├── position_service.py
│   └── risk_service.py
└── repositories/
    ├── order_repository.py
    └── position_repository.py

# Application Layer: Orchestrates domain objects
src/application/
├── services/
│   ├── trading_service.py
│   └── risk_service.py
├── use_cases/
│   ├── place_order_use_case.py
│   └── close_position_use_case.py
└── interfaces/
    └── external_services.py

# Infrastructure Layer: External concerns
src/infrastructure/
├── brokers/
│   ├── coinbase_broker.py
│   └── mock_broker.py
├── storage/
│   ├── database_storage.py
│   └── file_storage.py
└── monitoring/
    ├── metrics.py
    └── logging.py
```

#### 4.2 Dependency Direction Rules
Enforce strict dependency direction:

```python
# ✅ PREFER: Dependencies point inward
# Domain Layer (no dependencies on outer layers)
class OrderService:
    def __init__(self, order_repository: IOrderRepository):
        self._order_repository = order_repository

    def place_order(self, order_data: dict) -> Order:
        # Pure business logic
        order = Order(**order_data)
        self._validate_order(order)
        return self._order_repository.save(order)

# Application Layer (depends on domain)
class TradingApplication:
    def __init__(self, order_service: OrderService):
        self._order_service = order_service

    def execute_trade(self, trade_data: dict):
        # Orchestrates domain objects
        return self._order_service.place_order(trade_data)

# Infrastructure Layer (depends on application interfaces)
class CoinbaseBroker:
    def __init__(self, order_repository: IOrderRepository):
        self._order_repository = order_repository

    def place_order(self, order: Order):
        # External implementation
        return self._api_client.place_order(order.to_dict())
```

#### 4.3 Interface Segregation Principle
Define focused, cohesive interfaces:

```python
# ✅ PREFER: Focused interfaces following ISP
from typing import Protocol

# ❌ AVOID: Large, monolithic interfaces
class ITradingSystem(Protocol):
    def place_order(self, order: Order) -> Order: pass
    def cancel_order(self, order_id: str) -> bool: pass
    def get_positions(self) -> List[Position]: pass
    def get_account(self) -> Account: pass
    def calculate_risk(self, portfolio: Portfolio) -> RiskMetrics: pass
    def get_market_data(self, symbol: str) -> MarketData: pass
    def send_notification(self, message: str) -> None: pass

# ✅ PREFER: Focused, cohesive interfaces
class IOrderService(Protocol):
    def place_order(self, order: Order) -> Order: pass
    def cancel_order(self, order_id: str) -> bool: pass
    def get_order_history(self, account_id: str) -> List[Order]: pass

class IPositionService(Protocol):
    def get_positions(self, account_id: str) -> List[Position]: pass
    def update_position(self, position: Position) -> Position: pass
    def close_position(self, position_id: str) -> bool: pass

class IMarketDataService(Protocol):
    def get_market_data(self, symbol: str) -> MarketData: pass
    def subscribe_to_updates(self, symbol: str, callback: Callable) -> None: pass
```

---

## 6. Managing Legacy Code and Deprecation

### Problem Statement
The GPT-Trader codebase contains legacy code patterns and compatibility layers that create confusion and maintenance overhead.

### Best Practices

#### 5.1 Legacy Code Containment
Isolate legacy code with clear boundaries:

```python
# ✅ PREFER: Clear legacy code containment
src/
├── bot_v2/                    # Current active code
├── legacy/                      # Deprecated code clearly marked
│   ├── README.md               # Documentation of deprecation timeline
│   ├── v1/                     # Old version
│   │   ├── old_trading_bot.py
│   │   └── deprecated_utils.py
│   └── v2/                     # Previous version
│       ├── older_bot.py
│       └── old_models.py
└── archived/                     # Code no longer maintained
    └── old_experiments/
```

#### 5.2 Deprecation Strategy
Implement clear deprecation pathways:

```python
# ✅ PREFER: Clear deprecation with migration path
# In legacy/README.md
# Legacy Code Deprecation Plan

## Timeline
- **Phase 1** (Current): Mark deprecated with warnings
- **Phase 2** (Next 3 months): Update imports to use compatibility layer
- **Phase 3** (Next 6 months): Remove from active codebase
- **Phase 4** (Beyond 6 months): Archive completely

## Migration Strategy
1. Create compatibility adapters in `bot_v2/compatibility/`
2. Update imports gradually to use adapters
3. Provide clear migration documentation
4. Remove legacy code after deprecation period

## Current Status
- [ ] `old_trading_bot.py` - Deprecated in v2.1.0
- [ ] `deprecated_utils.py` - Deprecated in v2.1.0
- [x] `legacy_models.py` - Migrated in v2.0.0
```

#### 5.3 Compatibility Layers
Create thin compatibility layers for gradual migration:

```python
# ✅ PREFER: Compatibility layers for gradual migration
# bot_v2/compatibility/legacy_adapters.py
class LegacyTradingBotAdapter:
    """Adapter to bridge new architecture with legacy trading bot"""

    def __init__(self, legacy_bot: 'LegacyTradingBot'):
        self._legacy_bot = legacy_bot

    def place_order(self, order: Order) -> Order:
        # Translate new interface to legacy interface
        legacy_order = self._convert_to_legacy_format(order)
        return self._legacy_bot.execute_trade(legacy_order)

    def _convert_to_legacy_format(self, order: Order) -> 'LegacyOrder':
        # Convert new Order to legacy format
        return LegacyOrder(
            symbol=order.symbol,
            quantity=order.quantity,
            order_type=order.order_type.value
        )

# Usage in new code
from bot_v2.compatibility.legacy_adapters import LegacyTradingBotAdapter

# Initialize with legacy adapter for backward compatibility
legacy_bot = LegacyTradingBot()  # Old implementation
adapter = LegacyTradingBotAdapter(legacy_bot)
new_system = ModernTradingSystem(adapter)  # New system uses adapter
```

---

## 7. Configuration Organization

### Problem Statement
The GPT-Trader codebase has configuration complexity with scattered settings and mixed approaches, making it difficult to manage and validate configurations.

### Best Practices

#### 6.1 Configuration Hierarchy
Organize configuration by scope and environment:

```python
# ✅ PREFER: Hierarchical configuration organization
config/
├── environments/                 # Environment-specific configs
│   ├── development.yaml
│   ├── staging.yaml
│   └── production.yaml
├── profiles/                     # Feature-specific configs
│   ├── spot.yaml
│   ├── perps.yaml
│   └── arbitrage.yaml
├── services/                    # External service configs
│   ├── coinbase.yaml
│   └── database.yaml
└── defaults/                     # Default values
    ├── trading.yaml
    └── risk.yaml
```

#### 6.2 Configuration Schema Validation
Implement typed configuration with validation:

```python
# ✅ PREFER: Typed configuration with validation
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum

class TradingConfig(BaseModel):
    """Configuration for trading system with validation"""

    # Core trading settings
    symbols: List[str] = Field(..., min_items=1)
    max_position_size: float = Field(..., gt=0)
    risk_tolerance: float = Field(..., ge=0, le=1.0)

    # Broker settings
    broker_api_key: str = Field(..., min_length=32)
    broker_timeout: int = Field(default=30, gt=0, le=300)

    # Environment-specific overrides
    environment_overrides: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True

    @validator('environment_overrides')
    def validate_environment_overrides(cls, v):
        if v is not None:
            return v
        # Validate environment override structure
        if not isinstance(v, dict):
            raise ValueError("Environment overrides must be a dictionary")
        return v

# Usage with automatic validation
config = TradingConfig.parse_file("config/trading.yaml")
# Raises ValidationError with detailed messages if invalid
```

#### 6.3 Configuration Service Pattern
Implement centralized configuration management:

```python
# ✅ PREFER: Centralized configuration service
class ConfigurationService:
    """Centralized configuration management with change notification"""

    def __init__(self, config_path: str):
        self._config_path = config_path
        self._config = None
        self._change_listeners = []

    def load_config(self) -> TradingConfig:
        """Load configuration with validation"""
        self._config = TradingConfig.parse_file(self._config_path)
        self._notify_config_loaded()
        return self._config

    def get_config(self) -> TradingConfig:
        """Get current configuration"""
        if self._config is None:
            self.load_config()
        return self._config

    def update_config(self, updates: dict) -> None:
        """Update configuration with validation and notification"""
        if self._config is None:
            self.load_config()

        # Apply updates with validation
        updated_config = self._config.copy(update=updates)
        self._config = updated_config

        # Notify listeners of changes
        self._notify_config_changed(updates)

    def add_change_listener(self, listener: Callable[[dict], None]) -> None:
        """Add listener for configuration changes"""
        self._change_listeners.append(listener)

    def _notify_config_loaded(self) -> None:
        for listener in self._change_listeners:
            listener(self._config)

    def _notify_config_changed(self, changes: dict) -> None:
        for listener in self._change_listeners:
            listener(changes)
```

---

## 8. Dependency Management Patterns

### Problem Statement
The GPT-Trader codebase has circular dependencies and tight coupling, making it difficult to test, maintain, and extend components.

### Best Practices

#### 7.1 Dependency Injection Container
Implement a proper DI container:

```python
# ✅ PREFER: Dependency injection container
from typing import Dict, Type, Callable, Any
from abc import ABC, abstractmethod

class DIContainer:
    """Simple dependency injection container"""

    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}

    def register_singleton(self, interface: Type, implementation: Type) -> None:
        """Register a singleton implementation"""
        self._services[interface] = implementation

    def register_factory(self, interface: Type, factory: Callable) -> None:
        """Register a factory for creating instances"""
        self._factories[interface] = factory

    def get(self, interface: Type) -> Any:
        """Get instance of interface"""
        # Check for singleton
        if interface in self._services:
            return self._services[interface]

        # Check for factory
        if interface in self._factories:
            return self._factories[interface]()

        raise ValueError(f"No registration found for {interface}")

# Usage
container = DIContainer()
container.register_singleton(IOrderService, OrderService)
container.register_singleton(IPositionService, PositionService)
container.register_factory(IBroker, lambda config: CoinbaseBroker(config))

order_service = container.get(IOrderService)
```

#### 7.2 Circular Dependency Prevention
Establish clear dependency rules:

```python
# ✅ PREFER: Clear dependency rules to prevent cycles
# Dependency Rule: Dependencies can only point inward (toward domain)
# Domain Layer: No dependencies
# Application Layer: Depends on Domain
# Infrastructure Layer: Depends on Application

# ❌ AVOID: Circular dependencies
class OrderService:
    def __init__(self, position_service: PositionService):
        self._position_service = position_service

class PositionService:
    def __init__(self, order_service: OrderService):  # Creates circular dependency
        self._order_service = order_service

# ✅ PREFER: Proper dependency direction
class OrderService:
    def __init__(self, position_repository: IPositionRepository):
        self._position_repository = position_repository

class PositionService:
    def __init__(self, position_repository: IPositionRepository):
        self._position_repository = position_repository
```

#### 7.3 Interface Segregation
Create focused, cohesive interfaces:

```python
# ✅ PREFER: Interface segregation to reduce coupling
# Instead of large interfaces, create focused ones
class IOrderPlacementService(Protocol):
    def place_order(self, order: Order) -> Order: pass

class IOrderValidationService(Protocol):
    def validate_order(self, order: Order) -> ValidationResult: pass

class IOrderExecutionService(Protocol):
    def execute_order(self, order: Order) -> ExecutionResult: pass

# Implementation depends only on what it needs
class OrderService:
    def __init__(self,
                 placement_service: IOrderPlacementService,
                 validation_service: IOrderValidationService,
                 execution_service: IOrderExecutionService):
        self._placement_service = placement_service
        self._validation_service = validation_service
        self._execution_service = execution_service
```

---

## 9. Applying Best Practices to GPT-Trader

### 8.1 Refactoring PerpsBot Class

The current 786-line `PerpsBot` class should be refactored using the Builder pattern:

```python
# Current monolithic PerpsBot (786 lines)
class PerpsBot:
    def __init__(self, config, registry, event_store, orders_store,
                 session_guard, baseline_snapshot,
                 configuration_guardian=None, container=None):
        # 670 lines of mixed responsibilities
        # Environment parsing, storage setup, dependency creation,
        # Configuration validation, symbol normalization, service initialization

# Refactored using Builder pattern
class PerpsBotBuilder:
    def __init__(self):
        self._config = None
        self._registry = None
        self._event_store = None
        self._orders_store = None
        self._session_guard = None
        self._baseline_snapshot = None
        self._configuration_guardian = None

    def with_config(self, config: BotConfig) -> 'PerpsBotBuilder':
        self._config = config
        return self

    def with_registry(self, registry: ServiceRegistry) -> 'PerpsBotBuilder':
        self._registry = registry
        return self

    def with_event_store(self, event_store: EventStore) -> 'PerpsBotBuilder':
        self._event_store = event_store
        return self

    def with_orders_store(self, orders_store: OrdersStore) -> 'PerpsBotBuilder':
        self._orders_store = orders_store
        return self

    def with_session_guard(self, session_guard: SessionGuard) -> 'PerpsBotBuilder':
        self._session_guard = session_guard
        return self

    def with_baseline_snapshot(self, baseline_snapshot: Any) -> 'PerpsBotBuilder':
        self._baseline_snapshot = baseline_snapshot
        return self

    def with_configuration_guardian(self, guardian: ConfigurationGuardian) -> 'PerpsBotBuilder':
        self._configuration_guardian = guardian
        return self

    def build(self) -> 'PerpsBot':
        # Validate all required components
        self._validate_requirements()

        # Create bot with clean constructor
        return PerpsBot(
            config=self._config,
            registry=self._registry,
            event_store=self._event_store,
            orders_store=self._orders_store,
            session_guard=self._session_guard,
            baseline_snapshot=self._baseline_snapshot,
            configuration_guardian=self._configuration_guardian
        )

    def _validate_requirements(self):
        required = ['config', 'registry', 'event_store', 'orders_store']
        missing = [key for key in required if getattr(self, f'_{key}') is None]
        if missing:
            raise ValueError(f"Missing required components: {missing}")

# Simplified PerpsBot class
class PerpsBot:
    def __init__(self, config: BotConfig, registry: ServiceRegistry,
                 event_store: EventStore, orders_store: OrdersStore,
                 session_guard: SessionGuard, baseline_snapshot: Any,
                 configuration_guardian: ConfigurationGuardian):
        # Clean constructor with single responsibility
        self.config = config
        self.registry = registry
        self.event_store = event_store
        self.orders_store = orders_store
        self.session_guard = session_guard
        self.baseline_snapshot = baseline_snapshot
        self.configuration_guardian = configuration_guardian

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        # Focused initialization methods
        self._setup_coordinators()
        self._setup_services()
        self._setup_monitoring()
```

### 8.2 Reorganizing Directory Structure

Apply consistent module organization to GPT-Trader:

```
src/bot_v2/
├── core/                           # Core domain logic
│   ├── models/
│   │   ├── order.py
│   │   ├── position.py
│   │   ├── account.py
│   │   └── trading_session.py
│   ├── services/
│   │   ├── order_service.py
│   │   ├── position_service.py
│   │   ├── risk_service.py
│   │   └── account_service.py
│   └── repositories/
│       ├── order_repository.py
│       ├── position_repository.py
│       └── account_repository.py
├── infrastructure/                  # External concerns
│   ├── brokers/
│   │   ├── coinbase/
│   │   │   ├── coinbase_broker.py
│   │   │   ├── auth.py
│   │   │   └── websocket_handler.py
│   │   └── mock/
│   │       └── mock_broker.py
│   ├── storage/
│   │   ├── database/
│   │   │   ├── models.py
│   │   │   └── repositories.py
│   │   └── file/
│   │       ├── json_store.py
│   │       └── csv_store.py
│   └── monitoring/
│       ├── metrics.py
│       └── logging.py
├── application/                    # Application services
│   ├── coordinators/
│   │   ├── trading_coordinator.py
│   │   ├── risk_coordinator.py
│   │   └── execution_coordinator.py
│   ├── strategies/
│   │   ├── base_strategy.py
│   │   ├── momentum_strategy.py
│   │   └── mean_reversion_strategy.py
│   └── use_cases/
│       ├── place_order_use_case.py
│       ├── close_position_use_case.py
│       └── get_portfolio_use_case.py
└── interfaces/                      # Public interfaces
    ├── broker_interface.py
    ├── storage_interface.py
    └── strategy_interface.py
```

### 8.3 Implementing Configuration Management

Apply hierarchical configuration with validation:

```python
# config/environments/development.yaml
trading:
  symbols: ["BTC-USD", "ETH-USD"]
  max_position_size: 1.0
  risk_tolerance: 0.02

broker:
  api_key: "${COINBASE_API_KEY}"
  timeout: 30
  sandbox: true

risk:
  max_daily_loss: 1000
  position_size_limit: 5000

# config/environments/production.yaml
trading:
  symbols: ["BTC-USD", "ETH-USD", "SOL-USD"]
  max_position_size: 5.0
  risk_tolerance: 0.01

broker:
  api_key: "${COINBASE_API_KEY}"
  timeout: 60
  sandbox: false

risk:
  max_daily_loss: 10000
  position_size_limit: 50000

# bot_v2/core/config/configuration.py
from pydantic import BaseModel, Field
from typing import List, Optional
import yaml


class TradingSettings(BaseModel):
    symbols: List[str] = Field(..., min_items=1)
    max_position_size: float = Field(..., gt=0)
    risk_tolerance: float = Field(..., ge=0, le=1.0)


class BrokerConfig(BaseModel):
    api_key: str = Field(..., min_length=32)
    timeout: int = Field(default=30, gt=0, le=300)
    sandbox: bool = Field(default=True)


class RiskConfig(BaseModel):
    max_daily_loss: float = Field(..., gt=0)
    position_size_limit: float = Field(..., gt=0)


class BotConfig(BaseModel):
    """Top-level configuration container"""

    trading: TradingSettings
    broker: BrokerConfig
    risk: RiskConfig


class ConfigurationService:
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self._config: Optional[BotConfig] = None

    def load_config(self) -> BotConfig:
        config_path = f"config/environments/{self.environment}.yaml"
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        self._config = BotConfig(**config_data)
        return self._config

    def get_config(self) -> BotConfig:
        if self._config is None:
            self.load_config()
        return self._config
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
1. **Extract core models and services** from PerpsBot
2. **Implement dependency injection container**
3. **Create configuration service** with validation
4. **Set up testing infrastructure** for new structure

### Phase 2: Refactoring (Weeks 3-6)
1. **Refactor PerpsBot** using Builder pattern
2. **Reorganize directory structure** following domain boundaries
3. **Implement interface segregation** for major components
4. **Create compatibility layer** for legacy code

### Phase 3: Integration (Weeks 7-10)
1. **Migrate existing functionality** to new structure
2. **Update all imports** to use new organization
3. **Add comprehensive tests** for refactored components
4. **Update documentation** to reflect new architecture

### Success Metrics
- **File Size Reduction**: Target 50% reduction in largest files
- **Test Coverage**: Maintain >90% during refactoring
- **Import Complexity**: Reduce circular dependencies by 80%
- **Navigation Time**: Improve code location finding by 60%

---

## Conclusion

These best practices provide a comprehensive approach to addressing the specific pain points identified in the GPT-Trader codebase. By implementing these guidelines, the project will achieve:

1. **Improved Maintainability**: Smaller, focused modules with single responsibilities
2. **Enhanced Testability**: Clear interfaces and dependency injection enable better testing
3. **Better Navigation**: Consistent structure and naming make code easier to find
4. **Reduced Complexity**: Proper abstraction layers reduce cognitive load
5. **Clearer Evolution Paths**: Legacy code management and deprecation strategies
6. **Robust Configuration**: Hierarchical, validated configuration management
7. **Flexible Architecture**: Dependency management patterns enable easier extension

The implementation roadmap provides a phased approach that allows for gradual improvement without disrupting ongoing development.