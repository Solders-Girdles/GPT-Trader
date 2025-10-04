# State Repositories Refactor

**Date:** October 2025
**Status:** ✅ Complete
**Pattern:** Extract → Test → Integrate → Validate

## Executive Summary

Successfully refactored the monolithic `src/bot_v2/state/repositories.py` (619 lines) into a clean facade pattern with three tier-specific repository modules. Achieved 88% line reduction in the main file while adding 95 comprehensive unit tests.

### Key Results
- **Main file reduction:** 619 → 74 lines (-88.0%)
- **New tests added:** 134 (151 total passing)
- **Zero regressions:** All baseline tests remain green
- **Modules created:** 3 focused repository implementations
- **Telemetry:** Full Prometheus/Grafana metrics integration

## Motivation

### Problems with Monolithic Approach

The original `repositories.py` combined three distinct storage tier implementations (Redis, PostgreSQL, S3) in a single 619-line file, creating several issues:

1. **Poor Separation of Concerns**
   - HOT, WARM, and COLD tier logic intermingled
   - Difficult to understand tier-specific behavior
   - Changes to one tier risked affecting others

2. **Testing Challenges**
   - All tier logic tested through single test suite
   - Difficult to isolate tier-specific test failures
   - Mocking required understanding of all three implementations

3. **Maintenance Burden**
   - Large file difficult to navigate
   - Tier-specific optimizations required understanding entire file
   - Harder to onboard new developers

4. **Limited Extensibility**
   - Adding a new storage tier required modifying large file
   - Risk of breaking existing tier implementations
   - No clear template for new tier development

## Solution Architecture

### New Module Structure

```
src/bot_v2/state/repositories/
├── __init__.py                 # Facade (74 lines)
│   ├── StateRepository         # Protocol definition
│   └── Re-exports              # Redis, Postgres, S3 repositories
│
├── redis_repository.py         # HOT tier (171 lines)
│   └── RedisStateRepository
│       ├── TTL management
│       ├── Pipeline operations
│       └── JSON serialization
│
├── postgres_repository.py      # WARM tier (216 lines)
│   └── PostgresStateRepository
│       ├── Transaction handling
│       ├── UPSERT operations
│       └── Batch operations
│
└── s3_repository.py           # COLD tier (209 lines)
    └── S3StateRepository
        ├── Prefix management
        ├── Storage class handling
        └── Batch delete operations
```

### Design Principles

1. **Protocol-Based Interface**
   - `StateRepository` Protocol defines common interface
   - All tier repositories implement same 7 methods
   - Ensures consistent API across tiers

2. **Focused Modules**
   - Each repository handles single storage tier
   - Tier-specific optimizations isolated
   - Clear boundaries between implementations

3. **Comprehensive Testing**
   - Each repository has dedicated test suite
   - Tier-specific edge cases thoroughly covered
   - Mock adapters for isolated testing

4. **Clean Dependencies**
   - No circular dependencies
   - Each module imports only what it needs
   - Facade coordinates without coupling

## Implementation Details

### StateRepository Protocol

Defines the contract all tier repositories must implement:

```python
class StateRepository(Protocol):
    async def fetch(self, key: str) -> Any | None
    async def store(self, key: str, value: str, metadata: dict) -> bool
    async def delete(self, key: str) -> bool
    async def keys(self, pattern: str) -> list[str]
    async def stats(self) -> dict[str, Any]
    async def store_many(self, items: dict) -> set[str]
    async def delete_many(self, keys: list[str]) -> int
```

### Tier-Specific Implementations

#### RedisStateRepository (HOT Tier)
**Purpose:** Sub-second access for frequently accessed data

**Key Features:**
- TTL management (default + custom per-key)
- Efficient pipeline operations via `msetex`
- TTL grouping for batch operations
- JSON serialization/deserialization

**Optimizations:**
- Groups `store_many` items by TTL for pipeline efficiency
- Uses `delete_many` for batch deletions

#### PostgresStateRepository (WARM Tier)
**Purpose:** Recent data access with ~5s latency

**Key Features:**
- Transaction management (commit/rollback)
- UPSERT operations (INSERT ... ON CONFLICT UPDATE)
- Automatic timestamp tracking (`last_accessed`)
- Batch upsert via `batch_upsert`

**Optimizations:**
- Transactional batch operations
- Automatic version incrementing
- Checksum and size metadata tracking

#### S3StateRepository (COLD Tier)
**Purpose:** Long-term archival storage

**Key Features:**
- Prefix-based key organization
- STANDARD_IA storage class for cost optimization
- Checksum metadata for integrity
- Batch delete (up to 1000 keys via `delete_objects`)

**Optimizations:**
- Batch delete for efficiency
- Sequential `store_many` (S3 has no batch PUT)
- Helper methods for prefix management

## Telemetry & Observability

### Overview

All three tier repositories support optional metrics collection via `MetricsCollector` dependency injection. This enables Prometheus/Grafana monitoring of repository activity, error rates, and batch operation sizes without impacting performance when disabled.

### Metrics Architecture

**Design Principles:**
- **Zero overhead when disabled**: All metric recording wrapped in `if self.metrics_collector:` checks
- **Optional dependency injection**: `metrics_collector` parameter defaults to `None`
- **TYPE_CHECKING pattern**: Avoids circular imports with `MetricsCollector`
- **Tier-specific namespacing**: Each tier exports metrics under distinct namespace

### Exported Metrics

All metrics follow the naming convention: `state.repository.{tier}.operations.{metric_name}`

#### Redis (HOT Tier) Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `state.repository.redis.operations.fetch_total` | Counter | Total fetch operations |
| `state.repository.redis.operations.store_total` | Counter | Total store operations |
| `state.repository.redis.operations.delete_total` | Counter | Total delete operations |
| `state.repository.redis.operations.store_many_total` | Counter | Total batch store operations |
| `state.repository.redis.operations.delete_many_total` | Counter | Total batch delete operations |
| `state.repository.redis.operations.errors_total` | Counter | Total errors across all operations |
| `state.repository.redis.operations.batch_size` | Histogram | Distribution of batch operation sizes |

#### Postgres (WARM Tier) Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `state.repository.postgres.operations.fetch_total` | Counter | Total fetch operations |
| `state.repository.postgres.operations.store_total` | Counter | Total store (UPSERT) operations |
| `state.repository.postgres.operations.delete_total` | Counter | Total delete operations |
| `state.repository.postgres.operations.store_many_total` | Counter | Total batch UPSERT operations |
| `state.repository.postgres.operations.delete_many_total` | Counter | Total batch delete operations |
| `state.repository.postgres.operations.errors_total` | Counter | Total errors (includes rollbacks) |
| `state.repository.postgres.operations.batch_size` | Histogram | Distribution of batch operation sizes |

#### S3 (COLD Tier) Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `state.repository.s3.operations.fetch_total` | Counter | Total GET operations |
| `state.repository.s3.operations.store_total` | Counter | Total PUT operations |
| `state.repository.s3.operations.delete_total` | Counter | Total single DELETE operations |
| `state.repository.s3.operations.store_many_total` | Counter | Total batch PUT operations |
| `state.repository.s3.operations.delete_many_total` | Counter | Total batch DELETE operations |
| `state.repository.s3.operations.errors_total` | Counter | Total S3 API errors |
| `state.repository.s3.operations.batch_size` | Histogram | Distribution of batch operation sizes |

### Usage Example

```python
from bot_v2.monitoring.metrics_collector import MetricsCollector
from bot_v2.state.state_manager import StateManager

# Create metrics collector
metrics_collector = MetricsCollector()

# Pass to StateManager - automatically propagates to all repositories
state_manager = StateManager(
    config=state_config,
    redis_adapter=redis_adapter,
    postgres_adapter=postgres_adapter,
    s3_adapter=s3_adapter,
    metrics_collector=metrics_collector,  # Optional - omit to disable metrics
)

# All repository operations now export metrics
await state_manager.fetch("position:BTC")
# → Increments state.repository.redis.operations.fetch_total
```

### Prometheus Query Examples

**Hot tier fetch rate (ops/sec):**
```promql
rate(state_repository_redis_operations_fetch_total[1m])
```

**Error rate by tier:**
```promql
sum by (tier) (
  rate(state_repository_redis_operations_errors_total[5m]) or
  rate(state_repository_postgres_operations_errors_total[5m]) or
  rate(state_repository_s3_operations_errors_total[5m])
)
```

**Batch operation sizes (p95):**
```promql
histogram_quantile(0.95,
  rate(state_repository_redis_operations_batch_size_bucket[5m])
)
```

**Total operations across all tiers:**
```promql
sum(
  rate(state_repository_redis_operations_fetch_total[1m]) +
  rate(state_repository_postgres_operations_fetch_total[1m]) +
  rate(state_repository_s3_operations_fetch_total[1m])
)
```

### Grafana Dashboard Recommendations

**Panel 1: Operation Rate by Tier (Time Series)**
- Queries: `rate(state_repository_{tier}_operations_{op}_total[1m])`
- Legend: `{{tier}} - {{operation}}`
- Visualization: Stacked area chart

**Panel 2: Error Rate (Time Series)**
- Query: `rate(state_repository_{tier}_operations_errors_total[5m])`
- Legend: `{{tier}} errors`
- Alert: Trigger if rate > 10/min

**Panel 3: Batch Size Distribution (Heatmap)**
- Query: `state_repository_{tier}_operations_batch_size`
- Visualization: Histogram heatmap
- Helps identify batch operation patterns

**Panel 4: Tier Activity Breakdown (Pie Chart)**
- Query: `sum by (tier) (increase(state_repository_{tier}_operations_fetch_total[1h]))`
- Shows relative activity across HOT/WARM/COLD tiers

### Implementation Details

**Metrics Collection Integration:**

1. **StateManager** accepts optional `metrics_collector` parameter
2. **RepositoryFactory** receives collector and passes to each repository
3. **Repository constructors** accept `metrics_collector: "MetricsCollector | None" = None`
4. **All operations** check `if self.metrics_collector:` before recording
5. **Error paths** always record to `errors_total` counter

**Testing Strategy:**

Each repository test suite includes a `TestMetricsIntegration` class with 13 tests:
- Counter increments for each operation (fetch, store, delete)
- Error counter increments on exceptions
- Batch size histogram recording
- `collector=None` case (ensures no-op behavior)

See test files:
- `tests/unit/bot_v2/state/repositories/test_redis_repository.py::TestRedisRepositoryMetrics`
- `tests/unit/bot_v2/state/repositories/test_postgres_repository.py::TestPostgresRepositoryMetrics`
- `tests/unit/bot_v2/state/repositories/test_s3_repository.py::TestS3RepositoryMetrics`

## Testing Strategy

### Test Coverage by Repository

**RedisStateRepository (39 tests):**
- Store operations: default/custom TTL, failures, logging
- Fetch operations: JSON deserialization, missing keys, exceptions
- Delete operations: success/failure, logging
- Key operations: pattern matching, empty results
- Stats: key count, error handling
- Batch operations: TTL grouping, empty inputs, exceptions
- Metrics integration: counter/histogram recording, error tracking (13 tests)

**PostgresStateRepository (48 tests):**
- Store operations: UPSERT queries, metadata handling, transactions
- Fetch operations: data extraction, timestamp updates, rollback
- Delete operations: query execution, rollback handling
- Key operations: wildcard conversion, pattern matching
- Stats: COUNT queries, empty tables
- Batch operations: upsert record building, transactional rollback
- Metrics integration: counter/histogram recording, error tracking (13 tests)

**S3StateRepository (47 tests):**
- Store operations: prefix handling, storage class, metadata
- Fetch operations: JSON deserialization, missing objects
- Delete operations: prefix usage, error handling
- Key operations: wildcard→prefix conversion, key stripping
- Stats: KeyCount extraction, missing values
- Batch operations: sequential stores, partial failures, batch delete
- Metrics integration: counter/histogram recording, error tracking (13 tests)

### Baseline Tests (17 tests)
- Repository factory tests (creation, configuration)
- Repository access pattern tests (bundle retrieval)

**Total: 151 tests passing (zero regressions)**

## Migration Guide

### For Consumers (No Changes Required)

All existing imports continue to work:

```python
# This still works - no changes needed
from bot_v2.state.repositories import (
    StateRepository,
    RedisStateRepository,
    PostgresStateRepository,
    S3StateRepository,
)
```

The facade re-exports all classes, so consuming code is unaffected.

### For Repository Developers

When modifying a tier-specific repository:

1. **Find the right module:**
   - HOT tier → `redis_repository.py`
   - WARM tier → `postgres_repository.py`
   - COLD tier → `s3_repository.py`

2. **Run tier-specific tests:**
   ```bash
   pytest tests/unit/bot_v2/state/repositories/test_redis_repository.py
   ```

3. **Verify Protocol compliance:**
   - All 7 methods must be implemented
   - Method signatures must match Protocol
   - Return types must match Protocol

## Adding a New Storage Tier

To add a new tier (e.g., MemcachedStateRepository):

### 1. Create Repository Module

Create `src/bot_v2/state/repositories/memcached_repository.py`:

```python
"""
Memcached State Repository - FAST tier storage implementation

Provides distributed caching with microsecond access times.
"""

import logging
from typing import Any

from bot_v2.state.utils.adapters import MemcachedAdapter

logger = logging.getLogger(__name__)

__all__ = ["MemcachedStateRepository"]


class MemcachedStateRepository:
    """
    Memcached repository for FAST tier state storage.

    Provides distributed caching with microsecond access.
    """

    def __init__(self, adapter: MemcachedAdapter) -> None:
        self.adapter = adapter

    async def fetch(self, key: str) -> Any | None:
        """Fetch state from Memcached."""
        # Implementation
        ...

    async def store(self, key: str, value: str, metadata: dict[str, Any]) -> bool:
        """Store state in Memcached."""
        # Implementation
        ...

    # ... implement remaining Protocol methods
```

### 2. Update Facade

Edit `src/bot_v2/state/repositories/__init__.py`:

```python
# Add import
from .memcached_repository import MemcachedStateRepository

# Update __all__
__all__ = [
    "StateRepository",
    "RedisStateRepository",
    "PostgresStateRepository",
    "S3StateRepository",
    "MemcachedStateRepository",  # Add new repository
]
```

### 3. Create Test Suite

Create `tests/unit/bot_v2/state/repositories/test_memcached_repository.py`:

```python
"""Unit tests for MemcachedStateRepository - FAST tier storage."""

import pytest
from unittest.mock import Mock

from bot_v2.state.repositories.memcached_repository import MemcachedStateRepository

# ... implement 15-20 comprehensive tests
```

### 4. Update Repository Factory

Edit `src/bot_v2/state/repository_factory.py` to include creation logic.

### 5. Verify Integration

```bash
# Run all repository tests
pytest tests/unit/bot_v2/state/repositories/ -v

# Verify no regressions in baseline tests
pytest tests/unit/bot_v2/state/test_repository_*.py -v
```

## Metrics

### Before Refactor
- **File:** `src/bot_v2/state/repositories.py` (619 lines)
- **Tests:** 17 baseline tests
- **Structure:** Monolithic (3 implementations in 1 file)

### After Refactor
- **Facade:** `repositories/__init__.py` (74 lines)
- **Modules:** 3 focused implementations (596 lines total)
- **Tests:** 151 total (17 baseline + 134 new)
- **Structure:** Modular (1 implementation per file)
- **Telemetry:** Prometheus metrics on all operations

### Impact
- **Line reduction:** -545 lines (-88.0%) in main file
- **Test coverage:** +788% increase (17 → 151 tests)
- **Modularity:** 3 focused modules vs 1 monolith
- **Observability:** Full Prometheus/Grafana integration
- **Maintainability:** Significantly improved

## Lessons Learned

### What Worked Well

1. **Phase-by-phase approach**
   - Extracted one repository at a time
   - Validated tests after each phase
   - Easy to track progress and identify issues

2. **Comprehensive testing**
   - 95 new tests provided safety net
   - Caught edge cases during extraction
   - Enabled confident refactoring

3. **Protocol-based interface**
   - Clear contract for all implementations
   - Easy to verify compliance
   - Simplified facade design

4. **Mock-based testing**
   - Isolated unit tests for each tier
   - Fast test execution
   - No external dependencies

### Challenges Overcome

1. **Import restructuring**
   - Solution: Package-based structure with clean re-exports
   - Maintained backward compatibility

2. **Test organization**
   - Solution: Dedicated test file per repository
   - Clear test categorization by operation type

3. **Dependency management**
   - Solution: Minimal imports, no circular dependencies
   - Each module self-contained

## Future Enhancements

### Potential Improvements

1. **Async adapter support**
   - Current adapters are synchronous
   - Could benefit from true async I/O

2. **Latency tracking**
   - Add operation duration histograms
   - Track p50/p95/p99 latencies per tier

3. **Compression support**
   - Add optional compression for S3 storage
   - Reduce storage costs for large objects

4. **Connection pooling**
   - Enhance PostgreSQL connection management
   - Add Redis connection pooling

### Extension Points

1. **Additional tiers**
   - Memcached for ultra-fast caching
   - DynamoDB for serverless deployments
   - Local file system for development

2. **Tiering policies**
   - Automatic data migration between tiers
   - Age-based tier transitions
   - Access pattern-based optimization

3. **Replication strategies**
   - Cross-region S3 replication
   - Redis replica support
   - PostgreSQL read replicas

## References

- **Original file:** `src/bot_v2/state/repositories.py` (now replaced)
- **Progress log:** `/tmp/repositories_refactor_progress.md`
- **Related docs:** `docs/architecture/state_refactoring_opportunities.md`

## Conclusion

The State Repositories refactor successfully decomposed a monolithic 619-line file into a clean, maintainable architecture with focused modules and comprehensive test coverage. The 88% line reduction in the main file, combined with 95 new tests, demonstrates significant improvements in code quality and maintainability.

The Extract → Test → Integrate → Validate pattern proved highly effective and can be applied to other monolithic modules in the codebase. Each repository now lives in a dedicated, well-tested module, making the codebase more maintainable, testable, and extensible.
