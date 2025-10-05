# Orchestration Layer Analysis

**Generated**: 2025-10-05
**Purpose**: Identify refactoring opportunities in orchestration layer

---

## Summary

**Total Modules**: 37
**Total Lines**: 7,931
**Total Classes**: 44

### Complexity Indicators

- **Average Lines per Module**: 214
- **Largest Module**: streaming_service (477 lines)
- **Most Imports**: builders.perps_bot_builder (20 imports)

---

## Hotspot Analysis

Modules ranked by complexity (lines + classes*100 + imports*10):

| Rank | Module | Complexity | Lines | Classes | Imports |
|------|--------|------------|-------|---------|---------|
| 1 | `configuration` | 903 | 473 | 4 | 3 |
| 2 | `builders.perps_bot_builder` | 680 | 380 | 1 | 20 |
| 3 | `perps_bot` | 671 | 391 | 1 | 18 |
| 4 | `live_execution` | 652 | 402 | 2 | 5 |
| 5 | `lifecycle_service` | 603 | 293 | 3 | 1 |
| 6 | `runtime_coordinator` | 600 | 340 | 2 | 6 |
| 7 | `streaming_service` | 597 | 477 | 1 | 2 |
| 8 | `execution.guards` | 569 | 369 | 2 | 0 |
| 9 | `strategy_orchestrator` | 504 | 334 | 1 | 7 |
| 10 | `execution.order_placement` | 492 | 382 | 1 | 1 |
| 11 | `order_reconciler` | 460 | 260 | 2 | 0 |
| 12 | `deterministic_broker` | 457 | 357 | 1 | 0 |
| 13 | `guardrails` | 441 | 241 | 2 | 0 |
| 14 | `system_monitor` | 414 | 284 | 1 | 3 |
| 15 | `execution.validation` | 373 | 273 | 1 | 0 |

---

## Dependency Graph

### Core Dependencies (Most Imported)

- **configuration**: imported by 16 modules
- **service_registry**: imported by 7 modules
- **config_controller**: imported by 5 modules
- **perps_bot**: imported by 4 modules
- **live_execution**: imported by 4 modules
- **execution_coordinator**: imported by 3 modules
- **market_data_service**: imported by 3 modules
- **market_monitor**: imported by 3 modules
- **strategy_orchestrator**: imported by 3 modules
- **order_reconciler**: imported by 3 modules

### Circular Dependencies

✅ No circular dependencies detected

---

## Extraction Candidates

Modules that could be extracted to separate features:

### `system_monitor`
- **Lines**: 284
- **Dependencies**: 3 orchestration imports
- **Reason**: Domain-specific, low coupling

### `equity_calculator`
- **Lines**: 117
- **Dependencies**: 0 orchestration imports
- **Reason**: Domain-specific, low coupling

### `market_data_service`
- **Lines**: 131
- **Dependencies**: 0 orchestration imports
- **Reason**: Domain-specific, low coupling

### `symbols`
- **Lines**: 19
- **Dependencies**: 3 orchestration imports
- **Reason**: Domain-specific, low coupling

### `market_monitor`
- **Lines**: 68
- **Dependencies**: 0 orchestration imports
- **Reason**: Domain-specific, low coupling

### `streaming_service`
- **Lines**: 477
- **Dependencies**: 2 orchestration imports
- **Reason**: Domain-specific, low coupling

---

## Refactoring Recommendations

### High Priority

1. **configuration** (473 lines)
   - Split into smaller modules
   - Extract 4 classes to separate files

2. **builders.perps_bot_builder** (380 lines)
   - Split into smaller modules
   - Extract 1 classes to separate files

3. **perps_bot** (391 lines)
   - Split into smaller modules
   - Extract 1 classes to separate files

### Medium Priority

- **Extract Domain Modules**: Move domain-specific modules to features/
  - Market data → `features/market_data/`
  - Streaming → `features/streaming/`
  - Monitoring → Already in `monitoring/` (good!)

- **Reduce Core Dependencies**: Modules with >5 orchestration imports should be reviewed

### Low Priority

- **Consolidate Similar Modules**: Consider merging small, related modules
- **Documentation**: Add READMEs to execution/ subdir

---

## Proposed Structure (After Refactoring)

```
orchestration/
├── core/                    # Core orchestration logic
│   ├── bootstrap.py
│   ├── lifecycle.py
│   └── coordinator.py
├── services/                # Service management
│   ├── registry.py
│   ├── rebinding.py
│   └── telemetry.py
├── execution/               # Order execution (existing)
│   ├── ...
├── strategy/                # Strategy orchestration
│   ├── orchestrator.py
│   ├── executor.py
│   └── registry.py
└── config/                  # Configuration
    ├── controller.py
    └── models.py

# Extracted to features/
features/
├── market_data/
├── streaming/
└── guards/                  # Risk gates
```

---

**Next Steps**: Create detailed refactoring plan in `docs/architecture/orchestration_refactor.md`
