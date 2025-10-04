# Strategy Orchestrator Refactor - Phase 0-4 Complete

**Date:** October 2025
**Status:** ✅ Complete
**Original File:** `src/bot_v2/orchestration/strategy_orchestrator.py` (411 lines)
**Final Size:** 332 lines (-79 lines, -19.2% reduction)

## Overview

Refactored StrategyOrchestrator from a 411-line class into a lean orchestrator delegating to four specialized services. This improves testability, separation of concerns, and code maintainability while maintaining backward compatibility.

## Architecture

### Before: Monolithic StrategyOrchestrator
```
strategy_orchestrator.py (411 lines)
├── Strategy initialization (SPOT vs PERPS)
├── Equity calculation (cash + position value)
├── Risk gate validation (circuit breaker, staleness)
├── Strategy evaluation (timing, telemetry)
├── Decision recording
└── SPOT filter application (120 lines)
```

### After: Orchestrator + Specialized Services
```
StrategyOrchestrator (332 lines) - Main Orchestrator
├── EquityCalculator (120 lines)
│   ├── Cash balance extraction
│   └── Position value calculation
│
├── RiskGateValidator (131 lines)
│   ├── Volatility circuit breaker
│   └── Mark staleness check
│
├── StrategyRegistry (193 lines)
│   ├── Per-symbol strategies (SPOT)
│   ├── Shared strategy (PERPS)
│   ├── Config-driven initialization
│   └── Lazy strategy creation
│
└── StrategyExecutor (154 lines)
    ├── Strategy evaluation
    ├── Performance timing
    ├── Telemetry logging
    └── Decision recording
```

## Phased Extraction

### Phase 1: EquityCalculator (120 lines)
**Extracted:** Equity calculation logic
- `calculate()` - Total equity (cash + position value)
- `extract_cash_balance()` - USD/USDC balance extraction
- **Tests:** 17 unit tests
- **Line reduction:** 411 → 399 lines (-12 lines)

### Phase 2: RiskGateValidator (131 lines)
**Extracted:** Risk gate validation logic
- `validate_gates()` - Main validation orchestrator
- `_check_volatility_circuit_breaker()` - Volatility check
- `_check_mark_staleness()` - Data freshness check
- **Tests:** 17 unit tests
- **Line reduction:** 399 → 388 lines (-11 lines)
- **Pattern:** Lazy initialization via `@property` to handle construction ordering

### Phase 3: StrategyRegistry (193 lines)
**Extracted:** Strategy initialization and retrieval
- `initialize()` - Profile-specific strategy initialization
- `get_strategy()` - Strategy retrieval with lazy creation
- `_initialize_spot_strategies()` - Per-symbol SPOT strategies
- `_initialize_perps_strategy()` - Shared PERPS strategy
- **Tests:** 23 unit tests
- **Line reduction:** 388 → 353 lines (-35 lines)
- **Features:**
  - Config-driven window overrides from spot profiles
  - Position fraction overrides
  - Derivatives settings control
  - Backward compatibility (syncs to `bot.strategy` and `bot._symbol_strategies`)

### Phase 4: StrategyExecutor (154 lines)
**Extracted:** Strategy execution and decision recording
- `evaluate()` - Strategy evaluation with performance timing
- `record_decision()` - Decision storage and logging
- `evaluate_and_record()` - Combined flow for simple cases
- **Tests:** 13 unit tests
- **Line reduction:** 353 → 332 lines (-21 lines)
- **Features:**
  - Centralized telemetry logging
  - Split evaluate/record for SPOT filter support
  - Performance measurement

### Phase 5: Integration & Cleanup
**Activities:**
- Added 5 characterization tests verifying extracted services
- Verified all orchestration tests pass (566 tests)
- Updated test fixtures to use new service delegates
- Documented refactoring in architecture docs

## Component Responsibilities

### StrategyOrchestrator (Main Orchestrator)
- **Role:** Coordinates symbol processing workflow
- **Delegates to:**
  - `equity_calculator.calculate()` for equity computation
  - `risk_gate_validator.validate_gates()` for risk checks
  - `strategy_registry.get_strategy()` for strategy retrieval
  - `strategy_executor.evaluate()` for strategy execution
  - `strategy_executor.record_decision()` for decision logging
- **Maintains:**
  - SPOT filter application (`_apply_spot_filters()` - 121 lines, future extraction candidate)
  - Position state building
  - Balance/position fetching

### EquityCalculator
- **Responsibilities:**
  - Extract USD/USDC cash balance from broker balances
  - Calculate position value from quantity × mark price
  - Return total equity (cash + position value)
- **Key Methods:**
  - `calculate()` - Main calculation with error handling
  - `extract_cash_balance()` - Cash asset extraction

### RiskGateValidator
- **Responsibilities:**
  - Validate volatility circuit breaker (max change threshold)
  - Check mark data staleness (freshness requirement)
  - Early exit if any gate fails
- **Key Methods:**
  - `validate_gates()` - Orchestrates all checks
  - `_check_volatility_circuit_breaker()` - Price movement validation
  - `_check_mark_staleness()` - Data freshness validation
- **Pattern:** Lazy initialization to avoid construction-time dependencies

### StrategyRegistry
- **Responsibilities:**
  - Initialize strategies based on profile (SPOT vs PERPS)
  - Apply config overrides from spot profiles
  - Lazy strategy creation for missing symbols
  - Maintain backward compatibility with bot attributes
- **Key Methods:**
  - `initialize()` - Profile-specific initialization
  - `get_strategy()` - Retrieval with lazy creation
  - `_initialize_spot_strategies()` - Per-symbol initialization
  - `_initialize_perps_strategy()` - Shared strategy initialization
- **Configuration Sources:**
  - Global: `config.short_ma`, `config.long_ma`, `config.perps_position_fraction`
  - Per-symbol: `spot_profiles.load()` for window/fraction overrides
  - Derivatives: `config.derivatives_enabled`, `config.target_leverage`, `config.enable_shorts`

### StrategyExecutor
- **Responsibilities:**
  - Execute strategy with performance timing
  - Log telemetry (strategy duration)
  - Record decision to bot
  - Log decision at INFO level
- **Key Methods:**
  - `evaluate()` - Strategy execution with timing
  - `record_decision()` - Decision storage and logging
  - `evaluate_and_record()` - Combined flow (not used for SPOT due to filter insertion)
- **Telemetry:** Logs `strategy=<name>, duration_ms=<ms>` via performance logger

## Design Patterns

### Lazy Initialization
All extracted services use `@property` with lazy initialization:
```python
@property
def risk_gate_validator(self) -> RiskGateValidator:
    """Get or create risk gate validator (lazy initialization)."""
    if self._risk_gate_validator is None:
        self._risk_gate_validator = RiskGateValidator(self._bot.risk_manager)
    return self._risk_gate_validator
```
**Rationale:** Avoids construction-time dependencies (e.g., `risk_manager` not available during `__init__`)

### Dependency Injection
All services support constructor injection for testing:
```python
def __init__(
    self,
    bot: PerpsBot,
    equity_calculator: EquityCalculator | None = None,
    risk_gate_validator: RiskGateValidator | None = None,
    strategy_registry: StrategyRegistry | None = None,
    strategy_executor: StrategyExecutor | None = None,
) -> None:
    self._bot = bot
    self.equity_calculator = equity_calculator or EquityCalculator()
    self._risk_gate_validator = risk_gate_validator
    self._strategy_registry = strategy_registry
    self._strategy_executor = strategy_executor
```

### Backward Compatibility
StrategyRegistry syncs strategies to bot attributes:
```python
# Sync strategies to bot for backward compatibility
bot = self._bot
if bot.config.profile == Profile.SPOT:
    bot._symbol_strategies = self.strategy_registry.symbol_strategies
else:
    bot.strategy = self.strategy_registry.default_strategy
```
**Purpose:** Maintain existing code that accesses `bot.strategy` or `bot._symbol_strategies`

## Test Coverage

### Unit Tests
- **EquityCalculator:** 17 tests (256 lines)
- **RiskGateValidator:** 17 tests (290 lines)
- **StrategyRegistry:** 23 tests (378 lines)
- **StrategyExecutor:** 13 tests (282 lines)
- **Total:** 70 new unit tests

### Integration Tests
- **StrategyOrchestrator:** 49 tests (updated to use new services)
- **Characterization:** 5 tests verifying extracted services in end-to-end flow
- **Total Orchestration Suite:** 566 tests passing

### Test Patterns
- Mock injection for testing in isolation
- `caplog` for logging assertions
- `@pytest.mark.asyncio` for async tests
- Property access to trigger lazy initialization in tests

## Migration Notes

### Breaking Changes
None - full backward compatibility maintained.

### Deprecations
None - no methods marked for deprecation.

### Future Extraction Candidates
1. **SpotFilterService** (121 lines) - `_apply_spot_filters()` method
   - Volatility filter (ATR-based)
   - Volume filter (MA comparison)
   - Momentum filter (RSI-based)
   - Trend filter (MA slope)
   - Candle fetching and sorting

## Metrics

### Code Reduction
- **Total reduction:** 79 lines (-19.2%)
- **Extracted code:** 598 lines (4 new services)
- **Net complexity:** Significant reduction through separation of concerns

### Test Coverage
- **Before:** 513 tests
- **After:** 566 tests (+53 tests, +10.3%)
- **Coverage areas:** Equity calculation, risk validation, strategy management, execution telemetry

### Maintainability Improvements
- ✅ Single Responsibility: Each service has one clear purpose
- ✅ Testability: Services can be tested in isolation
- ✅ Reusability: Services can be used independently
- ✅ Extensibility: Easy to add new validation gates, filters, or telemetry
- ✅ Dependency Injection: All services support constructor injection

## Related Documentation
- `perps_bot_dependencies.md` - PerpsBot service architecture
- `../archive/refactoring-2025-q1/REFACTORING_PHASE_0_STATUS.md` - Overall refactoring status
- Individual test files in `tests/unit/bot_v2/orchestration/`

## Lessons Learned

1. **Lazy Initialization Pattern:** Essential for services with construction-time dependencies
2. **Split Methods:** `evaluate()` + `record_decision()` provides flexibility for intermediate processing (SPOT filters)
3. **Backward Compatibility:** Syncing to bot attributes maintains existing code without refactoring callsites
4. **Characterization Tests:** End-to-end integration tests catch regression risks early
5. **Phased Extraction:** Small, focused phases (50-150 lines each) minimize risk and provide clear progress milestones
