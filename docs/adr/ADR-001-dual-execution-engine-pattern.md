# ADR-001: Dual Execution Engine Pattern

**Status:** Accepted
**Date:** 2025-10-06
**Deciders:** System Architecture Team
**Technical Story:** Execution layer consolidation review (Phase 3 audit)

---

## Context

GPT-Trader requires flexible execution capabilities to support both baseline trading workflows and advanced features like dynamic position sizing and market impact estimation. Two execution engines currently exist:

1. **LiveExecutionEngine** (`orchestration/live_execution.py`) - 401 lines
   - Baseline execution with risk integration
   - Event store tracking, order preview, slippage multipliers
   - Delegates to orchestration/execution/ helpers

2. **AdvancedExecutionEngine** (`features/live_trade/advanced_execution.py`) - 456 lines
   - Advanced order workflows (TIF mapping, stop triggers, normalization)
   - Dynamic position sizing and market impact awareness
   - Delegates to 6+ specialized components

During the Phase 3 operational audit, we evaluated whether to consolidate these engines into a single implementation.

---

## Decision

**We will maintain the dual execution engine pattern with feature flag switching.**

The system will continue to use `ExecutionEngineFactory` to create either:
- `LiveExecutionEngine` (default) - when both feature flags are disabled
- `AdvancedExecutionEngine` (enhanced) - when any feature flag is enabled

Feature flags (in `RiskConfig`):
- `enable_dynamic_position_sizing` (env: `RISK_ENABLE_DYNAMIC_POSITION_SIZING`)
- `enable_market_impact_guard` (env: `RISK_ENABLE_MARKET_IMPACT_GUARD`)

---

## Rationale

### Why Keep Both Engines

1. **Progressive Enhancement**
   - LiveExecutionEngine provides stable baseline
   - AdvancedExecutionEngine adds opt-in complexity
   - Feature flags allow gradual rollout without code changes

2. **Separation of Concerns**
   - LiveExecutionEngine: Runtime guards, basic execution, event tracking
   - AdvancedExecutionEngine: Complex workflows, impact estimation, dynamic sizing
   - Each engine delegates to appropriate helpers (different helper modules)

3. **Testing Independence**
   - LiveExecutionEngine: 10+ test files covering baseline scenarios
   - AdvancedExecutionEngine: 6+ test files covering advanced features
   - Separate test suites verify each engine works independently
   - Consolidation would mix baseline + advanced tests, reducing clarity

4. **Rollback Safety**
   - Feature flags allow instant engine swap via config change
   - No code deployment needed to disable advanced features
   - Critical for production stability (rollback in seconds, not hours)

5. **Complexity Management**
   - LiveExecutionEngine: 401 lines (clean, focused)
   - AdvancedExecutionEngine: 456 lines (complex but modular)
   - Merged engine would be 800+ lines with conditional logic scattered throughout
   - Would violate Single Responsibility Principle

6. **Production Reality**
   - Current production config (`dev_dynamic.json`) enables both flags
   - AdvancedExecutionEngine is the active production engine
   - LiveExecutionEngine serves as fallback + test baseline
   - Both engines are actively maintained and tested

### Why Consolidation Would Fail

❌ **Loss of Simplicity:** Merged engine would be a god class (800+ lines)
❌ **Testing Nightmare:** Mixed baseline + advanced test scenarios
❌ **Conditional Complexity:** `if enable_feature:` logic scattered everywhere
❌ **Reduced Modularity:** Blur boundaries between helper components
❌ **Slower Rollback:** Config-only rollback becomes code change + deploy

---

## Consequences

### Positive

✅ **Feature Flag Safety:** Instant rollback via config (no redeploy)
✅ **Clean Separation:** Baseline vs advanced concerns clearly separated
✅ **Independent Testing:** Each engine has dedicated test suite
✅ **Progressive Enhancement:** Can enable features incrementally
✅ **Modularity:** Each engine delegates to appropriate helpers
✅ **Maintainability:** Two focused engines easier to maintain than one god class

### Negative

⚠️ **Code Duplication:** Some overlap in `place_order()` signatures
⚠️ **Factory Complexity:** Requires factory pattern to switch engines
⚠️ **Documentation Burden:** Need to explain two engines to new developers

### Mitigation

- **Duplication:** Minimal overlap (interface only), implementation differs significantly
- **Factory:** Well-established pattern, already implemented and tested
- **Documentation:** This ADR + ARCHITECTURE.md updates + factory docstrings

---

## Implementation Details

### Factory Pattern

`ExecutionEngineFactory.create_engine()` in `orchestration/execution/engine_factory.py`:

```python
@classmethod
def create_engine(cls, broker, risk_manager, event_store, bot_id, enable_preview):
    use_advanced = cls.should_use_advanced_engine(risk_manager)

    if use_advanced:
        from gpt_trader.features.live_trade.advanced_execution import AdvancedExecutionEngine
        engine = AdvancedExecutionEngine(broker=broker, risk_manager=risk_manager)
        logger.info("Initialized AdvancedExecutionEngine")
    else:
        from gpt_trader.orchestration.live_execution import LiveExecutionEngine
        engine = LiveExecutionEngine(
            broker=broker,
            risk_manager=risk_manager,
            event_store=event_store,
            bot_id=bot_id,
            enable_preview=enable_preview,
        )
        logger.info("Initialized LiveExecutionEngine")

    return engine
```

### Feature Flag Decision Logic

```python
@staticmethod
def should_use_advanced_engine(risk_manager: LiveRiskManager) -> bool:
    risk_config = getattr(risk_manager, "config", None)
    if risk_config is None:
        return False

    return (
        getattr(risk_config, "enable_dynamic_position_sizing", False) or
        getattr(risk_config, "enable_market_impact_guard", False)
    )
```

**Decision:** Use AdvancedExecutionEngine if **ANY** flag is enabled

### Production Configuration

`config/risk/dev_dynamic.json`:
```json
{
  "enable_market_impact_guard": true,
  "enable_dynamic_position_sizing": true
}
```

**Result:** Production uses `AdvancedExecutionEngine`

### Fallback Configuration

If both flags are false (or no config):
- Default to `LiveExecutionEngine`
- Provides baseline execution without advanced features

---

## Alternatives Considered

### Alternative 1: Consolidate into Single Engine

**Approach:** Merge both engines into one class with internal feature flags

**Rejected because:**
- Would create 800+ line god class violating SRP
- Testing would mix baseline + advanced scenarios (reduced clarity)
- Rollback becomes code change + deploy (slower, riskier)
- Conditional logic scattered throughout implementation
- Would reduce code quality and maintainability

### Alternative 2: Plugin Architecture

**Approach:** Base engine + pluggable feature modules

**Rejected because:**
- Over-engineering for two well-defined engines
- Would add complexity without clear benefit
- Current factory pattern already provides clean switching
- No requirement for runtime plugin loading

### Alternative 3: Inheritance Hierarchy

**Approach:** `BaseExecutionEngine` → `LiveExecutionEngine` → `AdvancedExecutionEngine`

**Rejected because:**
- Inheritance is more rigid than composition
- Both engines delegate to different helper modules (not a parent-child relationship)
- Factory pattern provides better flexibility
- Violates "favor composition over inheritance" principle

---

## Related Decisions

- **Phase 0:** MarketDataService extraction (service-based architecture)
- **Phase 1:** CLI modularization (command pattern)
- **Phase 2:** Live trade service extraction (vertical slice design)
- **Phase 3:** PerpsBotBuilder pattern (builder pattern for composition)

This ADR follows the same principle: **modular, composable services** with **feature flag control**.

---

## References

- **Analysis:** `docs/ops/phase3_execution_layer_analysis.md`
- **Factory:** `src/gpt_trader/orchestration/execution/engine_factory.py`
- **LiveExecutionEngine:** `src/gpt_trader/orchestration/live_execution.py`
- **AdvancedExecutionEngine:** `src/gpt_trader/features/live_trade/advanced_execution.py`
- **RiskConfig:** `src/gpt_trader/config/live_trade_config.py`
- **Production Config:** `config/risk/dev_dynamic.json`
- **Tests:**
  - LiveExecutionEngine: `tests/unit/gpt_trader/orchestration/test_live_execution.py` (+9 more)
  - AdvancedExecutionEngine: `tests/unit/gpt_trader/features/live_trade/test_advanced_execution.py` (+5 more)
  - Factory: `tests/unit/gpt_trader/orchestration/execution/test_engine_factory.py`

---

## Review and Update

- **Status:** Accepted (2025-10-06)
- **Next Review:** When adding third execution engine or major refactor
- **Owner:** System Architecture Team
- **Related ADRs:** (none yet - this is ADR-001)

---

## Notes

This ADR documents an **existing architectural pattern** identified during the Phase 3 operational audit. The dual-engine design was implemented organically during feature development (Phase 2: "Advanced execution, liquidity service, order policy composition"). This ADR formalizes the decision and rationale.

The pattern follows industry best practices:
- **Feature Flags:** Gradual rollout and safe rollback
- **Strategy Pattern:** Factory selects appropriate implementation
- **Single Responsibility:** Each engine has focused purpose
- **Composition over Inheritance:** Both engines delegate to helpers

**Key Insight:** Sometimes the right decision is to **keep** existing architecture rather than consolidate. The audit revealed this pattern is sound and should be preserved.
