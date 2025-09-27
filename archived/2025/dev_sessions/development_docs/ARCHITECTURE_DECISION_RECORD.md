# Architecture Decision Record: Bot_v2 Vertical Slices

**Date**: August 18, 2025  
**Decision**: Continue with Bot_v2 Vertical Slice Architecture  
**Status**: Approved and Active  

## Context

During EPIC-002 planning, we discovered a major inconsistency:
- The system had TWO parallel architectures: bot_v2 (vertical slices) and an experimental approach
- Bot_v2 was supposedly "archived" but remained active in `src/bot_v2/`
- An experimental structure was created but only scaffolded (empty)
- All 21 agents were configured to use bot_v2 paths
- Documentation incorrectly claimed the experimental architecture was complete

## Decision

**Keep the Bot_v2 Vertical Slice Architecture** as the primary system architecture.

## Rationale

1. **Already Working**: Bot_v2 has complete implementations with 11 feature slices
2. **75% Complete**: ML intelligence components already implemented and tested
3. **Agent Alignment**: All agents already configured for bot_v2 paths
4. **Proven Pattern**: Vertical slices provide excellent isolation (~500 tokens per slice)
5. **Unnecessary Complexity**: Alternative architecture adds complexity without clear benefits for this project size

## Consequences

### Positive
- No migration effort required (saves ~2 weeks)
- Agents continue working without reconfiguration
- Existing tests and demos remain valid
- Can immediately continue with feature development
- Maintains proven isolation and performance characteristics

### Negative
- Abandons experimental architecture exploration
- Some documentation needs updating to reflect reality
- May need refactoring if system grows significantly larger

## Implementation

1. **Keep Active**: `src/bot_v2/` remains the primary codebase
2. **Archive Experiments**: Move experimental structure to `archived/domain_exploration_20250818/`
3. **Update Docs**: Update CLAUDE.md and Command Center to reflect bot_v2 as active
4. **Agent Validation**: Verify all agents can access their configured paths
5. **Continue Development**: Proceed with EPIC-002 using bot_v2 structure

## Architecture Overview

```
src/bot_v2/features/           # PRIMARY ARCHITECTURE (Active)
├── adaptive_portfolio/        # Portfolio management (Week 5)
├── analyze/                   # Market analysis
├── backtest/                  # Historical testing
├── data/                      # Data management
├── live_trade/                # Live trading
├── market_regime/             # Regime detection (Week 3)
├── ml_strategy/               # ML strategy selection (Week 1-2)
├── monitor/                   # System monitoring
├── optimize/                  # Optimization
├── paper_trade/               # Paper trading
└── position_sizing/           # Position sizing (Week 4)
```

## Metrics for Success

- All 21 agents successfully access bot_v2 paths ✅
- No broken imports or missing dependencies
- Existing tests continue passing
- Development velocity improves without migration overhead

## Review and Approval

- **Proposed by**: System Architect
- **Reviewed by**: Development Team
- **Approved by**: Project Lead
- **Date**: August 18, 2025

## Future Considerations

If the system grows beyond 50 feature slices or requires multi-team development, we may reconsider alternative architectures. For now, vertical slices provide the optimal balance of simplicity and maintainability.