# Project Progress & Status (Claude Code session reference)

## Current Reality
- Architecture: Vertical Slice with complete slice isolation
- Active system: `src/bot_v2/` (11 slices)
- Tests: organized under `tests/integration/bot_v2/`
- Data provider abstraction: implemented (`src/bot_v2/data_providers`)

## Intelligence Components (Path B)
- Week 1–2: ML Strategy Selection — COMPLETE
  - Dynamic strategy switching, confidence scoring
- Week 3: Market Regime Detection — COMPLETE
  - 7 regimes; real-time monitoring
- Week 4: Intelligent Position Sizing — COMPLETE
  - Kelly Criterion; confidence- and regime-adjusted sizing
- Week 5: Adaptive Portfolio — COMPLETE
  - Configuration-first; tier-based; provider abstraction

## Current Status
- Core slices: operational (11/11)
- Tests: organized; integration coverage present
- Data provider: clean interfaces and fallbacks
- Root cleanup: historical assets archived

## Next Steps
- Phase 1: User configuration and capital setup
- Phase 2: Live deployment testing
- Phase 3: Production monitoring and optimization

## References
- Full control center: `docs/CLAUDE_FULL.md`
- Slice navigation: `src/bot_v2/SLICES.md`
- Agents & delegation: `.knowledge/AGENTS.md`, `.claude/agents/agent_mapping.yaml`
