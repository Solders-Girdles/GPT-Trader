# Agent Orchestration Guide - Maximizing Utilization

## Overview

This guide documents the enhanced agent orchestration workflow implemented in CLAUDE.md v4.0, designed to maximize agent utilization from the current ~20% to the target 80%+.

## Key Improvements Implemented

### 1. Automatic Agent Deployment Triggers
- **Pattern Recognition**: Automatically deploys agents based on task keywords
- **Parallel by Default**: All independent tasks run simultaneously
- **Team Deployments**: Pre-configured teams for common scenarios

### 2. Agent Utilization Metrics
- **Current**: ~20% utilization (too much direct work, not enough delegation)
- **Target**: 80%+ utilization (maximum parallel agent deployment)
- **Tracking**: Real-time metrics via orchestration script

### 3. Proactive Deployment Patterns

#### Discovery-First Approach
```python
# ALWAYS start with discovery agents
deploy_parallel([
    "code-archaeologist",   # Find relevant code
    "test-automator",       # Check test coverage
    "project-analyst"       # Map dependencies
])
```

#### Parallel Team Deployment
```python
# Deploy entire teams for common tasks
deploy_team("bug_fix")     # 4 agents working together
deploy_team("feature")     # 5 agents covering all aspects
deploy_team("validation")  # 4 agents for comprehensive checks
```

## Quick Start Commands

### Using the Orchestration Script

```bash
# Get agent recommendations for any task
python scripts/orchestrate_agents.py recommend "fix broken tests"

# Deploy predefined teams
python scripts/orchestrate_agents.py deploy-team bug_fix
python scripts/orchestrate_agents.py deploy-team feature
python scripts/orchestrate_agents.py deploy-team discovery

# Deploy Phase 5 recovery agents
python scripts/orchestrate_agents.py deploy-phase5
python scripts/orchestrate_agents.py deploy-phase5 Group_A_Discovery

# Check orchestration metrics
python scripts/orchestrate_agents.py metrics
```

## Task Pattern → Agent Mapping

| Task Contains | Agents Deployed | Execution |
|---------------|-----------------|-----------|
| "find", "search", "locate" | code-archaeologist + project-analyst | Parallel |
| "fix", "debug", "error" | debugger + backend-developer + test-automator | Parallel |
| "create", "implement", "add" | backend-developer + test-automator + doc-specialist | Parallel |
| "test", "verify", "check" | test-automator + code-reviewer | Parallel |
| "optimize", "performance" | performance-optimizer + backend-developer | Parallel |
| "review", "audit" | code-reviewer + test-automator + security-auditor | Parallel |

## Phase 5 Recovery Plan

### Group A: Discovery (4 agents, parallel)
- code-archaeologist: Find all broken tests
- test-automator: Audit test infrastructure
- project-analyst: Map missing strategies
- performance-optimizer: Baseline performance

### Group B: Strategy Expansion (4 agents, parallel)
- backend-developer: Mean reversion strategy
- backend-developer-2: Momentum strategy
- backend-developer-3: Volatility strategy
- trading-strategy-consultant: Validate logic

### Group C: Execution Layer (3 agents, parallel)
- backend-developer-4: Execution simulator
- backend-developer-5: Position tracking
- test-automator-2: Execution tests

### Group D: Validation (4 agents, sequential)
- test-automator-3: Fix failing tests
- code-reviewer: Review implementations
- performance-optimizer: Optimize paths
- documentation-specialist: Update docs

**Total**: 15 agents across 4 groups
**Time Estimate**: 2 hours parallel vs 8 hours sequential (4x speedup)

## Agent Fallback Chain

When an agent is not found, automatic fallbacks are used:

| Missing Agent | Fallback 1 | Fallback 2 | Final |
|---------------|------------|------------|-------|
| ml-engineer | backend-developer | data-scientist | general-purpose |
| python-pro | backend-developer | - | general-purpose |
| integration-engineer | backend-developer | - | general-purpose |
| quant-analyst | trading-strategy-consultant | - | general-purpose |

## Best Practices

### DO's ✅
1. **Always start with discovery agents** - Context prevents mistakes
2. **Deploy teams, not individuals** - Comprehensive coverage
3. **Use parallel execution** - 4x faster completion
4. **Clone agents for same-type work** - Maximum parallelization
5. **Deploy validation after changes** - Continuous quality

### DON'T's ❌
1. **Don't work directly when agents available** - Delegate everything
2. **Don't deploy sequentially without reason** - Parallel is faster
3. **Don't skip discovery phase** - Always get context first
4. **Don't ignore failing tests** - Deploy test team immediately
5. **Don't wait for perfect conditions** - Deploy and iterate

## Measuring Success

### Target Metrics
- Agent Utilization: >80% (current ~20%)
- Parallel Execution: >70% of tasks
- Speedup Factor: 4x vs sequential
- Test Pass Rate: >60% via agents
- Zero duplicate work

### Tracking Command
```bash
python scripts/orchestrate_agents.py metrics
```

## Implementation Status

- ✅ CLAUDE.md updated to v4.0 with orchestration patterns
- ✅ Orchestration automation script created
- ✅ Agent team compositions defined
- ✅ Phase 5 deployment plan ready
- ✅ Fallback chains implemented
- ✅ Metrics tracking system active

## Next Steps

1. **Immediate**: Deploy Phase 5 Group A discovery agents
2. **After Discovery**: Deploy Groups B & C in parallel
3. **Validation**: Deploy Group D validation swarm
4. **Continuous**: Monitor metrics and optimize

---

**Document Version**: 1.0
**Created**: August 15, 2025
**Status**: Ready for Implementation
EOF < /dev/null