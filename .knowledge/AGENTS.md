# ⚠️ DEPRECATED KNOWLEDGE LAYER

**This directory contains outdated information from August 2024.**

The project has migrated from Alpaca/Equities to Coinbase/Perpetuals.

**For current documentation, see: [docs/README.md](../docs/README.md)**

---

# 🤖 Agent Directory & Delegation Guide

Note: We primarily use Claude Code built-in agents. A small pilot set of custom specialists is being (re)introduced incrementally.

Source of truth for role mapping: `.claude/agents/agent_mapping.yaml`.

## Quick Reference: Who Does What

| Need | Primary Agent | Backup Agent |
|------|----------------|--------------|
| Code implementation | backend-developer | general-purpose |
| Architecture/analysis | project-analyst | general-purpose |
| Code review | code-reviewer | code-archaeologist |
| Documentation | documentation-specialist | code-archaeologist |
| Performance tuning | performance-optimizer | backend-developer |
| Test authoring | backend-developer | documentation-specialist |

## Built-ins We Rely On
backend-developer, project-analyst, code-reviewer, documentation-specialist, performance-optimizer, general-purpose, code-archaeologist, frontend-developer, api-architect, tailwind-frontend-expert, team-configurator.

## Draft Custom Agents (pilot)
- strategy-analyst: Validates strategy logic; designs test plans.
- backtest-specialist: Designs and runs backtests; reports metrics.
- test-engineer: Authors tests and coverage gates.
- perf-analyst: Profiles hotspots; proposes minimal safe optimizations.
- compliance-reviewer: Reviews risk/limits/security; blocks noncompliant changes.
- docs-editor: Updates CLAUDE.md, slice READMEs; keeps paths/examples correct.
- orchestrator-lite: Plans multi-step sequences with quality gates.

> Legacy custom agents (paused): Many legacy custom agents listed below are paused for redesign. Do not invoke them directly. Map to built-ins per `.claude/agents/agent_mapping.yaml` (e.g., compliance-officer → code-reviewer; trading-ops-lead → backend-developer).

### 🏢 Leadership & Strategy (5)
| Agent | Type | Purpose | Delegates To |
|-------|------|---------|--------------|
| **tech-lead-orchestrator** | Built-in | Architecture decisions | All departments |
| **ml-strategy-director** | Custom | ML pipeline oversight | feature-engineer, model-trainer |
| **trading-ops-lead** | Custom | Trading operations | backtest-engineer, paper-trade-manager |
| **devops-lead** | Custom | Infrastructure | deployment-engineer, monitoring-specialist |
| **trading-strategy-consultant** | Custom | Strategy validation | backtest-engineer, risk-analyst |

### 📈 Trading Operations (4)
| Agent | Type | Purpose | Reports To |
|-------|------|---------|------------|
| **backtest-engineer** | Custom | Historical validation | trading-ops-lead |
| **paper-trade-manager** | Custom | Paper trading | trading-ops-lead |
| **live-trade-operator** | Custom | Production trading | trading-ops-lead |
| **risk-analyst** | Custom | Risk management | trading-ops-lead |

### 🤖 Machine Learning (3)
| Agent | Type | Purpose | Reports To |
|-------|------|---------|------------|
| **feature-engineer** | Custom | Feature engineering | ml-strategy-director |
| **model-trainer** | Custom | Model training | ml-strategy-director |
| **gemini-gpt-hybrid** | Built-in | AI consultation | ml-strategy-director |

### 📊 Data Management (3)
| Agent | Type | Purpose | Reports To |
|-------|------|---------|------------|
| **data-pipeline-engineer** | Custom | ETL pipelines | devops-lead |
| **market-data-specialist** | Custom | Market feeds | data-pipeline-engineer |
| **project-analyst** | Built-in | Codebase analysis | tech-lead-orchestrator |

### 💻 Development (6)
| Agent | Type | Purpose | Works With |
|-------|------|---------|------------|
| **backend-developer** | Built-in | Server code | All teams |
| **frontend-developer** | Built-in | UI development | backend-developer |
| **api-architect** | Built-in | API design | backend-developer |
| **tailwind-frontend-expert** | Built-in | CSS styling | frontend-developer |
| **code-archaeologist** | Built-in | Legacy code | All teams |
| **documentation-specialist** | Built-in | Documentation | All teams |

### 🧪 Quality Assurance (6)
| Agent | Type | Purpose | Gates |
|-------|------|---------|-------|
| **code-reviewer** | Built-in | Code quality | All code changes |
| **agentic-code-reviewer** | Custom | PR reviews | Pull requests |
| **test-runner** | Custom | Test execution | All deployments |
| **debugger** | Custom | Bug fixes | Issue resolution |
| **adversarial-dummy** | Custom | Edge cases | New features |
| **performance-optimizer** | Built-in | Optimization | Performance issues |

### 🔧 Infrastructure & Operations (4)
| Agent | Type | Purpose | Reports To |
|-------|------|---------|------------|
| **deployment-engineer** | Custom | CI/CD | devops-lead |
| **monitoring-specialist** | Custom | Observability | devops-lead |
| **repo-structure-guardian** | Custom | Standards | tech-lead-orchestrator |
| **compliance-officer** | Custom | Regulatory | trading-ops-lead |

### 📝 Planning & Coordination (4)
| Agent | Type | Purpose | Coordinates |
|-------|------|---------|-------------|
| **planner** | Custom | Task planning | All teams |
| **team-configurator** | Built-in | Team setup | Initial setup |
| **general-purpose** | Built-in | Multi-domain | Complex tasks |
| **gemini-gpt-hybrid-hard** | Built-in | Aggressive AI | Rapid development |

## Delegation Patterns

### 🚀 New Feature Development
```
planner → tech-lead-orchestrator → backend-developer → code-reviewer → test-runner → deployment-engineer
```

### 📈 New Trading Strategy
```
trading-strategy-consultant → ml-strategy-director → backtest-engineer → risk-analyst → paper-trade-manager → live-trade-operator
```

### 🐛 Bug Investigation
```
debugger → code-archaeologist → backend-developer → test-runner → code-reviewer
```

### 🤖 ML Model Development
```
ml-strategy-director → feature-engineer → data-pipeline-engineer → model-trainer → backtest-engineer → risk-analyst
```

### 🔧 System Optimization
```
performance-optimizer → backend-developer → code-reviewer → test-runner → deployment-engineer
```

## Agent Communication Rules

### ✅ DO Delegate When:
- Task outside your expertise
- Specialized knowledge needed
- Multiple areas involved
- Risk assessment required
- Production deployment needed

### ❌ DON'T Delegate When:
- Simple file edits
- Reading/searching files
- Current task in progress
- You have the expertise

### 🔄 Parallel Execution
These can work simultaneously:
- Frontend & Backend development
- Testing & Documentation
- ML training & Backtesting
- Monitoring & Optimization

### ⛔ Veto Powers
These agents can block progress:
- **risk-analyst**: Can halt any trade or strategy
- **compliance-officer**: Can block for regulatory issues
- **trading-strategy-consultant**: Can reject strategy logic
- **tech-lead-orchestrator**: Can block architecture changes
- **code-reviewer**: Can reject code changes

## How to Call Agents

### Basic Delegation
```
"Use the [agent-name] to [specific task]"
"Have the [agent-name] review this for [specific concern]"
"Get the [agent-name] to validate [specific aspect]"
```

### Examples
- "Use the backtest-engineer to validate this strategy"
- "Have the risk-analyst check our exposure limits"
- "Get the code-reviewer to check for security issues"

## Agent Quality Status

### ✅ Production-Ready (15)
All specifications complete, ready for any task

### ⚠️ Functional (6)
Working but could use enhancement:
- adversarial-dummy
- agentic-code-reviewer  
- debugger
- planner
- repo-structure-guardian
- test-runner

## Important Notes
- **Custom agents** have definition files in `.claude/agents/`
- **Built-in agents** are always available in Claude Code
- **All agents** understand the vertical slice architecture
- **Delegation** should follow the patterns above
- **Communication** should be clear and specific
