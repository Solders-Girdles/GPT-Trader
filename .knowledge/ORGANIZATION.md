# ⚠️ DEPRECATED KNOWLEDGE LAYER

**This directory contains outdated information from August 2024.**

The project has migrated from Alpaca/Equities to Coinbase/Perpetuals.

**For current documentation, see: [docs/README.md](../docs/README.md)**

---

# 🏢 GPT-Trader Organization Structure

## Organizational Hierarchy

```
Executive Layer (Strategic Decisions)
    ├── Chief Architect (tech-lead-orchestrator)
    ├── Trading Director (trading-strategy-consultant)
    └── ML/AI Director (ml-strategy-director)
             ↓
Department Heads (Operations)
    ├── Development (backend-developer leads)
    ├── Quality Assurance (code-reviewer leads)
    ├── Trading Operations (trading-ops-lead)
    ├── Infrastructure (devops-lead)
    └── Data Management (data-pipeline-engineer)
             ↓
Specialists (Implementation)
    ├── ML Engineers (feature-engineer, model-trainer)
    ├── Trading Engineers (backtest, paper-trade, live-trade)
    ├── DevOps Engineers (deployment, monitoring)
    └── Risk & Compliance (risk-analyst, compliance-officer)
```

## Department Structure

### 💻 Development Department
**Lead:** backend-developer
**Team:** frontend-developer, api-architect, tailwind-frontend-expert
**Responsibilities:** All code implementation, APIs, UI/UX

### 🧪 Quality Assurance Department  
**Lead:** code-reviewer
**Team:** test-runner, debugger, adversarial-dummy, agentic-code-reviewer
**Responsibilities:** Code quality, testing, security, edge cases

### 📈 Trading Operations Department
**Lead:** trading-ops-lead
**Team:** backtest-engineer, paper-trade-manager, live-trade-operator
**Responsibilities:** Strategy execution, backtesting, paper/live trading

### 🔬 Research & ML Department
**Lead:** ml-strategy-director
**Team:** feature-engineer, model-trainer, gemini-gpt-hybrid
**Responsibilities:** ML models, features, research

### 🏗️ Infrastructure Department
**Lead:** devops-lead
**Team:** deployment-engineer, monitoring-specialist, repo-structure-guardian
**Responsibilities:** CI/CD, monitoring, infrastructure

### 📊 Data Department
**Lead:** data-pipeline-engineer
**Team:** market-data-specialist
**Responsibilities:** Data pipelines, quality, feeds

### ⚖️ Risk & Compliance
**Lead:** risk-analyst
**Team:** compliance-officer
**Responsibilities:** Risk management, regulatory compliance

## Decision Authority

### Who Can Make What Decisions

| Decision Type | Authority | Can Override |
|--------------|-----------|--------------|
| Architecture | tech-lead-orchestrator | No one |
| Trading Logic | trading-strategy-consultant | tech-lead only |
| ML Models | ml-strategy-director | tech-lead only |
| Risk Limits | risk-analyst | Can halt anything |
| Compliance | compliance-officer | Can halt anything |
| Code Quality | code-reviewer | Can block merges |
| Infrastructure | devops-lead | Can block deploys |
| Data Quality | data-pipeline-engineer | Can halt data flow |

## Reporting Structure

### Direct Reports
- **To tech-lead-orchestrator:** All department heads
- **To trading-ops-lead:** All trading engineers
- **To ml-strategy-director:** All ML engineers
- **To devops-lead:** All infrastructure engineers
- **To data-pipeline-engineer:** All data specialists

### Cross-Functional Teams
For complex tasks, these teams work together:

**Strategy Development Team:**
- Lead: trading-strategy-consultant
- Members: ml-strategy-director, backtest-engineer, risk-analyst

**Production Deployment Team:**
- Lead: devops-lead
- Members: deployment-engineer, monitoring-specialist, live-trade-operator

**Data Quality Team:**
- Lead: data-pipeline-engineer
- Members: market-data-specialist, feature-engineer

## Communication Protocols

### Escalation Path
1. Specialist → Department Head
2. Department Head → Executive
3. Executive → Final Decision

### Status Reporting
- **Daily:** Task progress in todos
- **Weekly:** Department summaries
- **Milestone:** Comprehensive reports

### Meeting Cadence
- **Daily Standup:** Quick sync (all active agents)
- **Weekly Review:** Department heads
- **Monthly Strategy:** Executive layer

## Key Principles

1. **Single Responsibility:** Each agent owns one domain
2. **Clear Hierarchy:** Escalation paths are defined
3. **Veto Powers:** Risk and compliance can stop anything
4. **Parallel Work:** Independent tasks run simultaneously
5. **Documentation:** All decisions must be recorded

## Quick Reference: Who Owns What

| Area | Owner | Backup |
|------|-------|--------|
| System Architecture | tech-lead-orchestrator | backend-developer |
| Trading Strategies | trading-strategy-consultant | trading-ops-lead |
| ML Models | ml-strategy-director | feature-engineer |
| Risk Management | risk-analyst | compliance-officer |
| Code Quality | code-reviewer | test-runner |
| Infrastructure | devops-lead | deployment-engineer |
| Data Pipeline | data-pipeline-engineer | market-data-specialist |
| Documentation | documentation-specialist | code-archaeologist |

For detailed agent capabilities, see `AGENTS.md`
For workflow patterns, see `DELEGATION_WORKFLOWS.md`