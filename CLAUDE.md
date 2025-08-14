# Claude Code Assistant Guide for GPT-Trader

## ⚠️ CRITICAL PROJECT STATUS - MUST READ

**Actual Completion: 35-45%** (NOT the 90%+ claimed in README)
**Current State: BROKEN** - CLI doesn't work, tests failing, no live trading

### What's Actually Working:
- Individual ML components (src/bot/ml/)
- Basic strategy framework (2 strategies)
- Some backtesting functionality
- Data download from yfinance

### What's BROKEN (Critical):
- **CLI completely broken** - ImportError: cannot import name 'BacktestEngine'
- **Test suite failing** - 76% of tests have errors
- **No production orchestrator** - File claimed in README doesn't exist
- **Modules disconnected** - Components exist but don't integrate

### False Claims to Ignore:
- ❌ "Production-ready" - FALSE
- ❌ "Phase 4 Complete" - FALSE
- ❌ "85%+ test success rate" - FALSE (actually 24%)
- ❌ "Real-time execution infrastructure" - MISSING

## Agent Instructions

**PRIORITY**: We are executing a 30-day recovery plan to bring the system from 35% to 65-75% functional.

**CURRENT PHASE**: Emergency Fixes (Week 1)
**ACTIVE BRANCH**: fix/critical-cli-imports

### How to Choose Tasks:
1. Check the Recovery Task List below
2. Find the first task marked [ ] (unchecked)
3. Work on that task using the context provided
4. Check it off [x] when complete
5. Commit with format: `[TASK-ID] type: description`

### Critical Context for Agents:
- **DO NOT TRUST README.md** - It contains false claims
- **TEST EVERYTHING** - Assume nothing works until proven
- **USE CORRECT IMPORTS** - BacktestEngine doesn't exist, use PortfolioBacktestEngine
- **CHECK ERROR LOGS** - Most features fail silently

---

## 🚨 30-Day Recovery Plan Task List

### Week 1: Emergency Fixes (Branch: fix/critical-cli-imports)
**Goal: Make the system minimally runnable**

#### Fix CLI Import Errors (CRITICAL)
- [ ] **CLI-001**: Find correct BacktestEngine class
  - Context: Check src/bot/backtest/engine_portfolio.py for PortfolioBacktestEngine
  - Error: ImportError in src/bot/cli.py line 10
  - Test: `python -c "from bot.cli import main"`

- [ ] **CLI-002**: Fix import in src/bot/cli.py
  - Current: `from bot.backtest.engine import BacktestEngine`
  - Fix to: `from bot.backtest.engine_portfolio import PortfolioBacktestEngine`
  - Verify: `poetry run gpt-trader --help` should work

- [ ] **CLI-003**: Audit all CLI module imports
  - Files: src/bot/cli/*.py
  - Test each: `python -c "from bot.cli.{module} import *"`

- [ ] **CLI-004**: Test each CLI command
  - Commands: backtest, optimize, paper, live
  - Document which work/fail

- [ ] **CLI-005**: Create CLI smoke test
  - File: scripts/test_cli_smoke.py
  - Add to CI pipeline

#### Repair Test Suite
- [ ] **TEST-001**: Fix test fixture imports
  - Files: tests/conftest.py, tests/factories.py
  - Current: 16 import errors during collection

- [ ] **TEST-002**: Fix unit test errors
  - Run: `pytest -v 2>&1 | grep ImportError`
  - Fix each systematically

- [ ] **TEST-003**: Fix integration tests
  - Focus: Database, ML pipeline, strategies

- [ ] **TEST-004**: Create minimal baseline
  - 20 critical tests that MUST pass

- [ ] **TEST-005**: Configure pytest properly
  - Update pytest.ini with markers

#### Create Working Demo
- [ ] **DEMO-001**: Fix standalone_demo.py
  - Current: Import errors
  - Goal: Runs end-to-end

- [ ] **DEMO-002**: Simple backtest demo
  - One strategy, one symbol, 30 days
  - Clear profit/loss output

- [ ] **DEMO-003**: Data download demo
  - Download AAPL, MSFT, GOOGL
  - Save to data/historical/

- [ ] **DEMO-004**: Document requirements
  - Create demos/README.md
  - Include .env.example

### Week 2: Core Integration (Branch: fix/core-integration)
**Goal: Connect the disconnected modules**

#### Create Production Orchestrator
- [ ] **ORCH-001**: Design architecture
  - File: docs/architecture/orchestrator_design.md
  - Define component interactions

- [ ] **ORCH-002**: Implement skeleton
  - File: src/bot/live/production_orchestrator.py
  - Basic event loop and registration

- [ ] **ORCH-003**: Wire data pipeline
  - Connect market data sources
  - Add validation and caching

- [ ] **ORCH-004**: Integrate strategies
  - Connect ML predictions to selection
  - Add performance tracking

- [ ] **ORCH-005**: Add risk management
  - Position limits, drawdown protection
  - Stop-loss logic

- [ ] **ORCH-006**: Connect execution
  - Order management
  - Trade tracking

- [ ] **ORCH-007**: Add monitoring
  - Health checks, metrics
  - Alert triggers

#### Fix Module Integration
- [ ] **INT-001**: Create event bus
  - File: src/bot/core/event_bus.py
  - Pub/sub pattern for modules

- [ ] **INT-002**: ML → Strategy connection
  - Predictions influence selection
  - Add confidence scoring

- [ ] **INT-003**: Strategy → Portfolio
  - Signals become positions
  - Apply constraints

- [ ] **INT-004**: Portfolio → Risk
  - Validate all changes
  - Continuous monitoring

- [ ] **INT-005**: Risk → Execution
  - Modify/block orders
  - Emergency controls

#### Database Integration
- [ ] **DB-001**: Design schema
  - Tables: trades, positions, performance, models
  - File: migrations/001_initial_schema.sql

- [ ] **DB-002**: Implement models
  - SQLAlchemy + Pydantic
  - File: src/bot/database/models.py

- [ ] **DB-003**: Data access layer
  - CRUD operations
  - File: src/bot/database/repository.py

- [ ] **DB-004**: Utilities
  - Backup, cleanup, monitoring scripts

### Week 3: Make It Usable (Branch: feature/working-strategies)
**Goal: Create actual user value**

#### Complete Working Strategies
- [ ] **STRAT-001**: Fix demo_ma
  - Signal generation errors
  - Add position sizing

- [ ] **STRAT-002**: Fix trend_breakout
  - ATR calculations
  - Entry/exit logic

- [ ] **STRAT-003**: Create momentum
  - RSI + volume strategy
  - New implementation

- [ ] **STRAT-004**: Validation framework
  - Test all strategies work

#### Paper Trading Pipeline
- [ ] **PAPER-001**: Alpaca integration
  - Fix authentication
  - Test order placement

- [ ] **PAPER-002**: Paper mode
  - Configuration switch
  - Simulated execution

- [ ] **PAPER-003**: Deployment
  - Script: deploy_paper.py
  - Auto-restart logic

- [ ] **PAPER-004**: Dashboard
  - Show positions, P&L
  - Trade history

#### Basic Monitoring
- [ ] **MON-001**: Core metrics
  - Returns, drawdown, Sharpe
  - Trade statistics

- [ ] **MON-002**: Alerting
  - Email, Slack integration
  - Critical event notifications

- [ ] **MON-003**: Dashboard
  - Streamlit UI
  - Real-time updates

- [ ] **MON-004**: Logging
  - Structured JSON logs
  - Rotation and search

### Week 4: Documentation Reality (Branch: docs/reality-update)
**Goal: Align docs with reality**

#### Update README
- [ ] **DOC-001**: Remove false claims
  - Remove "production-ready"
  - Fix percentages

- [ ] **DOC-002**: Document capabilities
  - What actually works
  - Known issues

- [ ] **DOC-003**: Realistic roadmap
  - Achievable timeline
  - Resource requirements

#### Fix Examples
- [ ] **EX-001**: Audit examples
  - Test each one
  - Document failures

- [ ] **EX-002**: Fix working ones
  - Update imports
  - Add error handling

- [ ] **EX-003**: Archive broken
  - Move to archived/
  - Add deprecation notes

- [ ] **EX-004**: Create new
  - Simple, working examples
  - Clear progression

#### Status Report
- [ ] **STAT-001**: Current state
  - Module percentages
  - Dependency tree

- [ ] **STAT-002**: Issues list
  - Categorized by severity
  - Reproduction steps

- [ ] **STAT-003**: Success metrics
  - Clear "done" criteria
  - Quality targets

---

## 🔧 Critical Quick Reference for Agents

### Correct Imports (Use These!)
```python
# WRONG (causes ImportError):
from bot.backtest.engine import BacktestEngine

# CORRECT:
from bot.backtest.engine_portfolio import PortfolioBacktestEngine

# WRONG:
from bot.live.production_orchestrator import ProductionOrchestrator

# CORRECT (file doesn't exist yet):
# Need to create this file first!
```

### Working File Locations
```
✅ WORKING:
- src/bot/ml/integrated_pipeline.py (ML orchestration)
- src/bot/ml/auto_retraining.py (retraining system)
- src/bot/strategy/demo_ma.py (simple strategy)
- src/bot/strategy/trend_breakout.py (breakout strategy)

❌ BROKEN/MISSING:
- src/bot/live/production_orchestrator.py (DOESN'T EXIST)
- src/bot/cli.py (import errors)
- Most tests in tests/ (76% failing)
- All examples in examples/ (import errors)
```

### Common Errors & Fixes
| Error | Cause | Fix |
|-------|-------|-----|
| `ImportError: cannot import name 'BacktestEngine'` | Class doesn't exist | Use PortfolioBacktestEngine |
| `ModuleNotFoundError: No module named 'bot'` | Python path issue | Add `sys.path.insert(0, 'src')` |
| `pytest: 16 errors` | Test fixtures broken | Fix imports in conftest.py |
| `No production_orchestrator` | File missing | Create it (ORCH-002) |

### Testing Commands
```bash
# Check if CLI works (currently broken):
poetry run gpt-trader --help

# Test specific import:
python -c "from bot.backtest.engine_portfolio import PortfolioBacktestEngine; print('Success')"

# Run tests (expect failures):
pytest -v 2>&1 | grep -E "passed|failed|error"

# Test data download (should work):
python -c "from bot.dataflow.sources.yfinance_source import YFinanceSource; print('Import OK')"
```

### Git Workflow for Recovery
```bash
# Start work on emergency fixes:
git checkout -b fix/critical-cli-imports

# Commit format:
git commit -m "[CLI-001] fix: locate BacktestEngine class"

# After completing Week 1 tasks:
git push origin fix/critical-cli-imports
# Create PR titled: "Emergency Fixes - Make System Runnable"
```

---

## 📊 Success Criteria by Phase

### Week 1 Success (Emergency Fixes)
**Must achieve ALL of these:**
- [ ] CLI loads without import errors
- [ ] At least 50% of tests pass (up from 24%)
- [ ] One demo runs completely end-to-end
- [ ] Can run: `poetry run gpt-trader backtest --help`

### Week 2 Success (Core Integration)
**Must achieve:**
- [ ] Orchestrator file exists and runs
- [ ] 3+ modules communicate via event bus
- [ ] Database stores at least one trade
- [ ] Can run basic backtest via CLI

### Week 3 Success (User Features)
**Must achieve:**
- [ ] 3 strategies complete backtests
- [ ] Paper trading places mock orders
- [ ] Dashboard displays some metrics
- [ ] Monitoring shows system health

### Week 4 Success (Documentation)
**Must achieve:**
- [ ] README honest about state
- [ ] 3+ examples run without errors
- [ ] Clear roadmap published
- [ ] Status report accurate

### Overall Recovery Success
**Target: 65-75% functional** (up from 35-45%)
- Working CLI with 5+ commands
- 3 validated strategies
- Paper trading operational
- 80%+ tests passing
- Honest documentation

---

## Project Structure (ACTUAL vs CLAIMED)

```
GPT-Trader/
├── src/bot/
│   ├── ml/                              ✅ 85% (WORKING)
│   │   ├── integrated_pipeline.py       ✅ Exists, complex
│   │   ├── auto_retraining.py          ✅ Exists, 1074 lines
│   │   └── deep_learning/              ✅ Exists, untested
│   ├── live/                            ⚠️  45% (PARTIAL)
│   │   ├── production_orchestrator.py  ❌ MISSING (claimed to exist)
│   │   ├── strategy_selector.py        ✅ Exists
│   │   └── trading_engine.py           ✅ Exists, not integrated
│   ├── strategy/                        ⚠️  75% (2 WORKING)
│   │   ├── demo_ma.py                  ✅ Works
│   │   ├── trend_breakout.py           ✅ Works
│   │   └── [others]                    ❌ Stubs/broken
│   ├── cli/                            ❌ 15% (BROKEN)
│   │   └── *.py                        ❌ Import errors
│   └── dashboard/                       ⚠️  25% (MINIMAL)
│       └── streamlit_app.py            ⚠️  UI only, no data
├── tests/                               ❌ 24% passing
│   └── 21 files with 16 errors
└── examples/                            ❌ 0% (ALL BROKEN)
    └── All have import errors
```

---

## Context Budget Policy
- **Main thread**: goals, current step, brief status, tiny diffs (<40 lines).
- **Subagents**: heavy reads, logs, wide diffs, repo scans.
  - Return a **10-bullet digest** + file paths + artifact links.
- Never paste >200 lines into main; summarize and reference paths.
- If a task lacks a clear 5–8 step plan, call **`planner`** first.

---

## Command Registry
# SoT & drift
- `python scripts/generate_filemap.py`
- `rg -n "src/bot/|python -m src\.bot|docker-compose|pytest" docs CLAUDE.md`
- `python scripts/doc_check.py --files CLAUDE.md docs/**/*.md`

# Test / perf (MOSTLY BROKEN)
- `pytest -q` (76% failing)
- `pytest tests/performance/benchmark_consolidated.py -q` (untested)

# Ops (BROKEN)
- `python -m src.bot.cli dashboard` (import errors)
- `python -m src.bot.cli backtest --symbol AAPL --start 2024-01-01 --end 2024-06-30 --strategy trend_breakout` (broken)
- `docker-compose -f deploy/postgres/docker-compose.yml up -d` (database only)

---

## 🤖 Agent-Specific Instructions

### For backend-developer Agent
**Focus**: Fix Python imports and module integration
**Priority Files**:
1. src/bot/cli.py (fix BacktestEngine import)
2. src/bot/live/production_orchestrator.py (create this)
3. tests/conftest.py (fix test fixtures)

### For python-pro Agent
**Focus**: Fix Python path issues and imports
**Key Tasks**:
- CLI-001 through CLI-005
- TEST-001 through TEST-005
**Known Issues**: Circular imports, missing __init__.py files

### For code-archaeologist Agent
**Focus**: Understand why system is broken
**Investigate**:
- When/why was BacktestEngine removed?
- What's the actual architecture?
- Which modules actually connect?

### For test-automator Agent
**Focus**: Fix test suite
**Current State**: 76% failing
**Goal**: 80% passing
**Start with**: tests/conftest.py

### For documentation-specialist Agent
**Focus**: Update docs to reality
**Priority**:
1. Fix README.md false claims
2. Update examples to working code
3. Create honest status report

### For performance-optimizer Agent
**Note**: DO NOT OPTIMIZE YET
**Reason**: System doesn't work at all
**When ready**: After Week 3 completion

### For trading-strategy-consultant Agent
**Focus**: Validate existing strategies
**Check**:
- src/bot/strategy/demo_ma.py
- src/bot/strategy/trend_breakout.py
**Create**: STRAT-003 momentum strategy

## ⚠️ Common Agent Pitfalls to Avoid

1. **Don't trust README.md** - It lies about capabilities
2. **Don't assume imports work** - Test everything
3. **Don't create complex features** - Fix basics first
4. **Don't optimize performance** - Make it work first
5. **Don't add new dependencies** - Fix with what exists
6. **Don't trust "Phase 4 Complete"** - It's not even Phase 1 complete

---

## 📝 Recovery Command Reference

### Diagnostic Commands
```bash
# Find the real BacktestEngine class:
grep -r "class.*BacktestEngine" src/

# See what's actually in backtest module:
ls -la src/bot/backtest/

# Check all import errors:
python -c "from bot.cli import main" 2>&1

# See which tests fail:
pytest --collect-only 2>&1 | grep ERROR

# Check if examples work:
for f in examples/*.py; do echo "=== $f ==="; python "$f" 2>&1 | head -5; done
```

### Fix Verification Commands
```bash
# After fixing CLI imports:
poetry run gpt-trader --help

# After fixing a strategy:
python -c "from bot.strategy.demo_ma import DemoMAStrategy; print('Import OK')"

# After fixing tests:
pytest tests/unit/test_config.py -v

# After creating orchestrator:
python -c "from bot.live.production_orchestrator import ProductionOrchestrator"
```

### Progress Tracking
```bash
# Count completed tasks:
grep -c "\[x\]" CLAUDE.md

# See remaining CLI tasks:
grep "CLI-" CLAUDE.md | grep "\[ \]"

# Check test pass rate:
pytest --tb=no | tail -1
```

---

## AI Subagent Reference Guide

### Overview
This section documents all available AI subagents, their specialized expertise, and appropriate use cases. Agents should be called proactively when their expertise matches the task at hand.

### Core Analysis & Configuration Agents

#### 1. **project-analyst**
- **Expertise**: Codebase analysis, framework detection, tech stack identification
- **When to use**: MUST BE USED for any new or unfamiliar codebase. Use PROACTIVELY to detect frameworks, tech stacks, and architecture before routing to specialists
- **Tools**: LS, Read, Grep, Glob, Bash

#### 2. **team-configurator**
- **Expertise**: AI team setup and configuration
- **When to use**: MUST BE USED to set up or refresh AI development team. Use PROACTIVELY on new repos, after major tech stack changes, or when user asks to configure the AI team
- **Tools**: LS, Read, WriteFile, Bash, Glob, Grep

#### 3. **tech-lead-orchestrator**
- **Expertise**: Strategic technical analysis and task planning
- **When to use**: MUST BE USED for multi-step development tasks, feature implementation, or architectural decisions. Returns structured findings and task breakdowns
- **Tools**: Read, Grep, Glob, LS, Bash

### Backend Development Agents

#### 4. **backend-developer**
- **Expertise**: General backend development across any language/stack
- **When to use**: MUST BE USED for server-side code when no framework-specific agent exists. Use PROACTIVELY for production-ready features
- **Tools**: Full access

#### 5. **python-pro**
- **Expertise**: Modern Python 3.11+ with type safety, async programming, data science, web frameworks
- **When to use**: Python development requiring Pythonic patterns and production-ready code quality
- **Tools**: Read, Write, MultiEdit, Bash, pip, pytest, black, mypy, poetry, ruff, bandit

### Quality & Documentation Agents

#### 6. **documentation-specialist**
- **Expertise**: Project documentation (READMEs, API specs, architecture guides)
- **When to use**: MUST BE USED for documentation. Use PROACTIVELY after major features, API changes, or when onboarding developers
- **Tools**: LS, Read, Grep, Glob, Bash, Write

#### 7. **code-reviewer**
- **Expertise**: Security-aware code review
- **When to use**: MUST BE USED after every feature, bug-fix, or pull-request. Use PROACTIVELY before merging to main
- **Tools**: LS, Read, Grep, Glob, Bash

#### 8. **agentic-code-reviewer**
- **Expertise**: Detecting AI-assisted development pitfalls (over-engineering, incomplete implementations, unnecessary complexity)
- **When to use**: After logical chunks of AI-generated code or when reviewing architectural decisions
- **Tools**: Full access

### Performance & Analysis Agents

#### 9. **performance-optimizer**
- **Expertise**: System performance optimization, bottleneck identification
- **When to use**: MUST BE USED for slowness, high cloud costs, or scaling concerns. Use PROACTIVELY before traffic spikes
- **Tools**: LS, Read, Grep, Glob, Bash

#### 10. **code-archaeologist**
- **Expertise**: Legacy/complex codebase exploration and documentation
- **When to use**: MUST BE USED for unfamiliar, legacy, or complex codebases. Use PROACTIVELY before refactors, onboarding, audits
- **Tools**: LS, Read, Grep, Glob, Bash

### Specialized Agents

#### 11. **repo-structure-guardian**
- **Expertise**: Project organization standards and file placement verification
- **When to use**: When adding new components, moving files, adding tests/documentation, or reviewing project organization
- **Tools**: Full access

#### 12. **trading-strategy-consultant**
- **Expertise**: Financial trading strategies, risk management, trading tools
- **When to use**: For trading strategy validation, tool recommendations, technical indicators, portfolio management, or backtesting methodologies
- **Tools**: Full access

### Testing & Quality Agents

#### 13. **test-automator**
- **Expertise**: Test frameworks, CI/CD integration, comprehensive test coverage
- **When to use**: Maintainable, scalable, efficient automated testing solutions
- **Tools**: Read, Write, selenium, cypress, playwright, pytest, jest, appium, k6, jenkins

#### 14. **qa-expert**
- **Expertise**: Comprehensive quality assurance, test strategy, quality metrics
- **When to use**: Manual and automated testing, test planning, quality processes
- **Tools**: Read, Grep, selenium, cypress, playwright, postman, jira, testrail, browserstack

### Database & Infrastructure Agents

#### 15. **database-administrator**
- **Expertise**: High-availability systems, performance optimization, disaster recovery
- **When to use**: PostgreSQL, MySQL, MongoDB, Redis operational excellence
- **Tools**: Read, Write, MultiEdit, Bash, psql, mysql, mongosh, redis-cli, pg_dump, percona-toolkit, pgbench

### DevOps & Cloud Agents

#### 16. **devops-engineer**
- **Expertise**: CI/CD, containerization, cloud platforms, automation
- **When to use**: Bridging development and operations with culture and collaboration focus
- **Tools**: Read, Write, MultiEdit, Bash, docker, kubernetes, terraform, ansible, prometheus, jenkins

### Agent Selection Best Practices

1. **Use framework-specific agents** when available
2. **Chain agents appropriately**: project-analyst → team-configurator → tech-lead-orchestrator → specific implementation agents
3. **Use PROACTIVELY** when agent descriptions indicate proactive use
4. **Launch multiple agents concurrently** when tasks are independent
5. **Trust agent outputs** - they are optimized for their specific domains
6. **For this GPT-Trader project specifically**:
   - Use backend-developer or python-pro for Python ML/trading logic
   - Use trading-strategy-consultant for strategy validation
   - Use test-automator for fixing test suite
   - Use code-archaeologist to understand why system is broken

---

## Code Style & Best Practices

1. **Always use type hints** for function parameters and returns
2. **Document with docstrings** (Google style)
3. **Handle errors gracefully** with try/except blocks
4. **Log important events** using structured logging
5. **Write comprehensive tests** (target 90% coverage - currently 24%)
6. **Use task IDs** for tracking (e.g., CLI-001, TEST-002)
7. **Implement in phases** with validation checkpoints
8. **Fix basics first** before adding features

---

## 🚀 Making Changes - Integration Guide

### Before Starting Any Task
1. Read this entire CLAUDE.md first
2. Check which tasks are already done [x]
3. Choose the next unchecked task [ ]
4. Read the context for that task
5. Test current state before changing

### Branch Strategy
```bash
# Week 1 (current):
git checkout -b fix/critical-cli-imports

# Week 2:
git checkout -b fix/core-integration

# Week 3:
git checkout -b feature/working-strategies

# Week 4:
git checkout -b docs/reality-update
```

### Commit Message Format
```
[TASK-ID] type: short description

Context: Why this change is needed
Impact: What this fixes/enables
Testing: How to verify

Example:
[CLI-001] fix: resolve BacktestEngine import error

Context: CLI is completely broken due to missing import
Impact: Enables all CLI commands to load
Testing: Run 'poetry run gpt-trader --help'
```

### Testing Your Changes
Always test in this order:
1. **Import Test**: `python -c "from module import Class"`
2. **Unit Test**: `pytest tests/unit/test_relevant.py`
3. **Integration Test**: Run the actual command
4. **Smoke Test**: `python scripts/test_cli_smoke.py`

### When You Complete a Task
1. Check it off in this file: `- [x] **TASK-ID**: Description`
2. Commit with proper format
3. Update progress tracking
4. Move to next unchecked task

---

## 🎯 Remember: Goal is 65-75% Functional in 30 Days

We're not trying to build a perfect system. We're trying to:
1. Make it actually run (Week 1)
2. Connect the pieces (Week 2)
3. Provide user value (Week 3)
4. Be honest about it (Week 4)

**Current Week: 1 - Emergency Fixes**
**Current Priority: Make CLI work**
**Success Metric: User can run a backtest**

---

## Important Technical Details

### Repository Cleanup & Single-Source-of-Truth (SoT) Program (ON HOLD)

**Status**: SoT program is ON HOLD until recovery plan completes. The system must work before we can standardize documentation.

Phase 0-2 completed:
- [x] SOT-001 through SOT-023 completed
- [ ] SOT-030 through SOT-053 deferred until system works

---

**Last Updated**: January 2025 - Recovery Plan Initiated
**True State**: 35-45% complete, major fixes needed
**Target State**: 65-75% functional in 30 days
