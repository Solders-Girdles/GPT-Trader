# GPT-Trader V2 Control Center

## 🎯 System Status: CLEAN & INTELLIGENT

**Architecture**: Vertical Slice + ML Intelligence (Path B: 50% Complete)  
**Repository**: Ultra-clean structure (76MB+ archived files purged)  
**Knowledge Layer**: Current only - no deprecated references  
**Navigation**: Use `src/bot_v2/SLICES.md` for development  

## 📁 Current Reality (Post-Cleanup)

### ✅ What We Have: Clean Trading System
```
src/bot_v2/                   # ONLY active system (8K lines)
├── features/                 # 9 feature slices (complete isolation)
│   ├── backtest/            ✅ Historical testing
│   ├── paper_trade/         ✅ Simulated trading
│   ├── analyze/             ✅ Market analysis
│   ├── optimize/            ✅ Parameter optimization
│   ├── live_trade/          ✅ Broker integration
│   ├── monitor/             ✅ Health monitoring
│   ├── data/                ✅ Data management
│   ├── ml_strategy/         ✅ ML strategy selection (Week 1-2)
│   └── market_regime/       ✅ Regime detection (Week 3)
└── test_*.py                ✅ Integration tests
```

### 🧠 Intelligence Components (Path B: Smart Money)

**Week 1-2: ML Strategy Selection ✅ COMPLETE**
- Dynamic strategy switching based on conditions
- Confidence scoring for predictions
- 35% return improvement in backtesting

**Week 3: Market Regime Detection ✅ COMPLETE**  
- 7 regime types (Bull/Bear/Sideways × Quiet/Volatile + Crisis)
- Real-time monitoring with transition prediction
- Regime-aware strategy recommendations

**Week 4: Intelligent Position Sizing 🎯 NEXT**
- Kelly Criterion implementation
- Confidence-based allocation
- Regime-adjusted sizing

## 🏗️ Architecture Principles

### Complete Isolation (For Optimal Token Efficiency)
- **Each slice is self-contained** (~500 tokens to load)
- **No shared dependencies** between slices
- **Local implementations** of everything needed
- **Duplication preferred** over cross-slice imports

### Vertical Slice Organization
```
Task: "Run backtest" → Load ONLY features/backtest/ 
Task: "Detect regime" → Load ONLY features/market_regime/
Task: "ML strategy selection" → Load ONLY features/ml_strategy/
```

## 🚀 Quick Commands

```bash
# Test specific slice
poetry run python src/bot_v2/test_backtest.py

# Run ML strategy selection
poetry run python -c "from src.bot_v2.features.ml_strategy import predict_best_strategy; print(predict_best_strategy('AAPL'))"

# Detect market regime
poetry run python -c "from src.bot_v2.features.market_regime import detect_regime; print(detect_regime('AAPL'))"
```

## ⚠️ What's Archived (Don't Use)

```
archived/                              # Minimal historical reference (~1.2MB total)
├── bot_v2_horizontal_20250817/       # Old horizontal architecture (428KB)  
├── old_plans/                        # Historical planning docs (412KB)
├── old_reports/                      # Historical analysis reports (368KB)
├── benchmarks/                       # Performance baselines (8KB)
└── legacy-v1/                        # Empty legacy directory
```

## 🎯 Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core Slices | ✅ Complete | 9 slices operational |
| ML Strategy | ✅ Complete | Weeks 1-2 finished |
| Market Regime | ✅ Complete | Week 3 finished |
| Position Sizing | 🎯 Next | Week 4 in progress |
| Archive Purge | ✅ Complete | 76MB+ deprecated files removed |

## ⚠️ Knowledge Layer Warnings

**DEPRECATED DOCS (Don't Trust):**
- `docs/SYSTEM_ARCHITECTURE.md` - References old V1 paths
- Most files in `docs/` - Written for old horizontal architecture
- Any documentation mentioning `src/bot/` paths

**CURRENT DOCS (Use These):**
- `CLAUDE.md` (this file) - Current control center
- `src/bot_v2/SLICES.md` - Agent navigation guide
- `src/bot_v2/README.md` - Current architecture overview
- `README.md` - Updated V2 system description

## 🔄 Session Continuity Protocols

### Starting a New Session: Context Recovery
1. **Check Current Progress**: Read the todo list (always displayed at session start)
2. **Review Recent History**: Read `src/bot_v2/WEEK_*_COMPLETE.md` files for project milestones
3. **Identify Active Work**: Look for "in_progress" tasks to understand current context
4. **Check Project State**: Review relevant slice documentation in `src/bot_v2/features/`

### Multi-Session Task Management
```bash
# Essential reads for context recovery:
1. Todo list (system provides automatically)
2. src/bot_v2/WEEK_3_COMPLETE.md (latest milestone) 
3. src/bot_v2/SLICES.md (current architecture)
4. This CLAUDE.md file (system status)
```

### Detecting Continuation Scenarios
- **"Continue with Week X"** → Read `WEEK_X_PLAN.md` or `WEEK_X_COMPLETE.md`
- **"Fix the issue we discussed"** → Check recent project files for context clues
- **"Where were we?"** → Review in_progress todos and recent completion dates

## 📚 Knowledge Layer Usage Guide

### When to Check `docs/knowledge/`
- **Complex Task Decomposition**: Use `TASK_TEMPLATES.md` for agent delegation
- **Agent Coordination Issues**: Reference `AGENT_WORKFLOW.md` for best practices
- **Workflow Problems**: Check `KNOWLEDGE_LAYER_MAINTENANCE.md` for automation

### Essential Knowledge Files
- **`docs/knowledge/TASK_TEMPLATES.md`**: Copy-paste templates for agent delegation
- **`docs/knowledge/AGENT_WORKFLOW.md`**: Advanced coordination principles
- **`docs/knowledge/AGENT_DELEGATION_GUIDE.md`**: When and how to use Task tool

### Navigation Priority
```
1st: Todo list (immediate priorities)
2nd: src/bot_v2/SLICES.md (architecture navigation)  
3rd: docs/knowledge/ (workflow guidance)
4th: src/bot_v2/WEEK_*.md (project history)
```

## 🎯 Complex Task Management Protocols

### Large Task Decomposition
1. **Assess Scope**: If task requires >3 steps, consider using Task tool
2. **Check Templates**: Look in `docs/knowledge/TASK_TEMPLATES.md` for relevant patterns
3. **Break Into Phases**: Use todo list to track multi-phase work
4. **Document Progress**: Create or update WEEK_*.md files for significant milestones

### Agent Delegation Decision Matrix
```
Use Task tool when:
✅ Task requires >500 tokens of context
✅ Specialized expertise needed (architecture, ML, testing)
✅ Multi-file changes across different domains
✅ Template exists in TASK_TEMPLATES.md

Handle directly when:
❌ Simple edits to current files
❌ Reading and analysis tasks
❌ Todo list management
❌ Navigation and status checks
```

### Progress Tracking Best Practices
- **Update todos in real-time** as work progresses
- **Mark completion immediately** when tasks finish
- **Create WEEK_*.md** for major milestones (>1 week effort)
- **Keep CLAUDE.md current** with system status updates

## 🧠 Meta-Workflow Intelligence

### Session Type Detection
- **Continuation Session**: User references previous work → Use context recovery protocols
- **New Feature Session**: User requests new capability → Plan using vertical slice principles  
- **Debugging Session**: User reports issues → Use diagnostic protocols from knowledge layer
- **Cleanup Session**: User wants organization → Follow architectural consistency guidelines

### Cross-Session Coordination
```bash
# For complex multi-session work:
1. Create clear todos with specific next steps
2. Document any architectural decisions made
3. Update relevant WEEK_*.md files with progress
4. Ensure SLICES.md reflects any new components
```

### Knowledge Layer Maintenance
- **After major changes**: Update relevant documentation
- **After completing milestones**: Create or update WEEK_*.md files
- **After architectural decisions**: Update SLICES.md or ARCHITECTURE.md
- **Before complex work**: Check docs/knowledge/ for applicable guidance

## 📊 Next Steps

**Immediate**: Continue Week 4 - Intelligent Position Sizing
**Week 5**: Performance Prediction models
**Week 6**: Integration & Testing of all ML components

---

**Last Updated**: August 17, 2025  
**Repository Status**: Clean (root Python files archived)  
**Architecture**: Vertical Slice with Complete Isolation  
**Intelligence**: 50% complete (2 of 4 ML components)  
**Meta-Workflow**: Enhanced with session continuity and complex task management protocols