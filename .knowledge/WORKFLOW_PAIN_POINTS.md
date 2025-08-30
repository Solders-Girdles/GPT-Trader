# Workflow Pain Points

## 🔍 Major Workflow Issues Resolved (Phase 1B - 2025-08-16)

### 1. TodoWrite Tool Misuse ✅ FIXED
**Problem**: Using TodoWrite for persistent project state instead of session-scoped tasks
**Root Cause**: Misunderstanding of Claude Code workflow patterns
**Solution**: 
- Knowledge layer (`.knowledge/`) for persistent state
- TodoWrite for session-scoped work only
- Clear session handoff protocols
**Result**: Proper workflow discipline established

### 2. Signal → Allocation Flow Opacity ✅ FIXED  
**Problem**: Strategies generate signals but 0 trades executed
**Root Cause**: Allocator only checking last bar signal value
**Solution**: Implemented 120-bar lookback window for recent signals
**Result**: Fixed allocator issue, 4/7 strategies now execute trades

### 3. Strategy Parameter Conservatism ✅ MAJOR BREAKTHROUGH
**Problem**: 4/7 strategies generating 0 signals due to overly conservative parameters
**Root Cause**: Parameters designed for extreme market conditions, not autonomous trading
**Solution**: 
- Mean Reversion: RSI 30/70 → 40/60, period 14 → 10
- Momentum: 3% → 1.5% threshold, 1.5x → 1.2x volume
- OptimizedMA: MA 10/20 → 5/15, disabled restrictive filters
- Enhanced Trend: 55 → 20 day lookback, disabled volume filter
**Result**: 1 strategy completely fixed, 3 generate signals but need allocator compatibility fix

### 3. Test Coverage Visibility Gap
**Problem**: Can't easily see which components have tests
**Current**: Must manually check test directories
**Needed**: Coverage report in knowledge layer

### 4. Column Name Inconsistency ✅ FIXED
**Problem**: YFinance returns "Close", strategies expect "close"
**Solution**: DataPipeline normalizes columns to lowercase
**Result**: Backtesting now works

### 5. Mock/Patch Testing Patterns ✅ FIXED  
**Problem**: Tests failing due to incorrect patch locations
**Solution**: Documented pattern in KNOWN_FAILURES.md
**Result**: 51 tests now passing

## 📊 Workflow Improvements Validated

### What's Working Well:
- Agent delegation with complete context
- Knowledge layer persistence between sessions
- Test-driven problem solving
- Organized documentation structure

### What Still Needs Work:
- Signal flow debugging tools
- Strategy parameter tuning guidance
- Test coverage reporting
- Performance metrics baselines

## 🎯 Next Actions

1. **Debug volatility strategy signals**
   - Add logging to signal generation
   - Verify ATR calculation working
   - Check threshold parameters

2. **Create signal flow visualizer**
   - Show strategy → signals → allocator → trades
   - Identify where flow breaks

3. **Add test coverage report**
   - Integrate into PROJECT_STATE.json
   - Show per-module coverage

## 📝 Notes

The workflow is significantly improved from initial state:
- Agents now have reliable context via knowledge layer
- Problems are documented with solutions
- Repository is organized and navigable

Main remaining issue is understanding why strategies that "work" don't generate trades. This is likely a parameter tuning or threshold issue rather than a code bug.