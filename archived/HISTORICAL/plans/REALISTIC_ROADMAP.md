# GPT-Trader Recovery Roadmap

## Executive Summary
**Current State**: 45% Functional
**Target State**: 75% Functional
**Timeline**: 8 weeks (realistic with buffer)
**Development Effort**: 20-30 hours/week

## System Health Assessment

### What's Actually Working (45%)
- ✅ CLI framework loads and displays help
- ✅ 2 strategies import successfully (demo_ma, trend_breakout)
- ✅ Data download from yfinance functional
- ✅ Poetry environment configured
- ✅ ML modules exist (not integrated)
- ✅ Basic project structure in place

### Critical Issues
- ❌ Backtest command has parameter mismatch
- ❌ No production orchestrator (missing file)
- ❌ Test suite broken (import errors)
- ❌ ML pipeline not connected to trading
- ❌ No paper trading implementation
- ❌ Monitoring/dashboards disconnected

## Phase 1: Emergency Stabilization (Weeks 1-2)
**Goal**: Make system minimally runnable
**Target**: 55% functional
**Hours Required**: 40-60 hours

### Week 1 Tasks (20-30 hours)
1. **Fix Backtest Parameter Mismatch** (2-3 hours)
   - Align CLI arguments with run_backtest function
   - Test with AAPL demo data
   - Verify output generation

2. **Repair Test Infrastructure** (8-10 hours)
   - Fix import paths in tests/
   - Update conftest.py fixtures
   - Get 100+ tests collecting
   - Target: 30% pass rate

3. **Create Working Demo** (4-6 hours)
   - Simple end-to-end backtest
   - Document all requirements
   - Add to examples/

4. **Document Current State** (3-4 hours)
   - Audit all modules
   - Create dependency map
   - List known failures

5. **Quick Wins** (3-4 hours)
   - Fix obvious import errors
   - Add missing __init__.py files
   - Update .env.template

### Week 2 Tasks (20-30 hours)
1. **Strategy Validation** (6-8 hours)
   - Test all strategies in isolation
   - Fix signal generation bugs
   - Add position sizing logic

2. **Basic Integration** (8-10 hours)
   - Connect data → strategy → backtest
   - Add error handling
   - Create integration tests

3. **CLI Enhancement** (6-8 hours)
   - Fix all command imports
   - Add validation
   - Improve error messages

4. **Documentation Update** (4-5 hours)
   - Update README with reality
   - Create QUICKSTART.md
   - Add troubleshooting guide

### Success Criteria Phase 1
- [ ] `gpt-trader backtest` runs without errors
- [ ] 150+ tests collect successfully
- [ ] 30% test pass rate achieved
- [ ] 3 working examples in examples/
- [ ] Clear documentation of what works/doesn't

## Phase 2: Core Integration (Weeks 3-4)
**Goal**: Connect disconnected modules
**Target**: 65% functional
**Hours Required**: 40-60 hours

### Week 3 Tasks (20-30 hours)
1. **Create Minimal Orchestrator** (10-12 hours)
   - File: src/bot/live/production_orchestrator.py
   - Basic event loop
   - Module registration
   - Health checks

2. **Wire Data Pipeline** (6-8 hours)
   - Connect data sources
   - Add caching layer
   - Implement validation

3. **Strategy Selection Logic** (4-6 hours)
   - Basic selection criteria
   - Performance tracking
   - Fallback mechanisms

### Week 4 Tasks (20-30 hours)
1. **Portfolio Management** (8-10 hours)
   - Position tracking
   - Risk constraints
   - Allocation logic

2. **Database Integration** (8-10 hours)
   - SQLite for simplicity
   - Trade/position storage
   - Performance metrics

3. **Basic Monitoring** (4-6 hours)
   - Log aggregation
   - Error tracking
   - Simple metrics

### Success Criteria Phase 2
- [ ] Orchestrator file exists and runs
- [ ] Data flows through 3+ modules
- [ ] Database stores trades
- [ ] 50% test pass rate
- [ ] Basic monitoring operational

## Phase 3: ML Integration (Weeks 5-6)
**Goal**: Connect ML pipeline to trading
**Target**: 70% functional
**Hours Required**: 40-60 hours

### Week 5 Tasks (20-30 hours)
1. **ML Pipeline Connection** (10-12 hours)
   - Integrate existing ML modules
   - Add prediction interface
   - Connect to strategy selection

2. **Model Training** (6-8 hours)
   - Enable basic training
   - Add validation
   - Store model artifacts

3. **Feature Engineering** (4-6 hours)
   - Connect to data pipeline
   - Add technical indicators
   - Implement caching

### Week 6 Tasks (20-30 hours)
1. **Model Serving** (8-10 hours)
   - Load trained models
   - Make predictions
   - Add confidence scoring

2. **Performance Tracking** (6-8 hours)
   - Track prediction accuracy
   - Monitor drift
   - Add retraining triggers

3. **Integration Tests** (6-8 hours)
   - End-to-end ML tests
   - Performance benchmarks
   - Stress testing

### Success Criteria Phase 3
- [ ] ML models train successfully
- [ ] Predictions influence trading
- [ ] Model performance tracked
- [ ] 60% test pass rate
- [ ] ML dashboard shows metrics

## Phase 4: Paper Trading (Weeks 7-8)
**Goal**: Enable simulated live trading
**Target**: 75% functional
**Hours Required**: 40-60 hours

### Week 7 Tasks (20-30 hours)
1. **Paper Trading Engine** (10-12 hours)
   - Alpaca paper integration
   - Order simulation
   - Position tracking

2. **Real-time Data** (6-8 hours)
   - Connect live feeds
   - Add buffering
   - Handle gaps/errors

3. **Risk Management** (4-6 hours)
   - Position limits
   - Stop losses
   - Drawdown protection

### Week 8 Tasks (20-30 hours)
1. **Dashboard Creation** (8-10 hours)
   - Streamlit interface
   - Real-time updates
   - Performance charts

2. **Deployment Scripts** (6-8 hours)
   - Docker configuration
   - Auto-restart logic
   - Health monitoring

3. **Documentation** (6-8 hours)
   - User guide
   - API documentation
   - Deployment guide

### Success Criteria Phase 4
- [ ] Paper trading places orders
- [ ] Dashboard displays positions
- [ ] Risk limits enforced
- [ ] 70% test pass rate
- [ ] System runs for 24 hours stable

## Resource Requirements

### Developer Time
- **Total Hours**: 320-480 hours
- **Weekly Commitment**: 20-30 hours
- **Duration**: 8 weeks minimum

### Technical Skills Needed
- Python (advanced)
- Pandas/NumPy (intermediate)
- ML basics (scikit-learn)
- API integration (REST)
- Testing (pytest)
- Basic DevOps (Docker)

### Infrastructure
- Development machine (8GB RAM minimum)
- GitHub for version control
- Basic CI/CD (GitHub Actions)
- Cloud instance for paper trading (optional)

### External Dependencies
- Alpaca API account (free)
- YFinance data access
- PostgreSQL or SQLite
- Redis for caching (optional)

## Risk Assessment

### High Risk Items
1. **ML Integration Complexity** (40% risk)
   - Mitigation: Start with simple models
   - Fallback: Rule-based strategies only

2. **Live Data Reliability** (30% risk)
   - Mitigation: Multiple data sources
   - Fallback: Delayed/batched processing

3. **Technical Debt** (50% risk)
   - Mitigation: Incremental refactoring
   - Fallback: Accept some inefficiency

### Medium Risk Items
1. **Testing Coverage** (30% risk)
   - Mitigation: Focus on critical paths
   - Fallback: Manual testing

2. **Performance Issues** (25% risk)
   - Mitigation: Profile and optimize
   - Fallback: Reduce data frequency

### Dependencies That Could Block
- Alpaca API changes
- YFinance rate limits
- Missing ML expertise
- Database corruption

## Quick Wins (Can Do Today)

### Top 5 Immediate Actions (< 2 hours each)
1. **Fix Backtest Parameters** (1 hour)
   ```python
   # In commands.py, change:
   # "start_date" → "start"
   # "end_date" → "end"
   ```

2. **Fix Test Imports** (2 hours)
   ```python
   # Add to tests/conftest.py:
   import sys
   sys.path.insert(0, 'src')
   ```

3. **Create Working Example** (1 hour)
   ```bash
   # examples/simple_backtest.py
   # Minimal working backtest
   ```

4. **Update README** (30 mins)
   - Remove "production-ready" claims
   - Add "under development" banner
   - List what actually works

5. **Add Health Check** (1 hour)
   ```python
   # scripts/health_check.py
   # Test all imports
   # Report status
   ```

## Recommended Team Structure

### Solo Developer Approach (Current)
- Focus on sequential progress
- Complete phases before moving on
- Document everything
- Test incrementally

### If Adding Help
1. **Backend Developer**
   - Focus: Core integration
   - Skills: Python, APIs, databases

2. **ML Engineer**
   - Focus: ML pipeline
   - Skills: scikit-learn, pandas, model serving

3. **DevOps Engineer**
   - Focus: Deployment, monitoring
   - Skills: Docker, CI/CD, cloud

## Success Metrics Summary

### End of Week 2 (55% functional)
- Backtest command works
- 30% tests pass
- 3 strategies validated
- Basic documentation complete

### End of Week 4 (65% functional)
- Orchestrator operational
- Data pipeline connected
- Database stores trades
- 50% tests pass

### End of Week 6 (70% functional)
- ML predictions working
- Models training/serving
- Performance tracked
- 60% tests pass

### End of Week 8 (75% functional)
- Paper trading live
- Dashboard operational
- Risk management active
- 70% tests pass

## Monthly Maintenance Post-Recovery

After reaching 75% functional:
- 10 hours/month bug fixes
- 10 hours/month feature additions
- 5 hours/month documentation
- 5 hours/month testing

## Notes on Realism

This roadmap assumes:
- No major architectural changes needed
- Existing code is salvageable
- Developer has Python experience
- 20-30 hours/week available
- Some delays and debugging time

Buffer time is built into each phase for:
- Unexpected bugs (common)
- Integration issues (very common)
- Testing and validation
- Documentation updates
- Code review and cleanup

## Next Steps

1. **Commit to Phase 1** (2 weeks)
2. **Fix critical bugs first**
3. **Document progress daily**
4. **Review/adjust after each phase**
5. **Celebrate small wins**

---

*Last Updated: January 2025*
*Status: Recovery Plan Initiated*
*True State: 45% complete*
*Target: 75% in 8 weeks*
EOF < /dev/null
