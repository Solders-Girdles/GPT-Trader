# ⚠️ DEPRECATED KNOWLEDGE LAYER

**This directory contains outdated information from August 2024.**

The project has migrated from Alpaca/Equities to Coinbase/Perpetuals.

**For current documentation, see: [docs/README.md](../docs/README.md)**

---

# 🤖 Autonomous Trading System Recommendations

## Your Goal: Seed & Run Autonomously
Based on your desire for a "set it and forget it" autonomous portfolio, here are my recommendations:

## 📊 Recommended Configuration

### 1. Trading Strategy Approach
**Recommendation: Multi-Strategy Ensemble**
- **Why**: Diversification reduces risk in autonomous operation
- **Components**:
  - Trend Following (30% allocation) - Catches major moves
  - Mean Reversion (30% allocation) - Profits from volatility
  - Momentum (20% allocation) - Rides winning stocks
  - ML-Enhanced Selection (20% allocation) - Adaptive to market

### 2. Risk Management
**Recommendation: Conservative-Moderate**
- **Maximum Drawdown**: 15% (stop all trading at 20%)
- **Position Size**: Max 5% per position (20 positions max)
- **Daily Loss Limit**: 2% (pause trading for day)
- **Leverage**: None initially (cash only)
- **Why**: Autonomous systems need strict guardrails

### 3. Performance Targets
**Recommendation: Realistic & Sustainable**
- **Annual Return Target**: 12-18%
- **Sharpe Ratio Target**: >1.5
- **Win Rate Target**: 55-60%
- **Why**: Better to aim for consistent returns than home runs

### 4. Market Selection
**Recommendation: Liquid US Equities**
- **Universe**: S&P 500 stocks
- **Minimum Volume**: 1M shares/day
- **Price Range**: $20-$500
- **Why**: High liquidity = better fills, less slippage

### 5. Trading Frequency
**Recommendation: Daily Rebalancing**
- **Signal Generation**: End of day
- **Order Execution**: Market open next day
- **Position Holding**: 1-30 days average
- **Why**: Reduces noise, lowers costs, more stable

### 6. Broker Selection
**Recommendation: Alpaca**
- **Pros**: 
  - Free API access
  - Paper trading for testing
  - Commission-free trades
  - Good API reliability
- **Alternative**: Interactive Brokers (more markets, higher costs)

### 7. Data Sources
**Primary Recommendation: YFinance + Alpaca**
- **Historical**: YFinance (free, reliable)
- **Real-time**: Alpaca data feed (included with account)
- **Alternative Data**: Start without, add later if needed
- **Cost**: ~$0/month initially

### 8. Capital Allocation
**Recommendation: Phased Approach**
- **Phase 1**: $10,000 paper trading (1 month)
- **Phase 2**: $5,000 real money (3 months)
- **Phase 3**: Add remaining capital if profitable
- **Reserve**: Keep 20% in cash for opportunities

### 9. Monitoring & Intervention
**Recommendation: Weekly Check-ins**
- **Automated Reports**: Daily email summaries
- **Dashboard**: Web interface for monitoring
- **Alerts**: Only for critical issues
- **Manual Override**: Available but rarely used

### 10. Safety Mechanisms
**Critical for Autonomous Operation:**
1. **Kill Switch**: One command stops everything
2. **Circuit Breakers**: Auto-pause on anomalies
3. **Position Limits**: Hard caps on exposure
4. **Correlation Limits**: Avoid concentration
5. **Data Validation**: Skip trading on bad data
6. **Heartbeat Monitor**: Detect system failures

## 🚀 Implementation Roadmap

### Month 1: Foundation
- [x] System setup (done!)
- [ ] Broker account (Alpaca paper)
- [ ] Backtest all strategies
- [ ] Optimize parameters

### Month 2: Paper Trading
- [ ] Deploy to paper trading
- [ ] Monitor daily performance
- [ ] Refine based on results
- [ ] Stress test edge cases

### Month 3: Gradual Live Deployment
- [ ] Start with $1,000 live
- [ ] Scale up weekly if profitable
- [ ] Full deployment by month end

### Month 4+: Autonomous Operation
- [ ] Weekly performance reviews
- [ ] Monthly strategy rebalancing
- [ ] Quarterly system updates

## 💡 Key Decisions to Make

### Required from You:
1. **Initial Capital**: How much to seed? ($10K-$100K recommended)
2. **Risk Tolerance**: Conservative (my recommendation) or Moderate?
3. **Time Horizon**: 1+ years? (recommended for compound growth)
4. **Geography**: US-only or international? (US recommended to start)

### Optional Preferences:
5. **ESG/Ethics**: Any stocks to exclude?
6. **Sectors**: Any to avoid or prefer?
7. **Tax**: Tax-advantaged account or taxable?
8. **Notifications**: How often want updates?

## 🎯 Success Metrics

### Good Autonomous System:
- Consistent returns (not necessarily highest)
- Low drawdowns (<15%)
- Minimal intervention needed
- Survives various market conditions
- Clear audit trail

### Warning Signs:
- Drawdown >15%
- Win rate <45%
- Excessive trading (>50 trades/day)
- Concentration in few positions
- System errors/crashes

## 📋 Next Steps

1. **Review these recommendations**
2. **Provide the 4 required decisions**
3. **I'll fill in PROJECT_CONTEXT.md**
4. **Set up Alpaca paper account**
5. **Begin backtesting phase**

## 💭 Philosophy for Autonomous Trading

**"Slow and steady wins the race"**

For autonomous systems, we prioritize:
- **Reliability** over returns
- **Consistency** over brilliance  
- **Safety** over speed
- **Diversification** over concentration
- **Adaptation** over rigidity

The goal is a system that can run for months without intervention, growing your capital steadily while you sleep.

---

**Ready to proceed?** Just let me know:
1. Your initial capital amount
2. Confirm conservative risk approach
3. Your investment time horizon
4. Any specific preferences or exclusions