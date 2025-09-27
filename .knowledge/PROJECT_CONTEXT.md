# ⚠️ DEPRECATED KNOWLEDGE LAYER

**This directory contains outdated information from August 2024.**

The project has migrated from Alpaca/Equities to Coinbase/Perpetuals.

**For current documentation, see: [docs/README.md](../docs/README.md)**

---

# 📋 Project Context - GPT-Trader

## Business Objectives
**Primary Goal**: Autonomous portfolio management - "seed and run"
- [x] Primary trading goal: **Autonomous wealth growth**
- [x] Target markets: **US Equities (S&P 500)**
- [x] Trading frequency: **Daily rebalancing**
- [ ] Capital allocation: **[AWAITING USER INPUT]**

## Risk Parameters
**Approach**: Conservative-Moderate for autonomous safety
- [x] Maximum drawdown: **15% (circuit breaker at 20%)**
- [x] Position size limits: **5% max per position**
- [x] Leverage constraints: **None (1.0x max)**
- [x] Stop-loss rules: **2% daily loss limit**

## Performance Targets
**Realistic & Sustainable**
- [x] Annual return target: **12-18%**
- [x] Sharpe ratio target: **>1.5**
- [x] Win rate target: **55-60%**
- [x] Maximum acceptable volatility: **20% annualized**

## Data Sources
**Cost-Effective Approach**
- [x] Primary data provider: **YFinance (historical)**
- [x] Real-time data: **Alpaca market data**
- [x] Alternative sources: **None initially**
- [x] Historical data range: **5 years for backtesting**

## Broker Integration
**Recommended**: Alpaca for autonomous trading
- [x] Broker: **Alpaca (paper then live)**
- [x] Account type: **Start paper, transition to live**
- [x] API credentials location: **.env file**
- [x] Order types supported: **Market, Limit**

## Compliance & Regulations
- [x] Regulatory jurisdiction: **United States**
- [x] Pattern day trader rules: **Will comply (>$25K or <3 day trades/week)**
- [x] Tax considerations: **Track all trades for tax reporting**
- [x] Audit requirements: **Full trade log maintained**

## ML/AI Strategy
**Adaptive Multi-Strategy Ensemble**
- [x] Prediction targets: **Price direction, volatility regimes**
- [x] Feature importance: **Technical indicators, market microstructure**
- [x] Model update frequency: **Weekly retrain, daily predictions**
- [x] Backtesting requirements: **5 years history, walk-forward validation**

## Infrastructure
- [x] Deployment environment: **Cloud (AWS/GCP) or Local server**
- [x] Computing resources: **2 CPU, 4GB RAM minimum**
- [x] Monitoring tools: **Custom dashboard + email alerts**
- [x] Backup/recovery plan: **Daily backups, 5-minute recovery**

## Trading Strategy Mix
**Multi-Strategy Ensemble (Recommended)**
1. **Trend Following** (30%): Catch major market moves
2. **Mean Reversion** (30%): Profit from volatility
3. **Momentum** (20%): Ride winning stocks
4. **ML-Enhanced** (20%): Adaptive selection

## Safety Mechanisms
**Critical for Autonomous Operation**
- [x] Kill switch implemented
- [x] Circuit breakers on losses
- [x] Position limits enforced
- [x] Data validation checks
- [x] Heartbeat monitoring
- [x] Daily reconciliation

## Monitoring Approach
**Light-Touch Oversight**
- Weekly performance review
- Daily automated reports via email
- Critical alerts only (drawdown, errors)
- Web dashboard for on-demand checking

## Implementation Status

### ✅ Completed
- System architecture (vertical slices)
- 31 agents configured (24 built-in + 7 pilot custom)
- ML infrastructure ready
- Risk management framework

### 🚧 In Progress  
- Strategy backtesting
- Paper trading setup
- Dashboard development

### 📅 Planned
- Alpaca integration
- Live deployment
- Performance optimization

## Decisions Needed from User

### Required:
1. **Initial Capital Amount**: $____________
2. **Risk Level Confirmation**: Conservative ☐ / Moderate ☐
3. **Investment Horizon**: _____ years
4. **Start Date**: When to begin paper trading?

### Optional:
5. **Excluded Sectors/Stocks**: _____________
6. **Tax Account Type**: Taxable ☐ / IRA ☐ / Other ☐
7. **Notification Frequency**: Daily ☐ / Weekly ☐ / On-Alert ☐
8. **Geographic Focus**: US-Only ☐ / International ☐

---
**Next Step**: Provide the 4 required decisions above, and we'll begin configuring your autonomous trading system!
