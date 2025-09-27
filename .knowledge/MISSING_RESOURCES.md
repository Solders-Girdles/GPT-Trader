# ⚠️ DEPRECATED KNOWLEDGE LAYER

**This directory contains outdated information from August 2024.**

The project has migrated from Alpaca/Equities to Coinbase/Perpetuals.

**For current documentation, see: [docs/README.md](../docs/README.md)**

---

# 🚨 Missing Resources Assessment

## Critical Missing Resources

### 1. Project Context & Business Logic
**What's Missing:**
- Trading strategy specifications
- Risk tolerance parameters
- Performance benchmarks/targets
- Regulatory compliance requirements
- Broker API credentials/configuration

**Impact:** Agents can't make informed decisions about trading logic without understanding business requirements.

**Solution:**
```markdown
Create: .knowledge/PROJECT_CONTEXT.md
- Business objectives
- Risk parameters
- Performance targets
- Compliance requirements
```

### 2. Data Sources & APIs
**What's Missing:**
- Available data sources documentation
- API endpoint specifications
- Rate limits and quotas
- Data quality expectations
- Historical data locations

**Impact:** Agents don't know where to get data or how to handle it properly.

**Solution:**
```markdown
Create: .knowledge/DATA_SOURCES.md
- YFinance configuration
- Alpaca API setup
- Alternative data sources
- Rate limit handling
```

### 3. Testing Infrastructure
**What's Missing:**
- Test data fixtures location
- Mock data generation guidelines
- Performance benchmarks
- Test coverage requirements
- CI/CD pipeline configuration

**Impact:** Agents can't properly test changes without knowing testing standards.

**Solution:**
```markdown
Create: .knowledge/TESTING_GUIDE.md
- Test data locations
- Coverage requirements
- Performance benchmarks
- Mock data patterns
```

### 4. Deployment & Operations
**What's Missing:**
- Deployment procedures
- Environment configurations
- Monitoring setup
- Alert thresholds
- Rollback procedures

**Impact:** Agents can't help with deployment or operations tasks.

**Solution:**
```markdown
Create: .knowledge/DEPLOYMENT.md
- Environment setup
- Deployment steps
- Monitoring configuration
- Incident response
```

### 5. ML Model Specifications
**What's Missing:**
- Model training parameters
- Feature engineering specs
- Model evaluation metrics
- Retraining schedules
- Model versioning strategy

**Impact:** ML agents lack context for model development and maintenance.

**Solution:**
```markdown
Create: .knowledge/ML_SPECIFICATIONS.md
- Feature definitions
- Training parameters
- Evaluation metrics
- Versioning strategy
```

## Moderate Priority Missing Resources

### 6. Performance Baselines
- Current system performance metrics
- Latency requirements
- Throughput targets
- Resource constraints

### 7. Error Handling Patterns
- Standard error responses
- Retry strategies
- Circuit breaker patterns
- Logging standards

### 8. Security Requirements
- Authentication methods
- API key management
- Data encryption requirements
- Audit logging needs

## Nice-to-Have Resources

### 9. Code Style Preferences
- Naming conventions beyond PEP8
- Documentation style
- Comment patterns
- File organization preferences

### 10. Communication Patterns
- How to report progress
- Error message formatting
- User notification preferences
- Log message standards

## Immediate Action Items

### Priority 1: Create Core Context Files
```bash
.knowledge/
├── PROJECT_CONTEXT.md      # Business logic & requirements
├── DATA_SOURCES.md         # API & data specifications
├── TESTING_GUIDE.md        # Testing infrastructure
├── DEPLOYMENT.md           # Operations procedures
└── ML_SPECIFICATIONS.md    # ML model requirements
```

### Priority 2: Update Existing Documentation
- Add "Available Resources" section to START_HERE.md
- Link AGENTS.md from START_HERE.md
- Create quick reference for common tasks

### Priority 3: Create Templates
```bash
.knowledge/templates/
├── feature_spec.md         # New feature template
├── bug_report.md          # Bug investigation template
├── performance_issue.md    # Performance analysis template
└── ml_experiment.md       # ML experiment template
```

## Questions for User

To properly equip agents, we need answers to:

1. **Trading Strategy**: What strategies should the system implement?
2. **Risk Limits**: What are acceptable drawdown/exposure limits?
3. **Data Sources**: Which brokers/data providers are you using?
4. **Performance Targets**: What returns/Sharpe ratio are you targeting?
5. **Deployment Target**: Where will this system run (cloud/local)?
6. **Compliance**: Any regulatory requirements to consider?
7. **ML Goals**: What should ML models predict/optimize?
8. **Testing Standards**: What test coverage is required?
9. **Monitoring**: What metrics need real-time monitoring?
10. **Incident Response**: How should agents handle failures?

## Summary

**Well-Equipped Areas:**
- ✅ Code structure (vertical slices)
- ✅ Architecture rules
- ✅ File placement guide
- ✅ Available agents directory

**Critical Gaps:**
- ❌ Business context
- ❌ Data specifications
- ❌ Testing infrastructure
- ❌ Deployment procedures
- ❌ ML requirements

**Recommendation:** Create the 5 priority context files to give agents the domain knowledge they need to be truly effective.