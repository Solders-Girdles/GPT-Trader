# GPT-Trader Operations Runbook
**Phase 3, Week 8: OPS-025 to OPS-032**

## Table of Contents
1. [System Overview](#system-overview)
2. [Incident Response](#incident-response)
3. [Alert Response Playbooks](#alert-response-playbooks)
4. [Troubleshooting Guide](#troubleshooting-guide)
5. [Performance Tuning](#performance-tuning)
6. [Monitoring Setup](#monitoring-setup)
7. [Best Practices](#best-practices)
8. [Emergency Procedures](#emergency-procedures)

---

## System Overview

### Architecture Components
- **ML Pipeline**: Feature engineering, model training, prediction services
- **Risk Management**: VaR calculation, stress testing, portfolio optimization
- **Trading Engine**: Order execution, position management
- **Data Pipeline**: Real-time feeds, historical data, validation
- **Monitoring**: Structured logging, intelligent alerts, dashboards

### Key Metrics to Monitor
| Metric | Normal Range | Alert Threshold | Critical Threshold |
|--------|--------------|-----------------|-------------------|
| Model Accuracy | 58-62% | < 55% | < 50% |
| Predictions/sec | 4500-5500 | < 4000 | < 3000 |
| API Latency | 10-30ms | > 50ms | > 100ms |
| Memory Usage | 40-70% | > 80% | > 90% |
| CPU Usage | 30-60% | > 75% | > 85% |
| Error Rate | < 0.1% | > 0.5% | > 1% |

---

## Incident Response

### Severity Levels

#### 🔴 **Critical (P1)**
**Response Time**: < 5 minutes
- Trading system down
- Data loss occurring
- Security breach
- Complete model failure

**Actions**:
1. Page on-call engineer immediately
2. Activate incident response team
3. Begin incident bridge
4. Notify stakeholders

#### 🟠 **High (P2)**
**Response Time**: < 30 minutes
- Performance degradation > 50%
- Model accuracy below threshold
- Risk limits breached
- Partial system failure

**Actions**:
1. Alert on-call engineer
2. Investigate root cause
3. Implement mitigation
4. Monitor recovery

#### 🟡 **Medium (P3)**
**Response Time**: < 2 hours
- Minor performance issues
- Non-critical component failure
- Warning thresholds exceeded

**Actions**:
1. Create incident ticket
2. Assign to appropriate team
3. Schedule resolution

#### 🔵 **Low (P4)**
**Response Time**: < 24 hours
- Informational alerts
- Minor configuration issues
- Documentation updates needed

---

## Alert Response Playbooks

### 📊 Model Performance Degradation

**Alert**: "Model accuracy dropped below 55%"

**Diagnosis Steps**:
```bash
# 1. Check recent model performance
python scripts/check_model_performance.py --last-hours 24

# 2. Verify data quality
python scripts/validate_data_quality.py --symbol ALL

# 3. Check for data drift
python scripts/detect_drift.py --model current --threshold 0.05

# 4. Review recent predictions
python scripts/analyze_predictions.py --count 1000
```

**Resolution**:
1. If data drift detected → Trigger retraining
2. If data quality issue → Fix data pipeline
3. If sudden market change → Adjust model parameters
4. If persistent → Rollback to previous model

### 💻 System Resource Alert

**Alert**: "Memory usage exceeds 85%"

**Diagnosis Steps**:
```bash
# 1. Identify memory consumers
ps aux | sort -rk 4 | head -20

# 2. Check for memory leaks
python scripts/memory_profiler.py --component all

# 3. Review cache usage
redis-cli INFO memory

# 4. Check database connections
psql -c "SELECT count(*) FROM pg_stat_activity;"
```

**Resolution**:
1. Clear unnecessary caches
2. Restart memory-intensive services
3. Scale horizontally if needed
4. Optimize memory-heavy operations

### 🚨 Trading System Error

**Alert**: "Order execution failed"

**Diagnosis Steps**:
```bash
# 1. Check broker connectivity
python scripts/test_broker_connection.py

# 2. Review order logs
tail -f logs/trading/orders.log | grep ERROR

# 3. Verify account status
python scripts/check_account_status.py

# 4. Check rate limits
python scripts/check_rate_limits.py
```

**Resolution**:
1. Retry failed orders (if safe)
2. Switch to backup broker
3. Adjust position sizes
4. Halt trading if critical

### 📈 Risk Limit Breach

**Alert**: "Portfolio VaR exceeds limit"

**Diagnosis Steps**:
```bash
# 1. Calculate current risk metrics
python scripts/calculate_risk_metrics.py

# 2. Review positions
python scripts/list_positions.py --sort-by risk

# 3. Check correlations
python scripts/check_correlations.py

# 4. Run stress test
python scripts/run_stress_test.py --scenario current
```

**Resolution**:
1. Reduce position sizes
2. Close high-risk positions
3. Increase hedging
4. Adjust risk parameters

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Slow Predictions
**Symptoms**: Prediction latency > 100ms

**Diagnosis**:
```python
# Profile prediction pipeline
from src.bot.ml import IntegratedMLPipeline
pipeline = IntegratedMLPipeline()
pipeline.profile_prediction()
```

**Solutions**:
1. Enable model caching
2. Reduce feature set
3. Use faster model variant
4. Scale prediction service

#### Issue: Database Connection Errors
**Symptoms**: "Connection pool exhausted"

**Diagnosis**:
```sql
-- Check active connections
SELECT count(*) FROM pg_stat_activity;

-- Find long-running queries
SELECT pid, age(clock_timestamp(), query_start), query 
FROM pg_stat_activity 
WHERE state != 'idle' 
ORDER BY query_start;
```

**Solutions**:
1. Increase connection pool size
2. Kill long-running queries
3. Optimize slow queries
4. Add read replicas

#### Issue: Data Feed Interruption
**Symptoms**: No new data for > 1 minute

**Diagnosis**:
```python
# Check data feed status
from src.bot.dataflow import RealtimeFeed
feed = RealtimeFeed()
feed.check_health()
```

**Solutions**:
1. Switch to backup data provider
2. Use cached/historical data
3. Restart data connectors
4. Verify API credentials

---

## Performance Tuning

### ML Pipeline Optimization

```python
# Optimize feature engineering
config.feature_engineering = {
    'n_jobs': -1,  # Use all cores
    'cache_features': True,
    'batch_size': 1000
}

# Optimize model training
config.model_training = {
    'early_stopping_rounds': 50,
    'use_gpu': True,
    'n_parallel_trees': 4
}

# Optimize predictions
config.prediction = {
    'batch_predictions': True,
    'cache_models': True,
    'prediction_threads': 4
}
```

### Database Optimization

```sql
-- Add indexes for common queries
CREATE INDEX idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX idx_positions_symbol ON positions(symbol);

-- Vacuum and analyze
VACUUM ANALYZE predictions;
VACUUM ANALYZE positions;

-- Configure autovacuum
ALTER TABLE predictions SET (autovacuum_vacuum_scale_factor = 0.1);
```

### System Resource Optimization

```bash
# Increase file descriptors
ulimit -n 65536

# Optimize TCP settings
sysctl -w net.core.rmem_max=134217728
sysctl -w net.core.wmem_max=134217728

# Configure swap
echo vm.swappiness=10 >> /etc/sysctl.conf
```

---

## Monitoring Setup

### Structured Logging Configuration

```python
from src.bot.monitoring import StructuredLogger

# Configure logger
logger = StructuredLogger(
    service="gpt-trader",
    log_level="INFO",
    enable_correlation=True,
    enable_tracing=True
)

# Use correlation context
with logger.correlation_context() as corr_id:
    logger.info("Processing trade", symbol="AAPL", quantity=100)
```

### Alert Rules Configuration

```python
from src.bot.monitoring import IntelligentAlertSystem, AlertRule

system = IntelligentAlertSystem()

# Add critical alert rules
system.add_rule(AlertRule(
    rule_id="model_accuracy",
    name="Model Accuracy Check",
    condition=lambda d: d.get('accuracy', 1.0) < 0.55,
    severity=AlertSeverity.HIGH,
    category=AlertCategory.MODEL
))

system.add_rule(AlertRule(
    rule_id="high_risk",
    name="Risk Limit Check",
    condition=lambda d: d.get('var', 0) > 50000,
    severity=AlertSeverity.CRITICAL,
    category=AlertCategory.RISK
))
```

### Dashboard Access

```bash
# Start operational dashboard
streamlit run src/bot/monitoring/ops_dashboard.py

# Access at http://localhost:8501
```

---

## Best Practices

### 1. **Proactive Monitoring**
- Review dashboards at market open/close
- Check model performance daily
- Monitor risk metrics continuously
- Review alert trends weekly

### 2. **Change Management**
- Test all changes in staging first
- Use gradual rollouts (10% → 50% → 100%)
- Maintain rollback procedures
- Document all changes

### 3. **Capacity Planning**
- Monitor resource trends
- Plan for 2x peak capacity
- Schedule heavy operations off-peak
- Regular performance testing

### 4. **Security**
- Rotate API keys monthly
- Review access logs daily
- Use encryption for sensitive data
- Regular security audits

### 5. **Documentation**
- Keep runbooks updated
- Document all incidents
- Share learnings with team
- Regular training sessions

---

## Emergency Procedures

### 🔴 **Emergency Trading Halt**

```bash
# 1. Immediately stop all trading
python scripts/emergency_stop.py --confirm

# 2. Cancel all open orders
python scripts/cancel_all_orders.py

# 3. Notify team
python scripts/send_alert.py --priority critical --message "Trading halted"

# 4. Begin investigation
python scripts/generate_incident_report.py
```

### 💾 **Data Recovery**

```bash
# 1. Check last backup
python scripts/check_backups.py

# 2. Restore from backup
python scripts/restore_backup.py --date 2024-01-15

# 3. Verify data integrity
python scripts/verify_data_integrity.py

# 4. Rebuild derived data
python scripts/rebuild_features.py
python scripts/rebuild_predictions.py
```

### 🔐 **Security Incident**

```bash
# 1. Isolate affected systems
python scripts/isolate_system.py --component affected

# 2. Rotate all credentials
python scripts/rotate_credentials.py --all

# 3. Audit access logs
python scripts/audit_access.py --last-hours 24

# 4. Generate security report
python scripts/security_report.py
```

---

## Contact Information

### Escalation Path
1. **L1 Support**: monitoring@gpt-trader.com
2. **L2 Engineering**: engineering@gpt-trader.com
3. **L3 Architecture**: architecture@gpt-trader.com
4. **Management**: management@gpt-trader.com

### On-Call Schedule
- **Primary**: Rotation A (Mon-Wed)
- **Secondary**: Rotation B (Thu-Sun)
- **Escalation**: Team Lead (24/7)

### External Contacts
- **Broker Support**: 1-800-BROKER
- **Data Provider**: support@data-provider.com
- **Cloud Provider**: support@cloud.com

---

## Appendix

### Useful Commands

```bash
# System health check
python scripts/health_check.py --full

# Generate performance report
python scripts/performance_report.py --period daily

# Run integration tests
python scripts/run_integration_tests.py

# Database maintenance
python scripts/database_maintenance.py --vacuum --analyze

# Cache management
python scripts/manage_cache.py --clear --component all
```

### Log Locations
- **Application Logs**: `/var/log/gpt-trader/app.log`
- **Trading Logs**: `/var/log/gpt-trader/trading.log`
- **Model Logs**: `/var/log/gpt-trader/ml.log`
- **System Logs**: `/var/log/syslog`

### Configuration Files
- **Main Config**: `/etc/gpt-trader/config.yaml`
- **Model Config**: `/etc/gpt-trader/models.yaml`
- **Risk Config**: `/etc/gpt-trader/risk.yaml`
- **Alert Rules**: `/etc/gpt-trader/alerts.yaml`

---

*Last Updated: 2025-08-14*
*Version: 1.0 (Phase 3, Week 8)*