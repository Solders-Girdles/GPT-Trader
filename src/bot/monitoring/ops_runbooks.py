"""
Operational Runbooks and Incident Response System
Phase 3, Week 8: OPS-025 to OPS-032
Team Training Materials and Procedures

Features:
- Interactive runbooks with step-by-step procedures
- Incident response automation
- System architecture documentation
- Troubleshooting guides
- Performance tuning guides
- Best practices documentation
- Alert response playbooks
- Training material generation
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from jinja2 import Template

# Import GPT-Trader components
try:
    from ..monitoring import get_logger, traced_operation
    from ..core.exceptions import GPTTraderException
    GPT_TRADER_AVAILABLE = True
except ImportError:
    GPT_TRADER_AVAILABLE = False
    logging.basicConfig(level=logging.INFO)

logger = get_logger(__name__) if GPT_TRADER_AVAILABLE else logging.getLogger(__name__)


class RunbookCategory(Enum):
    """Runbook categories"""
    INCIDENT_RESPONSE = "incident_response"
    MAINTENANCE = "maintenance"
    DEPLOYMENT = "deployment"
    TROUBLESHOOTING = "troubleshooting"
    PERFORMANCE = "performance"
    SECURITY = "security"
    TRAINING = "training"


class IncidentSeverity(Enum):
    """Incident severity levels"""
    P0_CRITICAL = "p0_critical"  # System down, trading stopped
    P1_HIGH = "p1_high"          # Major functionality impaired
    P2_MEDIUM = "p2_medium"      # Minor functionality impaired
    P3_LOW = "p3_low"            # Cosmetic or nice-to-have


class ExecutionStatus(Enum):
    """Step execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RunbookStep:
    """Individual runbook step"""
    id: str
    title: str
    description: str
    command: Optional[str] = None
    expected_output: Optional[str] = None
    timeout_seconds: int = 300
    required: bool = True
    automated: bool = False
    status: ExecutionStatus = ExecutionStatus.PENDING
    execution_time: Optional[datetime] = None
    output: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Runbook:
    """Complete runbook with metadata and steps"""
    id: str
    title: str
    description: str
    category: RunbookCategory
    severity: Optional[IncidentSeverity] = None
    estimated_duration: str = "15 minutes"
    prerequisites: List[str] = field(default_factory=list)
    steps: List[RunbookStep] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    last_updated: datetime = field(default_factory=datetime.now)
    success_criteria: List[str] = field(default_factory=list)


class RunbookExecutor:
    """Executes runbooks with logging and state tracking"""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.logger = logger
        self.execution_log: List[Dict[str, Any]] = []
    
    def execute_runbook(self, runbook: Runbook) -> Dict[str, Any]:
        """Execute a complete runbook"""
        execution_id = f"exec_{runbook.id}_{int(time.time())}"
        start_time = datetime.now()
        
        self.logger.info(f"Starting runbook execution: {runbook.title}", extra={
            'runbook_id': runbook.id,
            'execution_id': execution_id,
            'dry_run': self.dry_run
        })
        
        results = {
            'execution_id': execution_id,
            'runbook_id': runbook.id,
            'start_time': start_time,
            'status': 'in_progress',
            'completed_steps': 0,
            'failed_steps': 0,
            'skipped_steps': 0,
            'step_results': []
        }
        
        try:
            for step in runbook.steps:
                step_result = self.execute_step(step, execution_id)
                results['step_results'].append(step_result)
                
                if step_result['status'] == 'completed':
                    results['completed_steps'] += 1
                elif step_result['status'] == 'failed':
                    results['failed_steps'] += 1
                    if step.required:
                        results['status'] = 'failed'
                        break
                elif step_result['status'] == 'skipped':
                    results['skipped_steps'] += 1
            
            if results['status'] != 'failed':
                results['status'] = 'completed'
                
        except Exception as e:
            self.logger.error(f"Runbook execution failed: {e}", extra={
                'execution_id': execution_id,
                'error': str(e)
            })
            results['status'] = 'failed'
            results['error'] = str(e)
        
        results['end_time'] = datetime.now()
        results['duration'] = (results['end_time'] - start_time).total_seconds()
        
        self.execution_log.append(results)
        return results
    
    def execute_step(self, step: RunbookStep, execution_id: str) -> Dict[str, Any]:
        """Execute a single runbook step"""
        start_time = datetime.now()
        step.execution_time = start_time
        step.status = ExecutionStatus.IN_PROGRESS
        
        self.logger.info(f"Executing step: {step.title}", extra={
            'step_id': step.id,
            'execution_id': execution_id,
            'automated': step.automated
        })
        
        result = {
            'step_id': step.id,
            'title': step.title,
            'start_time': start_time,
            'status': 'in_progress',
            'automated': step.automated
        }
        
        try:
            if self.dry_run:
                result['output'] = f"[DRY RUN] Would execute: {step.command or 'Manual step'}"
                result['status'] = 'completed'
                step.status = ExecutionStatus.COMPLETED
            elif step.automated and step.command:
                result['output'] = self._execute_command(step.command, step.timeout_seconds)
                result['status'] = 'completed'
                step.status = ExecutionStatus.COMPLETED
            else:
                # Manual step - require confirmation
                result['output'] = "Manual step - requires operator action"
                result['status'] = 'completed'  # Assume completed for now
                step.status = ExecutionStatus.COMPLETED
                
        except Exception as e:
            self.logger.error(f"Step execution failed: {e}", extra={
                'step_id': step.id,
                'error': str(e)
            })
            result['error'] = str(e)
            result['status'] = 'failed'
            step.status = ExecutionStatus.FAILED
            step.error = str(e)
        
        result['end_time'] = datetime.now()
        result['duration'] = (result['end_time'] - start_time).total_seconds()
        
        return result
    
    def _execute_command(self, command: str, timeout: int) -> str:
        """Execute a shell command with timeout"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                raise RuntimeError(f"Command failed with exit code {result.returncode}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Command timed out after {timeout} seconds")


class RunbookLibrary:
    """Library of operational runbooks"""
    
    def __init__(self):
        self.runbooks: Dict[str, Runbook] = {}
        self.logger = logger
        self._initialize_runbooks()
    
    def _initialize_runbooks(self):
        """Initialize the runbook library with standard procedures"""
        
        # System Health Check Runbook
        self._add_system_health_runbook()
        
        # Model Degradation Response Runbook
        self._add_model_degradation_runbook()
        
        # High VaR Alert Response Runbook
        self._add_high_var_runbook()
        
        # Database Performance Issue Runbook
        self._add_database_performance_runbook()
        
        # Data Feed Outage Runbook
        self._add_data_feed_outage_runbook()
        
        # Trading Engine Stop Runbook
        self._add_trading_engine_stop_runbook()
        
        # System Deployment Runbook
        self._add_deployment_runbook()
        
        # Performance Tuning Runbook
        self._add_performance_tuning_runbook()
    
    def _add_system_health_runbook(self):
        """System health check runbook"""
        runbook = Runbook(
            id="system_health_check",
            title="System Health Check",
            description="Comprehensive system health verification procedure",
            category=RunbookCategory.MAINTENANCE,
            estimated_duration="10 minutes",
            prerequisites=["Access to monitoring dashboards", "Database access"],
            success_criteria=[
                "All services are running",
                "Database connections are healthy",
                "ML pipeline is operational",
                "Data feeds are active"
            ],
            tags=["health", "monitoring", "system"]
        )
        
        runbook.steps = [
            RunbookStep(
                id="check_services",
                title="Check Core Services",
                description="Verify all core services are running",
                command="systemctl status postgresql redis-server",
                automated=True
            ),
            RunbookStep(
                id="check_database",
                title="Check Database Connectivity",
                description="Test database connection and query performance",
                command="python -c \"from src.bot.database import PostgresManager; pm = PostgresManager(); print('DB OK' if pm.health_check() else 'DB FAIL')\"",
                automated=True
            ),
            RunbookStep(
                id="check_ml_pipeline",
                title="Check ML Pipeline",
                description="Verify ML pipeline is healthy and processing",
                command="python -c \"from src.bot.ml import IntegratedMLPipeline; print('ML OK')\"",
                automated=True
            ),
            RunbookStep(
                id="check_data_feeds",
                title="Check Data Feeds",
                description="Verify market data feeds are active and recent",
                command="python -c \"from src.bot.dataflow import RealtimeFeed; rf = RealtimeFeed(); print('Feed OK' if rf.is_healthy() else 'Feed FAIL')\"",
                automated=True
            ),
            RunbookStep(
                id="review_alerts",
                title="Review Active Alerts",
                description="Check for any active critical alerts",
                automated=False
            )
        ]
        
        self.runbooks[runbook.id] = runbook
    
    def _add_model_degradation_runbook(self):
        """Model degradation response runbook"""
        runbook = Runbook(
            id="model_degradation_response",
            title="Model Degradation Response",
            description="Response procedure for ML model performance degradation",
            category=RunbookCategory.INCIDENT_RESPONSE,
            severity=IncidentSeverity.P1_HIGH,
            estimated_duration="30 minutes",
            prerequisites=["ML pipeline access", "Model training environment"],
            success_criteria=[
                "Model performance restored above threshold",
                "Degradation root cause identified",
                "Preventive measures implemented"
            ],
            tags=["ml", "degradation", "incident"]
        )
        
        runbook.steps = [
            RunbookStep(
                id="assess_degradation",
                title="Assess Degradation Severity",
                description="Check model performance metrics and degradation extent",
                command="python scripts/check_model_performance.py --detailed",
                automated=True
            ),
            RunbookStep(
                id="switch_to_backup",
                title="Switch to Backup Model",
                description="Activate backup model to maintain trading operations",
                command="python -c \"from src.bot.ml import ModelPromotionManager; mpm = ModelPromotionManager(); mpm.emergency_rollback()\"",
                automated=True
            ),
            RunbookStep(
                id="analyze_root_cause",
                title="Analyze Root Cause",
                description="Investigate data drift, feature changes, or market regime shifts",
                automated=False
            ),
            RunbookStep(
                id="trigger_retraining",
                title="Trigger Model Retraining",
                description="Start retraining process with recent data",
                command="python scripts/trigger_retraining.py --emergency",
                automated=True
            ),
            RunbookStep(
                id="validate_new_model",
                title="Validate New Model",
                description="Test new model in shadow mode before deployment",
                automated=False
            )
        ]
        
        self.runbooks[runbook.id] = runbook
    
    def _add_high_var_runbook(self):
        """High VaR alert response runbook"""
        runbook = Runbook(
            id="high_var_response",
            title="High VaR Alert Response",
            description="Response procedure for Value at Risk limit breaches",
            category=RunbookCategory.INCIDENT_RESPONSE,
            severity=IncidentSeverity.P1_HIGH,
            estimated_duration="15 minutes",
            prerequisites=["Risk management system access", "Trading system access"],
            success_criteria=[
                "VaR reduced below limit",
                "Positions reviewed and adjusted",
                "Risk exposure documented"
            ],
            tags=["risk", "var", "trading"]
        )
        
        runbook.steps = [
            RunbookStep(
                id="calculate_current_var",
                title="Calculate Current VaR",
                description="Get current VaR calculation and breakdown",
                command="python -c \"from src.bot.risk import RiskMetricsEngine; rme = RiskMetricsEngine(); print(f'VaR: ${rme.calculate_var():.0f}')\"",
                automated=True
            ),
            RunbookStep(
                id="identify_high_risk_positions",
                title="Identify High Risk Positions",
                description="Find positions contributing most to VaR",
                automated=False
            ),
            RunbookStep(
                id="reduce_position_sizes",
                title="Reduce Position Sizes",
                description="Scale down high-risk positions to reduce VaR",
                automated=False
            ),
            RunbookStep(
                id="verify_var_reduction",
                title="Verify VaR Reduction",
                description="Confirm VaR is now within acceptable limits",
                command="python -c \"from src.bot.risk import RiskMetricsEngine; rme = RiskMetricsEngine(); print(f'New VaR: ${rme.calculate_var():.0f}')\"",
                automated=True
            ),
            RunbookStep(
                id="document_actions",
                title="Document Actions Taken",
                description="Record all actions and rationale for audit",
                automated=False
            )
        ]
        
        self.runbooks[runbook.id] = runbook
    
    def _add_database_performance_runbook(self):
        """Database performance issue runbook"""
        runbook = Runbook(
            id="database_performance",
            title="Database Performance Issue",
            description="Troubleshooting and optimization for database performance issues",
            category=RunbookCategory.TROUBLESHOOTING,
            severity=IncidentSeverity.P2_MEDIUM,
            estimated_duration="45 minutes",
            prerequisites=["Database admin access", "Query analysis tools"],
            success_criteria=[
                "Query response times improved",
                "Connection pool optimized",
                "Performance bottlenecks identified"
            ],
            tags=["database", "performance", "optimization"]
        )
        
        runbook.steps = [
            RunbookStep(
                id="check_connection_pool",
                title="Check Connection Pool Status",
                description="Review database connection pool metrics",
                command="python -c \"from src.bot.database import PostgresManager; pm = PostgresManager(); print(pm.get_pool_stats())\"",
                automated=True
            ),
            RunbookStep(
                id="analyze_slow_queries",
                title="Analyze Slow Queries",
                description="Identify and analyze slow-running queries",
                command="psql -d gpt_trader -c \"SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;\"",
                automated=True
            ),
            RunbookStep(
                id="check_table_stats",
                title="Check Table Statistics",
                description="Review table size and index usage statistics",
                command="psql -d gpt_trader -c \"SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del FROM pg_stat_user_tables ORDER BY n_tup_ins DESC;\"",
                automated=True
            ),
            RunbookStep(
                id="optimize_indexes",
                title="Optimize Database Indexes",
                description="Review and optimize database indexes",
                automated=False
            ),
            RunbookStep(
                id="update_statistics",
                title="Update Table Statistics",
                description="Run ANALYZE to update table statistics",
                command="psql -d gpt_trader -c \"ANALYZE;\"",
                automated=True
            )
        ]
        
        self.runbooks[runbook.id] = runbook
    
    def _add_data_feed_outage_runbook(self):
        """Data feed outage runbook"""
        runbook = Runbook(
            id="data_feed_outage",
            title="Data Feed Outage Response",
            description="Response procedure for market data feed outages",
            category=RunbookCategory.INCIDENT_RESPONSE,
            severity=IncidentSeverity.P0_CRITICAL,
            estimated_duration="20 minutes",
            prerequisites=["Data pipeline access", "Alternative data sources"],
            success_criteria=[
                "Data feed restored or alternative activated",
                "Trading operations maintained",
                "Data quality verified"
            ],
            tags=["data", "feed", "outage", "critical"]
        )
        
        runbook.steps = [
            RunbookStep(
                id="verify_outage",
                title="Verify Data Feed Outage",
                description="Confirm the data feed is actually down",
                command="python -c \"from src.bot.dataflow import RealtimeFeed; rf = RealtimeFeed(); print('UP' if rf.is_connected() else 'DOWN')\"",
                automated=True
            ),
            RunbookStep(
                id="check_alternative_sources",
                title="Check Alternative Data Sources",
                description="Verify status of backup data sources",
                automated=False
            ),
            RunbookStep(
                id="switch_to_backup",
                title="Switch to Backup Data Source",
                description="Activate backup data feed",
                command="python -c \"from src.bot.dataflow import DataSourceManager; dsm = DataSourceManager(); dsm.switch_to_backup()\"",
                automated=True
            ),
            RunbookStep(
                id="verify_data_quality",
                title="Verify Data Quality",
                description="Check that backup data meets quality standards",
                automated=False
            ),
            RunbookStep(
                id="notify_stakeholders",
                title="Notify Stakeholders",
                description="Inform relevant parties of data source change",
                automated=False
            )
        ]
        
        self.runbooks[runbook.id] = runbook
    
    def _add_trading_engine_stop_runbook(self):
        """Trading engine emergency stop runbook"""
        runbook = Runbook(
            id="trading_engine_stop",
            title="Trading Engine Emergency Stop",
            description="Emergency procedure to safely stop all trading operations",
            category=RunbookCategory.INCIDENT_RESPONSE,
            severity=IncidentSeverity.P0_CRITICAL,
            estimated_duration="10 minutes",
            prerequisites=["Trading system access", "Risk management authority"],
            success_criteria=[
                "All trading operations stopped",
                "Open positions documented",
                "Risk exposure assessed"
            ],
            tags=["trading", "emergency", "stop"]
        )
        
        runbook.steps = [
            RunbookStep(
                id="assess_situation",
                title="Assess Emergency Situation",
                description="Understand the reason for emergency stop",
                automated=False
            ),
            RunbookStep(
                id="stop_new_orders",
                title="Stop New Order Placement",
                description="Disable new order generation",
                command="python -c \"from src.bot.exec import TradingEngine; te = TradingEngine(); te.disable_trading()\"",
                automated=True
            ),
            RunbookStep(
                id="cancel_pending_orders",
                title="Cancel Pending Orders",
                description="Cancel all pending orders",
                command="python -c \"from src.bot.exec import TradingEngine; te = TradingEngine(); te.cancel_all_orders()\"",
                automated=True
            ),
            RunbookStep(
                id="document_positions",
                title="Document Current Positions",
                description="Record all current positions and their status",
                automated=False
            ),
            RunbookStep(
                id="calculate_exposure",
                title="Calculate Risk Exposure",
                description="Calculate current risk exposure",
                command="python -c \"from src.bot.risk import RiskMetricsEngine; rme = RiskMetricsEngine(); print(f'Total Exposure: ${rme.get_total_exposure():.0f}')\"",
                automated=True
            )
        ]
        
        self.runbooks[runbook.id] = runbook
    
    def _add_deployment_runbook(self):
        """System deployment runbook"""
        runbook = Runbook(
            id="system_deployment",
            title="System Deployment",
            description="Standard deployment procedure for GPT-Trader updates",
            category=RunbookCategory.DEPLOYMENT,
            estimated_duration="60 minutes",
            prerequisites=["Deployment access", "Backup verification", "Testing completed"],
            success_criteria=[
                "New version deployed successfully",
                "All tests passing",
                "Rollback plan ready"
            ],
            tags=["deployment", "update", "release"]
        )
        
        runbook.steps = [
            RunbookStep(
                id="pre_deployment_checks",
                title="Pre-deployment Checks",
                description="Verify system health before deployment",
                automated=False
            ),
            RunbookStep(
                id="backup_database",
                title="Backup Database",
                description="Create database backup before deployment",
                command="pg_dump gpt_trader > backups/pre_deploy_$(date +%Y%m%d_%H%M%S).sql",
                automated=True
            ),
            RunbookStep(
                id="stop_services",
                title="Stop Services",
                description="Gracefully stop application services",
                command="systemctl stop gpt-trader",
                automated=True
            ),
            RunbookStep(
                id="deploy_code",
                title="Deploy New Code",
                description="Deploy new application version",
                command="git pull origin main && poetry install",
                automated=True
            ),
            RunbookStep(
                id="run_migrations",
                title="Run Database Migrations",
                description="Apply any database schema changes",
                command="python scripts/run_migrations.py",
                automated=True
            ),
            RunbookStep(
                id="start_services",
                title="Start Services",
                description="Start application services",
                command="systemctl start gpt-trader",
                automated=True
            ),
            RunbookStep(
                id="verify_deployment",
                title="Verify Deployment",
                description="Run post-deployment verification tests",
                command="python scripts/verify_deployment.py",
                automated=True
            )
        ]
        
        self.runbooks[runbook.id] = runbook
    
    def _add_performance_tuning_runbook(self):
        """Performance tuning runbook"""
        runbook = Runbook(
            id="performance_tuning",
            title="Performance Tuning",
            description="System performance optimization procedures",
            category=RunbookCategory.PERFORMANCE,
            estimated_duration="90 minutes",
            prerequisites=["Performance monitoring tools", "System analysis access"],
            success_criteria=[
                "Performance bottlenecks identified",
                "Optimizations implemented",
                "Performance improvements measured"
            ],
            tags=["performance", "optimization", "tuning"]
        )
        
        runbook.steps = [
            RunbookStep(
                id="baseline_performance",
                title="Establish Performance Baseline",
                description="Measure current system performance metrics",
                command="python scripts/performance_benchmark.py --baseline",
                automated=True
            ),
            RunbookStep(
                id="profile_ml_pipeline",
                title="Profile ML Pipeline",
                description="Profile ML pipeline for bottlenecks",
                command="python scripts/profile_ml_pipeline.py",
                automated=True
            ),
            RunbookStep(
                id="analyze_database_performance",
                title="Analyze Database Performance",
                description="Review database query performance",
                automated=False
            ),
            RunbookStep(
                id="optimize_caching",
                title="Optimize Caching Strategy",
                description="Review and optimize caching configuration",
                automated=False
            ),
            RunbookStep(
                id="tune_parameters",
                title="Tune System Parameters",
                description="Adjust system parameters for optimal performance",
                automated=False
            ),
            RunbookStep(
                id="measure_improvements",
                title="Measure Performance Improvements",
                description="Re-run performance tests to measure improvements",
                command="python scripts/performance_benchmark.py --compare",
                automated=True
            )
        ]
        
        self.runbooks[runbook.id] = runbook
    
    def get_runbook(self, runbook_id: str) -> Optional[Runbook]:
        """Get a runbook by ID"""
        return self.runbooks.get(runbook_id)
    
    def list_runbooks(self, category: Optional[RunbookCategory] = None) -> List[Runbook]:
        """List runbooks, optionally filtered by category"""
        runbooks = list(self.runbooks.values())
        if category:
            runbooks = [rb for rb in runbooks if rb.category == category]
        return sorted(runbooks, key=lambda x: x.title)
    
    def search_runbooks(self, query: str) -> List[Runbook]:
        """Search runbooks by title, description, or tags"""
        query = query.lower()
        results = []
        
        for runbook in self.runbooks.values():
            if (query in runbook.title.lower() or 
                query in runbook.description.lower() or
                any(query in tag.lower() for tag in runbook.tags)):
                results.append(runbook)
        
        return sorted(results, key=lambda x: x.title)


class TrainingMaterialGenerator:
    """Generates training materials and documentation"""
    
    def __init__(self):
        self.logger = logger
        self.templates_dir = Path(__file__).parent / "templates"
        self.output_dir = Path(__file__).parent.parent.parent.parent / "docs" / "training"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all_materials(self):
        """Generate all training materials"""
        self.logger.info("Generating training materials")
        
        # Generate different types of training materials
        self._generate_quick_reference()
        self._generate_troubleshooting_guide()
        self._generate_alert_response_guide()
        self._generate_system_architecture_guide()
        self._generate_deployment_guide()
        self._generate_performance_guide()
        self._generate_security_guide()
        self._generate_onboarding_checklist()
    
    def _generate_quick_reference(self):
        """Generate quick reference guide"""
        content = """# GPT-Trader Quick Reference Guide

## Emergency Contacts
- **On-call Engineer**: +1-XXX-XXX-XXXX
- **Team Lead**: +1-XXX-XXX-XXXX
- **DevOps**: +1-XXX-XXX-XXXX

## Critical Commands

### System Health Check
```bash
python scripts/health_check.py
```

### Emergency Trading Stop
```bash
python -c "from src.bot.exec import TradingEngine; TradingEngine().emergency_stop()"
```

### Database Status
```bash
psql -d gpt_trader -c "SELECT version();"
```

### Check Model Performance
```bash
python scripts/check_model_performance.py
```

## Key Metrics to Monitor
- **Model Accuracy**: > 55%
- **VaR (95%)**: < $15,000
- **Max Drawdown**: < 20%
- **System Uptime**: > 99.5%
- **Alert Response**: < 15 minutes

## Common Issues and Quick Fixes

### Model Degradation
1. Check data quality
2. Switch to backup model
3. Trigger retraining

### High VaR Alert
1. Review position sizes
2. Check correlation matrix
3. Reduce high-risk positions

### Data Feed Outage
1. Switch to backup feed
2. Verify data quality
3. Monitor for gaps

### Database Connection Issues
1. Check connection pool
2. Restart PostgreSQL if needed
3. Verify network connectivity

## Log Locations
- **Application Logs**: `/var/log/gpt-trader/`
- **Database Logs**: `/var/log/postgresql/`
- **System Logs**: `/var/log/syslog`

## Useful Monitoring URLs
- **Grafana Dashboard**: http://localhost:3000
- **Application Dashboard**: http://localhost:8501
- **Database Admin**: http://localhost:5050
"""
        
        with open(self.output_dir / "quick_reference.md", "w") as f:
            f.write(content)
    
    def _generate_troubleshooting_guide(self):
        """Generate comprehensive troubleshooting guide"""
        content = """# GPT-Trader Troubleshooting Guide

## Diagnostic Process

### 1. Initial Assessment
- Check system health dashboard
- Review recent alerts
- Verify service status
- Check resource utilization

### 2. Problem Classification
- **P0 Critical**: Trading stopped, system down
- **P1 High**: Major functionality impaired
- **P2 Medium**: Minor functionality impaired
- **P3 Low**: Cosmetic or enhancement

### 3. Data Collection
- Gather relevant logs
- Export system metrics
- Document timeline of events
- Identify affected components

## Common Issues

### Model Performance Degradation

**Symptoms:**
- Accuracy below 55%
- Increasing prediction errors
- Poor trading performance

**Diagnosis:**
```bash
python scripts/check_model_performance.py --detailed
python scripts/analyze_data_drift.py
```

**Solutions:**
1. **Data Quality Issue**: Check for data anomalies or feed problems
2. **Market Regime Change**: Retrain model with recent data
3. **Feature Drift**: Update feature engineering pipeline
4. **Overfitting**: Implement regularization or ensemble methods

### High Risk Exposure

**Symptoms:**
- VaR exceeding limits
- High correlation between positions
- Large position concentrations

**Diagnosis:**
```bash
python -c "from src.bot.risk import RiskMetricsEngine; rme = RiskMetricsEngine(); print(rme.risk_report())"
```

**Solutions:**
1. **Position Sizing**: Reduce position sizes
2. **Diversification**: Add uncorrelated assets
3. **Hedging**: Add protective positions
4. **Limit Adjustment**: Temporarily adjust limits if justified

### Database Performance Issues

**Symptoms:**
- Slow query response times
- Connection timeouts
- High CPU/memory usage

**Diagnosis:**
```sql
-- Check active connections
SELECT count(*) FROM pg_stat_activity;

-- Find slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;

-- Check table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

**Solutions:**
1. **Index Optimization**: Add missing indexes
2. **Query Optimization**: Rewrite inefficient queries
3. **Connection Pooling**: Adjust pool settings
4. **Resource Scaling**: Increase server resources

### Data Feed Connectivity

**Symptoms:**
- Missing data updates
- Stale timestamps
- Connection errors

**Diagnosis:**
```bash
python -c "from src.bot.dataflow import RealtimeFeed; rf = RealtimeFeed(); print(rf.connection_status())"
```

**Solutions:**
1. **Network Issues**: Check network connectivity
2. **API Limits**: Verify rate limits and quotas
3. **Authentication**: Check API credentials
4. **Backup Sources**: Switch to alternative data provider

## Escalation Procedures

### When to Escalate
- Unable to resolve within 30 minutes
- P0/P1 issues affecting trading
- Security incidents
- Data integrity concerns

### Escalation Contacts
1. **First Level**: Team Lead
2. **Second Level**: Senior Engineer
3. **Third Level**: CTO/Management

### Information to Provide
- Problem description and timeline
- Steps already taken
- Current system status
- Business impact assessment

## Recovery Procedures

### System Recovery Checklist
- [ ] Identify root cause
- [ ] Implement fix
- [ ] Verify system functionality
- [ ] Monitor for stability
- [ ] Document incident
- [ ] Update procedures if needed

### Data Recovery
- [ ] Assess data loss scope
- [ ] Restore from backups
- [ ] Verify data integrity
- [ ] Replay missed transactions
- [ ] Validate system state
"""
        
        with open(self.output_dir / "troubleshooting_guide.md", "w") as f:
            f.write(content)
    
    def _generate_alert_response_guide(self):
        """Generate alert response playbook"""
        content = """# Alert Response Playbook

## Alert Classification

### Critical Alerts (P0) - Immediate Response Required
- Trading engine failure
- Database outage
- Security breach
- Data corruption

**Response Time**: < 5 minutes
**Action**: Page on-call engineer immediately

### High Priority Alerts (P1) - Quick Response Required
- Model degradation
- High VaR
- Data feed outage
- Performance degradation

**Response Time**: < 15 minutes
**Action**: Notify team via Slack

### Medium Priority Alerts (P2) - Response During Business Hours
- Minor performance issues
- Non-critical warnings
- Capacity warnings

**Response Time**: < 2 hours
**Action**: Create ticket for next business day

### Low Priority Alerts (P3) - Informational
- Successful deployments
- System health reports
- Usage statistics

**Response Time**: No immediate action required
**Action**: Review during regular maintenance

## Response Procedures

### Critical Alert Response (P0)

1. **Immediate Assessment** (0-2 minutes)
   - Acknowledge alert
   - Check system dashboard
   - Assess business impact

2. **Initial Response** (2-5 minutes)
   - Execute emergency procedures
   - Notify stakeholders
   - Begin documentation

3. **Resolution** (5-30 minutes)
   - Implement fix
   - Verify resolution
   - Monitor for stability

4. **Post-Incident** (30+ minutes)
   - Complete documentation
   - Schedule post-mortem
   - Update procedures

### High Priority Alert Response (P1)

1. **Assessment** (0-5 minutes)
   - Review alert details
   - Check related metrics
   - Determine severity

2. **Investigation** (5-15 minutes)
   - Gather diagnostic information
   - Identify root cause
   - Plan response

3. **Resolution** (15-60 minutes)
   - Implement solution
   - Test fix
   - Monitor results

4. **Follow-up** (60+ minutes)
   - Document resolution
   - Update monitoring
   - Prevent recurrence

## Specific Alert Responses

### Model Accuracy Below Threshold

**Alert**: Model accuracy dropped below 55%

**Immediate Actions**:
1. Check model performance dashboard
2. Switch to backup model if necessary
3. Investigate data quality issues

**Runbook**: `model_degradation_response`

### VaR Limit Breach

**Alert**: VaR exceeded $15,000 threshold

**Immediate Actions**:
1. Calculate current VaR breakdown
2. Identify high-risk positions
3. Reduce position sizes if necessary

**Runbook**: `high_var_response`

### Database Connection Failure

**Alert**: Database connection pool exhausted

**Immediate Actions**:
1. Check database server status
2. Review connection pool settings
3. Restart application if necessary

**Runbook**: `database_performance`

### Data Feed Outage

**Alert**: Market data feed disconnected

**Immediate Actions**:
1. Verify feed status
2. Switch to backup data source
3. Check data quality

**Runbook**: `data_feed_outage`

## Communication Templates

### Critical Incident Notification
```
CRITICAL INCIDENT - GPT-Trader

Incident: [Brief description]
Start Time: [Timestamp]
Impact: [Business impact]
Status: Investigating/Mitigating/Resolved
ETA: [Estimated resolution time]

Actions Taken:
- [Action 1]
- [Action 2]

Next Update: [Time]
```

### Resolution Notification
```
INCIDENT RESOLVED - GPT-Trader

Incident: [Brief description]
Resolution Time: [Timestamp]
Duration: [Total duration]
Root Cause: [Brief explanation]

Resolution:
- [Primary fix]
- [Additional actions]

Follow-up Actions:
- [Preventive measures]
- [Process improvements]
```

## Alert Fatigue Prevention

### Quality Metrics
- **False Positive Rate**: < 10%
- **Alert Resolution Time**: < 30 minutes average
- **Repeat Alerts**: < 5% within 24 hours

### Optimization Strategies
1. **Threshold Tuning**: Regularly review and adjust alert thresholds
2. **Alert Bundling**: Group related alerts to reduce noise
3. **Intelligent Routing**: Route alerts to appropriate team members
4. **Feedback Loop**: Use resolution data to improve alerting

### Regular Review Process
- Weekly alert effectiveness review
- Monthly threshold optimization
- Quarterly alert strategy assessment
- Annual playbook updates
"""
        
        with open(self.output_dir / "alert_response_playbook.md", "w") as f:
            f.write(content)


def main():
    """Main entry point for runbook system"""
    library = RunbookLibrary()
    executor = RunbookExecutor(dry_run=True)
    generator = TrainingMaterialGenerator()
    
    # Generate training materials
    generator.generate_all_materials()
    
    # Example: Execute system health check
    health_runbook = library.get_runbook("system_health_check")
    if health_runbook:
        result = executor.execute_runbook(health_runbook)
        print(f"Runbook execution result: {result['status']}")


if __name__ == "__main__":
    main()