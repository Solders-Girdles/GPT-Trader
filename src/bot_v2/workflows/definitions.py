"""
Workflow Definitions for Bot V2 Trading System

Provides predefined advanced workflows for multi-strategy orchestration,
conditional execution, and parallel processing.
"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import yaml
import json

# Workflow templates directory
WORKFLOW_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')


class WorkflowType(Enum):
    """Types of workflows"""
    BACKTEST = "backtest"
    PAPER_TRADE = "paper_trade"
    LIVE_TRADE = "live_trade"
    OPTIMIZATION = "optimization"
    ANALYSIS = "analysis"
    MONITORING = "monitoring"


class ExecutionMode(Enum):
    """Workflow execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    BACKGROUND = "background"
    SCHEDULED = "scheduled"


@dataclass
class WorkflowStep:
    """Individual workflow step definition"""
    name: str
    slice: Optional[str] = None
    action: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    timeout: int = 300
    retry_count: int = 3
    weight: float = 1.0
    outputs: List[str] = field(default_factory=list)


@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    name: str
    version: str
    description: str
    workflow_type: WorkflowType
    execution_mode: ExecutionMode
    steps: List[WorkflowStep]
    variables: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    event_handlers: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    schedule: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowDefinitions:
    """
    Manages predefined workflow definitions for the trading system.
    Provides templates for common trading workflows.
    """
    
    # Predefined workflow templates
    WORKFLOWS = {}
    
    @classmethod
    def register_workflow(cls, workflow: WorkflowDefinition):
        """Register a workflow definition"""
        cls.WORKFLOWS[workflow.name] = workflow
    
    @classmethod
    def get_workflow(cls, name: str) -> Optional[WorkflowDefinition]:
        """Get workflow by name"""
        return cls.WORKFLOWS.get(name)
    
    @classmethod
    def list_workflows(cls) -> List[str]:
        """List all available workflows"""
        return list(cls.WORKFLOWS.keys())
    
    @classmethod
    def initialize_default_workflows(cls):
        """Initialize default workflow definitions"""
        
        # 1. Simple Backtest Workflow
        cls.register_workflow(WorkflowDefinition(
            name="simple_backtest",
            version="1.0",
            description="Basic sequential backtest workflow",
            workflow_type=WorkflowType.BACKTEST,
            execution_mode=ExecutionMode.SEQUENTIAL,
            variables={
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 100000
            },
            steps=[
                WorkflowStep(
                    name="fetch_data",
                    slice="data",
                    action="get_historical_data",
                    params={"period": "1y", "interval": "1d"}
                ),
                WorkflowStep(
                    name="analyze_market",
                    slice="analyze",
                    action="find_patterns",
                    depends_on=["fetch_data"]
                ),
                WorkflowStep(
                    name="select_strategy",
                    slice="ml_strategy",
                    action="predict_best_strategy",
                    depends_on=["analyze_market"]
                ),
                WorkflowStep(
                    name="run_backtest",
                    slice="backtest",
                    action="execute",
                    depends_on=["select_strategy"]
                ),
                WorkflowStep(
                    name="generate_report",
                    slice="monitor",
                    action="create_performance_report",
                    depends_on=["run_backtest"]
                )
            ]
        ))

        # 1b. Paper Trading Workflow (lightweight)
        cls.register_workflow(WorkflowDefinition(
            name="paper_trading",
            version="1.0",
            description="Paper trading workflow",
            workflow_type=WorkflowType.PAPER_TRADE,
            execution_mode=ExecutionMode.SEQUENTIAL,
            steps=[
                WorkflowStep(name="fetch_data", slice="data", action="get_historical_data"),
                WorkflowStep(name="analyze_market", slice="analyze", action="find_patterns"),
                WorkflowStep(name="select_strategy", slice="ml_strategy", action="predict_best_strategy"),
                WorkflowStep(name="execute_paper", slice="paper_trade", action="execute"),
                WorkflowStep(name="generate_report", slice="monitor", action="create_performance_report"),
            ]
        ))
        
        # 2. Multi-Strategy Ensemble Workflow
        cls.register_workflow(WorkflowDefinition(
            name="multi_strategy_ensemble",
            version="2.0",
            description="Advanced multi-strategy trading with parallel execution",
            workflow_type=WorkflowType.PAPER_TRADE,
            execution_mode=ExecutionMode.PARALLEL,
            variables={
                "capital": 50000,
                "risk_budget": 0.02,
                "confidence_threshold": 0.7,
                "rebalance_frequency": "daily",
                "strategies": ["momentum", "mean_reversion", "ml_prediction"]
            },
            resources={
                "cpu_limit": "2",
                "memory_limit": "4Gi",
                "max_concurrent": 4
            },
            steps=[
                # Parallel data preparation
                WorkflowStep(
                    name="data_preparation",
                    execution_mode=ExecutionMode.PARALLEL,
                    params={
                        'nested_steps': [
                            WorkflowStep(
                                name="market_data",
                                slice="data",
                                action="fetch_historical",
                                params={"symbols": ["SPY", "QQQ", "IWM"], "period": "60d"}
                            ),
                            WorkflowStep(
                                name="alternative_data",
                                slice="data",
                                action="fetch_sentiment",
                                params={"sources": ["news", "social"]}
                            )
                        ]
                    }
                ),
                
                # Market regime detection
                WorkflowStep(
                    name="regime_analysis",
                    slice="market_regime",
                    action="detect_current_regime",
                    depends_on=["data_preparation"],
                    outputs=["regime_type", "confidence_score", "transition_probability"]
                ),
                
                # Parallel strategy analysis
                WorkflowStep(
                    name="strategy_ensemble",
                    execution_mode=ExecutionMode.PARALLEL,
                    depends_on=["regime_analysis"],
                    params={
                        'nested_steps': [
                            WorkflowStep(
                                name="momentum_analysis",
                                slice="analyze",
                                action="run_momentum_strategy",
                                weight=0.4,
                                conditions=[
                                    {"if": "regime_type in ['bull_trending', 'bear_trending']", "weight": 0.4},
                                    {"else": None, "weight": 0.1}
                                ]
                            ),
                            WorkflowStep(
                                name="mean_reversion_analysis",
                                slice="analyze",
                                action="run_mean_reversion_strategy",
                                weight=0.3,
                                conditions=[
                                    {"if": "regime_type == 'sideways_quiet'", "weight": 0.5},
                                    {"else": None, "weight": 0.2}
                                ]
                            ),
                            WorkflowStep(
                                name="ml_prediction",
                                slice="ml_strategy",
                                action="predict_ensemble",
                                params={"models": ["xgboost", "lstm", "random_forest"], "voting": "weighted"},
                                weight=0.3
                            )
                        ]
                    }
                ),
                
                # Risk management
                WorkflowStep(
                    name="risk_management",
                    slice="position_sizing",
                    action="calculate_optimal_sizes",
                    depends_on=["strategy_ensemble"],
                    params={
                        "method": "kelly_confidence",
                        "max_position_size": 0.1,
                        "correlation_limit": 0.6
                    }
                ),
                
                # Conditional execution
                WorkflowStep(
                    name="execution_decision",
                    execution_mode=ExecutionMode.CONDITIONAL,
                    depends_on=["risk_management"],
                    conditions=[
                        {
                            "if": "total_risk_score > 0.8",
                            "then": [
                                {
                                    "name": "risk_reduction",
                                    "slice": "position_sizing",
                                    "action": "reduce_exposure",
                                    "params": {"factor": 0.5}
                                }
                            ]
                        },
                        {
                            "elif": "total_risk_score > 0.6",
                            "then": [
                                {
                                    "name": "conservative_execution",
                                    "slice": "paper_trade",
                                    "action": "execute_trades",
                                    "params": {"execution_style": "conservative"}
                                }
                            ]
                        },
                        {
                            "else": [
                                {
                                    "name": "normal_execution",
                                    "slice": "paper_trade",
                                    "action": "execute_trades",
                                    "params": {"execution_style": "normal"}
                                }
                            ]
                        }
                    ]
                )
            ],
            event_handlers={
                "on_failure": [
                    {"action": "close_all_positions", "slice": "live_trade"},
                    {"action": "send_alert", "params": {"channels": ["email", "slack"]}}
                ],
                "on_success": [
                    {"action": "incremental_learning", "slice": "ml_strategy", "params": {"feedback_loop": True}}
                ]
            }
        ))
        
        # 3. Risk-Managed Live Trading Workflow
        cls.register_workflow(WorkflowDefinition(
            name="risk_managed_live_trading",
            version="1.0",
            description="Production live trading with comprehensive risk management",
            workflow_type=WorkflowType.LIVE_TRADE,
            execution_mode=ExecutionMode.CONDITIONAL,
            variables={
                "max_daily_loss": 0.02,
                "max_position_size": 0.05,
                "stop_loss": 0.01,
                "take_profit": 0.03,
                "trading_hours": {"start": "09:30", "end": "16:00"},
                "emergency_contact": "admin@trading.com"
            },
            steps=[
                # Pre-market checks
                WorkflowStep(
                    name="pre_market_validation",
                    execution_mode=ExecutionMode.SEQUENTIAL,
                    params={
                        'nested_steps': [
                            WorkflowStep(
                                name="system_health_check",
                                slice="monitor",
                                action="check_system_health",
                                conditions=[
                                    {"if": "health_status != 'healthy'", "abort": True}
                                ]
                            ),
                            WorkflowStep(
                                name="validate_broker_connection",
                                slice="live_trade",
                                action="validate_connection"
                            ),
                            WorkflowStep(
                                name="load_positions",
                                slice="live_trade",
                                action="get_current_positions"
                            ),
                            WorkflowStep(
                                name="check_risk_limits",
                                slice="position_sizing",
                                action="validate_risk_limits"
                            )
                        ]
                    }
                ),
                
                # Market hours trading loop
                WorkflowStep(
                    name="trading_loop",
                    execution_mode=ExecutionMode.BACKGROUND,
                    params={
                        'nested_steps': [
                            WorkflowStep(
                                name="fetch_real_time_data",
                                slice="data",
                                action="get_real_time_quotes"
                            ),
                            WorkflowStep(
                                name="analyze_signals",
                                slice="analyze",
                                action="generate_trading_signals",
                                depends_on=["fetch_real_time_data"]
                            ),
                            WorkflowStep(
                                name="risk_check",
                                slice="position_sizing",
                                action="check_risk_constraints",
                                depends_on=["analyze_signals"],
                                conditions=[
                                    {"if": "daily_loss >= max_daily_loss", "action": "halt_trading"},
                                    {"if": "position_concentration > 0.2", "action": "reduce_positions"}
                                ]
                            ),
                            WorkflowStep(
                                name="execute_orders",
                                slice="live_trade",
                                action="submit_orders",
                                depends_on=["risk_check"],
                                params={
                                    "order_type": "limit",
                                    "time_in_force": "DAY",
                                    "slippage_tolerance": 0.001
                                }
                            ),
                            WorkflowStep(
                                name="monitor_positions",
                                slice="monitor",
                                action="track_performance",
                                depends_on=["execute_orders"]
                            )
                        ]
                    }
                ),
                
                # Post-market reconciliation
                WorkflowStep(
                    name="post_market_tasks",
                    execution_mode=ExecutionMode.SEQUENTIAL,
                    params={
                        'nested_steps': [
                            WorkflowStep(
                                name="reconcile_positions",
                                slice="live_trade",
                                action="reconcile_with_broker"
                            ),
                            WorkflowStep(
                                name="calculate_performance",
                                slice="monitor",
                                action="calculate_daily_metrics"
                            ),
                            WorkflowStep(
                                name="update_models",
                                slice="ml_strategy",
                                action="update_with_daily_data"
                            ),
                            WorkflowStep(
                                name="send_daily_report",
                                slice="monitor",
                                action="send_performance_report",
                                params={"recipients": ["portfolio_manager@trading.com"]}
                            )
                        ]
                    }
                )
            ],
            event_handlers={
                "on_emergency": [
                    {"action": "close_all_positions", "slice": "live_trade"},
                    {"action": "halt_trading", "slice": "live_trade"},
                    {"action": "send_emergency_alert", "params": {"priority": "high"}}
                ],
                "on_daily_loss_limit": [
                    {"action": "stop_trading_for_day", "slice": "live_trade"},
                    {"action": "send_risk_alert", "params": {"type": "daily_loss_exceeded"}}
                ]
            }
        ))
        
        # 3b. Quick Test Workflow
        cls.register_workflow(WorkflowDefinition(
            name="quick_test",
            version="1.0",
            description="Quick two-step test workflow",
            workflow_type=WorkflowType.ANALYSIS,
            execution_mode=ExecutionMode.SEQUENTIAL,
            steps=[
                WorkflowStep(name="fetch_data", slice="data", action="get_historical_data"),
                WorkflowStep(name="analyze_market", slice="analyze", action="find_patterns"),
            ]
        ))

        # 4. Optimization Workflow
        cls.register_workflow(WorkflowDefinition(
            name="strategy_optimization",
            version="1.0",
            description="Optimize strategy parameters using historical data",
            workflow_type=WorkflowType.OPTIMIZATION,
            execution_mode=ExecutionMode.PARALLEL,
            variables={
                "optimization_method": "genetic_algorithm",
                "population_size": 100,
                "generations": 50,
                "objective_function": "sharpe_ratio",
                "cross_validation_folds": 5
            },
            steps=[
                WorkflowStep(
                    name="prepare_data",
                    slice="data",
                    action="prepare_optimization_data",
                    params={"split_ratio": 0.8, "walk_forward": True}
                ),
                WorkflowStep(
                    name="define_search_space",
                    slice="optimize",
                    action="create_parameter_space",
                    depends_on=["prepare_data"],
                    params={
                        "parameters": {
                            "lookback_period": {"min": 10, "max": 100, "step": 5},
                            "entry_threshold": {"min": 0.5, "max": 2.0, "step": 0.1},
                            "stop_loss": {"min": 0.005, "max": 0.05, "step": 0.005},
                            "take_profit": {"min": 0.01, "max": 0.1, "step": 0.01}
                        }
                    }
                ),
                WorkflowStep(
                    name="run_optimization",
                    slice="optimize",
                    action="optimize_parameters",
                    execution_mode=ExecutionMode.PARALLEL,
                    depends_on=["define_search_space"],
                    params={
                        "n_jobs": -1,  # Use all available cores
                        "cv_strategy": "time_series_split"
                    }
                ),
                WorkflowStep(
                    name="validate_results",
                    slice="optimize",
                    action="validate_optimal_parameters",
                    depends_on=["run_optimization"],
                    params={
                        "validation_metrics": ["sharpe", "calmar", "max_drawdown", "win_rate"]
                    }
                ),
                WorkflowStep(
                    name="generate_optimization_report",
                    slice="monitor",
                    action="create_optimization_report",
                    depends_on=["validate_results"]
                )
            ]
        ))

        # 6. ML-driven workflow (used by tests for continue_on_failure)
        cls.register_workflow(WorkflowDefinition(
            name="ml_driven",
            version="1.0",
            description="ML-driven analysis and optional optimization",
            workflow_type=WorkflowType.ANALYSIS,
            execution_mode=ExecutionMode.SEQUENTIAL,
            steps=[
                WorkflowStep(name="fetch_data", slice="data", action="get_historical_data"),
                WorkflowStep(name="analyze_market", slice="analyze", action="find_patterns"),
                WorkflowStep(name="select_strategy", slice="ml_strategy", action="predict_best_strategy"),
                # Optimization is optional; adapter marks optimize steps as continue_on_failure
                WorkflowStep(name="optimize", slice="optimize", action="optimize_parameters"),
                WorkflowStep(name="report", slice="monitor", action="create_performance_report"),
            ]
        ))
        
        # 5. Real-Time Monitoring Workflow
        cls.register_workflow(WorkflowDefinition(
            name="real_time_monitoring",
            version="1.0",
            description="Continuous monitoring and alerting workflow",
            workflow_type=WorkflowType.MONITORING,
            execution_mode=ExecutionMode.BACKGROUND,
            schedule="*/30 * * * * *",  # Every 30 seconds
            variables={
                "alert_thresholds": {
                    "drawdown": 0.05,
                    "daily_loss": 0.02,
                    "position_size": 0.1,
                    "correlation": 0.8
                },
                "monitoring_metrics": [
                    "pnl", "positions", "exposure", "risk_metrics", "system_health"
                ]
            },
            steps=[
                WorkflowStep(
                    name="collect_metrics",
                    execution_mode=ExecutionMode.PARALLEL,
                    params={
                        'nested_steps': [
                            WorkflowStep(
                                name="portfolio_metrics",
                                slice="monitor",
                                action="get_portfolio_metrics"
                            ),
                            WorkflowStep(
                                name="risk_metrics",
                                slice="position_sizing",
                                action="calculate_risk_metrics"
                            ),
                            WorkflowStep(
                                name="system_metrics",
                                slice="monitor",
                                action="get_system_metrics"
                            ),
                            WorkflowStep(
                                name="market_metrics",
                                slice="data",
                                action="get_market_indicators"
                            )
                        ]
                    }
                ),
                WorkflowStep(
                    name="analyze_metrics",
                    slice="monitor",
                    action="analyze_metric_trends",
                    depends_on=["collect_metrics"],
                    params={"lookback_window": 100, "anomaly_detection": True}
                ),
                WorkflowStep(
                    name="check_alerts",
                    slice="monitor",
                    action="evaluate_alert_conditions",
                    depends_on=["analyze_metrics"],
                    conditions=[
                        {
                            "if": "drawdown > alert_thresholds['drawdown']",
                            "then": [{"action": "send_drawdown_alert", "params": {"priority": "high"}}]
                        },
                        {
                            "if": "daily_loss > alert_thresholds['daily_loss']",
                            "then": [{"action": "send_loss_alert", "params": {"priority": "critical"}}]
                        },
                        {
                            "if": "system_health != 'healthy'",
                            "then": [{"action": "send_system_alert", "params": {"priority": "medium"}}]
                        }
                    ]
                ),
                WorkflowStep(
                    name="update_dashboard",
                    slice="monitor",
                    action="update_monitoring_dashboard",
                    depends_on=["check_alerts"],
                    params={"push_to_websocket": True}
                )
            ]
        ))
        
        # 6. Adaptive Strategy Selection Workflow
        cls.register_workflow(WorkflowDefinition(
            name="adaptive_strategy_selection",
            version="1.0",
            description="Dynamically select strategies based on market conditions",
            workflow_type=WorkflowType.ANALYSIS,
            execution_mode=ExecutionMode.CONDITIONAL,
            variables={
                "evaluation_period": "1h",
                "min_confidence": 0.6,
                "strategy_pool": [
                    "trend_following", "mean_reversion", "momentum",
                    "breakout", "volatility_arbitrage", "ml_ensemble"
                ]
            },
            steps=[
                WorkflowStep(
                    name="market_analysis",
                    slice="market_regime",
                    action="comprehensive_market_analysis",
                    outputs=["regime", "volatility", "trend", "correlation_matrix"]
                ),
                WorkflowStep(
                    name="strategy_scoring",
                    execution_mode=ExecutionMode.PARALLEL,
                    depends_on=["market_analysis"],
                    params={
                        'nested_steps': [
                            WorkflowStep(
                                name="score_trend_following",
                                slice="ml_strategy",
                                action="score_strategy_fitness",
                                params={"strategy": "trend_following", "lookback": 20}
                            ),
                            WorkflowStep(
                                name="score_mean_reversion",
                                slice="ml_strategy",
                                action="score_strategy_fitness",
                                params={"strategy": "mean_reversion", "lookback": 20}
                            ),
                            WorkflowStep(
                                name="score_momentum",
                                slice="ml_strategy",
                                action="score_strategy_fitness",
                                params={"strategy": "momentum", "lookback": 20}
                            )
                        ]
                    }
                ),
                WorkflowStep(
                    name="select_optimal_strategy",
                    slice="ml_strategy",
                    action="select_best_strategies",
                    depends_on=["strategy_scoring"],
                    params={
                        "selection_method": "probabilistic",
                        "max_strategies": 3,
                        "min_score": 0.6
                    }
                ),
                WorkflowStep(
                    name="allocate_capital",
                    slice="adaptive_portfolio",
                    action="optimize_allocation",
                    depends_on=["select_optimal_strategy"],
                    params={
                        "method": "risk_parity",
                        "constraints": {
                            "min_allocation": 0.1,
                            "max_allocation": 0.5,
                            "total_leverage": 1.0
                        }
                    }
                ),
                WorkflowStep(
                    name="deploy_strategies",
                    execution_mode=ExecutionMode.CONDITIONAL,
                    depends_on=["allocate_capital"],
                    conditions=[
                        {
                            "if": "total_confidence > 0.7",
                            "then": [
                                {
                                    "action": "deploy_to_paper_trade",
                                    "slice": "paper_trade",
                                    "params": {"mode": "aggressive"}
                                }
                            ]
                        },
                        {
                            "elif": "total_confidence > 0.5",
                            "then": [
                                {
                                    "action": "deploy_to_paper_trade",
                                    "slice": "paper_trade",
                                    "params": {"mode": "conservative"}
                                }
                            ]
                        },
                        {
                            "else": [
                                {
                                    "action": "maintain_current_positions",
                                    "slice": "paper_trade"
                                }
                            ]
                        }
                    ]
                )
            ]
        ))


# Initialize default workflows when module is imported
WorkflowDefinitions.initialize_default_workflows()

# Export workflows for easy access
ALL_WORKFLOWS = WorkflowDefinitions.WORKFLOWS


def load_workflow_from_yaml(filepath: str) -> WorkflowDefinition:
    """
    Load workflow definition from YAML file.
    
    Args:
        filepath: Path to YAML workflow definition
        
    Returns:
        WorkflowDefinition object
    """
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    
    # Parse workflow definition
    workflow = WorkflowDefinition(
        name=data['metadata']['name'],
        version=data['metadata']['version'],
        description=data['metadata'].get('description', ''),
        workflow_type=WorkflowType(data['spec'].get('type', 'backtest')),
        execution_mode=ExecutionMode(data['spec'].get('execution', {}).get('mode', 'sequential')),
        variables=data['spec'].get('variables', {}),
        resources=data['spec'].get('resources', {}),
        schedule=data['spec'].get('schedule'),
        event_handlers=data['spec'].get('event_handlers', {}),
        steps=[]
    )
    
    # Parse steps
    for step_data in data['spec'].get('steps', []):
        step = _parse_workflow_step(step_data)
        workflow.steps.append(step)
    
    return workflow


def _parse_workflow_step(step_data: Dict[str, Any]) -> WorkflowStep:
    """Parse individual workflow step from dictionary"""
    step = WorkflowStep(
        name=step_data['name'],
        slice=step_data.get('slice'),
        action=step_data.get('action'),
        params=step_data.get('params', {}),
        depends_on=step_data.get('depends_on', []),
        conditions=step_data.get('conditions', []),
        execution_mode=ExecutionMode(step_data.get('type', 'sequential')),
        timeout=step_data.get('timeout', 300),
        retry_count=step_data.get('retry_count', 3),
        weight=step_data.get('weight', 1.0),
        outputs=step_data.get('outputs', [])
    )
    
    # Handle nested steps for parallel execution
    if 'steps' in step_data:
        step.params['nested_steps'] = [
            _parse_workflow_step(sub_step) for sub_step in step_data['steps']
        ]
    
    return step


def save_workflow_to_yaml(workflow: WorkflowDefinition, filepath: str):
    """
    Save workflow definition to YAML file.
    
    Args:
        workflow: WorkflowDefinition object
        filepath: Path to save YAML file
    """
    data = {
        'apiVersion': 'bot_v2/workflows/v2',
        'kind': 'WorkflowDefinition',
        'metadata': {
            'name': workflow.name,
            'version': workflow.version,
            'description': workflow.description
        },
        'spec': {
            'type': workflow.workflow_type.value,
            'execution': {
                'mode': workflow.execution_mode.value
            },
            'variables': workflow.variables,
            'resources': workflow.resources,
            'schedule': workflow.schedule,
            'steps': [_step_to_dict(step) for step in workflow.steps],
            'event_handlers': workflow.event_handlers
        }
    }
    
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _step_to_dict(step: WorkflowStep) -> Dict[str, Any]:
    """Convert WorkflowStep to dictionary"""
    step_dict = {
        'name': step.name,
        'type': step.execution_mode.value
    }
    
    if step.slice:
        step_dict['slice'] = step.slice
    if step.action:
        step_dict['action'] = step.action
    if step.params:
        # Handle nested steps
        if 'nested_steps' in step.params:
            step_dict['steps'] = [
                _step_to_dict(sub_step) for sub_step in step.params['nested_steps']
            ]
            # Remove nested_steps from params
            params = step.params.copy()
            del params['nested_steps']
            if params:
                step_dict['params'] = params
        else:
            step_dict['params'] = step.params
    
    if step.depends_on:
        step_dict['depends_on'] = step.depends_on
    if step.conditions:
        step_dict['conditions'] = step.conditions
    if step.timeout != 300:
        step_dict['timeout'] = step.timeout
    if step.retry_count != 3:
        step_dict['retry_count'] = step.retry_count
    if step.weight != 1.0:
        step_dict['weight'] = step.weight
    if step.outputs:
        step_dict['outputs'] = step.outputs
    
    return step_dict


# Convenience functions
def get_simple_backtest_workflow() -> WorkflowDefinition:
    """Get simple backtest workflow"""
    return WorkflowDefinitions.get_workflow("simple_backtest")


def get_multi_strategy_workflow() -> WorkflowDefinition:
    """Get multi-strategy ensemble workflow"""
    return WorkflowDefinitions.get_workflow("multi_strategy_ensemble")


def get_live_trading_workflow() -> WorkflowDefinition:
    """Get risk-managed live trading workflow"""
    return WorkflowDefinitions.get_workflow("risk_managed_live_trading")


def get_optimization_workflow() -> WorkflowDefinition:
    """Get strategy optimization workflow"""
    return WorkflowDefinitions.get_workflow("strategy_optimization")


def get_monitoring_workflow() -> WorkflowDefinition:
    """Get real-time monitoring workflow"""
    return WorkflowDefinitions.get_workflow("real_time_monitoring")


def get_adaptive_workflow() -> WorkflowDefinition:
    """Get adaptive strategy selection workflow"""
    return WorkflowDefinitions.get_workflow("adaptive_strategy_selection")


# Initialize default workflows when module is imported
WorkflowDefinitions.initialize_default_workflows()

# Export ALL_WORKFLOWS for backwards compatibility
ALL_WORKFLOWS = WorkflowDefinitions.WORKFLOWS
