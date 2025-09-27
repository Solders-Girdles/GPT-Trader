"""Risk-Adjusted Trading Workflows

This module contains workflow definitions for risk-aware trading operations
including position sizing, stop-loss monitoring, and drawdown protection.
"""

from .engine import WorkflowStep

# Risk-Adjusted Trading Workflow
RISK_ADJUSTED_TRADING = [
    WorkflowStep(
        name="Fetch Market Data",
        function="fetch_data",
        description="Get market data for analysis",
        required_context=['symbol'],
        outputs=['market_data']
    ),
    WorkflowStep(
        name="Calculate Risk Metrics",
        function="calculate_risk_metrics",
        description="Calculate volatility, VaR, CVaR",
        required_context=['market_data'],
        outputs=['volatility', 'var_95', 'cvar_95', 'beta']
    ),
    WorkflowStep(
        name="Determine Risk Budget",
        function="determine_risk_budget",
        description="Calculate risk budget based on portfolio",
        required_context=['capital', 'risk_tolerance', 'volatility'],
        outputs=['risk_budget', 'max_position_size']
    ),
    WorkflowStep(
        name="Adjust Strategy Parameters",
        function="adjust_strategy_risk",
        description="Adjust strategy based on risk",
        required_context=['strategy', 'risk_budget', 'volatility'],
        outputs=['adjusted_strategy', 'stop_loss', 'take_profit']
    ),
    WorkflowStep(
        name="Calculate Risk-Adjusted Position",
        function="calculate_risk_position",
        description="Size position based on risk",
        required_context=['risk_budget', 'volatility', 'confidence'],
        outputs=['position_size', 'max_loss']
    ),
    WorkflowStep(
        name="Execute with Risk Controls",
        function="execute_risk_controlled",
        description="Execute trade with risk limits",
        required_context=['symbol', 'position_size', 'stop_loss', 'take_profit'],
        outputs=['execution_result']
    ),
    WorkflowStep(
        name="Monitor Initial Performance",
        function="monitor_initial_performance",
        description="Track trade performance immediately after execution",
        required_context=['execution_result'],
        outputs=['initial_pnl', 'risk_status']
    )
]

# Stop-Loss Monitoring Workflow
STOP_LOSS_MONITOR = [
    WorkflowStep(
        name="Get Open Positions",
        function="get_open_positions",
        description="Fetch all open positions",
        required_context=[],
        outputs=['open_positions']
    ),
    WorkflowStep(
        name="Check Stop Losses",
        function="check_stop_losses",
        description="Check if stop losses are triggered",
        required_context=['open_positions'],
        outputs=['triggered_stops', 'positions_at_risk']
    ),
    WorkflowStep(
        name="Calculate Unrealized PnL",
        function="calculate_unrealized_pnl",
        description="Calculate current unrealized gains/losses",
        required_context=['open_positions'],
        outputs=['unrealized_pnl', 'position_risks']
    ),
    WorkflowStep(
        name="Execute Stop Orders",
        function="execute_stop_orders",
        description="Close positions hitting stops",
        required_context=['triggered_stops'],
        outputs=['closed_positions'],
        continue_on_failure=False
    ),
    WorkflowStep(
        name="Update Risk Metrics",
        function="update_risk_metrics",
        description="Update portfolio risk after stops",
        required_context=['closed_positions'],
        outputs=['updated_risk']
    ),
    WorkflowStep(
        name="Log Risk Events",
        function="log_risk_events",
        description="Log all risk management actions",
        required_context=['closed_positions', 'updated_risk'],
        outputs=['risk_log']
    )
]

# Position Sizing Based on Risk Workflow
POSITION_SIZING_WORKFLOW = [
    WorkflowStep(
        name="Analyze Asset Volatility",
        function="analyze_volatility",
        description="Calculate historical volatility metrics",
        required_context=['symbol', 'lookback_period'],
        outputs=['volatility', 'volatility_percentile']
    ),
    WorkflowStep(
        name="Calculate Kelly Criterion",
        function="calculate_kelly",
        description="Calculate optimal position size using Kelly criterion",
        required_context=['win_rate', 'avg_win', 'avg_loss'],
        outputs=['kelly_fraction', 'kelly_position_size']
    ),
    WorkflowStep(
        name="Apply Risk Budget Constraints",
        function="apply_risk_constraints",
        description="Apply portfolio-level risk constraints",
        required_context=['kelly_position_size', 'portfolio_risk_budget', 'correlation'],
        outputs=['constrained_position_size']
    ),
    WorkflowStep(
        name="Calculate Value at Risk",
        function="calculate_var",
        description="Calculate position-level VaR",
        required_context=['constrained_position_size', 'volatility'],
        outputs=['position_var', 'position_cvar']
    ),
    WorkflowStep(
        name="Final Position Sizing",
        function="finalize_position_size",
        description="Final position size considering all constraints",
        required_context=['constrained_position_size', 'position_var', 'max_position_risk'],
        outputs=['final_position_size', 'expected_risk']
    )
]

# Drawdown Protection Workflow
DRAWDOWN_PROTECTION = [
    WorkflowStep(
        name="Calculate Current Drawdown",
        function="calculate_drawdown",
        description="Calculate portfolio drawdown",
        required_context=['capital'],
        outputs=['current_drawdown', 'max_drawdown']
    ),
    WorkflowStep(
        name="Check Drawdown Limits",
        function="check_drawdown_limits",
        description="Check if limits are breached",
        required_context=['current_drawdown', 'max_allowed_drawdown'],
        outputs=['drawdown_breach', 'severity']
    ),
    WorkflowStep(
        name="Calculate Drawdown Velocity",
        function="calculate_drawdown_velocity",
        description="Calculate rate of drawdown change",
        required_context=['current_drawdown'],
        outputs=['drawdown_velocity', 'acceleration']
    ),
    WorkflowStep(
        name="Assess Portfolio Correlation",
        function="assess_correlation",
        description="Check if positions are highly correlated",
        required_context=['open_positions'],
        outputs=['correlation_matrix', 'concentration_risk']
    ),
    WorkflowStep(
        name="Reduce Exposure",
        function="reduce_exposure",
        description="Reduce positions if needed",
        required_context=['drawdown_breach', 'severity', 'open_positions', 'concentration_risk'],
        outputs=['reduction_trades'],
        continue_on_failure=True
    ),
    WorkflowStep(
        name="Pause Trading",
        function="pause_trading",
        description="Pause new trades if severe",
        required_context=['severity'],
        outputs=['trading_paused']
    ),
    WorkflowStep(
        name="Schedule Recovery Plan",
        function="schedule_recovery",
        description="Create plan for recovery from drawdown",
        required_context=['severity', 'drawdown_velocity'],
        outputs=['recovery_plan', 'recovery_timeline']
    )
]

# Risk Monitoring Workflow
RISK_MONITORING_WORKFLOW = [
    WorkflowStep(
        name="Collect Portfolio Metrics",
        function="collect_portfolio_metrics",
        description="Gather all portfolio risk metrics",
        required_context=['portfolio'],
        outputs=['portfolio_metrics']
    ),
    WorkflowStep(
        name="Calculate Risk-Adjusted Returns",
        function="calculate_risk_adjusted_returns",
        description="Calculate Sharpe, Sortino, Calmar ratios",
        required_context=['portfolio_metrics'],
        outputs=['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']
    ),
    WorkflowStep(
        name="Monitor Risk Limits",
        function="monitor_risk_limits",
        description="Check all risk limits and constraints",
        required_context=['portfolio_metrics'],
        outputs=['limit_breaches', 'warning_signals']
    ),
    WorkflowStep(
        name="Generate Risk Report",
        function="generate_risk_report",
        description="Generate comprehensive risk report",
        required_context=['portfolio_metrics', 'limit_breaches', 'sharpe_ratio'],
        outputs=['risk_report']
    ),
    WorkflowStep(
        name="Alert Risk Manager",
        function="alert_risk_manager",
        description="Send alerts if necessary",
        required_context=['limit_breaches', 'warning_signals'],
        outputs=['alerts_sent']
    )
]

# Export all risk workflows
RISK_WORKFLOWS = {
    'risk_adjusted_trading': RISK_ADJUSTED_TRADING,
    'stop_loss_monitor': STOP_LOSS_MONITOR,
    'position_sizing': POSITION_SIZING_WORKFLOW,
    'drawdown_protection': DRAWDOWN_PROTECTION,
    'risk_monitoring': RISK_MONITORING_WORKFLOW
}

# Workflow metadata for easy discovery
WORKFLOW_METADATA = {
    'risk_adjusted_trading': {
        'description': 'Complete risk-aware trading workflow with position sizing and controls',
        'use_case': 'Primary trading workflow with integrated risk management',
        'frequency': 'Per trade execution'
    },
    'stop_loss_monitor': {
        'description': 'Continuous monitoring and execution of stop-loss orders',
        'use_case': 'Risk management for open positions',
        'frequency': 'Real-time monitoring'
    },
    'position_sizing': {
        'description': 'Advanced position sizing using multiple risk metrics',
        'use_case': 'Pre-trade position size calculation',
        'frequency': 'Before each trade'
    },
    'drawdown_protection': {
        'description': 'Portfolio-level drawdown monitoring and protection',
        'use_case': 'Portfolio risk management and capital preservation',
        'frequency': 'Daily or on significant losses'
    },
    'risk_monitoring': {
        'description': 'Comprehensive portfolio risk monitoring and reporting',
        'use_case': 'Regular risk assessment and reporting',
        'frequency': 'Daily or weekly'
    }
}