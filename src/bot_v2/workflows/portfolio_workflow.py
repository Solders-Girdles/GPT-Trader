"""
Portfolio-level workflows for multi-asset portfolio management.

This module defines comprehensive workflows for:
- Portfolio analysis and optimization
- Portfolio rebalancing
- Risk-parity portfolio construction
- Correlation analysis and diversification scoring
"""

from .engine import WorkflowStep
from typing import Dict, List

# Portfolio Analysis Workflow
PORTFOLIO_ANALYSIS = [
    WorkflowStep(
        name="Fetch Portfolio Data",
        function="fetch_portfolio_data",
        description="Get historical price data for all portfolio symbols",
        required_context=['symbols', 'capital'],
        outputs=['portfolio_data', 'correlation_matrix']
    ),
    WorkflowStep(
        name="Analyze Correlations",
        function="analyze_correlations", 
        description="Analyze cross-asset correlations and diversification benefits",
        required_context=['portfolio_data', 'correlation_matrix'],
        outputs=['correlation_analysis', 'diversification_score']
    ),
    WorkflowStep(
        name="Calculate Risk Metrics",
        function="calculate_portfolio_risk",
        description="Calculate portfolio-level risk metrics (VaR, Sharpe, drawdown)",
        required_context=['portfolio_data', 'correlation_matrix'],
        outputs=['portfolio_var', 'portfolio_sharpe', 'max_drawdown']
    ),
    WorkflowStep(
        name="Optimize Allocation",
        function="optimize_portfolio_allocation",
        description="Find optimal asset weights using modern portfolio theory",
        required_context=['portfolio_data', 'correlation_matrix', 'capital'],
        outputs=['optimal_weights', 'expected_return', 'expected_risk']
    ),
    WorkflowStep(
        name="Generate Portfolio Report",
        function="generate_portfolio_report",
        description="Create comprehensive portfolio analysis report",
        required_context=['optimal_weights', 'portfolio_var', 'diversification_score'],
        outputs=['portfolio_report']
    )
]

# Portfolio Rebalancing Workflow
PORTFOLIO_REBALANCE = [
    WorkflowStep(
        name="Get Current Positions",
        function="get_current_positions",
        description="Fetch current portfolio positions and market values",
        required_context=['symbols'],
        outputs=['current_positions', 'current_weights']
    ),
    WorkflowStep(
        name="Calculate Target Weights",
        function="calculate_target_weights",
        description="Determine target allocation based on strategy and risk tolerance",
        required_context=['symbols', 'capital', 'risk_tolerance'],
        outputs=['target_weights']
    ),
    WorkflowStep(
        name="Calculate Rebalancing Trades",
        function="calculate_rebalance_trades",
        description="Determine required trades to reach target allocation",
        required_context=['current_positions', 'current_weights', 'target_weights'],
        outputs=['rebalance_trades', 'estimated_costs']
    ),
    WorkflowStep(
        name="Validate Rebalancing",
        function="validate_rebalance",
        description="Validate rebalancing trades meet risk and cost constraints",
        required_context=['rebalance_trades', 'estimated_costs'],
        outputs=['validation_results']
    ),
    WorkflowStep(
        name="Execute Rebalancing",
        function="execute_rebalance",
        description="Execute validated rebalancing trades",
        required_context=['rebalance_trades', 'validation_results'],
        outputs=['execution_results'],
        continue_on_failure=False
    )
]

# Risk-Parity Portfolio Workflow
RISK_PARITY_PORTFOLIO = [
    WorkflowStep(
        name="Fetch Historical Data",
        function="fetch_portfolio_data",
        description="Get historical data for risk parity calculation",
        required_context=['symbols'],
        outputs=['portfolio_data']
    ),
    WorkflowStep(
        name="Calculate Asset Volatilities",
        function="calculate_volatilities",
        description="Calculate individual asset volatilities and correlations",
        required_context=['portfolio_data'],
        outputs=['volatilities', 'correlation_matrix']
    ),
    WorkflowStep(
        name="Calculate Risk Parity Weights",
        function="calculate_risk_parity",
        description="Calculate weights for equal risk contribution",
        required_context=['volatilities', 'correlation_matrix'],
        outputs=['risk_parity_weights']
    ),
    WorkflowStep(
        name="Validate Risk Parity",
        function="validate_risk_parity",
        description="Validate risk contributions are approximately equal",
        required_context=['risk_parity_weights', 'volatilities', 'correlation_matrix'],
        outputs=['risk_contribution_analysis']
    ),
    WorkflowStep(
        name="Backtest Risk Parity",
        function="backtest_portfolio",
        description="Test risk parity strategy performance",
        required_context=['portfolio_data', 'risk_parity_weights'],
        outputs=['backtest_results']
    )
]

# Momentum Portfolio Workflow
MOMENTUM_PORTFOLIO = [
    WorkflowStep(
        name="Fetch Price Data",
        function="fetch_portfolio_data",
        description="Get price data for momentum calculation",
        required_context=['symbols'],
        outputs=['portfolio_data']
    ),
    WorkflowStep(
        name="Calculate Momentum Scores",
        function="calculate_momentum_scores",
        description="Calculate momentum scores for each asset",
        required_context=['portfolio_data'],
        outputs=['momentum_scores', 'momentum_rankings']
    ),
    WorkflowStep(
        name="Apply Momentum Filter",
        function="apply_momentum_filter",
        description="Filter assets based on momentum criteria",
        required_context=['momentum_scores', 'momentum_rankings'],
        outputs=['filtered_symbols', 'momentum_weights']
    ),
    WorkflowStep(
        name="Optimize Momentum Portfolio",
        function="optimize_momentum_portfolio",
        description="Optimize weights for filtered momentum assets",
        required_context=['filtered_symbols', 'momentum_weights', 'portfolio_data'],
        outputs=['optimized_momentum_weights']
    ),
    WorkflowStep(
        name="Backtest Momentum Strategy",
        function="backtest_portfolio",
        description="Test momentum strategy performance",
        required_context=['portfolio_data', 'optimized_momentum_weights'],
        outputs=['momentum_backtest_results']
    )
]

# Mean Reversion Portfolio Workflow
MEAN_REVERSION_PORTFOLIO = [
    WorkflowStep(
        name="Fetch Portfolio Data",
        function="fetch_portfolio_data",
        description="Get data for mean reversion analysis",
        required_context=['symbols'],
        outputs=['portfolio_data']
    ),
    WorkflowStep(
        name="Calculate Mean Reversion Signals",
        function="calculate_mean_reversion_signals",
        description="Calculate mean reversion indicators for each asset",
        required_context=['portfolio_data'],
        outputs=['reversion_signals', 'z_scores']
    ),
    WorkflowStep(
        name="Apply Reversion Filter",
        function="apply_reversion_filter",
        description="Filter assets showing mean reversion opportunities",
        required_context=['reversion_signals', 'z_scores'],
        outputs=['reversion_candidates', 'reversion_weights']
    ),
    WorkflowStep(
        name="Optimize Reversion Portfolio",
        function="optimize_reversion_portfolio",
        description="Optimize portfolio for mean reversion strategy",
        required_context=['reversion_candidates', 'reversion_weights', 'portfolio_data'],
        outputs=['optimized_reversion_weights']
    ),
    WorkflowStep(
        name="Backtest Mean Reversion",
        function="backtest_portfolio",
        description="Test mean reversion strategy performance",
        required_context=['portfolio_data', 'optimized_reversion_weights'],
        outputs=['reversion_backtest_results']
    )
]

# Multi-Strategy Portfolio Workflow
MULTI_STRATEGY_PORTFOLIO = [
    WorkflowStep(
        name="Initialize Strategies",
        function="initialize_strategies",
        description="Initialize multiple trading strategies",
        required_context=['symbols', 'capital'],
        outputs=['strategy_configs']
    ),
    WorkflowStep(
        name="Generate Strategy Signals",
        function="generate_multi_strategy_signals",
        description="Generate signals from all strategies",
        required_context=['portfolio_data', 'strategy_configs'],
        outputs=['strategy_signals', 'signal_weights']
    ),
    WorkflowStep(
        name="Combine Strategy Signals",
        function="combine_strategy_signals",
        description="Combine signals using ensemble methods",
        required_context=['strategy_signals', 'signal_weights'],
        outputs=['combined_signals', 'ensemble_weights']
    ),
    WorkflowStep(
        name="Allocate Multi-Strategy Portfolio",
        function="allocate_multi_strategy",
        description="Allocate capital across combined strategy signals",
        required_context=['combined_signals', 'ensemble_weights', 'capital'],
        outputs=['multi_strategy_weights']
    ),
    WorkflowStep(
        name="Backtest Multi-Strategy",
        function="backtest_portfolio",
        description="Test combined multi-strategy performance",
        required_context=['portfolio_data', 'multi_strategy_weights'],
        outputs=['multi_strategy_results']
    )
]

# Export all portfolio workflows
PORTFOLIO_WORKFLOWS: Dict[str, List[WorkflowStep]] = {
    'portfolio_analysis': PORTFOLIO_ANALYSIS,
    'portfolio_rebalance': PORTFOLIO_REBALANCE,
    'risk_parity_portfolio': RISK_PARITY_PORTFOLIO,
    'momentum_portfolio': MOMENTUM_PORTFOLIO,
    'mean_reversion_portfolio': MEAN_REVERSION_PORTFOLIO,
    'multi_strategy_portfolio': MULTI_STRATEGY_PORTFOLIO
}

def get_portfolio_workflow(workflow_name: str) -> List[WorkflowStep]:
    """
    Get a portfolio workflow by name.
    
    Args:
        workflow_name: Name of the workflow to retrieve
        
    Returns:
        List of workflow steps
        
    Raises:
        KeyError: If workflow name not found
    """
    if workflow_name not in PORTFOLIO_WORKFLOWS:
        available = list(PORTFOLIO_WORKFLOWS.keys())
        raise KeyError(f"Portfolio workflow '{workflow_name}' not found. Available: {available}")
    
    return PORTFOLIO_WORKFLOWS[workflow_name]

def list_portfolio_workflows() -> List[str]:
    """
    List all available portfolio workflows.
    
    Returns:
        List of available workflow names
    """
    return list(PORTFOLIO_WORKFLOWS.keys())