"""
Workflow System for Bot V2 Trading System

EXPERIMENTAL: Generic workflow layer primarily used for demos/legacy flows.
Not required for the perps production pipeline.

Provides comprehensive workflow orchestration with support for:
- Sequential, parallel, conditional, background, and scheduled execution
- Safe condition evaluation with trading-specific logic
- Predefined workflow templates for common trading scenarios
- State management and progress tracking
"""

from .definitions import (
    WorkflowType,
    ExecutionMode,
    WorkflowStep,
    WorkflowDefinition,
    WorkflowDefinitions,
    ALL_WORKFLOWS,
    load_workflow_from_yaml,
    save_workflow_to_yaml,
    get_simple_backtest_workflow,
    get_multi_strategy_workflow,
    get_live_trading_workflow,
    get_optimization_workflow,
    get_monitoring_workflow,
    get_adaptive_workflow
)

from .engine import (
    WorkflowEngine
)

from .context import (
    WorkflowContext
)

__all__ = [
    # Definitions
    'WorkflowType',
    'ExecutionMode',
    'WorkflowStep',
    'WorkflowDefinition',
    'WorkflowDefinitions',
    'ALL_WORKFLOWS',
    'load_workflow_from_yaml',
    'save_workflow_to_yaml',
    'get_simple_backtest_workflow',
    'get_multi_strategy_workflow',
    'get_live_trading_workflow',
    'get_optimization_workflow',
    'get_monitoring_workflow',
    'get_adaptive_workflow',
    
    # Engine
    'WorkflowEngine',
    
    # Context
    'WorkflowContext'
]

# Adapt ALL_WORKFLOWS to engine-ready steps for convenience in tests
try:
    from .workflow_adapter import WorkflowAdapter
    # Ensure default workflows are initialized
    WorkflowDefinitions.initialize_default_workflows()
    _adapter = WorkflowAdapter()
    _adapted = {name: _adapter.convert_workflow(defn) for name, defn in WorkflowDefinitions.WORKFLOWS.items()}
    ALL_WORKFLOWS = _adapted  # override to expose engine steps
except Exception:
    pass

# Version
__version__ = '1.0.0'

# Marker used by tooling and documentation
__experimental__ = True
