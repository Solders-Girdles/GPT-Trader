"""
Workflow Adapter - Converts WorkflowDefinition to executable WorkflowSteps

This adapter bridges the gap between declarative workflow definitions
and the executable workflow engine format.
"""
import logging
from typing import Dict, Any, List, Callable
from dataclasses import dataclass, field

# Import definition types
from .definitions import WorkflowDefinition, WorkflowStep as DefinitionStep

logger = logging.getLogger(__name__)


# Define EngineStep locally to avoid circular import
@dataclass
class EngineStep:
    """Definition of a single workflow step for the engine"""
    name: str
    function: Callable
    description: str
    required_context: List[str] = None
    outputs: List[str] = None
    retry_count: int = 3
    continue_on_failure: bool = False
    # Extra metadata to aid dispatching or debugging
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.required_context = self.required_context or []
        self.outputs = self.outputs or []


class WorkflowAdapter:
    """
    Converts WorkflowDefinition objects to executable workflow steps
    that the WorkflowEngine can process.
    """
    
    def __init__(self):
        """Initialize the workflow adapter"""
        self.logger = logging.getLogger("workflow.adapter")
        
        # Map slice/action combinations to function names
        self.action_map = {
            ('data', 'fetch'): 'fetch_data',
            ('data', 'get_data'): 'fetch_data',
            ('data', 'fetch_historical'): 'fetch_data',
            ('data', 'get_historical_data'): 'fetch_data',
            ('data', 'prepare_optimization_data'): 'fetch_data',
            ('data', 'get_real_time_quotes'): 'fetch_data',
            ('analyze', 'analyze'): 'analyze_market',
            ('analyze', 'analyze_symbol'): 'analyze_market',
            ('analyze', 'find_patterns'): 'analyze_market',
            ('analyze', 'generate_trading_signals'): 'analyze_market',
            ('analyze', 'run_momentum_strategy'): 'analyze_market',
            ('analyze', 'run_mean_reversion_strategy'): 'analyze_market',
            ('market_regime', 'detect'): 'detect_regime',
            ('market_regime', 'detect_regime'): 'detect_regime',
            ('market_regime', 'detect_current_regime'): 'detect_regime',
            ('ml_strategy', 'predict'): 'select_strategy',
            ('ml_strategy', 'predict_best_strategy'): 'select_strategy',
            ('ml_strategy', 'predict_ensemble'): 'select_strategy',
            ('position_sizing', 'calculate'): 'calculate_position',
            ('position_sizing', 'calculate_position_size'): 'calculate_position',
            ('position_sizing', 'calculate_optimal_sizes'): 'calculate_position',
            ('position_sizing', 'validate_risk_limits'): 'calculate_position',
            ('position_sizing', 'check_risk_constraints'): 'calculate_position',
            ('backtest', 'run'): 'execute_backtest',
            ('backtest', 'run_backtest'): 'execute_backtest',
            ('backtest', 'execute'): 'execute_backtest',
            ('paper_trade', 'execute'): 'execute_paper_trade',
            ('paper_trade', 'execute_paper_trade'): 'execute_paper_trade',
            ('paper_trade', 'execute_trades'): 'execute_paper_trade',
            ('paper_trade', 'maintain_current_positions'): 'monitor_performance',
            ('live_trade', 'execute'): 'execute_live_trade',
            ('live_trade', 'execute_live_trade'): 'execute_live_trade',
            ('live_trade', 'submit_orders'): 'execute_live_trade',
            ('live_trade', 'validate_connection'): 'monitor_performance',
            ('live_trade', 'reconcile_with_broker'): 'monitor_performance',
            ('monitor', 'log'): 'monitor_performance',
            ('monitor', 'log_event'): 'monitor_performance',
            ('monitor', 'track_performance'): 'monitor_performance',
            ('monitor', 'check_system_health'): 'monitor_performance',
            ('monitor', 'create_performance_report'): 'monitor_performance',
            ('monitor', 'calculate_daily_metrics'): 'monitor_performance',
            ('monitor', 'send_performance_report'): 'monitor_performance',
            ('optimize', 'optimize'): 'optimize_parameters',
            ('optimize', 'optimize_strategy'): 'optimize_parameters',
            ('report', 'generate'): 'generate_report',
            ('*', 'report'): 'generate_report',
        }
    
    def convert_workflow(self, workflow_def: WorkflowDefinition) -> List[EngineStep]:
        """
        Convert a WorkflowDefinition to a list of executable WorkflowSteps.
        
        Args:
            workflow_def: The workflow definition to convert
            
        Returns:
            List of EngineStep objects that can be executed
        """
        engine_steps = []
        
        for def_step in workflow_def.steps:
            try:
                engine_step = self._convert_step(def_step, workflow_def)
                engine_steps.append(engine_step)
            except Exception as e:
                self.logger.error(f"Failed to convert step {def_step.name}: {e}")
                # Create a placeholder step that will fail gracefully
                engine_steps.append(EngineStep(
                    name=def_step.name,
                    function='unknown',
                    description=f"Failed to convert: {str(e)}",
                    required_context=[],
                    outputs=[],
                    retry_count=0,
                    continue_on_failure=True
                ))
        
        return engine_steps
    
    def _convert_step(self, def_step: DefinitionStep, workflow_def: WorkflowDefinition) -> EngineStep:
        """
        Convert a single definition step to an engine step.
        
        Args:
            def_step: The definition step to convert
            workflow_def: The parent workflow definition for context
            
        Returns:
            An EngineStep that can be executed
        """
        # Determine the function name from slice and action
        function_name = self._get_function_name(def_step)
        
        # Extract required context from dependencies and params
        required_context = self._extract_required_context(def_step)
        
        # Extract outputs
        outputs = def_step.outputs if def_step.outputs else self._infer_outputs(function_name)
        
        # Create the engine step
        engine_step = EngineStep(
            name=def_step.name,
            function=function_name,
            description=f"{def_step.slice}.{def_step.action if def_step.action else 'default'}",
            required_context=required_context,
            outputs=outputs,
            retry_count=def_step.retry_count,
            continue_on_failure=self._should_continue_on_failure(def_step),
            meta={
                'slice': def_step.slice,
                'action': def_step.action,
                'execution_mode': getattr(def_step, 'execution_mode', None).value if getattr(def_step, 'execution_mode', None) else None,
                'params': getattr(def_step, 'params', {})
            }
        )
        
        return engine_step
    
    def _get_function_name(self, def_step: DefinitionStep) -> str:
        """
        Map slice/action combination to a function name.
        
        Args:
            def_step: The definition step
            
        Returns:
            The function name to use in the engine
        """
        # Try exact match first
        key = (def_step.slice, def_step.action)
        if key in self.action_map:
            return self.action_map[key]
        
        # Try wildcard slice match
        wildcard_key = ('*', def_step.action)
        if wildcard_key in self.action_map:
            return self.action_map[wildcard_key]
        
        # Default mapping based on slice name
        default_functions = {
            'data': 'fetch_data',
            'analyze': 'analyze_market',
            'market_regime': 'detect_regime',
            'ml_strategy': 'select_strategy',
            'position_sizing': 'calculate_position',
            'backtest': 'execute_backtest',
            'paper_trade': 'execute_paper_trade',
            'live_trade': 'execute_live_trade',
            'monitor': 'monitor_performance',
            'optimize': 'optimize_parameters'
        }
        
        if def_step.slice in default_functions:
            return default_functions[def_step.slice]
        
        # If all else fails, use the action as the function name
        return def_step.action or 'unknown'
    
    def _extract_required_context(self, def_step: DefinitionStep) -> List[str]:
        """
        Extract required context variables from the step definition.
        
        Args:
            def_step: The definition step
            
        Returns:
            List of required context variable names
        """
        required = []
        
        # Add explicit dependencies
        required.extend(def_step.depends_on)
        
        # Add parameters that reference context variables
        for param_name, param_value in def_step.params.items():
            if isinstance(param_value, str) and param_value.startswith('${') and param_value.endswith('}'):
                # This is a context variable reference
                var_name = param_value[2:-1]  # Remove ${ and }
                if var_name not in required:
                    required.append(var_name)
        
        # Add common requirements based on function type
        if def_step.slice in ['analyze', 'market_regime', 'ml_strategy']:
            if 'market_data' not in required:
                required.append('market_data')
            if 'symbol' not in required:
                required.append('symbol')
        
        return required
    
    def _infer_outputs(self, function_name: str) -> List[str]:
        """
        Infer output variables based on function name.
        
        Args:
            function_name: The function name
            
        Returns:
            List of output variable names
        """
        output_map = {
            'fetch_data': ['market_data', 'symbol', 'rows'],
            'analyze_market': ['analysis'],
            'detect_regime': ['regime'],
            'select_strategy': ['strategy', 'confidence'],
            'calculate_position': ['position_size', 'position_value'],
            'execute_backtest': ['backtest_result'],
            'execute_paper_trade': ['trade_result'],
            'execute_live_trade': ['trade_result'],
            'monitor_performance': ['monitoring'],
            'optimize_parameters': ['optimization_result'],
            'generate_report': ['report']
        }
        
        return output_map.get(function_name, [])
    
    def _should_continue_on_failure(self, def_step: DefinitionStep) -> bool:
        """
        Determine if workflow should continue if this step fails.
        
        Args:
            def_step: The definition step
            
        Returns:
            True if workflow should continue on failure
        """
        # Steps that are optional or monitoring-related should not stop the workflow
        optional_slices = ['monitor', 'optimize', 'report']
        if def_step.slice in optional_slices:
            return True
        
        # Check for explicit conditions
        for condition in def_step.conditions:
            if 'optional' in condition and condition['optional']:
                return True
        
        # Default: stop on failure for critical steps
        return False
    
    def adapt_for_engine(self, workflow_name: str) -> Dict[str, List[EngineStep]]:
        """
        Adapt all registered workflows for the engine.
        
        Args:
            workflow_name: Name of the workflow to adapt (or 'all' for all workflows)
            
        Returns:
            Dictionary mapping workflow names to lists of EngineSteps
        """
        from .definitions import WorkflowDefinitions
        
        adapted_workflows = {}
        
        if workflow_name == 'all':
            # Adapt all workflows, but expose a curated core set for engine
            core_names = {
                'simple_backtest',
                'paper_trading',
                'multi_strategy_ensemble',
                'risk_managed_live_trading',
                'strategy_optimization',
                'quick_test'
            }
            for name in WorkflowDefinitions.list_workflows():
                if name not in core_names:
                    continue
                workflow_def = WorkflowDefinitions.get_workflow(name)
                if workflow_def:
                    adapted_workflows[name] = self.convert_workflow(workflow_def)
                    self.logger.info(f"Adapted workflow '{name}' with {len(adapted_workflows[name])} steps")
        else:
            # Adapt specific workflow
            workflow_def = WorkflowDefinitions.get_workflow(workflow_name)
            if workflow_def:
                adapted_workflows[workflow_name] = self.convert_workflow(workflow_def)
                self.logger.info(f"Adapted workflow '{workflow_name}' with {len(adapted_workflows[workflow_name])} steps")
            else:
                self.logger.error(f"Workflow '{workflow_name}' not found")
        
        return adapted_workflows


def create_workflow_adapter() -> WorkflowAdapter:
    """
    Factory function to create a workflow adapter instance.
    
    Returns:
        Configured WorkflowAdapter instance
    """
    return WorkflowAdapter()
