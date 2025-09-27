"""
Workflow Validators - Ensure workflow steps can execute successfully
"""
from typing import Dict, Any, List
import logging


class WorkflowValidator:
    """
    Validates workflow steps before execution to ensure prerequisites are met.
    """
    
    def __init__(self):
        """Initialize validator"""
        self.logger = logging.getLogger("workflow.validator")
    
    def validate_step(self, step, context) -> Dict[str, Any]:
        """
        Validate that a step can execute with current context.
        
        Args:
            step: WorkflowStep to validate
            context: Current WorkflowContext
            
        Returns:
            Validation result with 'valid' boolean and 'reason' if invalid
        """
        # Check required context keys exist
        if step.required_context:
            missing_keys = []
            for key in step.required_context:
                if not context.has(key):
                    missing_keys.append(key)
            
            if missing_keys:
                return {
                    'valid': False,
                    'reason': f"Missing required context keys: {missing_keys}"
                }
        
        # Validate specific step types
        if step.function == 'fetch_data':
            return self._validate_fetch_data(context)
        
        elif step.function == 'analyze_market':
            return self._validate_analyze_market(context)
        
        elif step.function == 'detect_regime':
            return self._validate_detect_regime(context)
        
        elif step.function == 'select_strategy':
            return self._validate_select_strategy(context)
        
        elif step.function == 'calculate_position':
            return self._validate_calculate_position(context)
        
        elif step.function == 'execute_backtest':
            return self._validate_execute_backtest(context)
        
        elif step.function == 'execute_paper_trade':
            return self._validate_execute_paper_trade(context)
        
        elif step.function == 'optimize_parameters':
            return self._validate_optimize_parameters(context)
        
        # Default validation passed
        return {'valid': True}
    
    def _validate_fetch_data(self, context) -> Dict[str, Any]:
        """Validate data fetching prerequisites"""
        # Need either symbol or symbols
        if not context.has('symbol') and not context.has('symbols'):
            return {
                'valid': False,
                'reason': "Neither 'symbol' nor 'symbols' found in context"
            }
        
        return {'valid': True}
    
    def _validate_analyze_market(self, context) -> Dict[str, Any]:
        """Validate market analysis prerequisites"""
        if not context.has('market_data'):
            return {
                'valid': False,
                'reason': "Market data not available for analysis"
            }
        
        data = context.get('market_data')
        if data is None or (hasattr(data, 'empty') and data.empty):
            return {
                'valid': False,
                'reason': "Market data is empty"
            }
        
        return {'valid': True}
    
    def _validate_detect_regime(self, context) -> Dict[str, Any]:
        """Validate regime detection prerequisites"""
        if not context.has('market_data'):
            return {
                'valid': False,
                'reason': "Market data required for regime detection"
            }
        
        return {'valid': True}
    
    def _validate_select_strategy(self, context) -> Dict[str, Any]:
        """Validate strategy selection prerequisites"""
        if not context.has('symbol'):
            return {
                'valid': False,
                'reason': "Symbol required for strategy selection"
            }
        
        return {'valid': True}
    
    def _validate_calculate_position(self, context) -> Dict[str, Any]:
        """Validate position calculation prerequisites"""
        if not context.has('confidence'):
            self.logger.warning("No confidence score, will use default")
        
        if not context.has('capital'):
            return {
                'valid': False,
                'reason': "Capital amount required for position sizing"
            }
        
        capital = context.get('capital')
        if capital <= 0:
            return {
                'valid': False,
                'reason': f"Invalid capital amount: {capital}"
            }
        
        return {'valid': True}
    
    def _validate_execute_backtest(self, context) -> Dict[str, Any]:
        """Validate backtest execution prerequisites"""
        if not context.has('symbol'):
            return {
                'valid': False,
                'reason': "Symbol required for backtesting"
            }
        
        if not context.has('strategy'):
            self.logger.warning("No strategy specified, will use default")
        
        if not context.has('market_data'):
            return {
                'valid': False,
                'reason': "Market data required for backtesting"
            }
        
        return {'valid': True}
    
    def _validate_execute_paper_trade(self, context) -> Dict[str, Any]:
        """Validate paper trade execution prerequisites"""
        if not context.has('symbol'):
            return {
                'valid': False,
                'reason': "Symbol required for paper trading"
            }
        
        if not context.has('position_value') and not context.has('position_size'):
            return {
                'valid': False,
                'reason': "Either position_value or position_size required"
            }
        
        return {'valid': True}
    
    def _validate_optimize_parameters(self, context) -> Dict[str, Any]:
        """Validate parameter optimization prerequisites"""
        if not context.has('symbol'):
            return {
                'valid': False,
                'reason': "Symbol required for optimization"
            }
        
        if not context.has('strategy'):
            self.logger.warning("No strategy specified, will use default")
        
        return {'valid': True}
    
    def validate_workflow(self, workflow: List, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate entire workflow before execution.
        
        Args:
            workflow: List of WorkflowSteps
            initial_context: Initial context data
            
        Returns:
            Validation result with details
        """
        issues = []
        warnings = []
        
        # Track what outputs will be available
        available_outputs = set(initial_context.keys())
        
        for i, step in enumerate(workflow):
            # Check if required inputs will be available
            if step.required_context:
                missing = [key for key in step.required_context if key not in available_outputs]
                if missing:
                    issues.append({
                        'step': i + 1,
                        'name': step.name,
                        'issue': f"Missing required inputs: {missing}"
                    })
            
            # Add step outputs to available
            if step.outputs:
                available_outputs.update(step.outputs)
            
            # Check for common issues
            if step.retry_count > 5:
                warnings.append({
                    'step': i + 1,
                    'name': step.name,
                    'warning': f"High retry count: {step.retry_count}"
                })
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'total_steps': len(workflow)
        }