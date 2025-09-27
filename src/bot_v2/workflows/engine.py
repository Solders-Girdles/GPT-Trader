"""
Workflow Engine - Core execution system for multi-step trading operations
"""
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

from .context import WorkflowContext
from .validators import WorkflowValidator

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """Definition of a single workflow step"""
    name: str
    function: Callable
    description: str
    required_context: List[str] = None
    outputs: List[str] = None
    retry_count: int = 3
    continue_on_failure: bool = False
    
    def __post_init__(self):
        self.required_context = self.required_context or []
        self.outputs = self.outputs or []


class WorkflowEngine:
    """
    Core workflow execution engine that orchestrates multi-step trading operations.
    Designed by trading-ops-lead for EPIC-002.5 Sprint 1 Day 2.
    """
    
    def __init__(self, orchestrator):
        """
        Initialize workflow engine with orchestrator reference.
        
        Args:
            orchestrator: TradingOrchestrator instance for accessing slices
        """
        self.orchestrator = orchestrator
        self.logger = logging.getLogger("workflow.engine")
        self.validator = WorkflowValidator()
        self.workflows = {}
        
        # Initialize workflow adapter (import here to avoid circular import)
        from .workflow_adapter import WorkflowAdapter
        self.adapter = WorkflowAdapter()
        
        # Import and adapt predefined workflows
        self._load_workflows()
        
        self.logger.info(f"Workflow engine initialized with {len(self.workflows)} workflows")
    
    def _load_workflows(self):
        """
        Load and adapt workflow definitions for execution.
        """
        try:
            from .definitions import WorkflowDefinitions
            
            # Initialize default workflows
            WorkflowDefinitions.initialize_default_workflows()
            
            # Adapt all workflows
            adapted_workflows = self.adapter.adapt_for_engine('all')
            self.workflows.update(adapted_workflows)
            
            self.logger.info(f"Loaded {len(self.workflows)} workflows")
            
        except ImportError as e:
            self.logger.error(f"Failed to import workflow definitions: {e}")
            # Try to load legacy workflows if available
            try:
                from .definitions import ALL_WORKFLOWS
                if isinstance(ALL_WORKFLOWS, dict):
                    # If ALL_WORKFLOWS is already in the right format, use it directly
                    self.workflows = ALL_WORKFLOWS
                else:
                    # Otherwise try to adapt
                    for workflow_name, workflow_def in ALL_WORKFLOWS.items():
                        if hasattr(workflow_def, 'steps'):
                            self.workflows[workflow_name] = self.adapter.convert_workflow(workflow_def)
            except Exception as legacy_error:
                self.logger.warning(f"Failed to load legacy workflows: {legacy_error}")
                # Create empty workflows dictionary
                self.workflows = {}
    
    def execute_workflow(self, workflow_name: str, 
                        initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a named workflow with given initial context.
        
        Args:
            workflow_name: Name of the workflow to execute
            initial_context: Initial data for the workflow
            
        Returns:
            Dict containing workflow results and execution details
        """
        if workflow_name not in self.workflows:
            # Try to adapt this workflow on the fly
            try:
                adapted = self.adapter.adapt_for_engine(workflow_name)
                if workflow_name in adapted:
                    self.workflows[workflow_name] = adapted[workflow_name]
                else:
                    return {
                        'status': 'error',
                        'error': f"Workflow '{workflow_name}' not found",
                        'available_workflows': list(self.workflows.keys())
                    }
            except Exception:
                return {
                    'status': 'error',
                    'error': f"Workflow '{workflow_name}' not found",
                    'available_workflows': list(self.workflows.keys())
                }
        
        workflow = self.workflows[workflow_name]
        context = WorkflowContext(initial_data=initial_context or {})
        
        start_time = datetime.now()
        self.logger.info(f"Starting workflow: {workflow_name}")
        
        results = {
            'workflow': workflow_name,
            'status': 'running',
            'steps_executed': [],
            'steps_failed': [],
            'context': {},
            'start_time': start_time.isoformat()
        }
        
        # Execute each step in the workflow
        for step_index, step in enumerate(workflow, 1):
            self.logger.info(f"Executing step {step_index}/{len(workflow)}: {step.name}")
            
            # Validate step prerequisites
            validation = self.validator.validate_step(step, context)
            if not validation['valid']:
                error_msg = f"Validation failed: {validation['reason']}"
                self.logger.error(f"Step {step.name}: {error_msg}")
                
                if not step.continue_on_failure:
                    results['status'] = 'failed'
                    results['steps_failed'].append({
                        'step': step.name,
                        'error': error_msg
                    })
                    break
                else:
                    results['steps_failed'].append({
                        'step': step.name,
                        'error': error_msg,
                        'continued': True
                    })
                    continue
            
            # Execute the step with retries
            step_success = False
            last_error = None
            
            for attempt in range(step.retry_count):
                try:
                    # Execute step function
                    step_result = self._execute_step(step, context)
                    
                    # Update context with outputs
                    for output_key in step.outputs:
                        if output_key in step_result:
                            context.set(output_key, step_result[output_key])
                    
                    results['steps_executed'].append(step.name)
                    step_success = True
                    self.logger.info(f"Step {step.name} completed successfully")
                    break
                    
                except Exception as e:
                    last_error = str(e)
                    self.logger.warning(f"Step {step.name} attempt {attempt + 1} failed: {e}")
                    if attempt < step.retry_count - 1:
                        continue
            
            if not step_success:
                error_entry = {
                    'step': step.name,
                    'error': last_error or "Unknown error"
                }
                
                if step.continue_on_failure:
                    error_entry['continued'] = True
                    results['steps_failed'].append(error_entry)
                    self.logger.warning(f"Step {step.name} failed but continuing workflow")
                else:
                    results['steps_failed'].append(error_entry)
                    results['status'] = 'failed'
                    self.logger.error(f"Step {step.name} failed, stopping workflow")
                    break
        
        # Finalize results
        end_time = datetime.now()
        results['end_time'] = end_time.isoformat()
        results['total_time'] = (end_time - start_time).total_seconds()
        results['context'] = context.get_all()
        
        if results['status'] == 'running':
            results['status'] = 'completed'
        
        self.logger.info(f"Workflow {workflow_name} {results['status']} in {results['total_time']:.2f}s")
        
        return results
    
    def _execute_step(self, step: WorkflowStep, context: WorkflowContext) -> Dict[str, Any]:
        """
        Execute a single workflow step.
        
        Args:
            step: WorkflowStep to execute
            context: Current workflow context
            
        Returns:
            Step execution results
        """
        # Handle both string function names and callable functions
        params = getattr(step, 'meta', {}).get('params', {}) if hasattr(step, 'meta') else {}
        if callable(step.function):
            # If function is callable, execute it directly
            return step.function(self.orchestrator, context)
        
        # Otherwise, map string function names to orchestrator methods
        if step.function == 'fetch_data':
            symbol = context.get('symbol') or context.get('symbols', ['AAPL'])[0]
            return self._fetch_data(symbol, params)
        
        elif step.function == 'analyze_market':
            return self._analyze_market(context, params)
        
        elif step.function == 'detect_regime':
            return self._detect_regime(context, params)
        
        elif step.function == 'select_strategy':
            return self._select_strategy(context, params)
        
        elif step.function == 'calculate_position':
            return self._calculate_position(context, params)
        
        elif step.function == 'execute_backtest':
            return self._execute_backtest(context, params)
        
        elif step.function == 'execute_paper_trade':
            return self._execute_paper_trade(context, params)
        
        elif step.function == 'monitor_performance':
            return self._monitor_performance(context, params)
        
        elif step.function == 'optimize_parameters':
            return self._optimize_parameters(context, params)
        
        elif step.function == 'generate_report':
            return self._generate_report(context, params)
        
        elif step.function == 'execute_live_trade':
            return self._execute_live_trade(context, params)
        
        else:
            raise ValueError(f"Unknown step function: {step.function}")
    
    def _fetch_data(self, symbol: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fetch market data using data provider abstraction"""
        try:
            from ..data_providers import get_data_provider
            provider = get_data_provider()
            p = params or {}
            period = p.get('period', "60d")
            interval = p.get('interval', "1d")
            data = provider.get_historical_data(symbol, period=period, interval=interval)
            return {
                'market_data': data,
                'symbol': symbol,
                'rows': len(data) if data is not None else 0
            }
        except Exception as e:
            self.logger.error(f"Data fetch failed: {e}")
            raise
    
    def _analyze_market(self, context: WorkflowContext, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze market using analyze slice"""
        symbol = context.get('symbol')
        lookback_days = (params or {}).get('lookback_days', 60)
        
        if self.orchestrator.analyzer:
            try:
                # Use the correct API: analyze_symbol(symbol, lookback_days)
                if hasattr(self.orchestrator.analyzer, 'analyze_symbol'):
                    analysis_result = self.orchestrator.analyzer.analyze_symbol(symbol, lookback_days=lookback_days)
                    return {
                        'analysis': {
                            'recommendation': analysis_result.recommendation,
                            'confidence': analysis_result.confidence,
                            'indicators': analysis_result.indicators,
                            'regime': analysis_result.regime
                        }
                    }
                else:
                    # Fallback to old API if needed
                    data = context.get('market_data')
                    analysis = self.orchestrator.analyzer.analyze(data, symbol)
                    return {'analysis': analysis}
            except Exception as e:
                self.logger.error(f"Analysis failed: {e}")
                return {'analysis': {'error': str(e)}}
        
        return {'analysis': {'error': 'Analyzer not available'}}
    
    def _detect_regime(self, context: WorkflowContext, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Detect market regime"""
        if self.orchestrator.market_regime:
            try:
                symbol = context.get('symbol')
                lookback_days = (params or {}).get('lookback_days', 60)
                # Use the correct API: detect_regime(symbol, lookback_days)
                if hasattr(self.orchestrator.market_regime, 'detect_regime'):
                    regime_analysis = self.orchestrator.market_regime.detect_regime(symbol, lookback_days=lookback_days)
                    regime = regime_analysis.current_regime.value if hasattr(regime_analysis.current_regime, 'value') else str(regime_analysis.current_regime)
                    return {
                        'regime': regime,
                        'regime_confidence': regime_analysis.confidence
                    }
                else:
                    # Fallback to old API if needed
                    data = context.get('market_data')
                    result = self.orchestrator.market_regime.detect_regime(data)
                    return {'regime': result.get('regime', 'unknown')}
            except Exception as e:
                self.logger.error(f"Regime detection failed: {e}")
                return {'regime': 'unknown'}
        
        return {'regime': 'unknown'}
    
    def _select_strategy(self, context: WorkflowContext, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Select best strategy using ML"""
        if self.orchestrator.ml_strategy:
            try:
                symbol = context.get('symbol')
                lookback_days = (params or {}).get('lookback_days', 60)
                top_n = (params or {}).get('top_n', 1)
                # Use the correct API: predict_best_strategy(symbol, lookback_days, top_n)
                if hasattr(self.orchestrator.ml_strategy, 'predict_best_strategy'):
                    predictions = self.orchestrator.ml_strategy.predict_best_strategy(symbol, lookback_days=lookback_days, top_n=top_n)
                    if predictions and len(predictions) > 0:
                        best_prediction = predictions[0]
                        strategy = best_prediction.strategy.value if hasattr(best_prediction.strategy, 'value') else str(best_prediction.strategy)
                        return {
                            'strategy': strategy,
                            'confidence': best_prediction.confidence,
                            'expected_return': best_prediction.expected_return
                        }
                else:
                    # Fallback to old API if needed
                    data = context.get('market_data')
                    result = self.orchestrator.ml_strategy.predict_best_strategy(symbol, data)
                    return {
                        'strategy': result.get('strategy', 'momentum'),
                        'confidence': result.get('confidence', 0.5)
                    }
            except Exception as e:
                self.logger.error(f"Strategy selection failed: {e}")
                return {'strategy': 'momentum', 'confidence': 0.5}
        
        return {'strategy': 'momentum', 'confidence': 0.5}
    
    def _calculate_position(self, context: WorkflowContext, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate position size"""
        confidence = context.get('confidence', 0.5)
        capital = context.get('capital', 10000)
        max_pos = (params or {}).get('max_position_size', 0.2)
        
        # Simple position sizing
        position_size = min(0.1 * confidence, max_pos)
        position_value = capital * position_size
        
        return {
            'position_size': position_size,
            'position_value': position_value
        }
    
    def _execute_backtest(self, context: WorkflowContext, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute backtest"""
        if self.orchestrator.backtest:
            try:
                from datetime import datetime, timedelta
                
                symbol = context.get('symbol')
                strategy = context.get('strategy', 'momentum')
                capital = context.get('capital', 10000)
                
                # Calculate date range for backtest
                end_date = datetime.now()
                window_days = (params or {}).get('window_days', 90)
                start_date = end_date - timedelta(days=window_days)  # window backtest
                
                # Use the correct API: run_backtest with proper parameters
                result = self.orchestrator.backtest.run_backtest(
                    strategy=strategy,
                    symbol=symbol,
                    start=start_date,
                    end=end_date,
                    initial_capital=capital
                )
                return {'backtest_result': result}
            except Exception as e:
                self.logger.error(f"Backtest failed: {e}")
                return {'backtest_result': {'error': str(e)}}
        
        return {'backtest_result': {'error': 'Backtest not available'}}
    
    def _execute_paper_trade(self, context: WorkflowContext, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute paper trade"""
        if self.orchestrator.paper_trade:
            try:
                symbol = context.get('symbol')
                position_value = context.get('position_value', 1000)
                strategy = context.get('strategy', 'momentum')
                confidence = context.get('confidence', 0.5)
                
                # Calculate shares from position value
                from ..data_providers import get_data_provider
                provider = get_data_provider()
                current_price = provider.get_current_price(symbol)
                shares = int(position_value / current_price)
                
                # Use the facade function with shares
                result = self.orchestrator.paper_trade.execute_paper_trade(
                    symbol=symbol,
                    action=(params or {}).get('execution_style', 'buy'),
                    quantity=shares,  # Pass shares, not dollars
                    strategy_info={'strategy': strategy, 'confidence': confidence}
                )
                return {'trade_result': result}
            except Exception as e:
                self.logger.error(f"Paper trade failed: {e}")
                return {'trade_result': {'error': str(e)}}
        
        return {'trade_result': {'error': 'Paper trading not available'}}
    
    def _monitor_performance(self, context: WorkflowContext, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Monitor performance"""
        if self.orchestrator.monitor:
            try:
                # Import log_event with correct signature
                from ..features.monitor.logger import log_event
                
                # Use correct signature: log_event(event_type, message, **kwargs)
                event_type = (params or {}).get('event_type', 'workflow_monitoring')
                message = (params or {}).get('message', f"Workflow monitoring for {context.get('symbol')}")
                log_event(
                    event_type,
                    message,
                    workflow=context.get('workflow_name', 'unknown'),
                    symbol=context.get('symbol'),
                    strategy=context.get('strategy'),
                    position_size=context.get('position_size')
                )
                return {'monitoring': 'active'}
            except ImportError:
                # Fallback to monitor module if available
                try:
                    event = {
                        'type': 'workflow_monitoring',
                        'workflow': context.get('workflow_name', 'unknown'),
                        'symbol': context.get('symbol'),
                        'strategy': context.get('strategy'),
                        'position_size': context.get('position_size')
                    }
                    if hasattr(self.orchestrator.monitor, 'log_event'):
                        self.orchestrator.monitor.log_event(event)
                    return {'monitoring': 'active'}
                except Exception as e:
                    self.logger.error(f"Monitoring failed: {e}")
                    return {'monitoring': 'failed'}
            except Exception as e:
                self.logger.error(f"Monitoring failed: {e}")
                return {'monitoring': 'failed'}
        
        return {'monitoring': 'unavailable'}
    
    def _optimize_parameters(self, context: WorkflowContext, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize strategy parameters (prefer optimize_strategy API)"""
        if self.orchestrator.optimizer:
            try:
                from datetime import datetime, timedelta
                symbol = context.get('symbol')
                strategy = context.get('strategy', 'momentum')

                # Parameter ranges (use provided params if any)
                param_ranges = (params or {}).get('parameters', {
                    'lookback': [10, 20, 30],
                    'threshold': [0.01, 0.02, 0.03]
                })

                # Default optimization window
                end = datetime.now()
                start = end - timedelta(days=(params or {}).get('window_days', 180))

                optimizer = self.orchestrator.optimizer
                if hasattr(optimizer, 'optimize_strategy'):
                    result = optimizer.optimize_strategy(
                        strategy=strategy,
                        symbol=symbol,
                        start_date=start,
                        end_date=end,
                        param_grid=param_ranges,
                        metric=(params or {}).get('metric', 'sharpe_ratio')
                    )
                elif hasattr(optimizer, 'optimize'):
                    # Fallback to a generic optimize if provided
                    result = optimizer.optimize(symbol, strategy, param_ranges)
                else:
                    return {'optimization_result': {'error': 'No optimization method available'}}

                return {'optimization_result': result}
            except Exception as e:
                self.logger.error(f"Optimization failed: {e}")
                return {'optimization_result': {'error': str(e)}}

        return {'optimization_result': {'error': 'Optimizer not available'}}
    
    def _generate_report(self, context: WorkflowContext, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate workflow report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'symbol': context.get('symbol'),
            'strategy': context.get('strategy'),
            'confidence': context.get('confidence'),
            'position_size': context.get('position_size'),
            'regime': context.get('regime'),
            'analysis': context.get('analysis', {})
        }
        
        if 'backtest_result' in context.data:
            report['backtest'] = context.get('backtest_result')
        
        if 'trade_result' in context.data:
            report['trade'] = context.get('trade_result')
        
        return {'report': report}
    
    def _execute_live_trade(self, context: WorkflowContext, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute live trade"""
        if self.orchestrator.live_trade:
            try:
                symbol = context.get('symbol')
                position_value = context.get('position_value', 1000)
                strategy = context.get('strategy', 'momentum')
                confidence = context.get('confidence', 0.5)
                
                # Calculate shares from position value
                from ..data_providers import get_data_provider
                provider = get_data_provider()
                current_price = provider.get_current_price(symbol)
                shares = int(position_value / current_price)
                
                # Use the facade function with shares
                result = self.orchestrator.live_trade.execute_live_trade(
                    symbol=symbol,
                    action=(params or {}).get('action', 'buy'),
                    quantity=shares,  # Pass shares, not dollars
                    strategy_info={'strategy': strategy, 'confidence': confidence}
                )
                return {'trade_result': result}
            except Exception as e:
                self.logger.error(f"Live trade failed: {e}")
                return {'trade_result': {'error': str(e)}}
        
        return {'trade_result': {'error': 'Live trading not available'}}
    
    def list_workflows(self) -> List[str]:
        """List all available workflows"""
        return list(self.workflows.keys())
    
    def get_workflow_info(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific workflow"""
        if workflow_name not in self.workflows:
            return None
        
        workflow = self.workflows[workflow_name]
        return {
            'name': workflow_name,
            'steps': len(workflow),
            'step_names': [step.name for step in workflow],
            'description': f"Workflow with {len(workflow)} steps"
        }
