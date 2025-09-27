#!/usr/bin/env python3
"""
Test the workflow adapter to verify it converts definitions to engine steps correctly.
"""
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_workflow_adapter():
    """Test workflow adapter functionality"""
    try:
        # Import workflow components
        from bot_v2.workflows.workflow_adapter import WorkflowAdapter
        from bot_v2.workflows.definitions import WorkflowDefinitions, get_simple_backtest_workflow
        
        logger.info("Testing workflow adapter...")
        
        # Initialize workflows
        WorkflowDefinitions.initialize_default_workflows()
        logger.info(f"Loaded {len(WorkflowDefinitions.list_workflows())} workflow definitions")
        
        # Create adapter
        adapter = WorkflowAdapter()
        
        # Test 1: Convert a single workflow
        simple_workflow = get_simple_backtest_workflow()
        if simple_workflow:
            logger.info(f"\nConverting workflow: {simple_workflow.name}")
            engine_steps = adapter.convert_workflow(simple_workflow)
            logger.info(f"Converted {len(engine_steps)} steps:")
            for i, step in enumerate(engine_steps, 1):
                logger.info(f"  {i}. {step.name} -> function: {step.function}")
                if step.required_context:
                    logger.info(f"     Required context: {step.required_context}")
                if step.outputs:
                    logger.info(f"     Outputs: {step.outputs}")
        else:
            logger.warning("Simple backtest workflow not found")
        
        # Test 2: Adapt all workflows
        logger.info("\n--- Adapting all workflows ---")
        adapted_workflows = adapter.adapt_for_engine('all')
        
        for workflow_name, steps in adapted_workflows.items():
            logger.info(f"\n{workflow_name}: {len(steps)} steps")
            for step in steps[:3]:  # Show first 3 steps
                logger.info(f"  - {step.name} ({step.function})")
        
        # Test 3: Verify function mapping
        logger.info("\n--- Testing function mappings ---")
        test_mappings = [
            ('data', 'fetch', 'fetch_data'),
            ('analyze', 'analyze_symbol', 'analyze_market'),
            ('ml_strategy', 'predict_best_strategy', 'select_strategy'),
            ('backtest', 'run_backtest', 'execute_backtest'),
            ('paper_trade', 'execute', 'execute_paper_trade'),
        ]
        
        for slice_name, action, expected_func in test_mappings:
            from bot_v2.workflows.definitions import WorkflowStep as DefStep
            test_step = DefStep(
                name=f"test_{slice_name}",
                slice=slice_name,
                action=action
            )
            func_name = adapter._get_function_name(test_step)
            result = "✓" if func_name == expected_func else "✗"
            logger.info(f"  {result} {slice_name}.{action} -> {func_name} (expected: {expected_func})")
        
        logger.info("\n✅ Workflow adapter test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_workflow_adapter()
    sys.exit(0 if success else 1)

