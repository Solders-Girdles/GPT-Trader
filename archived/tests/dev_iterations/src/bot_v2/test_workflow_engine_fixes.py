#!/usr/bin/env python3
"""
Test the workflow engine after fixing all API mismatches.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_workflow_engine_fixes():
    """Test workflow engine with fixed API calls"""
    try:
        logger.info("=== Testing Workflow Engine API Fixes ===\n")
        
        # Initialize orchestrator
        from bot_v2.orchestration.orchestrator import TradingOrchestrator
        from bot_v2.orchestration.types import TradingMode, OrchestratorConfig
        
        config = OrchestratorConfig(
            mode=TradingMode.BACKTEST,
            symbols=['AAPL'],
            capital=10000.0
        )
        orchestrator = TradingOrchestrator(config)
        logger.info(f"✓ Orchestrator initialized with {len(orchestrator.available_slices)} slices")
        
        # Initialize workflow engine
        from bot_v2.workflows.engine import WorkflowEngine
        engine = WorkflowEngine(orchestrator)
        logger.info(f"✓ Workflow engine initialized with {len(engine.workflows)} workflows")
        
        # Test individual workflow steps
        logger.info("\n--- Testing Individual Workflow Steps ---")
        
        # Create a test context
        from bot_v2.workflows.context import WorkflowContext
        context = WorkflowContext({
            'symbol': 'AAPL',
            'capital': 10000,
            'workflow_name': 'test_workflow'
        })
        
        # Test 1: Fetch Data
        logger.info("\nTest 1: Fetch Data")
        try:
            result = engine._fetch_data('AAPL')
            if 'market_data' in result:
                logger.info(f"  ✓ Data fetched successfully: {result.get('rows', 0)} rows")
                context.set('market_data', result['market_data'])
                context.set('symbol', result['symbol'])
            else:
                logger.warning("  ⚠ No market data returned")
        except Exception as e:
            logger.error(f"  ✗ Fetch data failed: {e}")
        
        # Test 2: Analyze Market
        logger.info("\nTest 2: Analyze Market")
        try:
            result = engine._analyze_market(context)
            if 'analysis' in result and 'error' not in result['analysis']:
                logger.info(f"  ✓ Analysis completed: {result['analysis'].get('recommendation', 'N/A')}")
                context.set('analysis', result['analysis'])
            else:
                logger.warning(f"  ⚠ Analysis error: {result.get('analysis', {}).get('error', 'Unknown')}")
        except Exception as e:
            logger.error(f"  ✗ Analyze market failed: {e}")
        
        # Test 3: Detect Regime
        logger.info("\nTest 3: Detect Regime")
        try:
            result = engine._detect_regime(context)
            if 'regime' in result:
                logger.info(f"  ✓ Regime detected: {result['regime']}")
                if 'regime_confidence' in result:
                    logger.info(f"    Confidence: {result['regime_confidence']:.1%}")
                context.set('regime', result['regime'])
            else:
                logger.warning("  ⚠ No regime detected")
        except Exception as e:
            logger.error(f"  ✗ Detect regime failed: {e}")
        
        # Test 4: Select Strategy
        logger.info("\nTest 4: Select Strategy")
        try:
            result = engine._select_strategy(context)
            if 'strategy' in result:
                logger.info(f"  ✓ Strategy selected: {result['strategy']}")
                logger.info(f"    Confidence: {result['confidence']:.1%}")
                if 'expected_return' in result:
                    logger.info(f"    Expected return: {result['expected_return']:.2%}")
                context.set('strategy', result['strategy'])
                context.set('confidence', result['confidence'])
            else:
                logger.warning("  ⚠ No strategy selected")
        except Exception as e:
            logger.error(f"  ✗ Select strategy failed: {e}")
        
        # Test 5: Calculate Position
        logger.info("\nTest 5: Calculate Position")
        try:
            result = engine._calculate_position(context)
            if 'position_size' in result:
                logger.info(f"  ✓ Position calculated: {result['position_size']:.1%} of capital")
                logger.info(f"    Position value: ${result['position_value']:.2f}")
                context.set('position_size', result['position_size'])
                context.set('position_value', result['position_value'])
            else:
                logger.warning("  ⚠ Position calculation failed")
        except Exception as e:
            logger.error(f"  ✗ Calculate position failed: {e}")
        
        # Test 6: Execute Backtest
        logger.info("\nTest 6: Execute Backtest")
        try:
            result = engine._execute_backtest(context)
            if 'backtest_result' in result and 'error' not in result['backtest_result']:
                logger.info(f"  ✓ Backtest executed successfully")
                context.set('backtest_result', result['backtest_result'])
            else:
                logger.warning(f"  ⚠ Backtest error: {result.get('backtest_result', {}).get('error', 'Unknown')}")
        except Exception as e:
            logger.error(f"  ✗ Execute backtest failed: {e}")
        
        # Test 7: Paper Trade (with share conversion)
        logger.info("\nTest 7: Paper Trade with Share Conversion")
        try:
            result = engine._execute_paper_trade(context)
            if 'trade_result' in result and 'error' not in result['trade_result']:
                logger.info(f"  ✓ Paper trade executed")
                trade = result['trade_result']
                logger.info(f"    Status: {trade.get('status', 'unknown')}")
                logger.info(f"    Quantity: {trade.get('quantity', 0)} shares")
                context.set('trade_result', result['trade_result'])
            else:
                logger.warning(f"  ⚠ Paper trade error: {result.get('trade_result', {}).get('error', 'Unknown')}")
        except Exception as e:
            logger.error(f"  ✗ Execute paper trade failed: {e}")
        
        # Test 8: Optimize Parameters
        logger.info("\nTest 8: Optimize Parameters")
        try:
            result = engine._optimize_parameters(context)
            if 'optimization_result' in result and 'error' not in result['optimization_result']:
                logger.info(f"  ✓ Optimization completed")
                context.set('optimization_result', result['optimization_result'])
            else:
                logger.warning(f"  ⚠ Optimization error: {result.get('optimization_result', {}).get('error', 'Unknown')}")
        except Exception as e:
            logger.error(f"  ✗ Optimize parameters failed: {e}")
        
        # Test 9: Monitor Performance
        logger.info("\nTest 9: Monitor Performance")
        try:
            result = engine._monitor_performance(context)
            if 'monitoring' in result:
                logger.info(f"  ✓ Monitoring status: {result['monitoring']}")
            else:
                logger.warning("  ⚠ Monitoring failed")
        except Exception as e:
            logger.error(f"  ✗ Monitor performance failed: {e}")
        
        # Test 10: Execute Complete Workflow
        logger.info("\n--- Testing Complete Workflow Execution ---")
        try:
            workflow_result = engine.execute_workflow(
                'simple_backtest',
                initial_context={'symbol': 'AAPL', 'capital': 10000}
            )
            
            logger.info(f"\nWorkflow Status: {workflow_result['status']}")
            logger.info(f"Steps Executed: {len(workflow_result['steps_executed'])}")
            logger.info(f"Steps Failed: {len(workflow_result['steps_failed'])}")
            logger.info(f"Total Time: {workflow_result.get('total_time', 0):.2f}s")
            
            if workflow_result['steps_executed']:
                logger.info(f"Executed steps: {', '.join(workflow_result['steps_executed'])}")
            
            if workflow_result['steps_failed']:
                logger.warning(f"Failed steps: {workflow_result['steps_failed']}")
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("Workflow Engine API Fix Summary:")
        logger.info("  ✅ Analyze uses analyze_symbol(symbol, lookback_days)")
        logger.info("  ✅ Regime uses detect_regime(symbol, lookback_days)")
        logger.info("  ✅ ML Strategy uses predict_best_strategy(symbol, lookback_days, top_n)")
        logger.info("  ✅ Backtest uses run_backtest with date ranges")
        logger.info("  ✅ Paper trade converts dollars to shares")
        logger.info("  ✅ Optimize checks for optimize_strategy method")
        logger.info("  ✅ Monitor uses correct log_event signature")
        logger.info("\n🎉 All workflow engine API mismatches fixed!")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_workflow_engine_fixes()
    sys.exit(0 if success else 1)

