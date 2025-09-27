#!/usr/bin/env python3
"""
Main entry point for GPT-Trader Bot V2

Usage:
    python -m bot_v2 --mode backtest --symbols BTC-USD ETH-USD
    python -m bot_v2 --mode paper --workflow paper_trading
    python -m bot_v2 --mode optimize --symbols BTC-USD --iterations 100
"""
import sys
import argparse
import json
import logging
import os
from typing import List
from pathlib import Path

# Ensure the repository's `src` directory is on sys.path so
# `import bot_v2` works when invoked via CLI entrypoints or `python -m bot_v2`.
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot_v2.orchestration import TradingOrchestrator, OrchestratorConfig, TradingMode
from bot_v2.logging_setup import configure_logging

# Try to import workflows (might not exist yet)
try:
    from bot_v2.workflows import WorkflowEngine, ALL_WORKFLOWS
    WORKFLOWS_AVAILABLE = True
except ImportError:
    WORKFLOWS_AVAILABLE = False
    ALL_WORKFLOWS = {}

# Configure logging (rotating files + console)
configure_logging()
logger = logging.getLogger(__name__)


def print_table(results: List):
    """Print results in a table format"""
    if not results:
        print("No results to display")
        return
    
    print("\n" + "="*80)
    print("TRADING RESULTS")
    print("="*80)
    
    for result in results:
        print(f"\nSymbol: {result.symbol}")
        print(f"Mode: {result.mode.value}")
        print(f"Success: {'✓' if result.success else '✗'}")
        print(f"Execution Time: {result.metrics.get('execution_time', 0):.2f}s")
        
        if result.data:
            if 'strategy' in result.data:
                print(f"Strategy: {result.data['strategy']}")
            if 'confidence' in result.data:
                print(f"Confidence: {result.data['confidence']:.2%}")
            if 'regime' in result.data:
                print(f"Market Regime: {result.data['regime']}")
            if 'position_size' in result.data:
                print(f"Position Size: {result.data['position_size']:.2%}")
        
        if result.errors:
            print(f"Errors ({len(result.errors)}):")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"  - {error}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='GPT-Trader Bot V2 - Unified Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python -m bot_v2 --mode backtest --symbols BTC-USD ETH-USD
  python -m bot_v2 --mode paper --symbols BTC-USD
  python -m bot_v2 --mode optimize --symbols BTC-USD
  python -m bot_v2 --status
        '''
    )
    
    parser.add_argument('--mode', 
                       choices=['backtest', 'paper', 'live', 'optimize'],
                       default='backtest',
                       help='Trading mode')
    parser.add_argument('--symbols',
                       nargs='+',
                       default=['BTC-USD'],
                       help='Symbols to trade (e.g., BTC-USD)')
    parser.add_argument('--capital',
                       type=float,
                       default=10000.0,
                       help='Starting capital')
    parser.add_argument('--config',
                       help='Config file path (JSON)')
    parser.add_argument('--output',
                       choices=['json', 'table', 'quiet'],
                       default='table',
                       help='Output format')
    parser.add_argument('--verbose',
                       action='store_true',
                       help='Verbose output')
    parser.add_argument('--status',
                       action='store_true',
                       help='Show orchestrator status and exit')
    parser.add_argument('--workflow',
                       help='Execute named workflow')
    parser.add_argument('--list-workflows',
                       action='store_true',
                       help='List available workflows and exit')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    # Allow PERPS_DEBUG=1 to elevate verbosity for selected modules
    if os.getenv('PERPS_DEBUG') == '1':
        logging.getLogger('bot_v2.features.brokerages.coinbase').setLevel(logging.DEBUG)
        logging.getLogger('bot_v2.orchestration').setLevel(logging.DEBUG)
    
    # Load config from file if provided
    config_dict = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
            logger.info(f"Loaded config from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return 1
    
    # Create config
    config = OrchestratorConfig(
        mode=TradingMode(args.mode),
        symbols=args.symbols,
        capital=args.capital,
        **config_dict
    )

    # Spot-first guard: warn on -PERP symbols when derivatives disabled
    try:
        broker_env = os.getenv('BROKER', 'coinbase').lower()
        deriv_enabled = os.getenv('COINBASE_ENABLE_DERIVATIVES', '0') in ('1', 'true', 'yes')
        if broker_env == 'coinbase' and not deriv_enabled:
            perps = [s for s in args.symbols if s.upper().endswith('-PERP')]
            if perps:
                logger.warning(
                    f"Perpetuals disabled (COINBASE_ENABLE_DERIVATIVES=0). "
                    f"Symbols {perps} may fail. Use spot symbols like BTC-USD, or enable derivatives if eligible."
                )
    except Exception:
        pass
    
    # Create orchestrator
    print(f"Starting GPT-Trader Bot V2 in {args.mode} mode...")
    try:
        orchestrator = TradingOrchestrator(config)
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        return 1
    
    # List workflows if requested
    if args.list_workflows:
        if not WORKFLOWS_AVAILABLE:
            print("Workflows module not available")
            return 1
        print("\nAvailable Workflows:")
        print("=" * 50)
        for name, workflow in ALL_WORKFLOWS.items():
            print(f"\n{name}:")
            try:
                # Handle both WorkflowDefinition and adapted engine steps
                if hasattr(workflow, 'steps'):
                    steps = workflow.steps
                else:
                    steps = workflow
                print(f"  Steps: {len(steps)}")
                step_names = [getattr(step, 'name', str(step)) for step in steps]
                print(f"  - {', '.join(step_names[:3])}...")
            except Exception:
                print("  (unable to list steps)")
        return 0
    
    # Show status if requested
    if args.status:
        status = orchestrator.get_status()
        print("\nOrchestrator Status:")
        print(f"  Available Slices: {status['slice_availability']}")
        print(f"  Mode: {status['mode']}")
        print(f"  Capital: ${status['capital']:,.2f}")
        print("\nSlices:")
        for slice, available in status['available_slices'].items():
            print(f"  {slice}: {'✓' if available else '✗'}")
        if status['failed_slices']:
            print("\nFailed Slices:")
            for slice, error in status['failed_slices'].items():
                print(f"  {slice}: {error[:50]}...")
        return 0
    
    # Run workflow if specified
    if args.workflow:
        try:
            engine = WorkflowEngine(orchestrator)
            # Pass both symbol and symbols for compatibility
            context = {
                'symbol': args.symbols[0] if args.symbols else 'BTC-USD',
                'symbols': args.symbols,
                'capital': args.capital
            }
            print(f"\nExecuting workflow: {args.workflow}")
            result = engine.execute_workflow(args.workflow, context)
            
            # Print workflow results
            print("\n" + "="*60)
            print(f"WORKFLOW: {args.workflow}")
            print("="*60)
            print(f"Status: {result['status']}")
            print(f"Steps Executed: {len(result['steps_executed'])}")
            if result['steps_executed']:
                print(f"  ✓ {', '.join(result['steps_executed'])}")
            print(f"Steps Failed: {len(result['steps_failed'])}")
            if result['steps_failed']:
                for failure in result['steps_failed']:
                    print(f"  ✗ {failure['step']}: {failure['error'][:50]}...")
            print(f"Total Time: {result.get('total_time', 0):.2f}s")
            
            return 0 if result['status'] == 'completed' else 1
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return 1
    
    # Run orchestrator (legacy mode)
    try:
        results = orchestrator.run(args.symbols)
    except Exception as e:
        logger.error(f"Orchestration failed: {e}")
        return 1
    
    # Format output
    if args.output == 'json':
        output = {
            'mode': args.mode,
            'symbols': args.symbols,
            'results': [r.to_dict() for r in results]
        }
        print(json.dumps(output, indent=2))
    elif args.output == 'table':
        print_table(results)
    
    # Summary
    if args.output != 'quiet':
        successful = sum(1 for r in results if r.success)
        total = len(results)
        print(f"\nCompleted: {successful}/{total} successful")
    
    # Return appropriate exit code
    return 0 if all(r.success for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
