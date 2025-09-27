# 🎯 EPIC-002.5 Sprint 1: Detailed Delegation Plan

**Critical Context**: We have 11 working but DISCONNECTED slices. No orchestration exists. Agents must BUILD from scratch, not look for existing integration.

## 📋 Pre-Delegation Checklist

### System State Verification
```bash
# What exists (✅)
src/bot_v2/features/          # 11 isolated slices
├── adaptive_portfolio/       # Works alone
├── analyze/                  # Works alone
├── backtest/                # Works alone
├── data/                    # Works alone
├── live_trade/              # Works alone (has circuit breakers)
├── market_regime/           # Works alone
├── ml_strategy/             # Works alone
├── monitor/                 # Works alone (has logging)
├── optimize/                # Works alone
├── paper_trade/             # Works alone
└── position_sizing/         # Works alone

# What DOESN'T exist (❌)
src/bot_v2/orchestration/     # DOESN'T EXIST - must create
src/bot_v2/workflows/         # DOESN'T EXIST - must create
src/bot_v2/__main__.py        # DOESN'T EXIST - must create
src/bot_v2/demo.py           # DOESN'T EXIST - must create
```

## 🎯 Day 1 Morning: Core Orchestrator

### Agent: ml-strategy-director
### Task: Create Core Orchestration Framework
### Duration: 4 hours

#### Context to Provide:
```markdown
CRITICAL CONTEXT:
- You are CREATING orchestration from scratch - it doesn't exist
- 11 slices in src/bot_v2/features/ work independently 
- NO unified pipeline exists (Sprint 2 work was never done)
- NO existing orchestration code to modify
- Slices have NEVER been connected before

WHAT EXISTS:
- src/bot_v2/features/data/data.py - Has get_data_provider() method
- src/bot_v2/features/analyze/analyze.py - Has analyze_market() function
- src/bot_v2/features/market_regime/market_regime.py - Has detect_regime()
- src/bot_v2/features/ml_strategy/ml_strategy.py - Has predict_best_strategy()
- src/bot_v2/features/position_sizing/position_sizing.py - Has calculate_position_size()
- src/bot_v2/features/backtest/backtest.py - Has run_backtest()
- src/bot_v2/features/monitor/logger.py - Has StructuredLogger (just added)
- src/bot_v2/features/live_trade/circuit_breakers.py - Has safety systems (just added)

WHAT TO CREATE:
src/bot_v2/orchestration/
├── __init__.py
├── orchestrator.py      # Main orchestrator class
├── types.py            # Type definitions
└── config.py           # Configuration management
```

#### Specific Instructions:
```python
# MUST CREATE src/bot_v2/orchestration/orchestrator.py

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import logging
from enum import Enum

# Import ALL slices - handle import errors gracefully
try:
    from ..features.data import get_data_provider
except ImportError as e:
    print(f"Warning: Could not import data: {e}")
    get_data_provider = None

# ... import all other slices similarly ...

class TradingMode(Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

@dataclass
class OrchestratorConfig:
    mode: TradingMode = TradingMode.BACKTEST
    symbols: List[str] = None
    capital: float = 10000.0
    
class TradingOrchestrator:
    def __init__(self, config: OrchestratorConfig):
        # Initialize ALL 11 slices
        # MUST handle missing imports
        
    def execute_trading_cycle(self, symbol: str):
        """Connect the slices in sequence:
        1. data.get_data()
        2. analyze.analyze_market()
        3. market_regime.detect_regime()
        4. ml_strategy.predict_best_strategy()
        5. position_sizing.calculate_position_size()
        6. Execute based on mode (backtest/paper/live)
        7. monitor.log_event()
        """
```

#### Success Criteria:
- [ ] Creates NEW orchestration/ directory
- [ ] Imports all 11 slices (with error handling)
- [ ] execute_trading_cycle() connects slices in sequence
- [ ] Handles missing/broken slices gracefully
- [ ] Basic logging shows flow

#### Common Pitfalls to Avoid:
- ❌ Don't look for existing orchestration (doesn't exist)
- ❌ Don't assume unified_pipeline.py exists (it doesn't)
- ❌ Don't assume slices are already integrated (they aren't)
- ❌ Don't modify slice internals (use their existing methods)

---

## 🎯 Day 1 Afternoon: Slice Registry

### Agent: data-pipeline-engineer  
### Task: Create Slice Registry and Adapters
### Duration: 4 hours

#### Context to Provide:
```markdown
CRITICAL CONTEXT:
- ml-strategy-director just created basic orchestrator
- Slices have INCONSISTENT interfaces
- Need to standardize without modifying slice code
- Create adapter pattern for incompatible methods

INTERFACE ISSUES TO SOLVE:
- data/data.py: Has get_data_provider() but needs get_data(symbol)
- analyze/analyze.py: Might expect different data format
- Some slices return dicts, others return custom types
- Error handling inconsistent across slices

WHAT TO CREATE:
src/bot_v2/orchestration/
├── registry.py         # NEW - Slice registry
└── adapters.py        # NEW - Interface adapters
```

#### Specific Instructions:
```python
# CREATE src/bot_v2/orchestration/registry.py

class SliceRegistry:
    def __init__(self):
        self.slices = {}
        self._discover_slices()
    
    def _discover_slices(self):
        """Dynamically load all slices from features/"""
        import importlib
        import os
        
        features_path = "src/bot_v2/features"
        for slice_name in os.listdir(features_path):
            if os.path.isdir(os.path.join(features_path, slice_name)):
                try:
                    module = importlib.import_module(f"bot_v2.features.{slice_name}")
                    self.slices[slice_name] = module
                except ImportError as e:
                    print(f"Could not load {slice_name}: {e}")

# CREATE src/bot_v2/orchestration/adapters.py

class DataAdapter:
    """Adapt data slice to standard interface"""
    def __init__(self, data_module):
        self.module = data_module
        
    def get_data(self, symbol: str):
        # Adapt to standard interface
        provider = self.module.get_data_provider()
        return provider.get_historical_data(symbol)

class AnalyzeAdapter:
    """Adapt analyze slice to standard interface"""
    # ... similar pattern ...
```

#### Success Criteria:
- [ ] SliceRegistry discovers all 11 slices
- [ ] Adapters created for incompatible interfaces  
- [ ] Orchestrator updated to use registry
- [ ] All slices callable through standard interface
- [ ] Graceful handling of missing slices

#### Common Pitfalls to Avoid:
- ❌ Don't modify the original slice files
- ❌ Don't assume all slices have same methods
- ❌ Don't break if a slice is missing
- ❌ Don't hardcode slice paths

---

## 🎯 Day 2: Workflow Engine

### Agent: trading-ops-lead
### Task: Create Workflow Execution Engine
### Duration: 6 hours

#### Context to Provide:
```markdown
CRITICAL CONTEXT:
- Orchestrator now exists and can call slices
- SliceRegistry provides unified access
- Need to define SEQUENCES of operations
- Different modes need different workflows

WORKFLOW EXAMPLES NEEDED:
1. simple_backtest: data → analyze → regime → strategy → position → backtest
2. paper_trading: real-time loop with all slices
3. optimization: multiple iterations with parameter tuning

WHAT TO CREATE:
src/bot_v2/workflows/
├── __init__.py
├── engine.py          # Workflow execution engine
├── definitions.py     # Workflow definitions
└── context.py        # Shared context between steps
```

#### Specific Instructions:
```python
# CREATE src/bot_v2/workflows/engine.py

from typing import Dict, List, Any
import time

class WorkflowStep:
    def __init__(self, slice_name: str, method_name: str, params: Dict = None):
        self.slice_name = slice_name
        self.method_name = method_name
        self.params = params or {}

class WorkflowEngine:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.workflows = {}
        self._load_default_workflows()
    
    def execute_workflow(self, workflow_name: str, context: Dict) -> Dict:
        """Execute named workflow with context"""
        workflow = self.workflows.get(workflow_name)
        if not workflow:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        results = {}
        for step in workflow:
            # Get slice from orchestrator
            # Call method with context
            # Store result in context
            # Handle errors
        
        return results

# CREATE src/bot_v2/workflows/definitions.py

SIMPLE_BACKTEST = [
    WorkflowStep("data", "get_data", {"period": "1y"}),
    WorkflowStep("analyze", "analyze_market"),
    WorkflowStep("market_regime", "detect_regime"),
    WorkflowStep("ml_strategy", "predict_best_strategy"),
    WorkflowStep("position_sizing", "calculate_position_size"),
    WorkflowStep("backtest", "run_backtest"),
    WorkflowStep("monitor", "log_results")
]
```

#### Success Criteria:
- [ ] WorkflowEngine executes steps sequentially
- [ ] Context passes between steps
- [ ] Error handling doesn't break flow
- [ ] At least 3 workflows defined
- [ ] Timing metrics collected

#### Common Pitfalls to Avoid:
- ❌ Don't assume all steps will succeed
- ❌ Don't lose context between steps
- ❌ Don't hardcode workflow logic
- ❌ Don't ignore performance metrics

---

## 🎯 Day 3 Morning: CLI Entry Point

### Agent: deployment-engineer
### Task: Create Main Entry Point and CLI
### Duration: 3 hours

#### Context to Provide:
```markdown
CRITICAL CONTEXT:
- Orchestrator works but needs user interface
- Workflows defined but need to be callable
- Docker needs entry point to run
- Currently NO WAY to run the system

WHAT TO CREATE:
src/bot_v2/
├── __main__.py       # Main entry point
├── cli.py           # CLI interface
└── demo.py          # Demo script

DOCKER UPDATE NEEDED:
- Dockerfile CMD should use new entry point
- Should be able to run: docker run gpt-trader
```

#### Specific Instructions:
```python
# CREATE src/bot_v2/__main__.py

import sys
import argparse
import json
from .orchestration.orchestrator import TradingOrchestrator, OrchestratorConfig, TradingMode
from .workflows.engine import WorkflowEngine

def main():
    parser = argparse.ArgumentParser(
        description='GPT-Trader Bot V2 - Unified Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python -m bot_v2 --mode backtest --symbols AAPL MSFT
  python -m bot_v2 --mode paper --workflow paper_trading
  python -m bot_v2 --mode optimize --symbols AAPL --iterations 100
        '''
    )
    
    parser.add_argument('--mode', 
                       choices=['backtest', 'paper', 'live', 'optimize'],
                       default='backtest',
                       help='Trading mode')
    parser.add_argument('--symbols',
                       nargs='+',
                       default=['AAPL'],
                       help='Symbols to trade')
    parser.add_argument('--workflow',
                       default='simple_backtest',
                       help='Workflow to execute')
    parser.add_argument('--config',
                       help='Config file path')
    parser.add_argument('--output',
                       choices=['json', 'table', 'quiet'],
                       default='table',
                       help='Output format')
    parser.add_argument('--verbose',
                       action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Create config
    config = OrchestratorConfig(
        mode=TradingMode(args.mode),
        symbols=args.symbols
    )
    
    # Run orchestrator
    print(f"Starting GPT-Trader Bot V2 in {args.mode} mode...")
    orchestrator = TradingOrchestrator(config)
    engine = WorkflowEngine(orchestrator)
    
    # Execute workflow
    result = engine.execute_workflow(args.workflow, {"symbols": args.symbols})
    
    # Format output
    if args.output == 'json':
        print(json.dumps(result, indent=2))
    elif args.output == 'table':
        print_table(result)
    
    return 0 if result.get('success') else 1

if __name__ == '__main__':
    sys.exit(main())
```

#### Success Criteria:
- [ ] Can run: `python -m bot_v2`
- [ ] Help text is clear and useful
- [ ] All modes work (or show clear errors)
- [ ] Output formatting works
- [ ] Docker CMD updated

#### Common Pitfalls to Avoid:
- ❌ Don't assume config files exist
- ❌ Don't crash on missing arguments
- ❌ Don't output raw Python objects
- ❌ Don't forget error codes

---

## 🎯 Day 3 Afternoon: Integration Testing

### Agent: test-runner
### Task: Create Comprehensive Integration Tests
### Duration: 3 hours

#### Context to Provide:
```markdown
CRITICAL CONTEXT:
- First time all slices work together
- Many integration issues expected
- Need to verify end-to-end flow
- Docker must be tested

TEST PRIORITIES:
1. Orchestrator loads all slices
2. Basic workflow executes
3. CLI commands work
4. Docker container runs
5. Error handling works

WHAT TO CREATE:
tests/integration/bot_v2/
├── test_orchestrator.py
├── test_workflows.py
├── test_cli.py
└── test_end_to_end.py
```

#### Specific Instructions:
```python
# CREATE tests/integration/bot_v2/test_orchestrator.py

import pytest
from bot_v2.orchestration.orchestrator import TradingOrchestrator, OrchestratorConfig

def test_orchestrator_initialization():
    """Test orchestrator can initialize all slices"""
    config = OrchestratorConfig()
    orchestrator = TradingOrchestrator(config)
    
    # Verify all 11 slices loaded
    assert orchestrator.data is not None
    assert orchestrator.analyzer is not None
    # ... check all slices ...

def test_trading_cycle_execution():
    """Test complete trading cycle"""
    config = OrchestratorConfig(symbols=['AAPL'])
    orchestrator = TradingOrchestrator(config)
    
    result = orchestrator.execute_trading_cycle('AAPL')
    assert result is not None
    assert 'error' not in result

def test_missing_slice_handling():
    """Test graceful handling of missing slices"""
    # Simulate missing slice
    # Should not crash
    pass

def test_docker_integration():
    """Test Docker container runs"""
    import subprocess
    result = subprocess.run(
        ['docker', 'run', 'gpt-trader-bot', '--mode', 'backtest'],
        capture_output=True,
        timeout=30
    )
    assert result.returncode == 0
```

#### Success Criteria:
- [ ] All slices load successfully
- [ ] Basic workflow completes
- [ ] CLI commands tested
- [ ] Docker container tested
- [ ] Error scenarios handled

#### Common Pitfalls to Avoid:
- ❌ Don't skip error scenarios
- ❌ Don't assume data exists
- ❌ Don't ignore performance
- ❌ Don't forget Docker test

---

## 📊 Delegation Execution Plan

### Timeline:
```
Day 1 Morning (9am-1pm):
  → ml-strategy-director: Core Orchestrator

Day 1 Afternoon (2pm-6pm):
  → data-pipeline-engineer: Slice Registry

Day 2 Morning (9am-3pm):
  → trading-ops-lead: Workflow Engine

Day 3 Morning (9am-12pm):
  → deployment-engineer: CLI Entry Point

Day 3 Afternoon (1pm-4pm):
  → test-runner: Integration Testing

Day 3 Evening (4pm-6pm):
  → Review and debugging
```

### Parallel Opportunities:
- While ml-strategy-director builds orchestrator, deployment-engineer can plan CLI
- While trading-ops-lead builds workflows, test-runner can prepare test framework

## 🎯 End Goal

By end of Day 3, we should be able to run:

```bash
# Simple backtest
$ python -m bot_v2 --mode backtest --symbols AAPL
Loading slices... ✓
Executing workflow: simple_backtest
[09:45:23] Fetching data for AAPL...
[09:45:24] Analyzing market patterns...
[09:45:24] Detecting market regime: BULL_QUIET
[09:45:25] Selecting strategy: momentum
[09:45:25] Calculating position size: 500 shares
[09:45:26] Running backtest...
[09:45:28] Backtest complete!

Results:
  Total Return: 15.3%
  Sharpe Ratio: 1.85
  Max Drawdown: -5.2%
  Trades: 42

# Docker
$ docker run gpt-trader-bot --mode paper
Starting GPT-Trader Bot V2...
System health check... ✓
Loading configuration... ✓
Connecting to market data... ✓
Paper trading started.
```

## ⚠️ Critical Reminders for All Agents

1. **BUILD, don't SEARCH** - The orchestration doesn't exist, create it
2. **CONNECT, don't MODIFY** - Use slices as they are, connect them
3. **HANDLE ERRORS** - Many things will fail, handle gracefully
4. **TEST EVERYTHING** - This is first integration, expect issues
5. **DOCUMENT CLEARLY** - Others need to understand your work

## 📝 Communication Protocol

Each agent should:
1. Start with "Beginning [task name]..."
2. Report issues immediately
3. Document assumptions made
4. Test their component works
5. End with "Completed [task name]. Result: [summary]"

---

**Ready to begin delegations with this detailed plan!**