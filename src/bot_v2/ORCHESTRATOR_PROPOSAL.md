# 🎯 Minimal Viable Orchestrator Proposal

> ARCHIVED NOTE: This document is historical and not part of the
> active perps trading path. See README section "What's Active Today".

**Objective**: Connect the 11 isolated slices into a working trading system  
**Timeline**: 2 days  
**Priority**: CRITICAL - Blocking all other work

## Day 1: Core Structure (4-6 hours)

### 1. Create `src/bot_v2/orchestrator.py`
```python
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
import logging

# Import all slices
from .features.data import DataProvider
from .features.analyze import MarketAnalyzer
from .features.market_regime import RegimeDetector
from .features.ml_strategy import StrategySelector
from .features.position_sizing import PositionSizer
from .features.backtest import Backtester
from .features.paper_trade import PaperTrader
from .features.live_trade import LiveTrader
from .features.monitor import SystemMonitor
from .features.optimize import Optimizer
from .features.adaptive_portfolio import AdaptiveManager

class TradingMode(Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"
    OPTIMIZE = "optimize"

@dataclass
class OrchestratorConfig:
    mode: TradingMode = TradingMode.BACKTEST
    symbols: List[str] = None
    strategies: List[str] = None
    capital: float = 10000.0
    risk_limits: Dict[str, float] = None

class TradingOrchestrator:
    """Minimal orchestrator to connect all slices"""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize slices
        self.data = DataProvider()
        self.analyzer = MarketAnalyzer()
        self.regime_detector = RegimeDetector()
        self.strategy_selector = StrategySelector()
        self.position_sizer = PositionSizer()
        self.monitor = SystemMonitor()
        
        # Mode-specific slices
        if config.mode == TradingMode.BACKTEST:
            self.executor = Backtester()
        elif config.mode == TradingMode.PAPER:
            self.executor = PaperTrader()
        elif config.mode == TradingMode.LIVE:
            self.executor = LiveTrader()
        else:
            self.executor = Optimizer()
    
    def run_trading_cycle(self, symbol: str) -> Dict[str, Any]:
        """Run one complete trading cycle for a symbol"""
        
        # 1. Fetch data
        market_data = self.data.get_data(symbol)
        
        # 2. Analyze market
        analysis = self.analyzer.analyze(market_data)
        regime = self.regime_detector.detect(market_data)
        
        # 3. Select strategy
        strategy = self.strategy_selector.select(
            symbol=symbol,
            regime=regime,
            analysis=analysis
        )
        
        # 4. Calculate position size
        position = self.position_sizer.calculate(
            confidence=strategy.confidence,
            regime=regime,
            capital=self.config.capital
        )
        
        # 5. Execute trade
        result = self.executor.execute(
            symbol=symbol,
            strategy=strategy,
            position=position
        )
        
        # 6. Monitor and log
        self.monitor.log_trade(result)
        
        return result
    
    def run(self) -> Dict[str, Any]:
        """Main orchestration loop"""
        results = []
        
        for symbol in self.config.symbols:
            try:
                result = self.run_trading_cycle(symbol)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
        
        return {
            'mode': self.config.mode.value,
            'symbols': self.config.symbols,
            'results': results,
            'summary': self._summarize_results(results)
        }
    
    def _summarize_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Summarize trading results"""
        # Basic summary logic
        return {
            'total_trades': len(results),
            'successful': sum(1 for r in results if r.get('success')),
            'failed': sum(1 for r in results if not r.get('success'))
        }
```

### 2. Create `src/bot_v2/__main__.py`
```python
import argparse
import json
from .orchestrator import TradingOrchestrator, OrchestratorConfig, TradingMode

def main():
    parser = argparse.ArgumentParser(description='GPT-Trader Bot V2')
    parser.add_argument('--mode', 
                       choices=['backtest', 'paper', 'live', 'optimize'],
                       default='backtest',
                       help='Trading mode')
    parser.add_argument('--symbols', 
                       nargs='+', 
                       default=['AAPL', 'MSFT'],
                       help='Symbols to trade')
    parser.add_argument('--capital', 
                       type=float, 
                       default=10000.0,
                       help='Starting capital')
    
    args = parser.parse_args()
    
    # Create configuration
    config = OrchestratorConfig(
        mode=TradingMode(args.mode),
        symbols=args.symbols,
        capital=args.capital
    )
    
    # Run orchestrator
    orchestrator = TradingOrchestrator(config)
    results = orchestrator.run()
    
    # Output results
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
```

### 3. Create Basic Demo
```python
# src/bot_v2/demo.py
from .orchestrator import TradingOrchestrator, OrchestratorConfig, TradingMode

def demo_backtest():
    """Demo backtest workflow"""
    config = OrchestratorConfig(
        mode=TradingMode.BACKTEST,
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        capital=10000.0
    )
    
    orchestrator = TradingOrchestrator(config)
    results = orchestrator.run()
    
    print("Backtest Results:")
    print(f"  Trades: {results['summary']['total_trades']}")
    print(f"  Success: {results['summary']['successful']}")

if __name__ == '__main__':
    demo_backtest()
```

## Day 2: Integration & Testing (4-6 hours)

### 1. Fix Import Issues
- Update each slice's `__init__.py` to expose clean interfaces
- Ensure all slices can be imported by orchestrator
- Handle any circular dependency issues

### 2. Create Adapters Where Needed
```python
# src/bot_v2/features/adapters.py
class SliceAdapter:
    """Adapt slice interfaces for orchestrator"""
    
    @staticmethod
    def adapt_strategy_result(result):
        """Convert strategy result to standard format"""
        return {
            'strategy': result.get('name'),
            'confidence': result.get('confidence', 0.5),
            'signal': result.get('signal', 'HOLD')
        }
```

### 3. Add Configuration
```python
# src/bot_v2/config.yaml
orchestrator:
  mode: backtest
  symbols:
    - AAPL
    - MSFT
  capital: 10000
  risk_limits:
    max_position: 0.2  # 20% max per position
    max_loss_daily: 0.02  # 2% max daily loss
```

### 4. Create Integration Test
```python
# tests/integration/bot_v2/test_orchestrator.py
def test_orchestrator_backtest():
    """Test orchestrator runs backtest"""
    config = OrchestratorConfig(
        mode=TradingMode.BACKTEST,
        symbols=['AAPL']
    )
    
    orchestrator = TradingOrchestrator(config)
    results = orchestrator.run()
    
    assert results is not None
    assert 'results' in results
    assert len(results['results']) > 0

def test_orchestrator_paper_trade():
    """Test orchestrator runs paper trading"""
    # Similar test for paper trading
    pass
```

### 5. Update Docker
```dockerfile
# Update Dockerfile CMD
CMD ["python", "-m", "bot_v2"]
```

## Expected Outcomes

### After Day 1:
- ✅ Basic orchestrator structure exists
- ✅ Can import and initialize all slices
- ✅ Simple demo runs (may have errors)
- ✅ Main entry point created

### After Day 2:
- ✅ Full integration working
- ✅ Can run complete trading cycle
- ✅ All modes (backtest, paper, live) supported
- ✅ Docker uses orchestrator as entry point
- ✅ Integration tests passing

## Why This is Critical

### Without Orchestrator:
```bash
# Current state - manual, disconnected
python -c "from bot_v2.features.data import get_data; print(get_data('AAPL'))"
python -c "from bot_v2.features.ml_strategy import select; print(select())"
# ... manually call each slice
```

### With Orchestrator:
```bash
# Desired state - automated, connected
python -m bot_v2 --mode backtest --symbols AAPL MSFT
# Runs complete trading workflow automatically
```

## Risk Assessment

### If We Don't Build This:
- **HIGH RISK**: Sprint 3 work (Docker, safety) applies to disconnected system
- **HIGH RISK**: Can't test end-to-end workflows
- **HIGH RISK**: Paper trading (EPIC-004) will be impossible
- **HIGH RISK**: No way to validate system actually works

### If We Build This:
- **LOW RISK**: 2-day delay in Sprint 3
- **HIGH VALUE**: All future work becomes meaningful
- **HIGH VALUE**: Can actually run and test the system
- **HIGH VALUE**: Docker deployment has entry point

## Recommendation: BUILD NOW

The orchestrator is the **keystone** that makes all other work valuable. Without it, we're adding production features to components that can't work together.

---

**Decision Required**: Pause Sprint 3 for 2 days to build orchestrator?
