#!/usr/bin/env python3
"""
Foundation Assessment: What Actually Works?
===========================================
Run this to get the brutal truth about system functionality.
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_signal_generation() -> Tuple[bool, str]:
    """Test if strategies actually generate signals"""
    try:
        from bot.strategy.demo_ma import DemoMAStrategy
        from bot.strategy.trend_breakout import TrendBreakoutStrategy
        
        # Create sample data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        data = pd.DataFrame({
            'open': [100 + i*0.5 for i in range(100)],
            'high': [101 + i*0.5 for i in range(100)],
            'low': [99 + i*0.5 for i in range(100)],
            'close': [100 + i*0.5 for i in range(100)],
            'volume': [1000000] * 100
        }, index=dates)
        
        results = {}
        
        # Test each strategy
        for name, strategy_class in [
            ("MA", DemoMAStrategy),
            ("Breakout", TrendBreakoutStrategy)
        ]:
            try:
                if name == "MA":
                    strategy = strategy_class(fast=10, slow=30)
                else:
                    strategy = strategy_class()
                
                signals = strategy.generate_signals(data)
                signal_count = len(signals[signals['signal'] != 0])
                results[name] = signal_count
            except Exception as e:
                results[name] = f"Error: {str(e)[:50]}"
        
        # Check results
        working = sum(1 for v in results.values() if isinstance(v, int) and v > 0)
        
        return working > 0, f"Signals: {results}"
        
    except Exception as e:
        return False, f"Failed: {str(e)[:100]}"


def test_signal_to_trade_conversion() -> Tuple[bool, str]:
    """Test if signals convert to trades"""
    try:
        from bot.portfolio import PortfolioAllocator
        from bot.strategy.demo_ma import DemoMAStrategy
        
        # Generate signals
        strategy = DemoMAStrategy(fast=10, slow=30)
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        data = pd.DataFrame({
            'open': [100 + i*0.5 for i in range(100)],
            'high': [101 + i*0.5 for i in range(100)],
            'low': [99 + i*0.5 for i in range(100)],
            'close': [100 + i*0.5 for i in range(100)],
            'volume': [1000000] * 100
        }, index=dates)
        
        signals = strategy.generate_signals(data)
        
        # Enrich signals with price data
        enriched_signals = signals.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                enriched_signals[col] = data[col]
        
        # Try allocation
        allocator = PortfolioAllocator()
        allocations = allocator.allocate_signals(
            signals={'AAPL': enriched_signals},
            market_data={'AAPL': data},
            portfolio_value=100000
        )
        
        # Check if allocations exist (allocations is a dict)
        if allocations is not None and allocations:
            positions = sum(1 for qty in allocations.values() if qty > 0)
            return True, f"Allocations created: {positions} positions"
        else:
            return False, "No allocations generated from signals"
            
    except Exception as e:
        return False, f"Conversion failed: {str(e)[:100]}"


def test_risk_management() -> Tuple[bool, str]:
    """Test if risk management works"""
    try:
        from bot.risk import SimpleRiskManager
        
        risk_mgr = SimpleRiskManager(
            max_position_size=0.1,
            max_portfolio_risk=0.02,
            stop_loss_pct=0.05
        )
        
        # Test position sizing
        position_size = risk_mgr.calculate_position_size(
            signal_strength=1.0,
            volatility=0.02,
            portfolio_value=100000
        )
        
        if position_size > 0 and position_size <= 10000:  # Max 10% position
            return True, f"Risk sizing works: ${position_size:.2f}"
        else:
            return False, f"Risk sizing issue: ${position_size:.2f}"
            
    except Exception as e:
        return False, f"Risk mgmt failed: {str(e)[:100]}"


def test_backtesting() -> Tuple[bool, str]:
    """Test if backtesting produces results"""
    try:
        from bot.integration.orchestrator import IntegratedOrchestrator
        from bot.strategy.demo_ma import DemoMAStrategy
        
        orchestrator = IntegratedOrchestrator()
        strategy = DemoMAStrategy(fast=10, slow=30)
        
        # Run minimal backtest
        results = orchestrator.run_backtest(
            strategy=strategy,
            symbols=['AAPL'],
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            initial_capital=100000,
            strategy_class=DemoMAStrategy,
            strategy_params={'fast': 5, 'slow': 10}
        )
        
        if results and hasattr(results, 'total_return'):
            return True, f"Backtest ran: {results.total_return:.2%} return"
        else:
            return False, "Backtest produced no results"
            
    except Exception as e:
        return False, f"Backtest failed: {str(e)[:100]}"


def test_paper_trading() -> Tuple[bool, str]:
    """Test if paper trading engine works"""
    try:
        from bot.paper_trading import PaperTradingEngine, PaperTradingConfig
        
        config = PaperTradingConfig(initial_capital=100000)
        engine = PaperTradingEngine(config=config)
        
        # Submit test order
        order_id = engine.submit_order(
            symbol='AAPL',
            quantity=100,
            order_type='market',
            side='buy'
        )
        
        # Check if order was submitted successfully
        if order_id:
            return True, f"Paper trading works: Order {order_id} submitted"
        else:
            return False, "Paper trading didn't create position"
            
    except Exception as e:
        return False, f"Paper trading failed: {str(e)[:100]}"


def count_test_status() -> Dict:
    """Count test collection and pass rate"""
    try:
        result = subprocess.run(
            ["poetry", "run", "pytest", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        collected = 0
        errors = 0
        
        for line in result.stdout.split('\n'):
            if 'collected' in line:
                parts = line.split()
                for part in parts:
                    if part.isdigit():
                        collected = int(part)
                        break
            if 'error' in line.lower():
                errors += 1
        
        # Now run actual tests on minimal baseline
        result2 = subprocess.run(
            ["poetry", "run", "pytest", "tests/minimal_baseline/", "-q"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        passed = result2.stdout.count(' passed')
        failed = result2.stdout.count(' failed')
        
        return {
            "collected": collected,
            "collection_errors": errors,
            "baseline_passed": passed,
            "baseline_failed": failed,
            "baseline_pass_rate": (passed / (passed + failed) * 100) if (passed + failed) > 0 else 0
        }
        
    except Exception as e:
        return {"error": str(e)}


def main():
    """Run complete foundation assessment"""
    
    print("="*60)
    print("🔍 FOUNDATION ASSESSMENT: What Actually Works?")
    print("="*60)
    print()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "core_functionality": {},
        "test_metrics": {},
        "trust_score": 0
    }
    
    # Test core functionality
    tests = [
        ("Signal Generation", test_signal_generation),
        ("Signal→Trade Conversion", test_signal_to_trade_conversion),
        ("Risk Management", test_risk_management),
        ("Backtesting", test_backtesting),
        ("Paper Trading", test_paper_trading),
    ]
    
    working_count = 0
    print("📊 Core Functionality Tests:\n")
    
    for name, test_func in tests:
        print(f"Testing {name}...")
        success, message = test_func()
        
        if success:
            print(f"  ✅ {name}: {message}")
            working_count += 1
        else:
            print(f"  ❌ {name}: {message}")
        
        results["core_functionality"][name] = {
            "working": success,
            "details": message
        }
        print()
    
    # Test metrics
    print("📈 Test Suite Metrics:\n")
    test_stats = count_test_status()
    
    if "error" not in test_stats:
        print(f"  Tests Collected: {test_stats['collected']}")
        print(f"  Collection Errors: {test_stats['collection_errors']}")
        print(f"  Baseline Passed: {test_stats['baseline_passed']}")
        print(f"  Baseline Failed: {test_stats['baseline_failed']}")
        print(f"  Baseline Pass Rate: {test_stats['baseline_pass_rate']:.1f}%")
    else:
        print(f"  ❌ Error collecting tests: {test_stats['error']}")
    
    results["test_metrics"] = test_stats
    
    # Calculate trust score
    functionality_score = (working_count / len(tests)) * 50  # 50% weight
    test_score = (test_stats.get('baseline_pass_rate', 0) / 100) * 50  # 50% weight
    trust_score = functionality_score + test_score
    
    results["trust_score"] = round(trust_score, 1)
    
    # Summary
    print("\n" + "="*60)
    print("📊 ASSESSMENT SUMMARY")
    print("="*60)
    print()
    print(f"Core Functions Working: {working_count}/{len(tests)}")
    print(f"Test Pass Rate: {test_stats.get('baseline_pass_rate', 0):.1f}%")
    print(f"Foundation Trust Score: {trust_score:.1f}%")
    print()
    
    # Diagnosis
    if trust_score >= 80:
        print("✅ Foundation is SOLID - ready to build on")
    elif trust_score >= 60:
        print("⚠️ Foundation is SHAKY - needs reinforcement")
    else:
        print("🔴 Foundation is WEAK - major repairs needed")
    
    print("\nKey Issues to Address:")
    for name, data in results["core_functionality"].items():
        if not data["working"]:
            print(f"  - Fix {name}: {data['details']}")
    
    # Save results
    output_file = Path(".claude_state") / "foundation_assessment.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Full report saved to: {output_file}")
    
    return 0 if trust_score >= 60 else 1


if __name__ == "__main__":
    sys.exit(main())