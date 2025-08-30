#!/usr/bin/env python3
"""
Quick Verification Suite for Claude Code Agents
================================================
Run this to instantly verify system health.
"""

import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, Tuple, List
import json

# Add meta_workflow to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.meta_workflow import STATE, quick_verify, get_state


class SystemVerifier:
    """Quick verification of all critical components"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def verify_imports(self) -> Dict[str, bool]:
        """Verify all critical imports"""
        print("🔍 Verifying imports...")
        
        imports = {
            "orchestrator": "from bot.integration.orchestrator import IntegratedOrchestrator",
            "paper_trading": "from bot.paper_trading import PaperTradingEngine",
            "strategies": "from bot.strategy.demo_ma import DemoMAStrategy",
            "risk": "from bot.risk.integration import RiskIntegration",
            "allocator": "from bot.portfolio.allocator import PortfolioAllocator",
            "data_pipeline": "from bot.dataflow.pipeline import DataPipeline",
            "config": "from bot.config import get_config",
            "cli": "from bot.cli.cli import main",
        }
        
        results = {}
        for name, import_cmd in imports.items():
            try:
                exec(import_cmd)
                results[name] = True
                print(f"  ✅ {name}")
            except Exception as e:
                results[name] = False
                print(f"  ❌ {name}: {str(e)[:50]}")
        
        return results
    
    def verify_tests(self) -> Tuple[int, int]:
        """Quick test collection check"""
        print("\n🧪 Verifying tests...")
        
        try:
            result = subprocess.run(
                ["poetry", "run", "pytest", "--collect-only", "-q"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Parse output for test count
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if "passed" in line or "collected" in line:
                    # Extract numbers
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        collected = int(numbers[0])
                        print(f"  📊 {collected} tests collected")
                        return collected, 0
            
            print("  ⚠️ Could not determine test count")
            return 0, 0
            
        except subprocess.TimeoutExpired:
            print("  ❌ Test collection timed out")
            return 0, 1
        except Exception as e:
            print(f"  ❌ Test verification failed: {e}")
            return 0, 1
    
    def verify_structure(self) -> Dict[str, bool]:
        """Verify expected directory structure"""
        print("\n📁 Verifying structure...")
        
        expected_dirs = [
            "src/bot",
            "src/bot/strategy",
            "src/bot/integration", 
            "src/bot/risk",
            "src/bot/portfolio",
            "src/bot/paper_trading",
            "src/bot/meta_workflow",
            "tests/unit",
            "tests/integration",
            "scripts/meta_workflow",
            "docs",
            "demos",
        ]
        
        results = {}
        for dir_path in expected_dirs:
            path = Path(dir_path)
            exists = path.exists() and path.is_dir()
            results[dir_path] = exists
            
            if exists:
                print(f"  ✅ {dir_path}")
            else:
                print(f"  ❌ {dir_path} missing")
        
        return results
    
    def verify_config(self) -> bool:
        """Verify configuration loads"""
        print("\n⚙️ Verifying configuration...")
        
        try:
            from bot.config import get_config
            config = get_config()
            print(f"  ✅ Config loaded: {config.app_name}")
            return True
        except Exception as e:
            print(f"  ❌ Config failed: {e}")
            return False
    
    def verify_state(self) -> Dict[str, any]:
        """Verify meta-workflow state"""
        print("\n💾 Verifying state management...")
        
        try:
            state = get_state()
            info = {
                "session_id": state.session_state.get("session_id"),
                "verified_components": state.verified_state.get("verified_count", 0),
                "active_tasks": len(state.session_state.get("active_tasks", [])),
                "completed_tasks": len(state.session_state.get("completed_tasks", [])),
            }
            
            print(f"  ✅ Session: {info['session_id']}")
            print(f"  📊 Verified components: {info['verified_components']}")
            print(f"  📋 Active tasks: {info['active_tasks']}")
            print(f"  ✅ Completed tasks: {info['completed_tasks']}")
            
            return info
        except Exception as e:
            print(f"  ❌ State verification failed: {e}")
            return {}
    
    def verify_critical_files(self) -> Dict[str, bool]:
        """Verify critical files exist"""
        print("\n📄 Verifying critical files...")
        
        critical_files = [
            "pyproject.toml",
            "poetry.lock",
            "README.md",
            "CLAUDE.md",
            "PROJECT_STRUCTURE.md",
            "src/bot/__init__.py",
            "src/bot/config/config.yaml",
        ]
        
        results = {}
        for file_path in critical_files:
            path = Path(file_path)
            exists = path.exists() and path.is_file()
            results[file_path] = exists
            
            if exists:
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ {file_path} missing")
        
        return results
    
    def run_quick_smoke_test(self) -> bool:
        """Run a minimal smoke test"""
        print("\n🔥 Running smoke test...")
        
        try:
            # Test basic import and execution
            code = """
from bot.strategy.demo_ma import DemoMAStrategy
from bot.dataflow.pipeline import DataPipeline
import pandas as pd
from datetime import datetime, timedelta

# Create strategy
strategy = DemoMAStrategy(fast=10, slow=30)

# Create minimal data
dates = pd.date_range(end=datetime.now(), periods=50)
data = pd.DataFrame({
    'close': [100 + i for i in range(50)],
    'volume': [1000000] * 50
}, index=dates)

# Generate signals
signals = strategy.generate_signals(data)
print(f"Signals generated: {len(signals)}")
"""
            
            result = subprocess.run(
                ["poetry", "run", "python", "-c", code],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and "Signals generated" in result.stdout:
                print(f"  ✅ Smoke test passed")
                return True
            else:
                print(f"  ❌ Smoke test failed")
                if result.stderr:
                    print(f"     Error: {result.stderr[:100]}")
                return False
                
        except Exception as e:
            print(f"  ❌ Smoke test error: {e}")
            return False
    
    def generate_report(self) -> Dict:
        """Generate comprehensive verification report"""
        
        # Run all verifications
        import_results = self.verify_imports()
        test_count, test_errors = self.verify_tests()
        structure_results = self.verify_structure()
        config_ok = self.verify_config()
        state_info = self.verify_state()
        files_results = self.verify_critical_files()
        smoke_ok = self.run_quick_smoke_test()
        
        # Calculate totals
        imports_ok = sum(1 for v in import_results.values() if v)
        structure_ok = sum(1 for v in structure_results.values() if v)
        files_ok = sum(1 for v in files_results.values() if v)
        
        elapsed = time.time() - self.start_time
        
        # Generate report
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed, 2),
            "summary": {
                "imports": f"{imports_ok}/{len(import_results)}",
                "tests_collected": test_count,
                "structure": f"{structure_ok}/{len(structure_results)}",
                "files": f"{files_ok}/{len(files_results)}",
                "config": config_ok,
                "smoke_test": smoke_ok,
            },
            "state": state_info,
            "details": {
                "imports": import_results,
                "structure": structure_results,
                "files": files_results,
            },
            "health_score": self._calculate_health_score(
                imports_ok, len(import_results),
                structure_ok, len(structure_results),
                files_ok, len(files_results),
                config_ok, smoke_ok, test_count
            )
        }
        
        return report
    
    def _calculate_health_score(self, imports_ok, imports_total, 
                               structure_ok, structure_total,
                               files_ok, files_total,
                               config_ok, smoke_ok, test_count) -> float:
        """Calculate overall system health score"""
        
        scores = []
        
        # Import health (30% weight)
        scores.append((imports_ok / imports_total) * 0.3 if imports_total > 0 else 0)
        
        # Structure health (20% weight)
        scores.append((structure_ok / structure_total) * 0.2 if structure_total > 0 else 0)
        
        # Files health (20% weight)
        scores.append((files_ok / files_total) * 0.2 if files_total > 0 else 0)
        
        # Config health (10% weight)
        scores.append(0.1 if config_ok else 0)
        
        # Smoke test (10% weight)
        scores.append(0.1 if smoke_ok else 0)
        
        # Test collection (10% weight)
        scores.append(0.1 if test_count > 0 else 0)
        
        return round(sum(scores) * 100, 1)


def main():
    """Run comprehensive system verification"""
    
    print("="*60)
    print("🚀 SYSTEM VERIFICATION SUITE")
    print("="*60)
    
    verifier = SystemVerifier()
    report = verifier.generate_report()
    
    # Display summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    print(f"\n⏱️ Completed in: {report['elapsed_seconds']}s")
    print(f"\n📈 System Health Score: {report['health_score']}%")
    
    print("\n📋 Results:")
    for key, value in report['summary'].items():
        status = "✅" if value not in [False, 0, "0/0"] else "❌"
        print(f"  {status} {key}: {value}")
    
    # Save report
    report_file = Path(".claude_state") / "last_verification.json"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved to: {report_file}")
    
    # Return exit code based on health
    if report['health_score'] >= 70:
        print("\n✅ System verification PASSED")
        return 0
    else:
        print("\n❌ System verification FAILED")
        print("   Run 'poetry install' and check error messages above")
        return 1


if __name__ == "__main__":
    sys.exit(main())