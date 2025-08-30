#!/usr/bin/env python3
"""
Run Test Suite and Generate Report
===================================
Runs all available tests and generates a comprehensive report.
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime


def run_tests():
    """Run pytest and collect results."""
    
    # Test directories to run
    test_paths = [
        "tests/unit/backtest",
        "tests/unit/strategy", 
        "tests/unit/indicators",
        "tests/unit/risk",
        "tests/unit/dataflow",
        "tests/integration/test_basic_fixtures.py",
        "tests/integration/test_orchestrator.py",
        "tests/unit/test_unified_config.py",
        "tests/unit/test_sizing_price.py",
    ]
    
    results = {}
    total_passed = 0
    total_failed = 0
    total_errors = 0
    
    for test_path in test_paths:
        path = Path(test_path)
        if not path.exists():
            continue
            
        print(f"\n{'='*60}")
        print(f"Running: {test_path}")
        print('='*60)
        
        cmd = [
            "poetry", "run", "pytest", str(test_path),
            "-v", "--tb=short", "--json-report", 
            "--json-report-file=/tmp/pytest_report.json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse results
        try:
            with open("/tmp/pytest_report.json", "r") as f:
                report = json.load(f)
                
            summary = report.get("summary", {})
            passed = summary.get("passed", 0)
            failed = summary.get("failed", 0) 
            errors = summary.get("error", 0)
            
            total_passed += passed
            total_failed += failed
            total_errors += errors
            
            results[test_path] = {
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "total": summary.get("total", 0)
            }
            
            print(f"✅ Passed: {passed}")
            print(f"❌ Failed: {failed}")
            print(f"⚠️  Errors: {errors}")
            
        except (FileNotFoundError, json.JSONDecodeError):
            # Fallback to parsing output
            output = result.stdout + result.stderr
            
            if "passed" in output:
                # Try to parse from output
                import re
                match = re.search(r"(\d+) passed", output)
                if match:
                    passed = int(match.group(1))
                    total_passed += passed
                    results[test_path] = {"passed": passed, "failed": 0, "errors": 0}
                    print(f"✅ Passed: {passed}")
    
    return results, total_passed, total_failed, total_errors


def generate_report(results, total_passed, total_failed, total_errors):
    """Generate test report."""
    
    print("\n" + "="*60)
    print("📊 TEST SUITE RESULTS")
    print("="*60)
    
    for test_path, result in results.items():
        if result.get("total", 0) > 0:
            pass_rate = (result["passed"] / result["total"]) * 100
            print(f"\n{test_path}:")
            print(f"  Pass Rate: {pass_rate:.1f}%")
            print(f"  Details: {result['passed']} passed, {result['failed']} failed")
    
    total = total_passed + total_failed
    if total > 0:
        overall_pass_rate = (total_passed / total) * 100
    else:
        overall_pass_rate = 0
    
    print("\n" + "="*60)
    print("📈 OVERALL SUMMARY")
    print("="*60)
    print(f"Total Tests Run: {total}")
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"⚠️  Errors: {total_errors}")
    print(f"📊 Pass Rate: {overall_pass_rate:.1f}%")
    
    # Save to file
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_errors": total_errors,
        "pass_rate": overall_pass_rate,
        "details": results
    }
    
    report_path = Path(".claude_state/test_report.json")
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n💾 Report saved to: {report_path}")
    
    return overall_pass_rate


def main():
    print("="*60)
    print("🧪 RUNNING TEST SUITE")
    print("="*60)
    
    # Check if pytest-json-report is installed
    subprocess.run(["pip", "install", "pytest-json-report"], 
                  capture_output=True, text=True)
    
    results, total_passed, total_failed, total_errors = run_tests()
    pass_rate = generate_report(results, total_passed, total_failed, total_errors)
    
    # Determine status
    if pass_rate >= 80:
        print("\n✅ FOUNDATION IS SOLID! (≥80% pass rate)")
    elif pass_rate >= 60:
        print("\n⚠️ FOUNDATION IS FUNCTIONAL (60-79% pass rate)")
    else:
        print("\n❌ FOUNDATION NEEDS WORK (<60% pass rate)")
    
    return 0 if pass_rate >= 60 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())