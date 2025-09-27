#!/usr/bin/env python3
"""
Calculate current test pass rate for GPT-Trader system.
"""

import subprocess
import re

def run_tests(test_path, description):
    """Run tests and get pass/fail counts."""
    try:
        result = subprocess.run(
            ["poetry", "run", "pytest", test_path, "-v", "--tb=no"],
            capture_output=True, text=True, timeout=60
        )
        
        output = result.stdout + result.stderr
        
        # Count passed and failed
        passed = len(re.findall(r' PASSED', output))
        failed = len(re.findall(r' FAILED', output))
        errors = len(re.findall(r' ERROR', output))
        
        # Check for collection errors
        collection_errors = 0
        if "error" in output.lower() and "collecting" in output.lower():
            collection_match = re.search(r'(\d+) errors? during collection', output)
            if collection_match:
                collection_errors = int(collection_match.group(1))
        
        return {
            'description': description,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'collection_errors': collection_errors,
            'total': passed + failed
        }
    except subprocess.TimeoutExpired:
        return {
            'description': description,
            'error': 'Timeout'
        }
    except Exception as e:
        return {
            'description': description,
            'error': str(e)
        }

def main():
    """Calculate overall test pass rate."""
    print("📊 GPT-Trader Test Pass Rate Analysis")
    print("=" * 50)
    
    # Test different suites
    test_suites = [
        ("tests/integration/test_orchestrator.py", "Orchestrator Tests"),
        ("tests/integration/test_basic_fixtures.py", "Basic Fixtures"),
        ("tests/minimal_baseline/", "Minimal Baseline"),
        ("tests/unit/dataflow/", "Dataflow Unit Tests"),
        ("tests/unit/risk/", "Risk Unit Tests"),
        ("tests/unit/integration/", "Integration Unit Tests"),
    ]
    
    total_passed = 0
    total_failed = 0
    total_errors = 0
    
    for test_path, description in test_suites:
        print(f"\n🧪 Testing: {description}")
        print("-" * 30)
        
        result = run_tests(test_path, description)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            if result['collection_errors'] > 0:
                print(f"⚠️  Collection errors: {result['collection_errors']}")
            
            if result['total'] > 0:
                pass_rate = (result['passed'] / result['total']) * 100
                print(f"✅ Passed: {result['passed']}")
                print(f"❌ Failed: {result['failed']}")
                print(f"📈 Pass Rate: {pass_rate:.1f}%")
                
                total_passed += result['passed']
                total_failed += result['failed']
                total_errors += result['errors']
            else:
                print("⚠️  No tests executed")
    
    # Overall summary
    print("\n" + "=" * 50)
    print("📊 OVERALL SUMMARY")
    print("=" * 50)
    
    total_tests = total_passed + total_failed
    if total_tests > 0:
        overall_pass_rate = (total_passed / total_tests) * 100
        print(f"✅ Total Passed: {total_passed}")
        print(f"❌ Total Failed: {total_failed}")
        print(f"⚠️  Total Errors: {total_errors}")
        print(f"📈 Overall Pass Rate: {overall_pass_rate:.1f}%")
        
        # Assessment
        print("\n📋 Assessment:")
        if overall_pass_rate >= 60:
            print("🎉 TARGET ACHIEVED! Pass rate is above 60%")
        else:
            needed = int(total_tests * 0.6) - total_passed
            print(f"📝 Need {needed} more passing tests to reach 60% target")
            print(f"   Current: {overall_pass_rate:.1f}% → Target: 60%")
    else:
        print("⚠️  Unable to calculate overall pass rate")
    
    # Additional metrics
    print("\n📊 Test Collection Status:")
    try:
        result = subprocess.run(
            ["poetry", "run", "pytest", "--co", "-q", "tests/"],
            capture_output=True, text=True, timeout=30
        )
        
        if "collected" in result.stdout:
            collected_match = re.search(r'(\d+) tests? collected', result.stdout)
            if collected_match:
                collected = int(collected_match.group(1))
                print(f"📁 Total tests collected: {collected}")
                if total_tests > 0:
                    execution_rate = (total_tests / collected) * 100
                    print(f"🔧 Execution rate: {execution_rate:.1f}% of collected tests")
    except:
        pass

if __name__ == "__main__":
    main()