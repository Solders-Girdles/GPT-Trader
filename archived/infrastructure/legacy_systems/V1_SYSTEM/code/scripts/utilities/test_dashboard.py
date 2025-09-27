#!/usr/bin/env python3
"""
Test script for dashboard functionality
"""

import sys
from pathlib import Path

def test_dashboard_files():
    """Test that dashboard files exist and are accessible"""
    
    files_to_check = [
        "src/bot/dashboard/realtime_dashboard.py",
        "src/bot/dashboard/data_fetcher.py", 
        "src/bot/dashboard/config.py",
        "src/bot/dashboard/app.py",
        "src/bot/dashboard/performance_dashboard.py"
    ]
    
    print("🔍 Testing dashboard files...")
    
    all_good = True
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_good = False
    
    return all_good

def test_basic_imports():
    """Test basic Python imports without dependencies"""
    
    print("\n🐍 Testing basic Python functionality...")
    
    try:
        import json
        import datetime
        import threading
        import time
        print("✅ Standard library imports work")
    except ImportError as e:
        print(f"❌ Standard library import failed: {e}")
        return False
    
    return True

def test_dashboard_structure():
    """Test dashboard file structure"""
    
    print("\n📁 Testing dashboard structure...")
    
    # Check if key files have expected content
    realtime_path = Path("src/bot/dashboard/realtime_dashboard.py")
    if realtime_path.exists():
        content = realtime_path.read_text()
        if "RealTimeDataProvider" in content and "render_system_overview" in content:
            print("✅ Real-time dashboard has expected content")
        else:
            print("❌ Real-time dashboard missing expected content")
            return False
    
    data_fetcher_path = Path("src/bot/dashboard/data_fetcher.py")
    if data_fetcher_path.exists():
        content = data_fetcher_path.read_text()
        if "RealTimeDataFetcher" in content and "get_system_health" in content:
            print("✅ Data fetcher has expected content")
        else:
            print("❌ Data fetcher missing expected content")
            return False
    
    return True

def test_launcher():
    """Test the dashboard launcher"""
    
    print("\n🚀 Testing dashboard launcher...")
    
    launcher_path = Path("launch_dashboard.py")
    if launcher_path.exists():
        content = launcher_path.read_text()
        if "launch_dashboard" in content and "realtime" in content:
            print("✅ Dashboard launcher has expected content")
        else:
            print("❌ Dashboard launcher missing expected content")
            return False
    else:
        print("❌ Dashboard launcher not found")
        return False
    
    return True

def main():
    """Run all tests"""
    
    print("🧪 GPT-Trader Dashboard Test Suite")
    print("=" * 40)
    
    tests = [
        ("Dashboard Files", test_dashboard_files),
        ("Basic Imports", test_basic_imports),
        ("Dashboard Structure", test_dashboard_structure),
        ("Launcher", test_launcher),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 40)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! Dashboard should be ready to launch.")
        print("\nNext steps:")
        print("1. Install dependencies: poetry install")
        print("2. Launch dashboard: python launch_dashboard.py --realtime")
        print("3. Or launch standard: python launch_dashboard.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())