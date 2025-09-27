#!/usr/bin/env python3
"""
Enhanced preflight checklist with proper CDP key detection and AT clarification.
"""

import os
import sys
import time
import json
import subprocess
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def check_environment():
    """Enhanced environment check with proper CDP detection."""
    print("\n🔧 Environment Check:")
    all_good = True
    
    # Check if we need to source the environment file
    env_file = "set_env.demo.sh" if os.getenv('COINBASE_SANDBOX') == '1' else "set_env.prod.sh"
    if os.path.exists(env_file) and not os.getenv('COINBASE_CDP_API_KEY'):
        print(f"  ⚠️  Environment not loaded. Run: source {env_file}")
        all_good = False
    
    # Check environment keys
    keys_status = {}
    required_keys = [
        'COINBASE_CDP_API_KEY',
        'COINBASE_CDP_PRIVATE_KEY',
        'COINBASE_API_MODE'
    ]
    
    for key in required_keys:
        value = os.getenv(key)
        if value:
            # Mask sensitive values
            if key == 'COINBASE_CDP_API_KEY':
                # Show key name prefix for CDP keys
                if value.startswith('organizations/'):
                    # Extract and show org/key pattern
                    parts = value.split('/')
                    if len(parts) >= 4:
                        masked = f"{parts[0]}/{parts[1][:8]}.../{parts[2]}/{parts[3][:8]}..."
                    else:
                        masked = value[:20] + '...' if len(value) > 20 else '***'
                else:
                    masked = value[:8] + '...' if len(value) > 8 else '***'
                keys_status[key] = f"✅ Set ({masked})"
            elif 'PRIVATE_KEY' in key:
                # Check if it's a file path
                if value.startswith('/') or value.startswith('.'):
                    if os.path.exists(value):
                        # Check file permissions
                        stat = os.stat(value)
                        mode = oct(stat.st_mode)[-3:]
                        if mode not in ['400', '600']:
                            keys_status[key] = f"⚠️  File exists but loose permissions: {mode}"
                            all_good = False
                        else:
                            keys_status[key] = f"✅ File exists: {value}"
                    else:
                        keys_status[key] = f"❌ File not found: {value}"
                        all_good = False
                else:
                    # It's inline PEM data
                    if 'BEGIN EC PRIVATE KEY' in value:
                        keys_status[key] = "✅ PEM data (inline)"
                    else:
                        keys_status[key] = "✅ Set (masked)"
            else:
                keys_status[key] = f"✅ Set: {value}"
        else:
            keys_status[key] = "❌ Not set"
            all_good = False
    
    # Display results
    for key, status in keys_status.items():
        print(f"  {status}")
    
    # Check API Mode and clarify endpoints
    api_mode = os.getenv('COINBASE_API_MODE', 'legacy')
    is_sandbox = os.getenv('COINBASE_SANDBOX') == '1'
    
    if api_mode == 'advanced':
        print("  ✅ API Mode: advanced (CDP)")
        if is_sandbox:
            print("  ℹ️  Note: Advanced Trade uses production endpoints (no public sandbox)")
            print("     Using tiny positions for safety in demo mode")
    else:
        print(f"  ⚠️  API Mode: {api_mode} (expected 'advanced' for CDP)")
        all_good = False
    
    # Check derivatives enabled
    derivatives = os.getenv('COINBASE_ENABLE_DERIVATIVES')
    if derivatives == '1':
        print("  ✅ Derivatives: Enabled")
    else:
        print("  ❌ Derivatives: Not enabled")
        all_good = False
    
    return all_good


def check_dependencies():
    """Check required Python packages."""
    print("\n📦 Dependency Check:")
    
    required = [
        'websockets',
        'cryptography',
        'aiohttp'
    ]
    
    # python-dotenv is optional if not using .env files
    optional = ['python-dotenv']
    
    all_good = True
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} not installed")
            print(f"     Run: python -m pip install {package}")
            all_good = False
    
    # Check optional packages
    for package in optional:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package} (optional)")
        except ImportError:
            if not os.path.exists('.env'):
                print(f"  ℹ️  {package} not needed (no .env file)")
            else:
                print(f"  ⚠️  {package} not installed but .env exists")
                print(f"     Run: python -m pip install {package}")
    
    return all_good


def check_connectivity():
    """Check network connectivity."""
    print("\n🌐 Connectivity Check:")
    
    is_sandbox = os.getenv('COINBASE_SANDBOX') == '1'
    api_mode = os.getenv('COINBASE_API_MODE', 'legacy')
    
    # Clarify endpoints based on mode
    if api_mode == 'advanced':
        # Advanced Trade always uses production endpoints
        endpoints = {
            'api': 'api.coinbase.com',
            'websocket': 'advanced-trade-ws.coinbase.com'
        }
        print(f"  Mode: Advanced Trade (production endpoints)")
    else:
        # Legacy/Exchange API
        if is_sandbox:
            endpoints = {
                'api': 'api-public.sandbox.exchange.coinbase.com',
                'websocket': 'ws-direct.sandbox.exchange.coinbase.com'
            }
        else:
            endpoints = {
                'api': 'api.exchange.coinbase.com',
                'websocket': 'ws-direct.exchange.coinbase.com'
            }
        print(f"  Mode: Exchange API ({'sandbox' if is_sandbox else 'production'})")
    
    all_good = True
    for name, host in endpoints.items():
        try:
            # Use ping to check connectivity
            result = subprocess.run(
                ['ping', '-c', '1', '-W', '2', host],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                # Extract latency
                output = result.stdout
                if 'time=' in output:
                    latency = output.split('time=')[1].split()[0]
                    print(f"  ✅ {name}: {host} ({latency})")
                else:
                    print(f"  ✅ {name}: {host} (reachable)")
            else:
                print(f"  ❌ {name}: {host} (unreachable)")
                all_good = False
        except Exception as e:
            print(f"  ❌ {name}: {host} (error: {e})")
            all_good = False
    
    return all_good


def check_safety_params():
    """Check safety parameters."""
    print("\n🛡️  Safety Parameters:")
    
    params = {
        'max_position_size': float(os.getenv('COINBASE_MAX_POSITION_SIZE', '0.01')),
        'daily_loss_limit': float(os.getenv('COINBASE_DAILY_LOSS_LIMIT', '100')),
        'max_impact_bps': int(os.getenv('COINBASE_MAX_IMPACT_BPS', '15')),
        'kill_switch': os.getenv('COINBASE_KILL_SWITCH', 'enabled'),
        'quiet_period_min': int(os.getenv('COINBASE_QUIET_MINUTES', '30'))
    }
    
    print(f"  Position Size: {params['max_position_size']} BTC")
    print(f"  Daily Loss Limit: ${params['daily_loss_limit']}")
    print(f"  Max Impact: {params['max_impact_bps']} bps")
    print(f"  Kill Switch: {params['kill_switch']}")
    print(f"  Quiet Period: {params['quiet_period_min']} min")
    
    warnings = []
    if params['max_position_size'] > 0.05:
        warnings.append("Position size > 0.05 BTC")
    if params['daily_loss_limit'] > 500:
        warnings.append("Daily loss limit > $500")
    if params['max_impact_bps'] > 20:
        warnings.append("Market impact > 20 bps")
    if params['kill_switch'] != 'enabled':
        warnings.append("Kill switch not enabled")
    
    if warnings:
        print("\n  ⚠️  Warnings:")
        for w in warnings:
            print(f"    - {w}")
        return False
    
    return True


def generate_report():
    """Generate consolidated preflight report."""
    print("\n" + "="*60)
    print("📋 PREFLIGHT REPORT")
    print("="*60)
    
    timestamp = datetime.now(timezone.utc).isoformat()
    environment = "demo" if os.getenv('COINBASE_SANDBOX') == '1' else "production"
    api_mode = os.getenv('COINBASE_API_MODE', 'legacy')
    
    report = {
        'timestamp': timestamp,
        'environment': environment,
        'api_mode': api_mode,
        'checks': {}
    }
    
    # Run all checks
    checks = [
        ('environment', check_environment),
        ('dependencies', check_dependencies),
        ('connectivity', check_connectivity),
        ('safety', check_safety_params)
    ]
    
    all_passed = True
    for name, check_func in checks:
        result = check_func()
        report['checks'][name] = 'PASS' if result else 'FAIL'
        if not result:
            all_passed = False
    
    # Summary
    print("\n📊 Summary:")
    for check, result in report['checks'].items():
        symbol = "✅" if result == 'PASS' else "❌"
        print(f"  {symbol} {check}: {result}")
    
    # Overall status
    if all_passed:
        print("\n🟢 PREFLIGHT PASSED - Ready for demo/canary")
        report['status'] = 'PASSED'
    else:
        print("\n🔴 PREFLIGHT FAILED - Fix issues before proceeding")
        report['status'] = 'FAILED'
    
    # Save report
    report_dir = Path("docs/ops/preflight")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = report_dir / f"preflight_{timestamp.replace(':', '-').replace('.', '-')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to: {report_file}")
    
    return all_passed


def main():
    """Run enhanced preflight checks."""
    print("🚀 ENHANCED PREFLIGHT CHECKLIST")
    print("="*60)
    
    # Check if environment is set
    if not os.getenv('COINBASE_CDP_API_KEY'):
        print("\n⚠️  Environment not configured!")
        print("Run one of:")
        print("  source set_env.demo.sh  # For demo/sandbox")
        print("  source set_env.prod.sh  # For production")
        sys.exit(1)
    
    success = generate_report()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()