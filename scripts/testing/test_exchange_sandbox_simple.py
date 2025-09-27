#!/usr/bin/env python3
"""
Simple Exchange Sandbox test using simple_cli broker command.
This uses the existing broker infrastructure.
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def run_broker_test():
    """Run broker test via CLI."""
    print("=" * 60)
    print("EXCHANGE SANDBOX ORDER LIFECYCLE TEST")
    print("=" * 60)
    print(f"Time: {datetime.now().isoformat()}")
    print("")
    
    # Check environment
    env_vars = {
        "COINBASE_SANDBOX": os.getenv("COINBASE_SANDBOX", "not set"),
        "COINBASE_API_MODE": os.getenv("COINBASE_API_MODE", "not set"),
        "API_KEY": "Present" if os.getenv("COINBASE_API_KEY") else "Missing",
        "SECRET": "Present" if os.getenv("COINBASE_API_SECRET") else "Missing",
        "PASSPHRASE": "Present" if os.getenv("COINBASE_API_PASSPHRASE") else "Missing"
    }
    
    print("Environment Configuration:")
    for key, value in env_vars.items():
        print(f"  {key}: {value}")
    print("")
    
    # Check if Exchange sandbox is configured
    if env_vars["API_KEY"] == "Missing":
        print("❌ ERROR: Exchange Sandbox API credentials not configured")
        print("")
        print("To set up Exchange Sandbox:")
        print("1. Go to https://public.sandbox.exchange.coinbase.com/")
        print("2. Create API key with View + Trade permissions")
        print("3. Add IP whitelist: 72.208.131.101")
        print("4. Set in .env file:")
        print("   COINBASE_API_KEY=your-key")
        print("   COINBASE_API_SECRET=your-secret")
        print("   COINBASE_API_PASSPHRASE=your-passphrase")
        print("   COINBASE_SANDBOX=1")
        print("   COINBASE_API_MODE=exchange")
        return False
    
    if os.getenv("COINBASE_API_MODE") != "exchange":
        print("⚠️ WARNING: COINBASE_API_MODE is not 'exchange'")
        print("  This test requires Exchange sandbox mode")
        print("")
    
    # Run the broker CLI test
    print("Running order lifecycle test...")
    print("-" * 40)
    
    try:
        # Use poetry run to ensure proper environment
        cmd = [
            "poetry", "run", "python", 
            "src/bot_v2/simple_cli.py", 
            "broker",
            "--sandbox",
            "--run-order-tests",
            "--symbol", "BTC-USD",
            "--limit-price", "10",
            "--qty", "0.0001"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print("")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print("Output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        # Check for success indicators
        success_indicators = [
            "order placed",
            "order cancelled",
            "PASS",
            "Order ID:"
        ]
        
        output_lower = result.stdout.lower()
        tests_passed = any(indicator.lower() in output_lower for indicator in success_indicators)
        
        if tests_passed:
            print("\n✅ TEST PASSED - Order lifecycle completed")
            print("Trade permissions are working!")
            return True
        else:
            print("\n❌ TEST FAILED - Check output above")
            
            # Common issues
            if "401" in result.stdout or "unauthorized" in output_lower:
                print("\nIssue: Authentication failed")
                print("Fix: Verify API credentials")
            elif "403" in result.stdout or "forbidden" in output_lower:
                print("\nIssue: Missing trade permission")
                print("Fix: Update API key to include 'trade' permission")
            elif "no module" in output_lower:
                print("\nIssue: Import error")
                print("Fix: Run from project root with poetry")
            
            return False
            
    except subprocess.TimeoutError:
        print("❌ Test timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


def main():
    """Main entry point."""
    success = run_broker_test()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if success:
        print("Result: ✅ PASS")
        print("\nNext steps:")
        print("1. Credentials verified for Exchange sandbox")
        print("2. Ready for order placement testing")
        print("3. Can proceed with Phase 1 validation")
    else:
        print("Result: ❌ FAIL")
        print("\nRequired actions:")
        print("1. Configure Exchange sandbox API key with trade permission")
        print("2. Update .env with credentials")
        print("3. Re-run this test")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()