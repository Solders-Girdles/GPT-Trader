#!/usr/bin/env python3
"""
Comprehensive preflight check for Coinbase perpetuals trading system.

This script validates:
1. Environment configuration
2. API credentials and permissions
3. Product discovery
4. Risk management systems
5. Test suite status
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from decimal import Decimal

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class PreflightChecker:
    """Comprehensive system preflight checker."""
    
    def __init__(self):
        self.results = []
        self.warnings = []
        self.errors = []
        self.status = "PASS"
        
    def add_result(self, category: str, item: str, status: str, message: str = ""):
        """Add a check result."""
        self.results.append({
            "category": category,
            "item": item,
            "status": status,
            "message": message
        })
        
        if status == "FAIL":
            self.status = "FAIL"
            self.errors.append(f"{category}/{item}: {message}")
        elif status == "WARN":
            if self.status != "FAIL":
                self.status = "WARN"
            self.warnings.append(f"{category}/{item}: {message}")
    
    def check_environment(self):
        """Check environment configuration."""
        print("\n🔧 Checking Environment Configuration...")
        
        required_vars = [
            "BROKER",
            "COINBASE_API_MODE",
            "COINBASE_ENABLE_DERIVATIVES",
            "COINBASE_SANDBOX",
            "COINBASE_PROD_CDP_API_KEY",
            "COINBASE_PROD_CDP_PRIVATE_KEY",
        ]
        
        optional_vars = [
            "RISK_MAX_LEVERAGE",
            "RISK_DAILY_LOSS_LIMIT",
            "ORDER_PREVIEW_ENABLED",
            "PERPS_ENABLE_STREAMING",
        ]
        
        # Check required variables
        for var in required_vars:
            value = os.environ.get(var)
            if value:
                # Don't print sensitive values
                if "KEY" in var or "SECRET" in var:
                    self.add_result("Environment", var, "PASS", "Set (hidden)")
                else:
                    self.add_result("Environment", var, "PASS", f"Value: {value}")
            else:
                self.add_result("Environment", var, "FAIL", "Not set")
        
        # Check optional variables
        for var in optional_vars:
            value = os.environ.get(var)
            if value:
                self.add_result("Environment", var, "INFO", f"Value: {value}")
        
        # Validate configuration consistency
        api_mode = os.environ.get("COINBASE_API_MODE")
        derivatives = os.environ.get("COINBASE_ENABLE_DERIVATIVES")
        sandbox = os.environ.get("COINBASE_SANDBOX")
        
        if derivatives == "1" and sandbox == "1":
            self.add_result("Environment", "Config Consistency", "FAIL", 
                          "Sandbox does not support derivatives! Set COINBASE_SANDBOX=0")
        elif derivatives == "1" and api_mode == "advanced":
            self.add_result("Environment", "Config Consistency", "PASS", 
                          "Valid perpetuals configuration")
        else:
            self.add_result("Environment", "Config Consistency", "WARN",
                          "Check if configuration supports perpetuals trading")
    
    def check_api_connectivity(self):
        """Check API connectivity and permissions."""
        print("\n🌐 Checking API Connectivity...")
        
        try:
            from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
            from bot_v2.features.brokerages.coinbase.cdp_auth import CDPAuth
            
            # Get credentials from environment
            api_key = os.environ.get("COINBASE_PROD_CDP_API_KEY")
            private_key = os.environ.get("COINBASE_PROD_CDP_PRIVATE_KEY")
            
            if not api_key or not private_key:
                self.add_result("API", "Credentials", "FAIL", "CDP credentials not found in environment")
                return
            
            # Initialize client
            auth = CDPAuth(api_key_name=api_key, private_key_pem=private_key)
            client = CoinbaseClient(
                base_url="https://api.coinbase.com",
                auth=auth,
                api_mode="advanced"
            )
            
            # Test basic connectivity
            try:
                accounts = client.get_accounts()
                if accounts and "accounts" in accounts:
                    self.add_result("API", "Account Access", "PASS", 
                                  f"Found {len(accounts['accounts'])} accounts")
                else:
                    self.add_result("API", "Account Access", "FAIL", "No accounts returned")
            except Exception as e:
                self.add_result("API", "Account Access", "FAIL", str(e))
            
            # Test CFM endpoints (futures)
            try:
                positions = client._request("GET", "/api/v3/brokerage/cfm/positions")
                self.add_result("API", "CFM Access", "PASS", "Futures API accessible")
            except Exception as e:
                if "401" in str(e) or "Unauthorized" in str(e):
                    self.add_result("API", "CFM Access", "FAIL", 
                                  "No futures permissions - enable in Coinbase CDP settings!")
                else:
                    self.add_result("API", "CFM Access", "FAIL", str(e))
            
            # Test product discovery
            try:
                products = client.get_products()
                if products and "products" in products:
                    total = len(products["products"])
                    perps = [p for p in products["products"] if "PERP" in p.get("product_id", "")]
                    
                    if perps and any("BTC-PERP" in p.get("product_id", "") for p in perps):
                        self.add_result("API", "Perpetuals Products", "PASS", 
                                      f"Found {len(perps)} perpetual products")
                    else:
                        self.add_result("API", "Perpetuals Products", "FAIL", 
                                      f"No BTC-PERP found (only {len(perps)} PERP products)")
                else:
                    self.add_result("API", "Perpetuals Products", "FAIL", "No products returned")
            except Exception as e:
                self.add_result("API", "Perpetuals Products", "FAIL", str(e))
                
        except ImportError as e:
            self.add_result("API", "Import", "FAIL", f"Cannot import required modules: {e}")
        except Exception as e:
            self.add_result("API", "General", "FAIL", str(e))
    
    def check_test_suite(self):
        """Check test suite status."""
        print("\n🧪 Checking Test Suite...")
        
        try:
            # Run tests with pytest
            result = subprocess.run(
                ["poetry", "run", "pytest", 
                 "tests/unit/bot_v2", "tests/unit/test_foundation.py",
                 "-q", "--tb=no", "--no-header"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            output = result.stdout + result.stderr
            
            # Parse test results
            if "passed" in output:
                # Extract pass/fail counts
                import re
                match = re.search(r"(\d+) passed", output)
                passed = int(match.group(1)) if match else 0
                
                match = re.search(r"(\d+) failed", output)
                failed = int(match.group(1)) if match else 0
                
                match = re.search(r"(\d+) error", output)
                errors = int(match.group(1)) if match else 0
                
                if failed == 0 and errors == 0:
                    self.add_result("Tests", "Unit Tests", "PASS", 
                                  f"{passed} tests passed")
                else:
                    self.add_result("Tests", "Unit Tests", "WARN", 
                                  f"{passed} passed, {failed} failed, {errors} errors")
            else:
                self.add_result("Tests", "Unit Tests", "FAIL", "Test suite failed to run")
                
        except subprocess.TimeoutExpired:
            self.add_result("Tests", "Unit Tests", "FAIL", "Tests timed out")
        except Exception as e:
            self.add_result("Tests", "Unit Tests", "FAIL", str(e))
    
    def check_risk_systems(self):
        """Check risk management configuration."""
        print("\n🛡️ Checking Risk Management Systems...")
        
        # Check risk configuration
        max_leverage = os.environ.get("RISK_MAX_LEVERAGE", "5")
        daily_loss = os.environ.get("RISK_DAILY_LOSS_LIMIT", "100")
        max_position = os.environ.get("RISK_MAX_POSITION_PCT_PER_SYMBOL", "0.25")
        
        try:
            leverage = float(max_leverage)
            if leverage > 10:
                self.add_result("Risk", "Max Leverage", "WARN", 
                              f"High leverage: {leverage}x (consider starting lower)")
            else:
                self.add_result("Risk", "Max Leverage", "PASS", f"{leverage}x")
        except:
            self.add_result("Risk", "Max Leverage", "FAIL", "Invalid leverage value")
        
        try:
            loss_limit = float(daily_loss)
            if loss_limit > 1000:
                self.add_result("Risk", "Daily Loss Limit", "WARN",
                              f"High daily loss limit: ${loss_limit}")
            else:
                self.add_result("Risk", "Daily Loss Limit", "PASS", f"${loss_limit}")
        except:
            self.add_result("Risk", "Daily Loss Limit", "FAIL", "Invalid loss limit")
        
        # Check order preview
        preview = os.environ.get("ORDER_PREVIEW_ENABLED", "0")
        if preview == "1":
            self.add_result("Risk", "Order Preview", "PASS", "Enabled (recommended)")
        else:
            self.add_result("Risk", "Order Preview", "WARN", "Disabled (enable for safety)")
    
    def check_canary_profile(self):
        """Check canary profile configuration."""
        print("\n🐤 Checking Canary Profile...")
        
        canary_path = "config/profiles/canary.yaml"
        if os.path.exists(canary_path):
            self.add_result("Profiles", "Canary Profile", "PASS", "Found")
            
            # Parse and validate canary settings
            try:
                import yaml
                with open(canary_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check critical safety settings
                max_position = config.get("trading", {}).get("position_sizing", {}).get("max_position_size")
                daily_loss = config.get("risk_management", {}).get("daily_loss_limit")
                reduce_only = config.get("trading", {}).get("mode")
                
                if max_position and float(max_position) <= 0.01:
                    self.add_result("Profiles", "Canary Position Size", "PASS", 
                                  f"Max {max_position} BTC")
                else:
                    self.add_result("Profiles", "Canary Position Size", "WARN",
                                  "Consider smaller position for testing")
                
                if daily_loss and float(daily_loss) <= 20:
                    self.add_result("Profiles", "Canary Loss Limit", "PASS",
                                  f"Max ${daily_loss} daily loss")
                else:
                    self.add_result("Profiles", "Canary Loss Limit", "WARN",
                                  "Consider lower loss limit for testing")
                
                if reduce_only == "reduce_only":
                    self.add_result("Profiles", "Canary Mode", "PASS",
                                  "Reduce-only mode enabled")
                    
            except Exception as e:
                self.add_result("Profiles", "Canary Validation", "FAIL", str(e))
        else:
            self.add_result("Profiles", "Canary Profile", "WARN", "Not found")
    
    def generate_report(self):
        """Generate final preflight report."""
        print("\n" + "=" * 70)
        print("📋 PREFLIGHT CHECK REPORT")
        print("=" * 70)
        print(f"Timestamp: {datetime.utcnow().isoformat()}")
        print(f"Overall Status: {self.status}")
        
        # Group results by category
        categories = {}
        for result in self.results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        
        # Print results by category
        for category, items in categories.items():
            print(f"\n{category}:")
            for item in items:
                status_icon = {
                    "PASS": "✅",
                    "FAIL": "❌",
                    "WARN": "⚠️",
                    "INFO": "ℹ️"
                }.get(item["status"], "❓")
                
                print(f"  {status_icon} {item['item']}: {item['message']}")
        
        # Print summary
        print("\n" + "-" * 70)
        if self.errors:
            print("\n❌ ERRORS (must fix before trading):")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS (review before trading):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        # Final recommendation
        print("\n" + "=" * 70)
        if self.status == "PASS":
            print("✅ READY FOR TESTING")
            print("Recommended next step: Run with canary profile in dry-run mode")
            print("Command: poetry run perps-bot --profile canary --dry-run")
        elif self.status == "WARN":
            print("⚠️  READY WITH WARNINGS")
            print("Review warnings above before proceeding")
            print("Test command: poetry run perps-bot --profile canary --dry-run")
        else:
            print("❌ NOT READY FOR TRADING")
            print("Critical issues must be resolved:")
            print("1. Enable futures/derivatives permissions on your CDP API key")
            print("2. Fix any failing tests")
            print("3. Ensure proper environment configuration")
        
        print("=" * 70)
        
        return self.status


def main():
    """Run preflight checks."""
    checker = PreflightChecker()
    
    print("=" * 70)
    print("🚀 GPT-TRADER PREFLIGHT CHECK")
    print("=" * 70)
    
    # Run all checks
    checker.check_environment()
    checker.check_api_connectivity()
    checker.check_test_suite()
    checker.check_risk_systems()
    checker.check_canary_profile()
    
    # Generate report
    status = checker.generate_report()
    
    # Exit with appropriate code
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()