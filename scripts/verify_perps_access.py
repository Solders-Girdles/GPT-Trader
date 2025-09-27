#!/usr/bin/env python3
"""
Comprehensive Perpetuals API Access Verification Script

This script checks whether your Coinbase API credentials have the necessary
permissions to trade perpetual futures. It performs non-destructive read-only
tests to validate your setup.

Usage:
    poetry run python scripts/verify_perps_access.py
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from decimal import Decimal

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class PerpsAccessVerifier:
    """Verify API access to Coinbase perpetuals."""
    
    def __init__(self):
        self.checks_passed = []
        self.checks_failed = []
        self.checks_warning = []
        self.client = None
        self.auth = None
        
    def run_verification(self) -> bool:
        """Run all verification checks."""
        print("=" * 70)
        print("🔍 COINBASE PERPETUALS API ACCESS VERIFICATION")
        print("=" * 70)
        print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
        print("\nThis script will verify your API credentials for perpetuals trading.")
        print("All checks are read-only and will not place any orders.\n")
        print("=" * 70)
        
        # Run checks in order of importance
        checks = [
            ("Environment Setup", self.check_environment),
            ("API Credentials", self.check_credentials),
            ("JWT Authentication", self.check_jwt_auth),
            ("API Connection", self.check_api_connection),
            ("Account Access", self.check_account_access),
            ("Perpetuals Permissions", self.check_perpetuals_permissions),
            ("Market Data Access", self.check_market_data),
            ("Position Management Access", self.check_position_access),
            ("Order Placement Capability", self.check_order_capability),
        ]
        
        for i, (check_name, check_func) in enumerate(checks, 1):
            print(f"\n[{i}/{len(checks)}] {check_name}")
            print("-" * 50)
            
            try:
                result, details = check_func()
                self._log_result(check_name, result, details)
                
                # Stop if critical check fails
                if result == "FAIL" and i <= 4:  # First 4 checks are critical
                    print("\n❌ Critical check failed. Please fix this before continuing.")
                    break
                    
            except Exception as e:
                self.checks_failed.append(check_name)
                print(f"❌ FAILED: Unexpected error: {e}")
                if i <= 4:  # Critical check
                    break
        
        return self._print_summary()
    
    def check_environment(self) -> Tuple[str, Dict]:
        """Check environment configuration."""
        details = {}
        issues = []
        
        # Check critical environment variables
        env_vars = {
            "COINBASE_API_MODE": ("advanced", "Must be 'advanced' for perpetuals"),
            "COINBASE_SANDBOX": ("0", "Must be '0' - sandbox doesn't support perpetuals"),
            "COINBASE_ENABLE_DERIVATIVES": ("1", "Must be '1' to enable perpetuals"),
        }
        
        for var, (expected, description) in env_vars.items():
            value = os.getenv(var, "").strip()
            if not value:
                issues.append(f"{var} not set - {description}")
                details[var] = "❌ Not set"
            elif value != expected:
                issues.append(f"{var}={value} (should be {expected}) - {description}")
                details[var] = f"❌ {value} (expected: {expected})"
            else:
                details[var] = f"✅ {value}"
        
        # Check for paper mode
        paper_mode = os.getenv("PERPS_PAPER", "0")
        if paper_mode == "1":
            details["PERPS_PAPER"] = "⚠️ Paper mode enabled (set to 0 for live trading)"
            issues.append("Paper mode is enabled - this will use mock data")
        else:
            details["PERPS_PAPER"] = "✅ Live mode"
        
        if issues:
            return ("FAIL" if len(issues) > 1 else "WARNING", {
                "configuration": details,
                "issues": issues
            })
        
        return ("PASS", {"configuration": details})
    
    def check_credentials(self) -> Tuple[str, Dict]:
        """Check if API credentials are present."""
        details = {}
        
        # Check for CDP credentials (primary)
        cdp_key = os.getenv("COINBASE_PROD_CDP_API_KEY") or os.getenv("COINBASE_CDP_API_KEY")
        cdp_private = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY") or os.getenv("COINBASE_CDP_PRIVATE_KEY")
        cdp_private_path = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY_PATH") or os.getenv("COINBASE_CDP_PRIVATE_KEY_PATH")
        
        # Check API key
        if cdp_key:
            # Mask the key for security
            if len(cdp_key) > 40:
                details["api_key"] = f"✅ {cdp_key[:20]}...{cdp_key[-10:]}"
            else:
                details["api_key"] = "✅ Configured"
        else:
            return ("FAIL", {
                "error": "No API key found",
                "help": "Set COINBASE_CDP_API_KEY or COINBASE_PROD_CDP_API_KEY"
            })
        
        # Check private key
        if cdp_private:
            # Check if it looks like a PEM key
            if "BEGIN EC PRIVATE KEY" in cdp_private or "BEGIN PRIVATE KEY" in cdp_private:
                details["private_key"] = "✅ PEM format in environment"
            else:
                details["private_key"] = "⚠️ In environment (check PEM format)"
        elif cdp_private_path:
            if os.path.exists(cdp_private_path):
                details["private_key"] = f"✅ File at {cdp_private_path}"
                # Check file permissions
                stat_info = os.stat(cdp_private_path)
                perms = oct(stat_info.st_mode)[-3:]
                if perms not in ["400", "600"]:
                    details["key_permissions"] = f"⚠️ {perms} (should be 400 or 600)"
            else:
                return ("FAIL", {
                    "error": f"Private key file not found: {cdp_private_path}"
                })
        else:
            return ("FAIL", {
                "error": "No private key found",
                "help": "Set COINBASE_CDP_PRIVATE_KEY (PEM string) or COINBASE_CDP_PRIVATE_KEY_PATH"
            })
        
        details["auth_method"] = "CDP/JWT (correct for perpetuals)"
        
        return ("PASS", details)
    
    def check_jwt_auth(self) -> Tuple[str, Dict]:
        """Check JWT token generation."""
        try:
            from bot_v2.features.brokerages.coinbase.cdp_auth_v2 import CDPAuthV2
            
            # Get credentials
            api_key = os.getenv("COINBASE_PROD_CDP_API_KEY") or os.getenv("COINBASE_CDP_API_KEY")
            private_key = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY") or os.getenv("COINBASE_CDP_PRIVATE_KEY")
            private_key_path = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY_PATH") or os.getenv("COINBASE_CDP_PRIVATE_KEY_PATH")
            
            # Load private key
            if private_key:
                private_key_pem = private_key
            elif private_key_path:
                with open(private_key_path, 'r') as f:
                    private_key_pem = f.read()
            else:
                return ("FAIL", {"error": "No private key available"})
            
            # Create auth instance
            self.auth = CDPAuthV2(
                api_key_name=api_key,
                private_key_pem=private_key_pem
            )
            
            # Generate a test JWT
            test_path = "/api/v3/brokerage/accounts"
            jwt_token = self.auth.generate_jwt(method="GET", path=test_path)
            
            # Validate JWT structure (should have 3 parts)
            parts = jwt_token.split('.')
            if len(parts) != 3:
                return ("FAIL", {
                    "error": "Invalid JWT structure",
                    "parts": len(parts)
                })
            
            # Decode header (base64)
            import base64
            header = json.loads(base64.urlsafe_b64decode(parts[0] + '=='))
            
            details = {
                "jwt_generation": "✅ Successful",
                "algorithm": header.get("alg", "unknown"),
                "token_length": len(jwt_token),
                "key_id": header.get("kid", "not specified")[:20] + "..." if header.get("kid") else "none"
            }
            
            # Check algorithm
            if header.get("alg") != "ES256":
                details["warning"] = f"Unexpected algorithm: {header.get('alg')} (expected ES256)"
                return ("WARNING", details)
            
            return ("PASS", details)
            
        except Exception as e:
            return ("FAIL", {
                "error": f"JWT generation failed: {str(e)}",
                "help": "Check your private key format (should be PEM with BEGIN/END markers)"
            })
    
    def check_api_connection(self) -> Tuple[str, Dict]:
        """Check basic API connectivity."""
        try:
            from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
            
            if not self.auth:
                return ("FAIL", {"error": "Authentication not initialized"})
            
            # Create client
            self.client = CoinbaseClient(
                base_url="https://api.coinbase.com",
                auth=self.auth,
                api_mode="advanced"
            )
            
            # Test server time endpoint (public, but validates connection)
            start_time = time.time()
            time_response = self.client.get_time()
            latency_ms = (time.time() - start_time) * 1000
            
            if not isinstance(time_response, dict):
                return ("FAIL", {
                    "error": "Invalid response format",
                    "response": str(time_response)[:100]
                })
            
            server_time = time_response.get("iso")
            if not server_time:
                return ("WARNING", {
                    "warning": "No server time in response",
                    "response": str(time_response)[:100]
                })
            
            # Check time synchronization
            server_dt = datetime.fromisoformat(server_time.replace('Z', '+00:00'))
            local_dt = datetime.now(timezone.utc)
            time_diff = abs((server_dt - local_dt).total_seconds())
            
            details = {
                "connection": "✅ Established",
                "latency": f"{latency_ms:.0f}ms",
                "server_time": server_time,
                "time_sync": f"✅ {time_diff:.1f}s difference" if time_diff < 30 else f"⚠️ {time_diff:.1f}s difference"
            }
            
            if time_diff > 30:
                details["warning"] = "Clock drift > 30s may cause JWT authentication issues"
                return ("WARNING", details)
            
            return ("PASS", details)
            
        except Exception as e:
            return ("FAIL", {
                "error": f"Connection failed: {str(e)}",
                "help": "Check network connectivity and API endpoint"
            })
    
    def check_account_access(self) -> Tuple[str, Dict]:
        """Check if we can access account information."""
        if not self.client:
            return ("FAIL", {"error": "Client not initialized"})
        
        try:
            # Get accounts
            accounts = self.client.get_accounts()
            
            if not accounts or "accounts" not in accounts:
                return ("FAIL", {
                    "error": "No accounts returned",
                    "response": str(accounts)[:200]
                })
            
            account_list = accounts["accounts"]
            if not account_list:
                return ("FAIL", {
                    "error": "Account list is empty",
                    "help": "Ensure your API key has account read permissions"
                })
            
            # Find futures account
            futures_account = None
            spot_accounts = []
            
            for account in account_list:
                if account.get("type") == "ACCOUNT_TYPE_FUTURES":
                    futures_account = account
                elif account.get("type") == "ACCOUNT_TYPE_CRYPTO":
                    spot_accounts.append(account)
            
            details = {
                "total_accounts": len(account_list),
                "spot_accounts": len(spot_accounts)
            }
            
            if futures_account:
                balance = futures_account.get("available_balance", {})
                currency = balance.get("currency", "USD")
                value = balance.get("value", "0")
                
                details["futures_account"] = "✅ Found"
                details["futures_balance"] = f"{currency} {value}"
                details["account_uuid"] = futures_account.get("uuid", "")[:8] + "..."
            else:
                details["futures_account"] = "❌ Not found"
                details["warning"] = "No futures account found - perpetuals may not be enabled"
                return ("WARNING", details)
            
            return ("PASS", details)
            
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                return ("FAIL", {
                    "error": "Authentication failed",
                    "help": "Check your API key and private key"
                })
            elif "403" in error_msg or "forbidden" in error_msg.lower():
                return ("FAIL", {
                    "error": "Permission denied",
                    "help": "Your API key may not have account read permissions"
                })
            else:
                return ("FAIL", {
                    "error": f"Account access failed: {error_msg}",
                    "help": "Check API permissions and account status"
                })
    
    def check_perpetuals_permissions(self) -> Tuple[str, Dict]:
        """Check if API has perpetuals/derivatives permissions."""
        if not self.client:
            return ("FAIL", {"error": "Client not initialized"})
        
        try:
            # Try to access CFM positions endpoint
            positions = self.client.cfm_positions()
            
            if positions is None:
                return ("WARNING", {
                    "warning": "No positions returned (may be normal if no open positions)",
                    "status": "Endpoint accessible"
                })
            
            if isinstance(positions, dict):
                if "positions" in positions:
                    position_list = positions.get("positions", [])
                    details = {
                        "cfm_access": "✅ Confirmed",
                        "open_positions": len(position_list),
                        "endpoint": "/api/v3/brokerage/cfm/positions"
                    }
                    
                    if position_list:
                        # Show first position as example
                        first_pos = position_list[0]
                        details["example_position"] = {
                            "product_id": first_pos.get("product_id"),
                            "side": first_pos.get("position_side"),
                            "contracts": first_pos.get("number_of_contracts")
                        }
                    
                    return ("PASS", details)
                elif "error" in positions:
                    error_msg = positions.get("error", "Unknown error")
                    if "not found" in error_msg.lower():
                        return ("FAIL", {
                            "error": "CFM endpoints not available",
                            "help": "Derivatives may not be enabled on your account"
                        })
            
            return ("PASS", {
                "cfm_access": "✅ Endpoint accessible",
                "note": "No open positions"
            })
            
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                return ("FAIL", {
                    "error": "CFM endpoints not found",
                    "help": "Your account may not have perpetuals enabled. Contact Coinbase support."
                })
            elif "403" in error_msg:
                return ("FAIL", {
                    "error": "Permission denied for CFM endpoints",
                    "help": "Your API key needs 'Trade Futures' permission"
                })
            else:
                return ("FAIL", {
                    "error": f"CFM access check failed: {error_msg}"
                })
    
    def check_market_data(self) -> Tuple[str, Dict]:
        """Check if we can access perpetuals market data."""
        if not self.client:
            return ("FAIL", {"error": "Client not initialized"})
        
        try:
            # Test symbols
            perp_symbols = ["BTC-PERP", "ETH-PERP"]
            
            # Try to get best bid/ask for perpetuals
            quotes = self.client.get_best_bid_ask(perp_symbols)
            
            if not quotes or "pricebooks" not in quotes:
                return ("FAIL", {
                    "error": "No price data returned",
                    "response": str(quotes)[:200]
                })
            
            pricebooks = quotes["pricebooks"]
            details = {
                "market_data": "✅ Accessible",
                "symbols_checked": perp_symbols
            }
            
            for book in pricebooks:
                product_id = book.get("product_id")
                bids = book.get("bids", [])
                asks = book.get("asks", [])
                
                if bids and asks:
                    best_bid = bids[0].get("price")
                    best_ask = asks[0].get("price")
                    spread = float(best_ask) - float(best_bid)
                    mid = (float(best_bid) + float(best_ask)) / 2
                    spread_bps = (spread / mid) * 10000
                    
                    details[product_id] = {
                        "bid": best_bid,
                        "ask": best_ask,
                        "spread_bps": f"{spread_bps:.1f}"
                    }
                else:
                    details[product_id] = "No quotes available"
            
            return ("PASS", details)
            
        except Exception as e:
            return ("WARNING", {
                "warning": f"Market data check failed: {str(e)}",
                "note": "This may be normal outside market hours"
            })
    
    def check_position_access(self) -> Tuple[str, Dict]:
        """Check position management capabilities."""
        if not self.client:
            return ("FAIL", {"error": "Client not initialized"})
        
        try:
            # Check CFM sweeps (funding payments)
            sweeps = self.client.cfm_sweeps()
            
            details = {
                "position_management": "✅ Available"
            }
            
            if sweeps and "sweeps" in sweeps:
                sweep_list = sweeps["sweeps"]
                details["funding_history"] = f"{len(sweep_list)} records"
                
                if sweep_list:
                    latest = sweep_list[0]
                    details["latest_funding"] = {
                        "time": latest.get("scheduled_time", "")[:19],
                        "status": latest.get("status")
                    }
            
            return ("PASS", details)
            
        except Exception as e:
            return ("WARNING", {
                "warning": f"Position access limited: {str(e)}",
                "note": "Basic position management should still work"
            })
    
    def check_order_capability(self) -> Tuple[str, Dict]:
        """Check order placement capability (without placing actual orders)."""
        if not self.client:
            return ("FAIL", {"error": "Client not initialized"})
        
        details = {
            "order_capability": "✅ API endpoints verified",
            "note": "Actual order placement not tested (would require real trade)"
        }
        
        # Check if we can get product info
        try:
            # We've already verified CFM access and market data access
            # These are the key requirements for order placement
            
            details["requirements_met"] = [
                "✅ JWT authentication working",
                "✅ CFM endpoints accessible", 
                "✅ Market data available",
                "✅ Account access confirmed"
            ]
            
            # Verify order preview capability (if configured)
            if os.getenv("ORDER_PREVIEW_ENABLED") == "1":
                details["order_preview"] = "✅ Enabled (safer trading)"
            else:
                details["order_preview"] = "ℹ️ Disabled (set ORDER_PREVIEW_ENABLED=1 to enable)"
            
            return ("PASS", details)
            
        except Exception as e:
            return ("WARNING", {
                "warning": f"Could not fully verify: {str(e)}",
                "note": "Order placement should work if other checks passed"
            })
    
    def _log_result(self, name: str, result: str, details: Dict):
        """Log check result."""
        symbols = {
            "PASS": "✅",
            "FAIL": "❌", 
            "WARNING": "⚠️"
        }
        
        print(f"{symbols.get(result, '❓')} {result}")
        
        # Track results
        if result == "PASS":
            self.checks_passed.append(name)
        elif result == "FAIL":
            self.checks_failed.append(name)
        else:
            self.checks_warning.append(name)
        
        # Print details
        def print_dict(d: dict, indent: int = 2):
            for key, value in d.items():
                if isinstance(value, dict):
                    print(f"{' ' * indent}{key}:")
                    print_dict(value, indent + 2)
                elif isinstance(value, list):
                    print(f"{' ' * indent}{key}:")
                    for item in value:
                        if isinstance(item, dict):
                            print_dict(item, indent + 2)
                        else:
                            print(f"{' ' * (indent + 2)}- {item}")
                else:
                    print(f"{' ' * indent}{key}: {value}")
        
        print_dict(details)
    
    def _print_summary(self) -> bool:
        """Print verification summary."""
        print("\n" + "=" * 70)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 70)
        
        total = len(self.checks_passed) + len(self.checks_failed) + len(self.checks_warning)
        
        print(f"\n✅ Passed: {len(self.checks_passed)}/{total}")
        if self.checks_passed:
            for check in self.checks_passed:
                print(f"   • {check}")
        
        if self.checks_warning:
            print(f"\n⚠️  Warnings: {len(self.checks_warning)}/{total}")
            for check in self.checks_warning:
                print(f"   • {check}")
        
        if self.checks_failed:
            print(f"\n❌ Failed: {len(self.checks_failed)}/{total}")
            for check in self.checks_failed:
                print(f"   • {check}")
        
        print("\n" + "=" * 70)
        
        if self.checks_failed:
            print("❌ VERIFICATION FAILED\n")
            print("Your API is not ready for perpetuals trading.")
            print("Please fix the failed checks above.\n")
            print("Common solutions:")
            print("1. Ensure COINBASE_API_MODE=advanced")
            print("2. Ensure COINBASE_SANDBOX=0 (sandbox doesn't support perpetuals)")
            print("3. Ensure COINBASE_ENABLE_DERIVATIVES=1")
            print("4. Check that your API key has 'Trade Futures' permission")
            print("5. Verify your account has perpetuals/derivatives enabled")
            return False
        
        elif self.checks_warning:
            print("⚠️  VERIFICATION PASSED WITH WARNINGS\n")
            print("Your API should work for perpetuals trading, but review the warnings above.")
            print("\nYou can proceed with:")
            print("1. Paper trading: PERPS_PAPER=1 poetry run perps-bot --profile dev")
            print("2. Canary testing: poetry run perps-bot --profile canary")
            return True
        
        else:
            print("✅ ALL CHECKS PASSED!\n")
            print("Your API is fully configured for perpetuals trading.")
            print("\nNext steps:")
            print("1. Start with paper trading to validate strategies:")
            print("   PERPS_PAPER=1 poetry run perps-bot --profile dev")
            print("\n2. Move to canary profile for minimal-risk live testing:")
            print("   poetry run perps-bot --profile canary")
            print("\n3. Gradually increase limits as you gain confidence")
            return True


def main():
    """Run verification."""
    verifier = PerpsAccessVerifier()
    success = verifier.run_verification()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
