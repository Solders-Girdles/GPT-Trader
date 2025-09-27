#!/usr/bin/env python3
"""
Comprehensive API key permissions checker for Coinbase CDP.

This script checks all available permissions that can be granted to API keys
according to the Coinbase CDP documentation.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dotenv import load_dotenv
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


class PermissionChecker:
    """Utility class for checking API key permissions."""
    
    # All available permissions according to CDP documentation
    ALL_PERMISSIONS = ["can_view", "can_trade", "can_transfer"]
    
    # Permission requirements for different operations
    OPERATION_REQUIREMENTS = {
        "read_only": ["can_view"],
        "trading": ["can_view", "can_trade"],
        "full_access": ["can_view", "can_trade", "can_transfer"],
    }
    
    def __init__(self, client):
        """Initialize with a Coinbase client."""
        self.client = client
        self.permissions = None
        self.check_time = None
        
    def fetch_permissions(self) -> Dict[str, Any]:
        """Fetch current API key permissions."""
        try:
            self.permissions = self.client.get_key_permissions()
            self.check_time = datetime.now()
            return self.permissions
        except Exception as e:
            raise Exception(f"Failed to fetch permissions: {e}")
            
    def check_permission(self, permission: str) -> bool:
        """Check if a specific permission is granted."""
        if self.permissions is None:
            self.fetch_permissions()
        return self.permissions.get(permission, False)
        
    def check_operation_permissions(self, operation: str) -> Tuple[bool, List[str]]:
        """
        Check if permissions are sufficient for a specific operation.
        
        Returns:
            Tuple of (has_permissions, missing_permissions)
        """
        if operation not in self.OPERATION_REQUIREMENTS:
            raise ValueError(f"Unknown operation: {operation}")
            
        required = self.OPERATION_REQUIREMENTS[operation]
        missing = []
        
        for perm in required:
            if not self.check_permission(perm):
                missing.append(perm)
                
        return len(missing) == 0, missing
        
    def get_permission_summary(self) -> Dict[str, Any]:
        """Get a summary of all permissions."""
        if self.permissions is None:
            self.fetch_permissions()
            
        return {
            "timestamp": self.check_time.isoformat() if self.check_time else None,
            "permissions": {
                perm: self.permissions.get(perm, False) 
                for perm in self.ALL_PERMISSIONS
            },
            "portfolio": {
                "uuid": self.permissions.get("portfolio_uuid"),
                "type": self.permissions.get("portfolio_type"),
            },
            "operations": {
                op: self.check_operation_permissions(op)[0]
                for op in self.OPERATION_REQUIREMENTS
            }
        }
        
    def print_detailed_report(self):
        """Print a detailed permission report."""
        summary = self.get_permission_summary()
        
        print("=" * 70)
        print("COINBASE CDP API KEY PERMISSIONS REPORT")
        print("=" * 70)
        print(f"\n📅 Checked at: {summary['timestamp']}")
        
        # Individual permissions
        print("\n📋 INDIVIDUAL PERMISSIONS:")
        print("-" * 40)
        for perm, granted in summary["permissions"].items():
            icon = "✅" if granted else "❌"
            status = "GRANTED" if granted else "DENIED"
            print(f"  {icon} {perm:<20} : {status}")
            
        # Portfolio information
        print("\n📁 PORTFOLIO INFORMATION:")
        print("-" * 40)
        portfolio = summary["portfolio"]
        if portfolio["uuid"]:
            print(f"  UUID: {portfolio['uuid']}")
            print(f"  Type: {portfolio['type']}")
        else:
            print("  ⚠️ No portfolio associated")
            
        # Operation capabilities
        print("\n🚀 OPERATION CAPABILITIES:")
        print("-" * 40)
        for op, capable in summary["operations"].items():
            icon = "✅" if capable else "❌"
            op_name = op.replace("_", " ").title()
            status = "AVAILABLE" if capable else "UNAVAILABLE"
            print(f"  {icon} {op_name:<20} : {status}")
            
            if not capable:
                _, missing = self.check_operation_permissions(op)
                print(f"      Missing: {', '.join(missing)}")
                
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        
        if not summary["permissions"]["can_view"]:
            print("  🔴 CRITICAL: No view permission - API key cannot access any data")
            print("     Action: Enable 'View' permission in CDP Console")
            
        if summary["permissions"]["can_view"] and not summary["permissions"]["can_trade"]:
            print("  🟡 WARNING: Read-only access - cannot place orders")
            print("     Action: Enable 'Trade' permission for trading capabilities")
            
        if not summary["permissions"]["can_transfer"]:
            print("  🔵 INFO: Transfer disabled - cannot move funds between accounts")
            print("     Action: Enable 'Transfer' only if fund movements are needed")
            
        if all(summary["permissions"].values()):
            print("  🟢 EXCELLENT: All permissions granted - full access available")
            
        return summary


def test_endpoint_access(client, checker: PermissionChecker):
    """Test actual endpoint access with current permissions."""
    print("\n" + "=" * 70)
    print("ENDPOINT ACCESS VERIFICATION")
    print("=" * 70)
    
    endpoints = [
        {
            "name": "Server Time",
            "func": client.get_time,
            "requires": [],
            "description": "Public endpoint - should always work"
        },
        {
            "name": "Account List",
            "func": client.get_accounts,
            "requires": ["can_view"],
            "description": "Requires view permission"
        },
        {
            "name": "Portfolio List",
            "func": client.list_portfolios,
            "requires": ["can_view"],
            "description": "Requires view permission"
        },
        {
            "name": "Transaction Summary",
            "func": client.get_transaction_summary,
            "requires": ["can_view"],
            "description": "Requires view permission"
        },
        {
            "name": "Best Bid/Ask",
            "func": lambda: client.get_best_bid_ask(["BTC-USD"]),
            "requires": ["can_view"],
            "description": "May require additional market data permissions"
        },
    ]
    
    results = []
    
    for endpoint in endpoints:
        print(f"\n📍 Testing: {endpoint['name']}")
        print(f"   {endpoint['description']}")
        
        # Check if we have required permissions
        has_perms = all(checker.check_permission(p) for p in endpoint["requires"])
        
        if not has_perms and endpoint["requires"]:
            print(f"   ⏭️ Skipped - missing permissions: {endpoint['requires']}")
            results.append({
                "endpoint": endpoint["name"],
                "status": "skipped",
                "reason": "missing_permissions"
            })
            continue
            
        try:
            result = endpoint["func"]()
            if result:
                print(f"   ✅ Success")
                # Show sample of response
                if isinstance(result, dict):
                    if "iso" in result:  # Time endpoint
                        print(f"      Server time: {result['iso']}")
                    elif "accounts" in result:
                        print(f"      Found {len(result['accounts'])} accounts")
                    elif "portfolios" in result:
                        print(f"      Found {len(result['portfolios'])} portfolios")
                results.append({
                    "endpoint": endpoint["name"],
                    "status": "success",
                    "sample": str(result)[:100] if result else None
                })
            else:
                print(f"   ⚠️ Empty response")
                results.append({
                    "endpoint": endpoint["name"],
                    "status": "empty",
                })
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg:
                print(f"   ❌ Unauthorized")
                if endpoint["name"] == "Best Bid/Ask":
                    print(f"      Note: Market data often requires additional permissions")
            elif "403" in error_msg:
                print(f"   ❌ Forbidden")
            else:
                print(f"   ❌ Error: {error_msg[:100]}")
            results.append({
                "endpoint": endpoint["name"],
                "status": "error",
                "error": error_msg[:200]
            })
            
    return results


def main():
    """Main function to check all permissions."""
    load_dotenv()
    
    print("\n" + "🔐" * 35)
    print("COMPREHENSIVE CDP API PERMISSIONS CHECK")
    print("🔐" * 35)
    
    # Import client
    from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
    from bot_v2.features.brokerages.coinbase.cdp_auth_v2 import CDPAuthV2
    
    # Get credentials
    api_key = os.getenv("COINBASE_PROD_CDP_API_KEY") or os.getenv("COINBASE_CDP_API_KEY")
    private_key = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY") or os.getenv("COINBASE_CDP_PRIVATE_KEY")
    
    if not api_key or not private_key:
        print("\n❌ ERROR: Missing CDP API credentials")
        print("\nRequired environment variables:")
        print("  - COINBASE_PROD_CDP_API_KEY (or COINBASE_CDP_API_KEY)")
        print("  - COINBASE_PROD_CDP_PRIVATE_KEY (or COINBASE_CDP_PRIVATE_KEY)")
        return 1
        
    # Truncate key for display
    key_display = api_key[:50] + "..." if len(api_key) > 50 else api_key
    print(f"\n🔑 Using API Key: {key_display}")
    
    try:
        # Setup auth and client
        auth = CDPAuthV2(
            api_key_name=api_key,
            private_key_pem=private_key,
            base_host="api.coinbase.com"
        )
        
        client = CoinbaseClient(
            base_url="https://api.coinbase.com",
            auth=auth,
            api_mode="advanced"
        )
        
        # Create permission checker
        checker = PermissionChecker(client)
        
        # Print detailed permission report
        summary = checker.print_detailed_report()
        
        # Test endpoint access
        endpoint_results = test_endpoint_access(client, checker)
        
        # Final summary
        print("\n" + "=" * 70)
        print("FINAL STATUS")
        print("=" * 70)
        
        # Determine overall status
        has_view = checker.check_permission("can_view")
        has_trade = checker.check_permission("can_trade")
        has_transfer = checker.check_permission("can_transfer")
        
        if not has_view:
            print("\n🔴 CRITICAL: API key has no permissions")
            print("\nAction Required:")
            print("1. Go to https://portal.cdp.coinbase.com/")
            print("2. Edit your API key")
            print("3. Enable required permissions:")
            print("   - ✅ View (minimum required)")
            print("   - ✅ Trade (for trading)")
            print("   - ⬜ Transfer (optional)")
            return 1
            
        elif has_view and not has_trade:
            print("\n🟡 LIMITED: Read-only access")
            print("\nCurrent capabilities:")
            print("  ✅ View account data")
            print("  ✅ Get market data")
            print("  ❌ Place orders")
            print("  ❌ Cancel orders")
            print("\nFor trading, enable 'Trade' permission in CDP Console")
            return 0
            
        elif has_view and has_trade and not has_transfer:
            print("\n🟢 READY: Trading permissions active")
            print("\nCurrent capabilities:")
            print("  ✅ View account data")
            print("  ✅ Place and manage orders")
            print("  ❌ Transfer funds (not required for trading)")
            print("\nYou're ready to run:")
            print("  poetry run perps-bot --profile canary --dry-run")
            return 0
            
        else:
            print("\n🟢 FULL ACCESS: All permissions granted")
            print("\nCurrent capabilities:")
            print("  ✅ Full account access")
            print("  ✅ Trading operations")
            print("  ✅ Fund transfers")
            print("\nReady for production trading!")
            return 0
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Verify API key is active in CDP Console")
        print("2. Check private key format (should be PEM format)")
        print("3. Ensure you're using production keys (not sandbox)")
        return 1


if __name__ == "__main__":
    sys.exit(main())