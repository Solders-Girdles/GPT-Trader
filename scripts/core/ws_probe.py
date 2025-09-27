#!/usr/bin/env python3
"""
WebSocket connectivity probe for Coinbase Advanced Trade.
Tests WebSocket connections and authentication.
"""

import os
import sys
import asyncio
import json
import time
import ssl
import websockets
from datetime import datetime, timezone
from typing import Dict, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class WebSocketProbe:
    """Test WebSocket connectivity to Coinbase."""
    
    def __init__(self):
        self.results = {}
        self.sandbox = os.getenv("COINBASE_SANDBOX", "0") == "1"
        
    async def run_all_tests(self):
        """Run all WebSocket connectivity tests."""
        print("🔍 WEBSOCKET CONNECTIVITY PROBE")
        print("="*60)
        print(f"Environment: {'SANDBOX' if self.sandbox else 'PRODUCTION'}")
        print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
        print("="*60)
        
        # Test different endpoints
        endpoints = self.get_endpoints()
        
        for name, url in endpoints.items():
            print(f"\n📡 Testing {name}: {url}")
            print("-"*40)
            
            # Basic connectivity
            await self.test_connectivity(name, url)
            
            # If connected, test subscribe
            if self.results.get(f"{name}_connect") == "PASS":
                await self.test_subscribe(name, url)
        
        # Generate report
        self.generate_report()
    
    def get_endpoints(self) -> Dict[str, str]:
        """Get WebSocket endpoints to test."""
        if self.sandbox:
            return {
                # Coinbase Exchange sandbox public feed (valid sandbox WS)
                "exchange_sandbox_feed": "wss://ws-feed-public.sandbox.exchange.coinbase.com",
                # Advanced Trade has no public sandbox; production WS is reachable for public data
                "advanced_trade": "wss://advanced-trade-ws.coinbase.com",
            }
        else:
            return {
                "production_direct": "wss://ws-direct.coinbase.com",
                "production_feed": "wss://ws-feed.coinbase.com",
                "advanced_trade": "wss://advanced-trade-ws.coinbase.com"
            }
    
    async def test_connectivity(self, name: str, url: str):
        """Test basic WebSocket connectivity."""
        try:
            # Create SSL context
            ssl_context = ssl.create_default_context()
            
            # Attempt connection with timeout
            print(f"  Connecting to {url}...")
            start_time = time.time()
            
            async with websockets.connect(
                url,
                ssl=ssl_context,
                ping_interval=None,
                close_timeout=10
            ) as ws:
                connect_time = (time.time() - start_time) * 1000
                print(f"  ✅ Connected in {connect_time:.0f}ms")
                
                # Test ping
                pong_waiter = await ws.ping()
                await asyncio.wait_for(pong_waiter, timeout=5)
                print(f"  ✅ Ping/Pong successful")
                
                self.results[f"{name}_connect"] = "PASS"
                self.results[f"{name}_latency"] = connect_time
                
                # Close gracefully
                await ws.close()
                
        except asyncio.TimeoutError:
            print(f"  ❌ Connection timeout")
            self.results[f"{name}_connect"] = "TIMEOUT"
            
        except Exception as e:
            print(f"  ❌ Connection failed: {e}")
            self.results[f"{name}_connect"] = f"FAIL: {e}"
    
    async def test_subscribe(self, name: str, url: str):
        """Test WebSocket subscription."""
        try:
            print(f"  Testing subscription...")
            
            ssl_context = ssl.create_default_context()
            
            async with websockets.connect(
                url,
                ssl=ssl_context,
                ping_interval=None,
                close_timeout=10
            ) as ws:
                # Subscribe to ticker channel (public, no auth needed)
                subscribe_msg = {
                    "type": "subscribe",
                    "channels": ["ticker"],
                    "product_ids": ["BTC-USD"]
                }
                
                await ws.send(json.dumps(subscribe_msg))
                print(f"  📤 Sent subscribe message")
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=5)
                    data = json.loads(response)
                    
                    if data.get("type") == "subscriptions":
                        print(f"  ✅ Subscription confirmed")
                        print(f"     Channels: {data.get('channels', [])}")
                        self.results[f"{name}_subscribe"] = "PASS"
                    else:
                        print(f"  ⚠️  Unexpected response: {data.get('type')}")
                        self.results[f"{name}_subscribe"] = "UNEXPECTED"
                        
                except asyncio.TimeoutError:
                    print(f"  ❌ No response to subscription")
                    self.results[f"{name}_subscribe"] = "NO_RESPONSE"
                    
                # Wait for a ticker message
                try:
                    ticker = await asyncio.wait_for(ws.recv(), timeout=5)
                    ticker_data = json.loads(ticker)
                    if ticker_data.get("type") == "ticker":
                        print(f"  ✅ Received ticker data")
                        print(f"     Price: ${ticker_data.get('price', 'N/A')}")
                        self.results[f"{name}_data"] = "PASS"
                except asyncio.TimeoutError:
                    print(f"  ⚠️  No ticker data received")
                    self.results[f"{name}_data"] = "NO_DATA"
                
                await ws.close()
                
        except Exception as e:
            print(f"  ❌ Subscribe test failed: {e}")
            self.results[f"{name}_subscribe"] = f"FAIL: {e}"
    
    async def test_authenticated_channel(self):
        """Test authenticated WebSocket channels (requires JWT)."""
        print("\n🔐 Testing Authenticated Channels")
        print("-"*40)
        
        try:
            # Check if we have JWT capability
            cdp_key = os.getenv("COINBASE_CDP_API_KEY")
            private_key_path = os.getenv("COINBASE_CDP_PRIVATE_KEY_PATH")
            private_key = os.getenv("COINBASE_CDP_PRIVATE_KEY")
            
            if not cdp_key:
                print("  ⚠️  CDP API key not configured")
                self.results["auth_channel"] = "NO_KEY"
                return
            
            if not (private_key_path or private_key):
                print("  ⚠️  Private key not configured")
                self.results["auth_channel"] = "NO_PRIVATE_KEY"
                return
            
            # Try to generate JWT
            from bot_v2.features.brokerages.coinbase.cdp_auth_v2 import CDPAuthV2
            
            private_key_pem = private_key or (open(private_key_path).read() if private_key_path else None)
            if not private_key_pem:
                raise ValueError("Private key not found")

            auth = CDPAuthV2(
                api_key_name=cdp_key,
                private_key_pem=private_key_pem
            )
            jwt_token = auth.generate_jwt(method="GET", path="/api/v3/brokerage/accounts")
            
            print(f"  ✅ JWT token generated")
            
            # Connect and authenticate
            url = "wss://advanced-trade-ws.coinbase.com"
            ssl_context = ssl.create_default_context()
            
            async with websockets.connect(url, ssl=ssl_context) as ws:
                # Send authenticated subscribe
                auth_msg = {
                    "type": "subscribe",
                    "channel": "user",
                    "jwt": jwt_token,
                    "product_ids": ["BTC-PERP"] if self.sandbox else ["BTC-USD"]
                }
                
                await ws.send(json.dumps(auth_msg))
                print(f"  📤 Sent authenticated subscribe")
                
                # Wait for response
                response = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(response)
                
                if data.get("type") == "subscriptions":
                    print(f"  ✅ Authenticated channel subscribed")
                    self.results["auth_channel"] = "PASS"
                else:
                    print(f"  ❌ Auth failed: {data}")
                    self.results["auth_channel"] = "AUTH_FAIL"
                
                await ws.close()
                
        except ImportError:
            print("  ⚠️  Auth module not available")
            self.results["auth_channel"] = "NO_MODULE"
        except Exception as e:
            print(f"  ❌ Auth channel test failed: {e}")
            self.results["auth_channel"] = f"FAIL: {e}"
    
    def generate_report(self):
        """Generate connectivity report."""
        print("\n" + "="*60)
        print("📊 WEBSOCKET PROBE SUMMARY")
        print("="*60)
        
        # Connectivity results
        print("\n🌐 Connectivity Results:")
        for key, value in self.results.items():
            if "_connect" in key:
                endpoint = key.replace("_connect", "")
                status = "✅" if value == "PASS" else "❌"
                print(f"  {status} {endpoint}: {value}")
                
                # Show latency if available
                latency_key = f"{endpoint}_latency"
                if latency_key in self.results:
                    print(f"     Latency: {self.results[latency_key]:.0f}ms")
        
        # Subscription results
        print("\n📡 Subscription Results:")
        for key, value in self.results.items():
            if "_subscribe" in key:
                endpoint = key.replace("_subscribe", "")
                status = "✅" if value == "PASS" else "❌"
                print(f"  {status} {endpoint}: {value}")
        
        # Data flow results
        print("\n📈 Data Flow Results:")
        for key, value in self.results.items():
            if "_data" in key:
                endpoint = key.replace("_data", "")
                status = "✅" if value == "PASS" else "⚠️"
                print(f"  {status} {endpoint}: {value}")
        
        # Authentication
        if "auth_channel" in self.results:
            print("\n🔐 Authentication:")
            auth_result = self.results["auth_channel"]
            status = "✅" if auth_result == "PASS" else "❌"
            print(f"  {status} Authenticated channels: {auth_result}")
        
        # Overall assessment
        print("\n🎯 Overall Assessment:")
        passed = sum(1 for v in self.results.values() if v == "PASS")
        total = len(self.results)
        
        if passed == total:
            print("  🟢 All WebSocket tests PASSED")
        elif passed > total / 2:
            print("  🟡 Partial connectivity - review failures")
        else:
            print("  🔴 WebSocket connectivity issues detected")
        
        # Recommendations
        print("\n💡 Recommendations:")
        if any("TIMEOUT" in str(v) for v in self.results.values()):
            print("  - Check firewall settings for WebSocket (port 443)")
            print("  - Verify no proxy interference")
            print("  - Test from different network")
        
        if any("NO_RESPONSE" in str(v) for v in self.results.values()):
            print("  - Verify correct endpoint URLs")
            print("  - Check if API changes occurred")
        
        if self.results.get("auth_channel") not in ["PASS", None]:
            print("  - Verify JWT configuration")
            print("  - Check CDP API key and private key")
            print("  - Ensure derivatives are enabled")
        
        # Save report
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": "sandbox" if self.sandbox else "production",
            "egress_ip": os.popen("curl -s ifconfig.me 2>/dev/null").read().strip() or "unknown",
            "results": self.results,
            "staleness_threshold_ms": 5000,
            "reconnect_count": self.results.get("reconnect_count", 0),
            "auth_success": self.results.get("auth_channel") == "PASS",
            "summary": {
                "passed": passed,
                "total": total,
                "success_rate": (passed / total * 100) if total > 0 else 0
            }
        }
        
        # Save to preflight directory
        report_dir = "docs/ops/preflight"
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, "ws_probe_report.json")
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Report saved to: {report_file}")


async def main():
    """Run WebSocket probe."""
    import argparse
    
    parser = argparse.ArgumentParser(description="WebSocket Connectivity Probe")
    parser.add_argument("--auth", action="store_true",
                       help="Test authenticated channels")
    parser.add_argument("--sandbox", action="store_true",
                       help="Test sandbox endpoints")
    
    args = parser.parse_args()
    
    if args.sandbox:
        os.environ["COINBASE_SANDBOX"] = "1"
    
    probe = WebSocketProbe()
    await probe.run_all_tests()
    
    if args.auth:
        await probe.test_authenticated_channel()


if __name__ == "__main__":
    asyncio.run(main())
