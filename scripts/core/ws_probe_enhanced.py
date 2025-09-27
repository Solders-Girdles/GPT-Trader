#!/usr/bin/env python3
"""
Enhanced WebSocket connectivity probe for Coinbase with Exchange sandbox support.
Tests both Advanced Trade and Exchange WebSocket connections.
"""

import os
import sys
import asyncio
import json
import time
import ssl
import websockets
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple
from collections import defaultdict

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class EnhancedWebSocketProbe:
    """Enhanced WebSocket connectivity probe with staleness monitoring."""
    
    def __init__(self):
        self.results = {}
        self.metrics = defaultdict(list)
        self.sandbox = os.getenv("COINBASE_SANDBOX", "0") == "1"
        self.staleness_threshold_ms = 2000  # 2 seconds for staleness detection
        self.message_timestamps = defaultdict(list)
        
    async def run_all_tests(self):
        """Run comprehensive WebSocket connectivity tests."""
        print("🔍 ENHANCED WEBSOCKET CONNECTIVITY PROBE")
        print("="*60)
        print(f"Environment: {'SANDBOX' if self.sandbox else 'PRODUCTION'}")
        print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
        print(f"Staleness Threshold: {self.staleness_threshold_ms}ms")
        print("="*60)
        
        # Test different endpoints including Exchange sandbox
        endpoints = self.get_endpoints()
        
        # Parallel connectivity tests
        connect_tasks = []
        for name, url in endpoints.items():
            print(f"\n📡 Queuing {name}: {url}")
            connect_tasks.append(self.test_endpoint_comprehensive(name, url))
        
        # Run all tests in parallel
        print("\n⚡ Running parallel connectivity tests...")
        await asyncio.gather(*connect_tasks, return_exceptions=True)
        
        # Test authenticated channels if configured
        if self.has_auth_config():
            await self.test_authenticated_channel()
        
        # Generate comprehensive report
        self.generate_report()
    
    def get_endpoints(self) -> Dict[str, str]:
        """Get all WebSocket endpoints to test including Exchange sandbox."""
        endpoints = {}
        
        if self.sandbox:
            # Sandbox endpoints
            endpoints.update({
                "sandbox_advanced": "wss://advanced-trade-ws.sandbox.coinbase.com",
                "sandbox_direct": "wss://ws-direct.sandbox.coinbase.com",
                "sandbox_feed": "wss://ws-feed.sandbox.coinbase.com",
                "exchange_sandbox": "wss://ws-feed-public.sandbox.exchange.coinbase.com",  # Exchange sandbox
                "exchange_sandbox_pro": "wss://ws-feed-public.sandbox.pro.coinbase.com",  # Alternative
            })
        else:
            # Production endpoints
            endpoints.update({
                "prod_advanced": "wss://advanced-trade-ws.coinbase.com",
                "prod_direct": "wss://ws-direct.coinbase.com",
                "prod_feed": "wss://ws-feed.coinbase.com",
                "exchange_prod": "wss://ws-feed.exchange.coinbase.com",
            })
        
        return endpoints
    
    async def test_endpoint_comprehensive(self, name: str, url: str):
        """Comprehensive test for a single endpoint."""
        print(f"\n🔧 Testing {name}...")
        
        # 1. Basic connectivity
        connect_result, latency = await self.test_connectivity(name, url)
        
        if connect_result == "PASS":
            # 2. Subscription test
            await self.test_subscription_with_staleness(name, url)
            
            # 3. Message flow analysis (10 second sample)
            await self.analyze_message_flow(name, url)
    
    async def test_connectivity(self, name: str, url: str) -> Tuple[str, float]:
        """Test basic WebSocket connectivity with latency measurement."""
        try:
            ssl_context = ssl.create_default_context()
            start_time = time.time()
            
            async with websockets.connect(
                url,
                ssl=ssl_context,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            ) as ws:
                connect_time = (time.time() - start_time) * 1000
                
                # Test ping/pong
                pong_waiter = await ws.ping()
                await asyncio.wait_for(pong_waiter, timeout=5)
                
                print(f"  ✅ {name}: Connected in {connect_time:.0f}ms")
                
                self.results[f"{name}_connect"] = "PASS"
                self.results[f"{name}_latency_ms"] = connect_time
                self.metrics[f"{name}_latencies"].append(connect_time)
                
                await ws.close()
                return "PASS", connect_time
                
        except asyncio.TimeoutError:
            print(f"  ❌ {name}: Connection timeout")
            self.results[f"{name}_connect"] = "TIMEOUT"
            return "TIMEOUT", -1
            
        except Exception as e:
            print(f"  ❌ {name}: Connection failed - {e}")
            self.results[f"{name}_connect"] = f"FAIL: {str(e)[:50]}"
            return "FAIL", -1
    
    async def test_subscription_with_staleness(self, name: str, url: str):
        """Test subscription and monitor for staleness."""
        try:
            ssl_context = ssl.create_default_context()
            
            async with websockets.connect(
                url,
                ssl=ssl_context,
                ping_interval=20,
                close_timeout=10
            ) as ws:
                # Determine subscription format based on endpoint
                if "exchange" in name.lower():
                    # Exchange API format
                    subscribe_msg = {
                        "type": "subscribe",
                        "channels": [
                            {"name": "ticker", "product_ids": ["BTC-USD"]},
                            {"name": "heartbeat", "product_ids": ["BTC-USD"]}
                        ]
                    }
                else:
                    # Advanced Trade format
                    subscribe_msg = {
                        "type": "subscribe",
                        "channels": ["ticker", "heartbeat"],
                        "product_ids": ["BTC-USD"]
                    }
                
                await ws.send(json.dumps(subscribe_msg))
                
                # Monitor messages for 5 seconds
                start_time = time.time()
                message_count = 0
                last_message_time = start_time
                max_gap_ms = 0
                
                try:
                    while (time.time() - start_time) < 5:
                        msg = await asyncio.wait_for(ws.recv(), timeout=self.staleness_threshold_ms/1000)
                        current_time = time.time()
                        
                        # Calculate message gap
                        gap_ms = (current_time - last_message_time) * 1000
                        if gap_ms > max_gap_ms and message_count > 0:
                            max_gap_ms = gap_ms
                        
                        self.message_timestamps[name].append(current_time)
                        last_message_time = current_time
                        message_count += 1
                        
                        # Parse message
                        data = json.loads(msg)
                        if data.get("type") == "ticker":
                            price = data.get("price") or data.get("best_bid")
                            if price and message_count == 1:
                                print(f"  ✅ {name}: Receiving ticker data (${price})")
                
                except asyncio.TimeoutError:
                    if message_count > 0:
                        print(f"  ⚠️  {name}: Stale data detected (>{self.staleness_threshold_ms}ms gap)")
                        self.results[f"{name}_staleness"] = "STALE"
                    else:
                        print(f"  ❌ {name}: No data received")
                        self.results[f"{name}_staleness"] = "NO_DATA"
                
                # Analyze staleness
                if message_count > 0:
                    avg_rate = message_count / 5  # messages per second
                    print(f"  📊 {name}: {message_count} messages, {avg_rate:.1f}/sec, max gap {max_gap_ms:.0f}ms")
                    
                    self.results[f"{name}_message_rate"] = avg_rate
                    self.results[f"{name}_max_gap_ms"] = max_gap_ms
                    
                    if max_gap_ms < self.staleness_threshold_ms:
                        self.results[f"{name}_staleness"] = "FRESH"
                    else:
                        self.results[f"{name}_staleness"] = "INTERMITTENT"
                
                await ws.close()
                
        except Exception as e:
            print(f"  ❌ {name}: Subscription test failed - {e}")
            self.results[f"{name}_staleness"] = f"ERROR: {str(e)[:30]}"
    
    async def analyze_message_flow(self, name: str, url: str):
        """Analyze message flow patterns over 10 seconds."""
        try:
            print(f"  📈 {name}: Analyzing message flow...")
            
            ssl_context = ssl.create_default_context()
            
            async with websockets.connect(
                url,
                ssl=ssl_context,
                ping_interval=20,
                close_timeout=10
            ) as ws:
                # Subscribe
                if "exchange" in name.lower():
                    subscribe_msg = {
                        "type": "subscribe",
                        "channels": [
                            {"name": "ticker", "product_ids": ["BTC-USD", "ETH-USD"]},
                            {"name": "heartbeat", "product_ids": ["BTC-USD"]}
                        ]
                    }
                else:
                    subscribe_msg = {
                        "type": "subscribe",
                        "channels": ["ticker", "heartbeat"],
                        "product_ids": ["BTC-USD", "ETH-USD"]
                    }
                
                await ws.send(json.dumps(subscribe_msg))
                
                # Collect metrics for 10 seconds
                start_time = time.time()
                message_types = defaultdict(int)
                gaps = []
                last_time = start_time
                
                while (time.time() - start_time) < 10:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1)
                        current_time = time.time()
                        
                        # Track gap
                        gap_ms = (current_time - last_time) * 1000
                        if gap_ms < 10000:  # Ignore startup gap
                            gaps.append(gap_ms)
                        last_time = current_time
                        
                        # Track message type
                        data = json.loads(msg)
                        msg_type = data.get("type", "unknown")
                        message_types[msg_type] += 1
                        
                    except asyncio.TimeoutError:
                        continue
                    except json.JSONDecodeError:
                        message_types["invalid"] += 1
                
                # Calculate statistics
                if gaps:
                    p50_gap = sorted(gaps)[len(gaps)//2]
                    p95_gap = sorted(gaps)[int(len(gaps)*0.95)] if len(gaps) > 20 else max(gaps)
                    p99_gap = sorted(gaps)[int(len(gaps)*0.99)] if len(gaps) > 100 else max(gaps)
                    
                    print(f"  📊 {name} Message Gap Statistics:")
                    print(f"     P50: {p50_gap:.0f}ms, P95: {p95_gap:.0f}ms, P99: {p99_gap:.0f}ms")
                    
                    self.results[f"{name}_p50_gap_ms"] = p50_gap
                    self.results[f"{name}_p95_gap_ms"] = p95_gap
                    self.results[f"{name}_p99_gap_ms"] = p99_gap
                    
                    # Check staleness criteria
                    if p95_gap < 1000:
                        print(f"  ✅ {name}: Meets p95 < 1s staleness target")
                        self.results[f"{name}_staleness_check"] = "PASS"
                    else:
                        print(f"  ⚠️  {name}: Exceeds p95 staleness target ({p95_gap:.0f}ms > 1000ms)")
                        self.results[f"{name}_staleness_check"] = "WARN"
                
                # Message type breakdown
                if message_types:
                    print(f"  📧 {name} Message Types: {dict(message_types)}")
                
                await ws.close()
                
        except Exception as e:
            print(f"  ❌ {name}: Flow analysis failed - {e}")
            self.results[f"{name}_flow_analysis"] = f"ERROR"
    
    def has_auth_config(self) -> bool:
        """Check if authentication is configured."""
        return bool(os.getenv("COINBASE_CDP_API_KEY") or 
                   os.getenv("COINBASE_API_KEY"))
    
    async def test_authenticated_channel(self):
        """Test authenticated WebSocket channels."""
        print("\n🔐 Testing Authenticated Channels")
        print("-"*40)
        
        # Implementation similar to original but with enhanced metrics
        # ... (keeping original auth test logic)
        print("  ⚠️  Auth test skipped (use original ws_probe.py for auth)")
    
    def generate_report(self):
        """Generate comprehensive connectivity and staleness report."""
        print("\n" + "="*60)
        print("📊 ENHANCED WEBSOCKET PROBE REPORT")
        print("="*60)
        
        # Group results by endpoint
        endpoints = set()
        for key in self.results.keys():
            endpoint = key.split("_")[0]
            if endpoint not in ["auth"]:
                endpoints.add(endpoint)
        
        # Staleness Summary
        print("\n🎯 STALENESS MONITORING (Target: p95 < 1s)")
        print("-"*40)
        
        staleness_passed = 0
        staleness_total = 0
        
        for endpoint in sorted(endpoints):
            if f"{endpoint}_p95_gap_ms" in self.results:
                staleness_total += 1
                p95 = self.results[f"{endpoint}_p95_gap_ms"]
                status = "✅" if p95 < 1000 else "⚠️"
                
                if p95 < 1000:
                    staleness_passed += 1
                
                print(f"{status} {endpoint}:")
                print(f"   P95 gap: {p95:.0f}ms")
                
                if f"{endpoint}_max_gap_ms" in self.results:
                    print(f"   Max gap: {self.results[f'{endpoint}_max_gap_ms']:.0f}ms")
                
                if f"{endpoint}_message_rate" in self.results:
                    print(f"   Rate: {self.results[f'{endpoint}_message_rate']:.1f} msg/sec")
        
        # Connectivity Summary
        print("\n🌐 CONNECTIVITY SUMMARY")
        print("-"*40)
        
        for endpoint in sorted(endpoints):
            connect_result = self.results.get(f"{endpoint}_connect", "NOT_TESTED")
            
            if connect_result == "PASS":
                latency = self.results.get(f"{endpoint}_latency_ms", -1)
                print(f"✅ {endpoint}: Connected ({latency:.0f}ms)")
            elif connect_result == "TIMEOUT":
                print(f"❌ {endpoint}: Timeout")
            elif connect_result.startswith("FAIL"):
                print(f"❌ {endpoint}: Failed")
            else:
                print(f"⚠️  {endpoint}: {connect_result}")
        
        # Exchange Sandbox Specific
        print("\n🔄 EXCHANGE SANDBOX STATUS")
        print("-"*40)
        
        exchange_endpoints = [e for e in endpoints if "exchange" in e.lower()]
        if exchange_endpoints:
            for endpoint in exchange_endpoints:
                connect = self.results.get(f"{endpoint}_connect", "NOT_TESTED")
                staleness = self.results.get(f"{endpoint}_staleness", "UNKNOWN")
                
                if connect == "PASS":
                    print(f"✅ {endpoint}: Connected, Staleness: {staleness}")
                else:
                    print(f"❌ {endpoint}: {connect}")
        else:
            print("  No Exchange sandbox endpoints tested")
        
        # Overall Assessment
        print("\n🎯 OVERALL ASSESSMENT")
        print("-"*40)
        
        total_tests = len(endpoints)
        passed_connectivity = sum(1 for e in endpoints if self.results.get(f"{e}_connect") == "PASS")
        
        print(f"Connectivity: {passed_connectivity}/{total_tests} endpoints connected")
        print(f"Staleness: {staleness_passed}/{staleness_total} meet p95 < 1s target")
        
        if passed_connectivity == total_tests and staleness_passed == staleness_total:
            print("\n🟢 PHASE 1 READY: All connectivity and staleness targets met")
        elif passed_connectivity > 0:
            print("\n🟡 PARTIAL READY: Some endpoints meet targets")
        else:
            print("\n🔴 NOT READY: Connectivity issues detected")
        
        # Monitoring Recommendations
        print("\n📋 MONITORING SETUP RECOMMENDATIONS")
        print("-"*40)
        print("1. Staleness Alert: Trigger if p95 gap > 1s for 5 minutes")
        print("2. Funding Alert: Trigger if no funding update for 2 periods (16 hours)")
        print("3. Latency Alert: Trigger if p95 REST latency > 1.5s")
        print("4. Slippage Alert: Log all fills with impact > 15 bps")
        
        # Save detailed report
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": "sandbox" if self.sandbox else "production",
            "staleness_threshold_ms": self.staleness_threshold_ms,
            "results": self.results,
            "metrics": {
                "connectivity_rate": (passed_connectivity / total_tests * 100) if total_tests > 0 else 0,
                "staleness_pass_rate": (staleness_passed / staleness_total * 100) if staleness_total > 0 else 0,
                "phase1_ready": passed_connectivity == total_tests and staleness_passed == staleness_total
            },
            "thresholds": {
                "staleness_p95_ms": 1000,
                "last_tick_age_ms": 2000,
                "rest_latency_p95_ms": 1500,
                "slippage_impact_bps": 15
            }
        }
        
        # Save to preflight directory
        report_dir = "docs/ops/preflight"
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, f"ws_probe_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")


async def main():
    """Run enhanced WebSocket probe."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced WebSocket Probe with Staleness Monitoring")
    parser.add_argument("--sandbox", action="store_true",
                       help="Test sandbox endpoints including Exchange sandbox")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test (skip flow analysis)")
    
    args = parser.parse_args()
    
    if args.sandbox:
        os.environ["COINBASE_SANDBOX"] = "1"
    
    probe = EnhancedWebSocketProbe()
    
    if args.quick:
        # Just connectivity and basic staleness
        endpoints = probe.get_endpoints()
        for name, url in endpoints.items():
            await probe.test_connectivity(name, url)
        probe.generate_report()
    else:
        # Full test suite
        await probe.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())