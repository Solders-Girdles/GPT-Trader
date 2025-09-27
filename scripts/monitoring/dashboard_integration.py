#!/usr/bin/env python3
"""
Dashboard Integration for Canary Monitoring

Sends real-time metrics to dashboard endpoints for visualization.
Supports both push (HTTP POST) and pull (metrics endpoint) modes.

Usage:
  from scripts.monitoring.dashboard_integration import DashboardClient
  
  client = DashboardClient()
  await client.send_metrics(metrics_dict)
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    name: str
    value: float
    tags: Dict[str, str] = None
    
    def to_dict(self) -> dict:
        d = {
            "timestamp": self.timestamp.isoformat(),
            "name": self.name,
            "value": self.value
        }
        if self.tags:
            d["tags"] = self.tags
        return d


@dataclass
class Alert:
    """Alert/violation for dashboard"""
    timestamp: datetime
    severity: str  # info, warning, critical
    message: str
    component: str
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "message": self.message,
            "component": self.component
        }


class DashboardClient:
    """Client for sending metrics to monitoring dashboard"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("DASHBOARD_URL", "http://localhost:8080")
        self.session = None
        self.metrics_buffer: List[MetricPoint] = []
        self.alerts_buffer: List[Alert] = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def send_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Send metrics batch to dashboard"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            # Convert to metric points
            points = []
            timestamp = datetime.now(timezone.utc)
            
            # Position metrics
            for symbol, pos_data in metrics.get("positions", {}).items():
                points.append(MetricPoint(
                    timestamp=timestamp,
                    name="position.size",
                    value=pos_data.get("size", 0),
                    tags={"symbol": symbol}
                ))
                points.append(MetricPoint(
                    timestamp=timestamp,
                    name="position.pnl",
                    value=pos_data.get("unrealized_pnl", 0),
                    tags={"symbol": symbol}
                ))
            
            # Summary metrics
            points.extend([
                MetricPoint(timestamp, "total.pnl", metrics.get("total_pnl", 0)),
                MetricPoint(timestamp, "orders.placed", metrics.get("orders_placed", 0)),
                MetricPoint(timestamp, "orders.filled", metrics.get("orders_filled", 0)),
                MetricPoint(timestamp, "max.drawdown", metrics.get("max_drawdown", 0)),
            ])
            
            # Latency metrics
            if latencies := metrics.get("api_latencies", []):
                avg_latency = sum(latencies) / len(latencies)
                points.append(MetricPoint(timestamp, "api.latency.avg", avg_latency))
                points.append(MetricPoint(timestamp, "api.latency.max", max(latencies)))
            
            # Send to dashboard
            payload = {
                "metrics": [p.to_dict() for p in points],
                "timestamp": timestamp.isoformat()
            }
            
            async with self.session.post(
                f"{self.base_url}/api/metrics",
                json=payload,
                timeout=5
            ) as resp:
                return resp.status == 200
                
        except Exception as e:
            print(f"⚠️ Dashboard send failed: {e}")
            return False
    
    async def send_alert(self, severity: str, message: str, component: str = "canary") -> bool:
        """Send alert to dashboard"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            alert = Alert(
                timestamp=datetime.now(timezone.utc),
                severity=severity,
                message=message,
                component=component
            )
            
            async with self.session.post(
                f"{self.base_url}/api/alerts",
                json=alert.to_dict(),
                timeout=5
            ) as resp:
                return resp.status == 200
                
        except Exception as e:
            print(f"⚠️ Alert send failed: {e}")
            return False
    
    async def get_health(self) -> Dict[str, Any]:
        """Get dashboard health status"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.get(
                f"{self.base_url}/health",
                timeout=5
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"status": "unhealthy", "code": resp.status}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}


class MetricsExporter:
    """Export metrics in Prometheus format for pull-based monitoring"""
    
    def __init__(self, port: int = 9090):
        self.port = port
        self.metrics = {}
        self.server = None
        
    async def start(self):
        """Start metrics HTTP server"""
        from aiohttp import web
        
        app = web.Application()
        app.router.add_get("/metrics", self.handle_metrics)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        
        print(f"📊 Metrics exporter running on http://0.0.0.0:{self.port}/metrics")
        
    async def handle_metrics(self, request):
        """Handle /metrics endpoint"""
        from aiohttp import web
        
        # Format as Prometheus metrics
        lines = []
        for name, value in self.metrics.items():
            # Convert metric name to Prometheus format
            prom_name = name.replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE {prom_name} gauge")
            lines.append(f"{prom_name} {value}")
        
        return web.Response(text="\n".join(lines), content_type="text/plain")
    
    def update_metric(self, name: str, value: float):
        """Update a metric value"""
        self.metrics[name] = value
    
    def update_metrics_batch(self, metrics: Dict[str, Any]):
        """Update multiple metrics"""
        # Flatten nested metrics
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        self.metrics[f"{key}_{sub_key}"] = sub_value
            elif isinstance(value, (int, float)):
                self.metrics[key] = value


async def demo():
    """Demo dashboard integration"""
    async with DashboardClient() as client:
        # Check health
        health = await client.get_health()
        print(f"Dashboard health: {health}")
        
        # Send test metrics
        test_metrics = {
            "positions": {
                "BTC-PERP": {"size": 0.001, "unrealized_pnl": 10.50},
                "ETH-PERP": {"size": 0.01, "unrealized_pnl": -5.25}
            },
            "total_pnl": 5.25,
            "orders_placed": 10,
            "orders_filled": 8,
            "api_latencies": [50, 75, 100, 125]
        }
        
        success = await client.send_metrics(test_metrics)
        print(f"Metrics sent: {success}")
        
        # Send test alert
        alert_sent = await client.send_alert(
            severity="warning",
            message="Test alert from canary monitor",
            component="test"
        )
        print(f"Alert sent: {alert_sent}")


if __name__ == "__main__":
    asyncio.run(demo())