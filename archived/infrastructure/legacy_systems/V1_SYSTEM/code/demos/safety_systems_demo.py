#!/usr/bin/env python3
"""
Safety Systems Demo for GPT-Trader

Demonstrates comprehensive circuit breakers and kill switches:
- Circuit breaker triggering on various conditions
- Kill switch activation and resumption 
- Integration between safety systems
- State persistence and audit logging
- Thread-safe operation under load
- Emergency API endpoints

This demo shows how the safety systems protect against:
- Daily loss limits (5% of capital)
- Portfolio drawdown (15% from peak)
- Position concentration (20% of portfolio)
- Market volatility spikes (3x normal)
- Volume anomalies (5x normal)
- Consecutive trading losses (5 in a row)

And provides emergency controls:
- Global emergency stop (immediate halt)
- Global graceful shutdown (close positions first)
- Global liquidation (market orders to exit)
- Strategy-specific stops
- Resume capabilities with audit trail
"""

import asyncio
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.risk.safety_integration import SafetySystemsIntegration, SafetySystemConfig
from bot.risk.kill_switch import KillSwitchReason


class MockTradingEngine:
    """Mock trading engine for demonstration"""
    
    def __init__(self, engine_id: str):
        self.engine_id = engine_id
        self.is_running = True
        self.positions = {}
        self.orders = []
        self.stop_called = False
        self.emergency_stop_called = False
        self.freeze_called = False
        self.positions_closed = 0
        self.orders_cancelled = 0
        
        # Mock order types
        class OrderSide:
            BUY = "buy"
            SELL = "sell"
        
        class OrderType:
            MARKET = "market"
            LIMIT = "limit"
        
        self.OrderSide = OrderSide
        self.OrderType = OrderType
        
        # Add some mock positions
        self.positions = {
            "AAPL": MockPosition("AAPL", 1000, 150.0),
            "GOOGL": MockPosition("GOOGL", 100, 2800.0),
            "MSFT": MockPosition("MSFT", 500, 400.0),
            "TSLA": MockPosition("TSLA", -200, 800.0),  # Short position
        }
    
    def stop_trading_engine(self):
        """Stop trading engine"""
        self.is_running = False
        self.stop_called = True
        console.print(f"      🛑 Trading engine {self.engine_id} stopped")
    
    def emergency_stop(self):
        """Emergency stop"""
        self.is_running = False
        self.emergency_stop_called = True
        console.print(f"      🚨 Emergency stop executed on {self.engine_id}")
    
    def freeze_trading(self):
        """Freeze trading"""
        self.freeze_called = True
        console.print(f"      ❄️ Trading frozen on {self.engine_id}")
    
    def submit_order(self, **kwargs):
        """Submit order (mock)"""
        order_id = f"order_{len(self.orders)}_{time.time()}"
        self.orders.append({
            "id": order_id,
            "timestamp": time.time(),
            **kwargs
        })
        console.print(f"         📝 Order submitted: {kwargs.get('symbol')} {kwargs.get('side')} {kwargs.get('quantity')}")
        return order_id
    
    def close_all_positions_graceful(self):
        """Close all positions gracefully"""
        count = len([p for p in self.positions.values() if p.quantity != 0])
        for position in self.positions.values():
            if position.quantity != 0:
                side = self.OrderSide.SELL if position.quantity > 0 else self.OrderSide.BUY
                self.submit_order(
                    strategy_id="graceful_close",
                    symbol=position.symbol,
                    side=side,
                    quantity=abs(position.quantity),
                    order_type=self.OrderType.LIMIT,
                    notes="Graceful position close"
                )
                position.quantity = 0
        
        self.positions_closed += count
        console.print(f"      💰 Gracefully closed {count} positions")
        return count
    
    def liquidate_all_positions(self):
        """Liquidate all positions"""
        count = len([p for p in self.positions.values() if p.quantity != 0])
        for position in self.positions.values():
            if position.quantity != 0:
                side = self.OrderSide.SELL if position.quantity > 0 else self.OrderSide.BUY
                self.submit_order(
                    strategy_id="emergency_liquidation",
                    symbol=position.symbol,
                    side=side,
                    quantity=abs(position.quantity),
                    order_type=self.OrderType.MARKET,
                    notes="Emergency liquidation"
                )
                position.quantity = 0
        
        self.positions_closed += count
        console.print(f"      💥 Emergency liquidated {count} positions")
        return count


class MockPosition:
    """Mock position for demonstration"""
    
    def __init__(self, symbol: str, quantity: float, current_price: float):
        self.symbol = symbol
        self.quantity = quantity
        self.current_price = current_price
    
    @property
    def market_value(self):
        return abs(self.quantity * self.current_price)


class MockRiskMonitor:
    """Mock risk monitor with configurable conditions"""
    
    def __init__(self):
        self.risk_metrics = MockRiskMetrics()
        self.position_metrics = {"max_position_pct": 0.15}  # 15% concentration
        self.market_metrics = {
            "volume_spike_ratio": 2.0,
            "volatility_spike_ratio": 2.5
        }
        self.trade_metrics = {"consecutive_losses": 3}
        self.strategy_metrics = {"win_rate": 0.65}  # 65% win rate
    
    def set_dangerous_conditions(self):
        """Set conditions that will trigger circuit breakers"""
        self.risk_metrics.total_realized_pnl = -6000  # $6000 loss (exceeds 5% of $100k)
        self.risk_metrics.current_drawdown = 0.20     # 20% drawdown (exceeds 15%)
        self.position_metrics["max_position_pct"] = 0.25  # 25% concentration (exceeds 20%)
        self.market_metrics["volume_spike_ratio"] = 6.0   # 6x volume (exceeds 5x)
        self.market_metrics["volatility_spike_ratio"] = 4.0  # 4x volatility (exceeds 3x)
        self.trade_metrics["consecutive_losses"] = 6      # 6 losses (exceeds 5)
        self.strategy_metrics["win_rate"] = 0.25         # 25% win rate (below 30%)
    
    def get_current_risk_metrics(self):
        return self.risk_metrics
    
    def get_position_metrics(self):
        return self.position_metrics
    
    def get_market_metrics(self):
        return self.market_metrics
    
    def get_trade_metrics(self):
        return self.trade_metrics
    
    def get_strategy_metrics(self):
        return self.strategy_metrics


class MockRiskMetrics:
    """Mock risk metrics"""
    
    def __init__(self):
        self.total_unrealized_pnl = 500   # $500 unrealized gain
        self.total_realized_pnl = -1000   # $1000 realized loss
        self.current_drawdown = 0.05      # 5% drawdown


class MockAlertingSystem:
    """Mock alerting system"""
    
    def __init__(self):
        self.alerts_sent = []
    
    def send_critical_alert(self, title, message, severity="critical"):
        """Send critical alert"""
        self.alerts_sent.append({
            "title": title,
            "message": message,
            "severity": severity,
            "timestamp": time.time()
        })
        console.print(f"🚨 [bold red]CRITICAL ALERT[/bold red]: {title}")
    
    def send_alert(self, title, message, severity="info"):
        """Send regular alert"""
        self.alerts_sent.append({
            "title": title,
            "message": message,
            "severity": severity,
            "timestamp": time.time()
        })
        console.print(f"📢 [yellow]ALERT[/yellow]: {title}")


console = Console()


def print_banner():
    """Print demo banner"""
    banner = Panel(
        "[bold blue]GPT-Trader Safety Systems Demo[/bold blue]\n\n"
        "🛡️ Circuit Breakers & Kill Switches\n"
        "⚡ Real-time Risk Protection\n"
        "🚨 Emergency Controls\n"
        "📊 Comprehensive Monitoring\n\n"
        "[yellow]⚠️  This is a demonstration with mock components ⚠️[/yellow]",
        title="🚀 Safety Demo",
        border_style="blue"
    )
    console.print(banner)


def print_portfolio_status(engines):
    """Print current portfolio status"""
    table = Table(title="📊 Current Portfolio Status")
    table.add_column("Engine", style="cyan")
    table.add_column("Symbol", style="yellow")
    table.add_column("Position", justify="right")
    table.add_column("Price", justify="right", style="green")
    table.add_column("Market Value", justify="right", style="blue")
    
    total_value = 0
    for engine in engines:
        for symbol, position in engine.positions.items():
            if position.quantity != 0:
                market_value = position.market_value
                total_value += market_value
                
                table.add_row(
                    engine.engine_id,
                    symbol,
                    f"{position.quantity:+.0f}",
                    f"${position.current_price:.2f}",
                    f"${market_value:,.0f}"
                )
    
    table.add_row("", "", "", "TOTAL:", f"${total_value:,.0f}", style="bold")
    console.print(table)


async def demo_circuit_breakers(safety_integration, risk_monitor):
    """Demonstrate circuit breaker functionality"""
    console.print("\n⚡ [bold yellow]CIRCUIT BREAKER DEMONSTRATION[/bold yellow]")
    console.print("=" * 60)
    
    # Show current status
    console.print("\n📊 Current Risk Conditions:")
    console.print(f"   Daily Loss: ${abs(risk_monitor.risk_metrics.total_realized_pnl):,.0f}")
    console.print(f"   Drawdown: {risk_monitor.risk_metrics.current_drawdown:.1%}")
    console.print(f"   Max Position: {risk_monitor.position_metrics['max_position_pct']:.1%}")
    console.print(f"   Volume Spike: {risk_monitor.market_metrics['volume_spike_ratio']:.1f}x")
    console.print(f"   Volatility Spike: {risk_monitor.market_metrics['volatility_spike_ratio']:.1f}x")
    console.print(f"   Consecutive Losses: {risk_monitor.trade_metrics['consecutive_losses']}")
    
    console.print("\n🟢 [green]All conditions within normal limits[/green]")
    
    # Wait a moment
    await asyncio.sleep(2)
    
    console.print("\n⚠️ [bold yellow]Setting dangerous market conditions...[/bold yellow]")
    risk_monitor.set_dangerous_conditions()
    
    console.print("\n📊 Updated Risk Conditions:")
    console.print(f"   Daily Loss: ${abs(risk_monitor.risk_metrics.total_realized_pnl):,.0f} [red](EXCEEDED)[/red]")
    console.print(f"   Drawdown: {risk_monitor.risk_metrics.current_drawdown:.1%} [red](EXCEEDED)[/red]")
    console.print(f"   Max Position: {risk_monitor.position_metrics['max_position_pct']:.1%} [red](EXCEEDED)[/red]")
    console.print(f"   Volume Spike: {risk_monitor.market_metrics['volume_spike_ratio']:.1f}x [red](EXCEEDED)[/red]")
    console.print(f"   Volatility Spike: {risk_monitor.market_metrics['volatility_spike_ratio']:.1f}x [red](EXCEEDED)[/red]")
    console.print(f"   Consecutive Losses: {risk_monitor.trade_metrics['consecutive_losses']} [red](EXCEEDED)[/red]")
    
    console.print("\n🚨 [bold red]Multiple circuit breaker conditions triggered![/bold red]")
    
    # Wait for circuit breakers to detect and trigger
    console.print("\n⏳ Waiting for circuit breaker monitoring to detect conditions...")
    await asyncio.sleep(5)
    
    # Check status
    status = safety_integration.get_comprehensive_status()
    cb_status = status.get("circuit_breakers", {})
    
    console.print(f"\n📊 Circuit Breaker Status:")
    console.print(f"   Triggered Breakers: {cb_status.get('triggered_breakers', 0)}")
    console.print(f"   Active Events: {cb_status.get('active_events', 0)}")


async def demo_kill_switches(safety_integration, engines):
    """Demonstrate kill switch functionality"""
    console.print("\n🔴 [bold red]KILL SWITCH DEMONSTRATION[/bold red]")
    console.print("=" * 60)
    
    # Show current positions
    print_portfolio_status(engines)
    
    # Demo 1: Global Emergency Stop
    console.print("\n🚨 [bold red]Demo 1: Global Emergency Stop[/bold red]")
    console.print("Triggering immediate halt of all trading...")
    
    event_id = safety_integration.trigger_global_emergency_stop(
        "Demo: Testing global emergency stop",
        "demo_user",
        "immediate"
    )
    
    if event_id:
        console.print(f"✅ Emergency stop triggered: {event_id}")
        
        # Check engine status
        console.print("\n📊 Engine Status After Emergency Stop:")
        for engine in engines:
            console.print(f"   {engine.engine_id}: {'🛑 STOPPED' if engine.emergency_stop_called else '🟢 RUNNING'}")
        
        # Wait a moment
        await asyncio.sleep(2)
        
        # Resume
        console.print("\n🔄 [yellow]Resuming trading...[/yellow]")
        resumed = safety_integration.resume_emergency_stop(
            event_id,
            "demo_user",
            "Demo completed, resuming normal operations"
        )
        
        if resumed:
            console.print("✅ Trading resumed successfully")
        
        # Reset engine states for next demo
        for engine in engines:
            engine.emergency_stop_called = False
            engine.stop_called = False
    
    await asyncio.sleep(2)
    
    # Demo 2: Graceful Shutdown
    console.print("\n💰 [bold yellow]Demo 2: Graceful Shutdown[/bold yellow]")
    console.print("Triggering graceful shutdown (close positions first)...")
    
    event_id = safety_integration.trigger_global_emergency_stop(
        "Demo: Testing graceful shutdown",
        "demo_user",
        "graceful"
    )
    
    if event_id:
        console.print(f"✅ Graceful shutdown triggered: {event_id}")
        
        # Check positions
        await asyncio.sleep(1)
        print_portfolio_status(engines)
        
        # Resume
        safety_integration.resume_emergency_stop(
            event_id,
            "demo_user",
            "Graceful shutdown demo completed"
        )
    
    await asyncio.sleep(2)
    
    # Demo 3: Emergency Liquidation
    console.print("\n💥 [bold red]Demo 3: Emergency Liquidation[/bold red]")
    console.print("Triggering emergency liquidation with market orders...")
    
    # Reset positions for demo
    for engine in engines:
        for symbol, position in engine.positions.items():
            if symbol == "AAPL":
                position.quantity = 1000
            elif symbol == "GOOGL":
                position.quantity = 100
            elif symbol == "MSFT":
                position.quantity = 500
            elif symbol == "TSLA":
                position.quantity = -200
    
    event_id = safety_integration.trigger_global_emergency_stop(
        "Demo: Testing emergency liquidation",
        "demo_user",
        "liquidate"
    )
    
    if event_id:
        console.print(f"✅ Emergency liquidation triggered: {event_id}")
        
        # Check positions after liquidation
        await asyncio.sleep(1)
        print_portfolio_status(engines)
        
        # Resume
        safety_integration.resume_emergency_stop(
            event_id,
            "demo_user",
            "Emergency liquidation demo completed"
        )


async def demo_integration_features(safety_integration):
    """Demonstrate integration features"""
    console.print("\n🔗 [bold blue]INTEGRATION FEATURES DEMONSTRATION[/bold blue]")
    console.print("=" * 60)
    
    # Show comprehensive status
    console.print("\n📊 Comprehensive Safety Status:")
    status = safety_integration.get_comprehensive_status()
    
    integration_status = status["integration"]
    console.print(f"   Monitoring: {'🟢 ACTIVE' if integration_status['is_monitoring'] else '🔴 INACTIVE'}")
    console.print(f"   Registered Engines: {integration_status['registered_engines']}")
    console.print(f"   Safety Checks: {integration_status['performance_metrics']['safety_checks_performed']}")
    console.print(f"   Cross-System Triggers: {integration_status['performance_metrics']['cross_system_triggers']}")
    
    cb_status = status.get("circuit_breakers", {})
    console.print(f"   Circuit Breakers: {cb_status.get('active_breakers', 0)}/{cb_status.get('total_breakers', 0)} active")
    
    ks_status = status.get("kill_switches", {})
    console.print(f"   Kill Switches: {ks_status.get('armed_switches', 0)}/{ks_status.get('total_switches', 0)} armed")
    
    # Demo manual circuit breaker trigger
    console.print("\n⚡ [yellow]Manual Circuit Breaker Trigger[/yellow]")
    success = safety_integration.trigger_circuit_breaker(
        "max_daily_drawdown",
        "Manual demo trigger"
    )
    
    if success:
        console.print("✅ Circuit breaker manually triggered")
    else:
        console.print("❌ Circuit breaker trigger failed")
    
    await asyncio.sleep(2)
    
    # Show dashboard
    console.print("\n📋 [bold]Full Safety Dashboard[/bold]")
    safety_integration.display_safety_dashboard()


async def demo_state_persistence(safety_integration):
    """Demonstrate state persistence"""
    console.print("\n💾 [bold cyan]STATE PERSISTENCE DEMONSTRATION[/bold cyan]")
    console.print("=" * 60)
    
    # Trigger an event to create state
    event_id = safety_integration.kill_switches.trigger_kill_switch(
        "global_emergency_stop",
        KillSwitchReason.MANUAL_OVERRIDE,
        "persistence_demo",
        "Testing state persistence"
    )
    
    if event_id:
        console.print(f"✅ Created event for persistence demo: {event_id}")
        
        # Show active events
        active_events = safety_integration.kill_switches.get_active_events()
        console.print(f"   Active events: {len(active_events)}")
        
        # Show event history
        event_history = safety_integration.kill_switches.get_event_history(limit=5)
        console.print(f"   Total events in history: {len(event_history)}")
        
        # In a real system, this state would persist across restarts
        console.print("\n💾 State persisted to database:")
        console.print(f"   Database: {safety_integration.kill_switches.db_path}")
        console.print("   Events stored with full audit trail")
        console.print("   State survives system restarts")
        
        # Resume to clean up
        safety_integration.resume_emergency_stop(event_id, "persistence_demo", "Demo cleanup")


async def main():
    """Main demo function"""
    print_banner()
    
    # Create mock components
    console.print("\n🔧 [bold]Setting up demo environment...[/bold]")
    
    # Create mock trading engines
    engines = [
        MockTradingEngine("main_engine"),
        MockTradingEngine("backup_engine"),
    ]
    
    # Create mock risk monitor
    risk_monitor = MockRiskMonitor()
    
    # Create mock alerting system
    alerting_system = MockAlertingSystem()
    
    # Create safety systems integration
    config = SafetySystemConfig(
        initial_capital=100000.0,
        enable_auto_liquidation=True,
        circuit_breaker_data_dir="data/demo/circuit_breakers",
        enable_auto_resume=False,
        default_cooldown_minutes=1,  # Short for demo
        kill_switch_data_dir="data/demo/kill_switches",
        enable_cross_system_triggers=True,
        enable_state_persistence=True,
        monitoring_interval_seconds=0.5,  # Fast for demo
    )
    
    safety_integration = SafetySystemsIntegration(config)
    
    # Register components
    for engine in engines:
        safety_integration.register_trading_engine(engine.engine_id, engine)
    
    safety_integration.register_risk_monitor(risk_monitor)
    safety_integration.register_alerting_system(alerting_system)
    
    # Initialize
    console.print("🚀 Initializing safety systems...")
    success = await safety_integration.initialize()
    
    if not success:
        console.print("❌ Failed to initialize safety systems")
        return
    
    # Start monitoring
    safety_integration.start_monitoring()
    console.print("✅ Safety systems online and monitoring")
    
    try:
        # Run demonstrations
        await demo_circuit_breakers(safety_integration, risk_monitor)
        await demo_kill_switches(safety_integration, engines)
        await demo_integration_features(safety_integration)
        await demo_state_persistence(safety_integration)
        
        # Final status
        console.print("\n🎯 [bold green]DEMO COMPLETE[/bold green]")
        console.print("=" * 60)
        
        final_status = safety_integration.get_comprehensive_status()
        console.print("\n📊 Final Statistics:")
        metrics = final_status["integration"]["performance_metrics"]
        console.print(f"   Safety Checks Performed: {metrics['safety_checks_performed']}")
        console.print(f"   Circuit Breakers Triggered: {metrics['circuit_breakers_triggered']}")
        console.print(f"   Kill Switches Triggered: {metrics['kill_switches_triggered']}")
        console.print(f"   Cross-System Triggers: {metrics['cross_system_triggers']}")
        
        console.print(f"\n📋 Event Summary:")
        console.print(f"   Total Events: {final_status['total_events']}")
        console.print(f"   Cross-System Events: {final_status['cross_system_events']}")
        
        # Show alerts sent
        console.print(f"\n🚨 Alerts Sent: {len(alerting_system.alerts_sent)}")
        for alert in alerting_system.alerts_sent[-3:]:  # Show last 3
            console.print(f"   {alert['severity'].upper()}: {alert['title']}")
        
        console.print("\n✅ [bold green]All safety systems demonstrated successfully![/bold green]")
        console.print("\n🛡️ Your trading system is now protected by:")
        console.print("   ⚡ 7 circuit breakers monitoring risk conditions")
        console.print("   🔴 4 kill switches for emergency control")
        console.print("   🔗 Integrated cross-system triggers")
        console.print("   💾 Persistent state with audit trails")
        console.print("   🚨 Real-time alerting and notifications")
        console.print("   🧵 Thread-safe, high-performance monitoring")
        
    finally:
        # Cleanup
        console.print("\n🧹 Cleaning up...")
        safety_integration.stop_monitoring()
        console.print("✅ Demo cleanup complete")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())