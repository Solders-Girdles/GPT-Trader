"""
Real-time WebSocket Server for Risk Updates
Phase 3, Week 3: RISK-001
WebSocket server for streaming risk metrics to clients
"""

import asyncio
import json
import logging
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import websockets
from websockets.legacy.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types"""

    RISK_UPDATE = "risk_update"
    POSITION_UPDATE = "position_update"
    ALERT = "alert"
    HEARTBEAT = "heartbeat"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    ERROR = "error"


class SubscriptionType(Enum):
    """Subscription types for clients"""

    ALL = "all"
    RISK_METRICS = "risk_metrics"
    POSITIONS = "positions"
    ALERTS = "alerts"
    VAR = "var"
    EXPOSURE = "exposure"
    GREEKS = "greeks"


@dataclass
class RiskUpdate:
    """Risk metric update message"""

    timestamp: datetime
    metric_type: str
    value: float
    change_pct: float | None = None
    status: str = "normal"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type,
            "value": self.value,
            "change_pct": self.change_pct,
            "status": self.status,
        }


@dataclass
class WebSocketMessage:
    """WebSocket message structure"""

    type: MessageType
    data: Any
    client_id: str | None = None
    timestamp: datetime | None = None

    def to_json(self) -> str:
        """Convert to JSON string"""
        msg = {"type": self.type.value, "timestamp": (self.timestamp or datetime.now()).isoformat()}

        if isinstance(self.data, dict):
            msg["data"] = self.data
        elif hasattr(self.data, "to_dict"):
            msg["data"] = self.data.to_dict()
        else:
            msg["data"] = str(self.data)

        if self.client_id:
            msg["client_id"] = self.client_id

        return json.dumps(msg)


class RiskWebSocketServer:
    """
    WebSocket server for real-time risk updates.

    Features:
    - Real-time risk metric streaming
    - Client subscription management
    - Message broadcasting
    - Connection monitoring
    - Automatic reconnection handling
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        """
        Initialize WebSocket server.

        Args:
            host: Server host address
            port: Server port
        """
        self.host = host
        self.port = port
        self.clients: set[WebSocketServerProtocol] = set()
        self.subscriptions: dict[WebSocketServerProtocol, set[SubscriptionType]] = {}
        self.message_queue: queue.Queue = queue.Queue()
        self.server = None
        self.running = False
        self.server_thread = None

        # Risk data cache
        self.latest_metrics: dict[str, Any] = {}
        self.metric_history: list[RiskUpdate] = []
        self.max_history = 100

    async def handler(self, websocket: WebSocketServerProtocol, path: str):
        """
        Handle new WebSocket connection.

        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        # Register client
        self.clients.add(websocket)
        self.subscriptions[websocket] = {SubscriptionType.ALL}

        logger.info(f"New client connected: {websocket.remote_address}")

        # Send welcome message with current state
        welcome_msg = WebSocketMessage(type=MessageType.RISK_UPDATE, data=self.get_current_state())
        await websocket.send(welcome_msg.to_json())

        try:
            # Handle incoming messages
            async for message in websocket:
                await self.process_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        finally:
            # Unregister client
            self.clients.remove(websocket)
            if websocket in self.subscriptions:
                del self.subscriptions[websocket]

    async def process_client_message(self, websocket: WebSocketServerProtocol, message: str):
        """
        Process message from client.

        Args:
            websocket: Client connection
            message: Message string
        """
        try:
            data = json.loads(message)
            msg_type = MessageType(data.get("type"))

            if msg_type == MessageType.SUBSCRIBE:
                # Handle subscription request
                sub_types = data.get("subscriptions", ["all"])
                self.subscriptions[websocket] = {SubscriptionType(st) for st in sub_types}

                # Send confirmation
                response = WebSocketMessage(
                    type=MessageType.SUBSCRIBE,
                    data={"status": "subscribed", "subscriptions": sub_types},
                )
                await websocket.send(response.to_json())

            elif msg_type == MessageType.UNSUBSCRIBE:
                # Handle unsubscribe request
                unsub_types = data.get("subscriptions", [])
                for st in unsub_types:
                    self.subscriptions[websocket].discard(SubscriptionType(st))

                # Send confirmation
                response = WebSocketMessage(
                    type=MessageType.UNSUBSCRIBE,
                    data={"status": "unsubscribed", "subscriptions": unsub_types},
                )
                await websocket.send(response.to_json())

            elif msg_type == MessageType.HEARTBEAT:
                # Respond to heartbeat
                response = WebSocketMessage(type=MessageType.HEARTBEAT, data={"status": "alive"})
                await websocket.send(response.to_json())

        except Exception as e:
            logger.error(f"Error processing client message: {e}")
            error_msg = WebSocketMessage(type=MessageType.ERROR, data={"error": str(e)})
            await websocket.send(error_msg.to_json())

    async def broadcast(
        self, message: WebSocketMessage, subscription_type: SubscriptionType = SubscriptionType.ALL
    ):
        """
        Broadcast message to subscribed clients.

        Args:
            message: Message to broadcast
            subscription_type: Required subscription type
        """
        if self.clients:
            # Filter clients by subscription
            target_clients = [
                client
                for client in self.clients
                if subscription_type in self.subscriptions.get(client, set())
                or SubscriptionType.ALL in self.subscriptions.get(client, set())
            ]

            # Send to all target clients
            if target_clients:
                message_str = message.to_json()
                await asyncio.gather(
                    *[client.send(message_str) for client in target_clients], return_exceptions=True
                )

    def update_risk_metric(self, metric_type: str, value: float, status: str = "normal"):
        """
        Update risk metric and broadcast to clients.

        Args:
            metric_type: Type of metric (e.g., 'var_95', 'sharpe_ratio')
            value: Metric value
            status: Metric status
        """
        # Calculate change percentage
        change_pct = None
        if metric_type in self.latest_metrics:
            old_value = self.latest_metrics[metric_type].get("value", 0)
            if old_value != 0:
                change_pct = (value - old_value) / abs(old_value)

        # Create update
        update = RiskUpdate(
            timestamp=datetime.now(),
            metric_type=metric_type,
            value=value,
            change_pct=change_pct,
            status=status,
        )

        # Store in cache
        self.latest_metrics[metric_type] = {
            "value": value,
            "status": status,
            "timestamp": update.timestamp,
        }

        # Add to history
        self.metric_history.append(update)
        if len(self.metric_history) > self.max_history:
            self.metric_history.pop(0)

        # Queue for broadcast
        message = WebSocketMessage(type=MessageType.RISK_UPDATE, data=update)
        self.message_queue.put((message, SubscriptionType.RISK_METRICS))

    def send_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """
        Send alert to clients.

        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity
        """
        alert_data = {
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
        }

        msg = WebSocketMessage(type=MessageType.ALERT, data=alert_data)
        self.message_queue.put((msg, SubscriptionType.ALERTS))

    def get_current_state(self) -> dict[str, Any]:
        """
        Get current state of all metrics.

        Returns:
            Current state dictionary
        """
        return {
            "metrics": self.latest_metrics,
            "history_count": len(self.metric_history),
            "connected_clients": len(self.clients),
            "server_time": datetime.now().isoformat(),
        }

    async def message_sender(self):
        """Background task to send queued messages"""
        while self.running:
            try:
                # Check for messages (non-blocking)
                if not self.message_queue.empty():
                    message, sub_type = self.message_queue.get_nowait()
                    await self.broadcast(message, sub_type)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in message sender: {e}")
                await asyncio.sleep(0.1)

    async def run_server(self):
        """Run the WebSocket server"""
        self.running = True

        # Start message sender task
        sender_task = asyncio.create_task(self.message_sender())

        # Start WebSocket server
        async with websockets.serve(self.handler, self.host, self.port):
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")

            # Keep server running
            while self.running:
                await asyncio.sleep(1)

                # Send periodic heartbeat
                if len(self.clients) > 0:
                    heartbeat = WebSocketMessage(
                        type=MessageType.HEARTBEAT, data={"status": "server_alive"}
                    )
                    await self.broadcast(heartbeat)

        # Cancel sender task
        sender_task.cancel()

    def start(self):
        """Start WebSocket server in background thread"""

        def run_async_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.run_server())

        self.server_thread = threading.Thread(target=run_async_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        logger.info("WebSocket server thread started")

    def stop(self):
        """Stop WebSocket server"""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=5)
        logger.info("WebSocket server stopped")

    def simulate_updates(self):
        """Simulate risk metric updates for testing"""
        import random
        import time

        metrics = [
            ("var_95", 0.05, 0.01),
            ("var_99", 0.10, 0.02),
            ("sharpe_ratio", 1.2, 0.3),
            ("max_drawdown", 0.15, 0.05),
            ("gross_exposure", 1.8, 0.2),
            ("net_exposure", 0.4, 0.1),
        ]

        while self.running:
            for metric_name, base_value, volatility in metrics:
                # Generate random walk
                value = base_value + random.gauss(0, volatility)

                # Determine status
                if metric_name.startswith("var"):
                    status = "warning" if value > base_value * 1.2 else "normal"
                elif metric_name == "max_drawdown":
                    status = "critical" if value > 0.20 else "warning" if value > 0.15 else "normal"
                else:
                    status = "normal"

                self.update_risk_metric(metric_name, value, status)

            # Occasionally send alert
            if random.random() < 0.1:
                self.send_alert(
                    "risk_breach",
                    f"Risk limit approaching: {random.choice(['VaR', 'Drawdown', 'Exposure'])}",
                    "warning",
                )

            time.sleep(2)


def demonstrate_websocket_server():
    """Demonstrate WebSocket server functionality"""
    print("WebSocket Server Demonstration")
    print("=" * 60)

    # Create server
    server = RiskWebSocketServer(host="localhost", port=8765)

    print("\nStarting WebSocket server...")
    server.start()

    print("Server running on ws://localhost:8765")
    print("\nSimulating risk updates...")

    # Simulate some updates
    import time

    server.update_risk_metric("var_95", 0.048, "normal")
    time.sleep(1)
    server.update_risk_metric("var_99", 0.095, "normal")
    time.sleep(1)
    server.update_risk_metric("sharpe_ratio", 1.35, "normal")
    time.sleep(1)

    # Send alert
    server.send_alert("limit_breach", "VaR 95% approaching limit", "warning")

    print("\nCurrent state:")
    state = server.get_current_state()
    for metric, data in state["metrics"].items():
        print(f"  {metric}: {data['value']:.3f} ({data['status']})")

    print("\n✅ WebSocket server operational!")
    print("\nTo test with a client, use:")
    print("  wscat -c ws://localhost:8765")
    print("  or use a WebSocket client library")

    # Keep running for demo
    try:
        print("\nPress Ctrl+C to stop the server...")
        server.simulate_updates()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()


if __name__ == "__main__":
    demonstrate_websocket_server()
