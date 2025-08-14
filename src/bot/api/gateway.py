"""
Comprehensive API Gateway

Production-ready REST and WebSocket API:
- Strategy management endpoints
- Real-time data streaming
- Trading operations
- System monitoring
- Authentication and rate limiting
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import jwt

# FastAPI for high-performance async API
try:
    from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Redis for caching and session management
try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

import numpy as np

# --- Data Models ---


class UserRole(str, Enum):
    """User roles for authorization"""

    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    VIEWER = "viewer"


class OrderType(str, Enum):
    """Order types"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(str, Enum):
    """Order sides"""

    BUY = "buy"
    SELL = "sell"


class StrategyStatus(str, Enum):
    """Strategy execution status"""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


# --- Request/Response Models ---

if FASTAPI_AVAILABLE:

    class LoginRequest(BaseModel):
        """Login request"""

        username: str
        password: str

    class TokenResponse(BaseModel):
        """JWT token response"""

        access_token: str
        token_type: str = "bearer"
        expires_in: int = 3600

    class StrategyConfig(BaseModel):
        """Strategy configuration"""

        name: str
        strategy_type: str
        parameters: dict[str, Any]
        symbols: list[str]
        allocation: float = Field(gt=0, le=1.0)
        max_positions: int = Field(gt=0, default=10)
        risk_limit: float = Field(gt=0, default=0.02)

    class OrderRequest(BaseModel):
        """Order placement request"""

        symbol: str
        side: OrderSide
        quantity: float = Field(gt=0)
        order_type: OrderType = OrderType.MARKET
        limit_price: float | None = None
        stop_price: float | None = None
        time_in_force: str = "DAY"

    class BacktestRequest(BaseModel):
        """Backtest request"""

        strategy_config: StrategyConfig
        start_date: str
        end_date: str
        initial_capital: float = Field(gt=0, default=100000)
        commission: float = Field(ge=0, default=0.001)

    class MarketDataRequest(BaseModel):
        """Market data request"""

        symbols: list[str]
        data_type: str = "bars"  # bars, quotes, trades
        interval: str = "1min"
        limit: int = Field(gt=0, le=1000, default=100)


# --- Security ---


class SecurityManager:
    """
    Handles authentication and authorization.

    Features:
    - JWT token generation and validation
    - Role-based access control
    - Rate limiting
    - Session management
    """

    def __init__(
        self,
        secret_key: str = None,
        algorithm: str = "HS256",
        token_expiry_hours: int = 24,
    ) -> None:
        # Use environment variable for secret key, with secure fallback
        self.secret_key = secret_key or os.environ.get("JWT_SECRET_KEY", os.urandom(32).hex())
        self.algorithm = algorithm
        self.token_expiry_hours = token_expiry_hours
        self.logger = logging.getLogger(__name__)

        # Mock user database (in production, use proper database)
        # Get passwords from environment variables with validation
        admin_password = os.environ.get("ADMIN_PASSWORD")
        trader_password = os.environ.get("TRADER_PASSWORD")

        # Validate that passwords are set for security
        if not admin_password:
            raise ValueError("ADMIN_PASSWORD environment variable must be set")
        if not trader_password:
            raise ValueError("TRADER_PASSWORD environment variable must be set")

        # Additional validation for production environments
        env = os.environ.get("ENVIRONMENT", "development")
        if env.lower() in ["production", "prod"]:
            if admin_password in ["change_admin_password", "admin123", "password"]:
                raise ValueError("Production environment requires a strong admin password")
            if trader_password in ["change_trader_password", "trader123", "password"]:
                raise ValueError("Production environment requires a strong trader password")

        self.users = {
            "admin": {
                "password_hash": self._hash_password(admin_password),
                "role": UserRole.ADMIN,
                "api_key": "ak_admin_" + uuid.uuid4().hex[:16],
            },
            "trader": {
                "password_hash": self._hash_password(trader_password),
                "role": UserRole.TRADER,
                "api_key": "ak_trader_" + uuid.uuid4().hex[:16],
            },
        }

        # Rate limiting storage
        self.rate_limits = {}  # user -> list of timestamps

    def _hash_password(self, password: str) -> str:
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def authenticate_user(self, username: str, password: str) -> dict[str, Any] | None:
        """Authenticate user with username and password"""
        user = self.users.get(username)
        if not user:
            return None

        if user["password_hash"] != self._hash_password(password):
            return None

        return {"username": username, "role": user["role"], "api_key": user["api_key"]}

    def create_token(self, username: str, role: str) -> str:
        """Create JWT token"""
        expiry = datetime.utcnow() + timedelta(hours=self.token_expiry_hours)

        payload = {"sub": username, "role": role, "exp": expiry, "iat": datetime.utcnow()}

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token

    def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid token: {e}")
            return None

    def check_rate_limit(
        self, user: str, max_requests: int = 100, window_seconds: int = 60
    ) -> bool:
        """Check if user exceeded rate limit"""
        now = time.time()

        if user not in self.rate_limits:
            self.rate_limits[user] = []

        # Remove old timestamps
        self.rate_limits[user] = [ts for ts in self.rate_limits[user] if now - ts < window_seconds]

        # Check limit
        if len(self.rate_limits[user]) >= max_requests:
            return False

        # Add current request
        self.rate_limits[user].append(now)
        return True

    def has_permission(self, role: str, resource: str, action: str) -> bool:
        """Check if role has permission for resource/action"""
        permissions = {
            UserRole.ADMIN: ["*"],  # All permissions
            UserRole.TRADER: ["trade:*", "strategy:*", "data:read", "monitor:read"],
            UserRole.ANALYST: ["strategy:read", "data:read", "backtest:*", "monitor:read"],
            UserRole.VIEWER: ["data:read", "monitor:read"],
        }

        user_permissions = permissions.get(role, [])

        # Check for wildcard or specific permission
        permission_string = f"{resource}:{action}"

        for perm in user_permissions:
            if perm == "*" or perm == permission_string:
                return True
            if perm.endswith(":*") and perm.split(":")[0] == resource:
                return True

        return False


# --- API Gateway ---


class APIGateway:
    """
    Main API gateway application.

    Features:
    - REST endpoints for all operations
    - WebSocket support for real-time data
    - Authentication and authorization
    - Rate limiting and caching
    - Error handling and logging
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required. Install with: pip install fastapi")

        self.config = config or {}
        self.app = FastAPI(
            title="GPT-Trader API",
            description="Advanced AI-Powered Trading Platform API",
            version="4.0.0",
        )

        # Initialize components
        self.security = SecurityManager()
        self.logger = logging.getLogger(__name__)

        # WebSocket connections
        self.websocket_clients: set[WebSocket] = set()

        # Redis client for caching
        self.redis_client = None
        if REDIS_AVAILABLE and self.config.get("redis_enabled", False):
            self._init_redis()

        # Setup middleware
        self._setup_middleware()

        # Setup routes
        self._setup_routes()

        # Mock data stores (in production, use proper databases)
        self.strategies = {}
        self.positions = {}
        self.orders = []

    def _init_redis(self) -> None:
        """Initialize Redis connection"""
        try:
            redis_url = self.config.get("redis_url", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url)
            self.logger.info("Redis cache initialized")
        except Exception as e:
            self.logger.error(f"Redis initialization failed: {e}")
            self.redis_client = None

    def _setup_middleware(self) -> None:
        """Setup API middleware"""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get("cors_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Request logging
        @self.app.middleware("http")
        async def log_requests(request, call_next):
            start_time = time.time()
            response = await call_next(request)
            duration = time.time() - start_time

            self.logger.info(
                f"{request.method} {request.url.path} "
                f"completed in {duration:.3f}s "
                f"with status {response.status_code}"
            )

            return response

    def _setup_routes(self) -> None:
        """Setup API routes"""

        # --- Authentication ---

        @self.app.post("/api/v1/auth/login", response_model=TokenResponse)
        async def login(request: LoginRequest):
            """User login"""
            user = self.security.authenticate_user(request.username, request.password)

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
                )

            token = self.security.create_token(user["username"], user["role"])

            return TokenResponse(
                access_token=token, expires_in=self.security.token_expiry_hours * 3600
            )

        # --- Dependency for authenticated routes ---

        security = HTTPBearer()

        async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
            """Get current authenticated user"""
            token = credentials.credentials
            payload = self.security.verify_token(token)

            if not payload:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
                )

            # Check rate limit
            if not self.security.check_rate_limit(payload["sub"]):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded"
                )

            return payload

        # --- Strategy Management ---

        @self.app.post("/api/v1/strategies")
        async def create_strategy(config: StrategyConfig, user: dict = Depends(get_current_user)):
            """Create new strategy"""
            if not self.security.has_permission(user["role"], "strategy", "create"):
                raise HTTPException(status_code=403, detail="Permission denied")

            strategy_id = str(uuid.uuid4())
            self.strategies[strategy_id] = {
                "id": strategy_id,
                "config": config.dict(),
                "status": StrategyStatus.IDLE,
                "created_by": user["sub"],
                "created_at": datetime.utcnow().isoformat(),
                "performance": {},
            }

            return {"strategy_id": strategy_id, "status": "created"}

        @self.app.get("/api/v1/strategies")
        async def list_strategies(user: dict = Depends(get_current_user)):
            """List all strategies"""
            if not self.security.has_permission(user["role"], "strategy", "read"):
                raise HTTPException(status_code=403, detail="Permission denied")

            return list(self.strategies.values())

        @self.app.post("/api/v1/strategies/{strategy_id}/start")
        async def start_strategy(strategy_id: str, user: dict = Depends(get_current_user)):
            """Start strategy execution"""
            if not self.security.has_permission(user["role"], "strategy", "execute"):
                raise HTTPException(status_code=403, detail="Permission denied")

            if strategy_id not in self.strategies:
                raise HTTPException(status_code=404, detail="Strategy not found")

            self.strategies[strategy_id]["status"] = StrategyStatus.RUNNING
            self.strategies[strategy_id]["started_at"] = datetime.utcnow().isoformat()

            # In production, would actually start strategy execution

            return {"status": "started"}

        @self.app.post("/api/v1/strategies/{strategy_id}/stop")
        async def stop_strategy(strategy_id: str, user: dict = Depends(get_current_user)):
            """Stop strategy execution"""
            if not self.security.has_permission(user["role"], "strategy", "execute"):
                raise HTTPException(status_code=403, detail="Permission denied")

            if strategy_id not in self.strategies:
                raise HTTPException(status_code=404, detail="Strategy not found")

            self.strategies[strategy_id]["status"] = StrategyStatus.STOPPED
            self.strategies[strategy_id]["stopped_at"] = datetime.utcnow().isoformat()

            return {"status": "stopped"}

        # --- Trading Operations ---

        @self.app.post("/api/v1/orders")
        async def place_order(order: OrderRequest, user: dict = Depends(get_current_user)):
            """Place trading order"""
            if not self.security.has_permission(user["role"], "trade", "execute"):
                raise HTTPException(status_code=403, detail="Permission denied")

            order_id = str(uuid.uuid4())
            order_data = {
                "id": order_id,
                "user": user["sub"],
                "timestamp": datetime.utcnow().isoformat(),
                **order.dict(),
            }

            self.orders.append(order_data)

            # In production, would send to broker

            return {"order_id": order_id, "status": "submitted"}

        @self.app.get("/api/v1/orders")
        async def list_orders(limit: int = 100, user: dict = Depends(get_current_user)):
            """List orders"""
            if not self.security.has_permission(user["role"], "trade", "read"):
                raise HTTPException(status_code=403, detail="Permission denied")

            # Filter by user if not admin
            if user["role"] != UserRole.ADMIN:
                user_orders = [o for o in self.orders if o["user"] == user["sub"]]
                return user_orders[-limit:]

            return self.orders[-limit:]

        @self.app.get("/api/v1/positions")
        async def get_positions(user: dict = Depends(get_current_user)):
            """Get current positions"""
            if not self.security.has_permission(user["role"], "trade", "read"):
                raise HTTPException(status_code=403, detail="Permission denied")

            # In production, would fetch from broker/database
            return self.positions

        # --- Market Data ---

        @self.app.post("/api/v1/data/market")
        async def get_market_data(
            request: MarketDataRequest, user: dict = Depends(get_current_user)
        ):
            """Get market data"""
            if not self.security.has_permission(user["role"], "data", "read"):
                raise HTTPException(status_code=403, detail="Permission denied")

            # Check cache if available
            if self.redis_client:
                cache_key = f"market:{':'.join(request.symbols)}:{request.interval}"
                cached = await self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)

            # Generate mock data for demo
            data = self._generate_mock_market_data(request)

            # Cache result
            if self.redis_client:
                await self.redis_client.setex(cache_key, 60, json.dumps(data))  # 1 minute TTL

            return data

        # --- Backtesting ---

        @self.app.post("/api/v1/backtest")
        async def run_backtest(request: BacktestRequest, user: dict = Depends(get_current_user)):
            """Run strategy backtest"""
            if not self.security.has_permission(user["role"], "backtest", "execute"):
                raise HTTPException(status_code=403, detail="Permission denied")

            # In production, would run actual backtest
            # For demo, return mock results

            results = {
                "backtest_id": str(uuid.uuid4()),
                "status": "completed",
                "metrics": {
                    "total_return": 0.25,
                    "sharpe_ratio": 1.8,
                    "max_drawdown": -0.12,
                    "win_rate": 0.58,
                    "num_trades": 150,
                },
                "equity_curve": [100000 + i * 1000 + np.random.normal(0, 500) for i in range(100)],
            }

            return results

        # --- System Monitoring ---

        @self.app.get("/api/v1/system/health")
        async def health_check():
            """System health check"""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "4.0.0",
            }

        @self.app.get("/api/v1/system/metrics")
        async def get_metrics(user: dict = Depends(get_current_user)):
            """Get system metrics"""
            if not self.security.has_permission(user["role"], "monitor", "read"):
                raise HTTPException(status_code=403, detail="Permission denied")

            # In production, would fetch actual metrics
            return {
                "cpu_usage": np.random.uniform(20, 40),
                "memory_usage": np.random.uniform(30, 50),
                "active_strategies": len(
                    [s for s in self.strategies.values() if s["status"] == StrategyStatus.RUNNING]
                ),
                "total_orders": len(self.orders),
                "websocket_clients": len(self.websocket_clients),
            }

        # --- WebSocket for Real-time Data ---

        @self.app.websocket("/ws/market/{symbol}")
        async def websocket_market_data(websocket: WebSocket, symbol: str) -> None:
            """WebSocket endpoint for real-time market data"""
            await websocket.accept()
            self.websocket_clients.add(websocket)

            try:
                while True:
                    # Send market data every second
                    data = {
                        "symbol": symbol,
                        "price": 100 + np.random.normal(0, 1),
                        "volume": np.random.uniform(1000, 10000),
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    await websocket.send_json(data)
                    await asyncio.sleep(1)

            except WebSocketDisconnect:
                self.websocket_clients.remove(websocket)
                self.logger.info("WebSocket client disconnected")
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                self.websocket_clients.discard(websocket)

    def _generate_mock_market_data(self, request: MarketDataRequest) -> dict[str, Any]:
        """Generate mock market data for demo"""
        data = {}

        for symbol in request.symbols:
            prices = 100 + np.cumsum(np.random.normal(0, 1, request.limit))

            data[symbol] = {
                "bars": [
                    {
                        "timestamp": (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
                        "open": prices[i] + np.random.uniform(-0.5, 0.5),
                        "high": prices[i] + np.random.uniform(0, 1),
                        "low": prices[i] - np.random.uniform(0, 1),
                        "close": prices[i],
                        "volume": np.random.uniform(1000, 10000),
                    }
                    for i in range(request.limit)
                ]
            }

        return data

    def run(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Run the API server"""
        import uvicorn

        # Use environment variable for host binding, default to localhost for security
        actual_host = os.environ.get("API_HOST", host)
        self.logger.info(f"Starting API Gateway on {actual_host}:{port}")
        uvicorn.run(self.app, host=actual_host, port=port)


# --- Helper Functions ---


def create_api_gateway(config: dict[str, Any] | None = None) -> APIGateway:
    """Create and configure API gateway"""
    default_config = {
        "redis_enabled": True,
        "redis_url": "redis://localhost:6379",
        "cors_origins": ["http://localhost:3000", "https://app.gpt-trader.com"],
        "rate_limit": 100,
        "rate_limit_window": 60,
    }

    if config:
        default_config.update(config)

    return APIGateway(default_config)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create and run API gateway
    gateway = create_api_gateway()

    # Note: In production, use proper ASGI server like Gunicorn
    # gunicorn -w 4 -k uvicorn.workers.UvicornWorker bot.api.gateway:app

    print("🚀 GPT-Trader API Gateway")
    print("=" * 50)
    print("📍 Endpoints:")
    print("   REST API: http://localhost:8000")
    print("   WebSocket: ws://localhost:8000/ws/market/{symbol}")
    print("   Docs: http://localhost:8000/docs")
    # Don't print actual credentials for security
    print("\n🔑 Authentication:")
    print("   Set ADMIN_PASSWORD and TRADER_PASSWORD environment variables")
    print("   Default users: admin, trader")

    gateway.run()
