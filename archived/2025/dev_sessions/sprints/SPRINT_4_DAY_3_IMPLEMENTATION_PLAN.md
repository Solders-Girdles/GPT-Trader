# Sprint 4 Day 3: CLI & API Layer Implementation Plan

## Overview
Create the user-facing interface layer that makes the bot_v2 system accessible via CLI, REST API, and WebSocket connections.

## Architecture Vision

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interfaces                          │
├──────────────┬──────────────┬────────────────┬──────────────┤
│     CLI      │   REST API   │   WebSocket    │   Dashboard  │
├──────────────┴──────────────┴────────────────┴──────────────┤
│                    __main__.py Entry Point                   │
├───────────────────────────────────────────────────────────────┤
│                    Integration Layer                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Workflows | Optimization | State | Feature Slices    │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

## Implementation Tasks

### Task 1: Main Entry Point (__main__.py)
**Agent**: deployment-engineer  
**Priority**: CRITICAL - Must be done first

**Requirements**:
```python
# Core responsibilities
1. Parse command-line arguments
2. Load configuration (YAML/JSON/ENV)
3. Initialize optimization layer (cache, pools, lazy loading)
4. Setup logging with structured output
5. Handle signals (SIGTERM, SIGINT) gracefully
6. Route to appropriate interface (CLI/API/WebSocket)
7. Manage lifecycle (startup, shutdown, cleanup)
```

**Key Features**:
- Multi-mode operation (cli, api, websocket, all)
- Configuration hierarchy (file → env → args)
- Health check endpoint
- Graceful shutdown with resource cleanup
- Error recovery and restart capabilities

### Task 2: REST API (api/rest.py)
**Agent**: deployment-engineer (continues from Task 1)  
**Priority**: HIGH

**FastAPI Implementation**:
```python
# Core endpoints structure
/api/v1/
├── /workflows/
│   ├── GET /list                    # List available workflows
│   ├── POST /execute/{workflow_id}  # Execute workflow
│   ├── GET /status/{execution_id}   # Get execution status
│   └── DELETE /cancel/{execution_id} # Cancel execution
├── /strategies/
│   ├── GET /list                    # List strategies
│   ├── POST /backtest               # Run backtest
│   ├── GET /performance/{strategy}  # Get performance metrics
│   └── POST /optimize               # Optimize parameters
├── /market/
│   ├── GET /data/{symbol}          # Get market data
│   ├── GET /regime                 # Current market regime
│   └── POST /analyze               # Market analysis
├── /portfolio/
│   ├── GET /positions              # Current positions
│   ├── GET /performance            # Portfolio metrics
│   ├── POST /rebalance            # Trigger rebalancing
│   └── GET /risk                  # Risk metrics
├── /system/
│   ├── GET /health                # Health check
│   ├── GET /metrics               # System metrics
│   ├── GET /cache/stats          # Cache statistics
│   └── POST /cache/clear         # Clear cache
```

**Features**:
- JWT authentication with refresh tokens
- Rate limiting per endpoint
- Request/response validation with Pydantic
- Automatic OpenAPI documentation
- CORS configuration for web clients
- WebSocket upgrade support
- Async request handling
- Background task support

### Task 3: WebSocket Server (api/websocket.py)
**Agent**: devops-lead  
**Priority**: HIGH

**Real-Time Streaming**:
```python
# WebSocket channels
/ws/v1/
├── /market/stream      # Real-time market data
├── /portfolio/updates  # Portfolio changes
├── /trades/live       # Trade execution events
├── /system/events     # System notifications
└── /custom/{channel}  # Custom subscriptions
```

**Protocol Design**:
```json
// Subscribe message
{
  "action": "subscribe",
  "channels": ["market:AAPL", "portfolio:updates"],
  "auth": "jwt_token"
}

// Data message
{
  "channel": "market:AAPL",
  "type": "quote",
  "data": {
    "symbol": "AAPL",
    "price": 150.25,
    "volume": 1000000,
    "timestamp": "2025-08-19T10:30:00Z"
  }
}

// Heartbeat
{
  "type": "ping",
  "timestamp": 1234567890
}
```

**Features**:
- Connection pooling and management
- Automatic reconnection handling
- Message queuing for reliability
- Subscription management
- Authentication via JWT
- Rate limiting per connection
- Binary and JSON message support
- Compression for large messages

### Task 4: CLI Commands (cli/commands.py)
**Agent**: trading-ops-lead  
**Priority**: HIGH

**Command Structure**:
```bash
gpt-trader [OPTIONS] COMMAND [ARGS]

Commands:
  run          Run trading strategy
  backtest     Run historical backtest
  optimize     Optimize strategy parameters
  monitor      Monitor live trading
  status       Show system status
  config       Manage configuration
  cache        Cache management
  workflow     Workflow operations
  
Examples:
  gpt-trader run --strategy momentum --symbol AAPL --mode paper
  gpt-trader backtest --from 2024-01-01 --to 2024-12-31 --strategy all
  gpt-trader optimize --strategy mean_reversion --metric sharpe
  gpt-trader monitor --dashboard --port 8080
  gpt-trader status --format json
  gpt-trader workflow execute multi_strategy_ensemble --async
```

**Features**:
- Rich terminal output with colors
- Progress bars for long operations
- Interactive mode with prompts
- Configuration file support (~/.gpt-trader/config.yaml)
- Output formats (table, json, csv)
- Async operation support
- Command completion
- Help system with examples

## Integration Architecture

### 1. Shared Service Layer
```python
# services/__init__.py
class ServiceRegistry:
    """Central registry for all services"""
    
    def __init__(self):
        self.workflow_executor = None
        self.cache_manager = None
        self.connection_pools = None
        self.lazy_loader = None
        self.batch_processor = None
        self.state_manager = None
    
    async def initialize(self, config: Dict[str, Any]):
        """Initialize all services with configuration"""
        # Initialize optimization layer
        self.cache_manager = await initialize_cache(config['cache'])
        self.connection_pools = await initialize_pools(config['connections'])
        self.lazy_loader = LazyLoader(config['lazy_loading'])
        self.batch_processor = BatchProcessor(config['batch'])
        
        # Initialize core services
        self.workflow_executor = WorkflowExecutor()
        self.state_manager = StateManager()
        
        # Start monitoring
        await initialize_metrics_monitoring(self.cache_manager)
```

### 2. Request Flow

```
CLI Command → __main__.py → Command Handler → Service Layer → Feature Slices
     ↓                              ↓                ↓
REST API → FastAPI Router → Service Layer → Feature Slices
     ↓                              ↓                ↓
WebSocket → Connection Handler → Service Layer → Feature Slices
```

### 3. Configuration Management
```yaml
# config/default.yaml
system:
  mode: development
  log_level: INFO
  
api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  cors_origins: ["http://localhost:3000"]
  
websocket:
  port: 8001
  max_connections: 100
  heartbeat_interval: 30
  
cli:
  color_output: true
  progress_bars: true
  default_format: table
  
optimization:
  cache:
    enabled: true
    redis_url: redis://localhost:6379
  lazy_loading:
    enabled: true
    preload_critical: true
  batch_processing:
    max_batch_size: 1000
```

## Delegation Strategy

### Parallel Development Plan

**Morning Session (3 tasks in parallel)**:
1. **deployment-engineer**: Create __main__.py entry point
2. **devops-lead**: Start WebSocket server structure
3. **trading-ops-lead**: Begin CLI command framework

**Afternoon Session (2 tasks)**:
1. **deployment-engineer**: Complete REST API
2. **devops-lead + trading-ops-lead**: Integration testing

### Task Dependencies
```
__main__.py (MUST BE FIRST)
    ├── REST API (depends on main)
    ├── WebSocket (depends on main)
    └── CLI (depends on main)
         └── Integration Tests (depends on all)
```

## Technical Decisions

### Framework Choices
- **CLI**: Click (mature, feature-rich, good async support)
- **REST API**: FastAPI (async, automatic docs, WebSocket support)
- **WebSocket**: FastAPI WebSocket + python-socketio for fallback
- **Configuration**: Pydantic Settings + YAML/JSON support

### Authentication Strategy
- JWT tokens for API authentication
- API keys for programmatic access
- Session-based auth for WebSocket
- CLI uses local config file

### Error Handling
- Global exception handlers
- Graceful degradation
- Detailed error messages in development
- Sanitized errors in production
- Error tracking and reporting

### Performance Considerations
- Connection pooling for all external services
- Request caching with TTL
- Rate limiting to prevent abuse
- Async/await throughout
- Background task processing

## Success Metrics

### Functional Requirements
- [ ] All 11 feature slices accessible via API
- [ ] All 6 workflows executable via CLI
- [ ] Real-time data streaming working
- [ ] Authentication and authorization functional
- [ ] Configuration management working

### Performance Requirements
- [ ] API response time < 100ms (p95)
- [ ] WebSocket latency < 50ms
- [ ] CLI startup time < 1 second
- [ ] Support 100+ concurrent connections
- [ ] Memory usage < 500MB baseline

### Quality Requirements
- [ ] OpenAPI documentation complete
- [ ] CLI help for all commands
- [ ] Error messages helpful and clear
- [ ] Logging comprehensive
- [ ] Graceful shutdown working

## Testing Strategy

### Unit Tests
- API endpoint testing with pytest-asyncio
- WebSocket protocol testing
- CLI command testing with click.testing
- Configuration validation tests

### Integration Tests
- End-to-end workflow execution
- Multi-client WebSocket testing
- API authentication flow
- CLI to API communication

### Load Tests
- 1000+ concurrent API requests
- 100+ WebSocket connections
- Large batch operations
- Memory leak detection

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Complex integration | Start with __main__.py to establish patterns |
| Authentication complexity | Use proven JWT libraries |
| WebSocket stability | Implement reconnection and heartbeat |
| CLI usability | Include examples and interactive mode |
| Performance issues | Use optimization layer from Day 2 |

## Implementation Order

1. **__main__.py** - Central entry point (CRITICAL)
2. **services/__init__.py** - Service registry
3. **api/rest.py** - REST API endpoints
4. **api/websocket.py** - WebSocket server
5. **cli/commands.py** - CLI commands
6. **config/settings.py** - Configuration management
7. **Integration tests** - Verify everything works

This plan ensures systematic implementation with proper integration of all components built in previous sprints.