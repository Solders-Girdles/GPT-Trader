#!/bin/bash
# Sandbox Soak Test Deployment Script
# Phase 3.2/3.3 Validation - 24-48 Hour Test
set -e

PROJECT_ROOT="/Users/rj/PycharmProjects/GPT-Trader"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Sandbox Soak Test - Deployment Script"
echo "Phase 3.2/3.3 Validation"
echo "=========================================="
echo ""

# Check prerequisites
echo "Step 1/7: Checking prerequisites..."
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found"
    echo "   Please copy .env.sandbox.example to .env and configure your Coinbase Sandbox credentials"
    echo ""
    echo "   cp .env.sandbox.example .env"
    echo "   # Then edit .env with your sandbox API key/secret"
    exit 1
fi

# Verify required environment variables
source .env

# Check for CDP credentials (preferred for production/advanced mode)
if [ -n "$COINBASE_CDP_API_KEY" ] && [ -n "$COINBASE_CDP_PRIVATE_KEY" ]; then
    echo "✓ CDP credentials configured (Advanced Trade API)"
# Check for legacy HMAC credentials (sandbox/exchange mode)
elif [ -n "$COINBASE_API_KEY" ] && [ -n "$COINBASE_API_SECRET" ]; then
    if [ "$COINBASE_API_KEY" == "your_sandbox_api_key_here" ] || [ "$COINBASE_API_SECRET" == "your_sandbox_api_secret_here" ]; then
        echo "❌ Error: COINBASE_API_KEY/SECRET contain placeholder values"
        echo "   Update .env with real credentials"
        exit 1
    fi
    echo "✓ HMAC credentials configured (Exchange API)"
else
    echo "❌ Error: No Coinbase credentials configured in .env"
    echo ""
    echo "   Required: Either CDP credentials OR HMAC credentials"
    echo ""
    echo "   Option 1 - CDP (Advanced Trade API, recommended):"
    echo "   COINBASE_CDP_API_KEY=<your-api-key-uuid>"
    echo "   COINBASE_CDP_PRIVATE_KEY=<your-ec-private-key-pem>"
    echo ""
    echo "   Option 2 - HMAC (Exchange API, sandbox only):"
    echo "   COINBASE_API_KEY=<your-api-key>"
    echo "   COINBASE_API_SECRET=<your-api-secret>"
    echo "   COINBASE_API_PASSPHRASE=<your-passphrase>"
    exit 1
fi

# Set defaults if not provided
export ADMIN_PASSWORD="${ADMIN_PASSWORD:-admin123}"
export DATABASE_PASSWORD="${DATABASE_PASSWORD:-trader}"

echo "✓ Environment configured"
echo ""

# Check Docker
echo "Step 2/7: Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker not found. Please install Docker Desktop."
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "❌ Error: Docker daemon not running. Please start Docker Desktop."
    exit 1
fi
echo "✓ Docker ready"
echo ""

# Check Poetry
echo "Step 3/7: Checking Poetry and dependencies..."
if ! command -v poetry &> /dev/null; then
    echo "❌ Error: Poetry not found. Please install Poetry."
    exit 1
fi

poetry install --sync
echo "✓ Dependencies installed"
echo ""

# Start monitoring stack
echo "Step 4/7: Starting monitoring stack..."
cd monitoring
docker-compose down 2>/dev/null || true
docker-compose up -d

# Wait for services to be ready
echo "   Waiting for Prometheus..."
for i in {1..30}; do
    if curl -s http://localhost:9091/-/ready &> /dev/null; then
        break
    fi
    sleep 1
done

echo "   Waiting for Grafana..."
for i in {1..30}; do
    if curl -s http://localhost:3000/api/health &> /dev/null; then
        break
    fi
    sleep 1
done

echo "   Waiting for Alertmanager..."
for i in {1..30}; do
    if curl -s http://localhost:9093/-/ready &> /dev/null; then
        break
    fi
    sleep 1
done

cd ..
echo "✓ Monitoring stack running"
echo "   - Prometheus:    http://localhost:9091"
echo "   - Grafana:       http://localhost:3000 (admin/admin123)"
echo "   - Alertmanager:  http://localhost:9093"
echo ""

# Validate configuration
echo "Step 5/7: Validating bot configuration..."
poetry run python -m bot_v2.cli \
    --profile canary \
    --dry-run \
    --validate-only

echo "✓ Configuration valid"
echo ""

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/sandbox_soak_${TIMESTAMP}.log"

echo "Step 6/7: Deployment ready. Review configuration:"
echo "   Profile:          canary"
echo "   Environment:      sandbox"
echo "   Symbol:           BTC-USD"
echo "   Daily Loss Limit: \$10"
echo "   Max Trade Value:  \$100"
echo "   Position Cap:     0.01 BTC"
echo "   Streaming:        Enabled (level 1, 5s REST fallback)"
echo "   Log File:         $LOG_FILE"
echo ""

read -p "Ready to start soak test? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Deployment cancelled."
    exit 0
fi

echo ""
echo "Step 7/7: Starting bot for soak test..."
echo "=========================================="
echo "Bot starting - press Ctrl+C to stop"
echo "Monitor at: http://localhost:3000"
echo "Health: curl http://localhost:9090/health | jq ."
echo "Metrics: curl http://localhost:9090/metrics | grep bot_"
echo "=========================================="
echo ""

# Start bot
poetry run python -m bot_v2.cli \
    --profile canary \
    --max-trade-value 100 \
    --streaming-rest-poll-interval 5.0 \
    2>&1 | tee "$LOG_FILE"
