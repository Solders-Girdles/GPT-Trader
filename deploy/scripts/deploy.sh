#!/bin/bash

# GPT-Trader ML System Deployment Script
# Production deployment automation

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DEPLOY_ENV=${1:-production}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPLOY_DIR="$PROJECT_ROOT/deploy"
BACKUP_DIR="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GPT-Trader ML System Deployment${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Environment: $DEPLOY_ENV"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed${NC}"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Docker Compose is not installed${NC}"
        exit 1
    fi

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python 3 is not installed${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ All prerequisites met${NC}"
}

# Function to backup existing data
backup_data() {
    echo -e "${YELLOW}Backing up existing data...${NC}"

    mkdir -p "$BACKUP_DIR"

    # Backup databases
    if [ -f "$PROJECT_ROOT/data/trading.db" ]; then
        cp "$PROJECT_ROOT/data/trading.db" "$BACKUP_DIR/"
        echo "  - Database backed up"
    fi

    # Backup models
    if [ -d "$PROJECT_ROOT/models" ]; then
        cp -r "$PROJECT_ROOT/models" "$BACKUP_DIR/"
        echo "  - Models backed up"
    fi

    # Backup configuration
    if [ -d "$PROJECT_ROOT/config" ]; then
        cp -r "$PROJECT_ROOT/config" "$BACKUP_DIR/"
        echo "  - Configuration backed up"
    fi

    echo -e "${GREEN}✓ Backup complete: $BACKUP_DIR${NC}"
}

# Function to validate configuration
validate_config() {
    echo -e "${YELLOW}Validating configuration...${NC}"

    # Check for required config files
    required_files=(
        "config/trading.yaml"
        "config/ml_config.yaml"
        ".env"
    )

    for file in "${required_files[@]}"; do
        if [ ! -f "$PROJECT_ROOT/$file" ]; then
            echo -e "${YELLOW}  ! Missing: $file${NC}"
            echo "    Creating template..."

            case "$file" in
                ".env")
                    cat > "$PROJECT_ROOT/.env" << EOF
# GPT-Trader Environment Variables
TRADING_MODE=paper
INITIAL_CAPITAL=100000
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///data/trading.db

# ML Configuration
ML_ENABLED=true
RETRAIN_INTERVAL_DAYS=7
MIN_CONFIDENCE=0.6

# API Keys (add your keys here)
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPHA_VANTAGE_KEY=
EOF
                    ;;
                "config/trading.yaml")
                    mkdir -p "$PROJECT_ROOT/config"
                    cat > "$PROJECT_ROOT/config/trading.yaml" << EOF
# Trading Configuration
trading:
  mode: paper
  initial_capital: 100000
  max_positions: 10
  position_size_pct: 0.1

risk:
  stop_loss_pct: 0.05
  take_profit_pct: 0.15
  max_drawdown: 0.20

execution:
  commission_rate: 0.001
  slippage_rate: 0.0005
EOF
                    ;;
                "config/ml_config.yaml")
                    cat > "$PROJECT_ROOT/config/ml_config.yaml" << EOF
# ML Configuration
models:
  regime_detector:
    n_regimes: 5
    covariance_type: full

  strategy_selector:
    n_estimators: 100
    max_depth: 10
    learning_rate: 0.1

optimization:
  risk_free_rate: 0.02
  optimization_method: SLSQP

rebalancing:
  threshold: 0.05
  min_interval_hours: 24
EOF
                    ;;
            esac
        fi
    done

    echo -e "${GREEN}✓ Configuration validated${NC}"
}

# Function to build Docker images
build_images() {
    echo -e "${YELLOW}Building Docker images...${NC}"

    cd "$DEPLOY_DIR/docker"

    # Build main application image
    docker-compose build --no-cache ml-trader

    # Build dashboard image if exists
    if [ -f "Dockerfile.dashboard" ]; then
        docker-compose build dashboard
    fi

    echo -e "${GREEN}✓ Docker images built${NC}"
}

# Function to run tests
run_tests() {
    echo -e "${YELLOW}Running tests...${NC}"

    # Run unit tests
    docker run --rm \
        -v "$PROJECT_ROOT:/app" \
        -w /app \
        gpt-trader-ml:latest \
        python -m pytest tests/ml/ -v --tb=short

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ All tests passed${NC}"
    else
        echo -e "${RED}✗ Tests failed${NC}"
        exit 1
    fi
}

# Function to deploy services
deploy_services() {
    echo -e "${YELLOW}Deploying services...${NC}"

    cd "$DEPLOY_DIR/docker"

    case "$DEPLOY_ENV" in
        development)
            docker-compose up -d redis postgres
            docker-compose up -d ml-trader dashboard
            ;;
        staging)
            docker-compose up -d
            ;;
        production)
            # Production deployment with scaling
            docker-compose up -d --scale ml-trader=2
            ;;
        *)
            echo -e "${RED}Unknown environment: $DEPLOY_ENV${NC}"
            exit 1
            ;;
    esac

    # Wait for services to be healthy
    echo "Waiting for services to be healthy..."
    sleep 10

    # Check service health
    docker-compose ps

    echo -e "${GREEN}✓ Services deployed${NC}"
}

# Function to run database migrations
run_migrations() {
    echo -e "${YELLOW}Running database migrations...${NC}"

    docker-compose exec ml-trader python -m src.bot.database.migrate

    echo -e "${GREEN}✓ Migrations complete${NC}"
}

# Function to initialize ML models
initialize_models() {
    echo -e "${YELLOW}Initializing ML models...${NC}"

    docker-compose exec ml-trader python -c "
from src.bot.paper_trading.ml_paper_trader import MLPaperTrader
import asyncio

async def init():
    trader = MLPaperTrader()
    await trader.initialize()
    print('Models initialized')

asyncio.run(init())
"

    echo -e "${GREEN}✓ ML models initialized${NC}"
}

# Function to setup monitoring
setup_monitoring() {
    echo -e "${YELLOW}Setting up monitoring...${NC}"

    # Create monitoring directories
    mkdir -p "$DEPLOY_DIR/monitoring/prometheus"
    mkdir -p "$DEPLOY_DIR/monitoring/grafana/dashboards"
    mkdir -p "$DEPLOY_DIR/monitoring/grafana/datasources"

    # Create Prometheus config if not exists
    if [ ! -f "$DEPLOY_DIR/monitoring/prometheus.yml" ]; then
        cat > "$DEPLOY_DIR/monitoring/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ml-trader'
    static_configs:
      - targets: ['ml-trader:8000']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
EOF
    fi

    echo -e "${GREEN}✓ Monitoring configured${NC}"
}

# Function to display deployment info
display_info() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Services:"
    echo "  - ML Trader: http://localhost:8000"
    echo "  - Dashboard: http://localhost:8501"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000 (admin/admin)"
    echo "  - Redis: localhost:6379"
    echo "  - PostgreSQL: localhost:5432"
    echo ""
    echo "Commands:"
    echo "  - View logs: docker-compose logs -f ml-trader"
    echo "  - Stop services: docker-compose down"
    echo "  - Restart services: docker-compose restart"
    echo "  - View status: docker-compose ps"
    echo ""
    echo "Backup location: $BACKUP_DIR"
    echo ""
}

# Main deployment flow
main() {
    echo "Starting deployment process..."
    echo ""

    check_prerequisites
    backup_data
    validate_config
    build_images

    if [ "$DEPLOY_ENV" != "development" ]; then
        run_tests
    fi

    deploy_services

    if [ "$DEPLOY_ENV" == "production" ]; then
        run_migrations
        initialize_models
        setup_monitoring
    fi

    display_info
}

# Handle script interruption
trap 'echo -e "\n${RED}Deployment interrupted${NC}"; exit 1' INT TERM

# Run main function
main

exit 0
