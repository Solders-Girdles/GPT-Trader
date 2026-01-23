#!/usr/bin/env bash

set -euo pipefail

RED='[0;31m'
GREEN='[0;32m'
YELLOW='[1;33m'
NC='[0m'

DEPLOY_ENV="${1:-production}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_DIR="$PROJECT_ROOT/deploy/gpt_trader/docker"
COMPOSE_FILE="$COMPOSE_DIR/docker-compose.yaml"
COMPOSE_INFRA_FILE="$COMPOSE_DIR/docker-compose.infrastructure.yaml"
COMPOSE_ENV_FILE="$COMPOSE_DIR/.env"
ENV_TEMPLATE="$PROJECT_ROOT/config/environments/.env.template"
BACKUP_DIR="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"

# Allow callers to override build target via BUILD_TARGET env
if [[ -z "${BUILD_TARGET:-}" ]]; then
    case "$DEPLOY_ENV" in
        production) BUILD_TARGET="production" ;;
        staging) BUILD_TARGET="testing" ;;
        *) BUILD_TARGET="development" ;;
    esac
fi

LOG_LEVEL="${LOG_LEVEL:-INFO}"
RUNTIME_ENV="${BOT_ENV:-$DEPLOY_ENV}"
ENABLE_OBSERVABILITY="${ENABLE_OBSERVABILITY:-0}"
WITH_PROXY="${WITH_PROXY:-${WITH_INFRA:-0}}"

compose() {
    local files=(-f "$COMPOSE_FILE")
    if [[ "$DEPLOY_ENV" == "production" || "$WITH_PROXY" == "1" ]]; then
        files+=(-f "$COMPOSE_INFRA_FILE")
    fi
    docker compose "${files[@]}" "$@"
}

print_header() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}GPT-Trader gpt_trader Deployment${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo "Environment: $DEPLOY_ENV"
    echo "Project Root: $PROJECT_ROOT"
    echo "Compose Dir : $COMPOSE_DIR"
    echo "Build Target: $BUILD_TARGET"
    echo ""
}

check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"

    if ! command -v docker >/dev/null 2>&1; then
        echo -e "${RED}Docker is not installed${NC}"
        exit 1
    fi

    if ! docker compose version >/dev/null 2>&1; then
        echo -e "${RED}Docker Compose v2 plugin is not available${NC}"
        exit 1
    fi

    if [[ ! -d "$COMPOSE_DIR" || ! -f "$COMPOSE_FILE" ]]; then
        echo -e "${RED}gpt_trader deployment assets not found (${COMPOSE_FILE})${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Prerequisites ok${NC}"
}

backup_artifacts() {
    echo -e "${YELLOW}Backing up configuration artifacts...${NC}"
    mkdir -p "$BACKUP_DIR"

    for path in config var; do
        src="$PROJECT_ROOT/$path"
        if [[ -e "$src" ]]; then
            cp -R "$src" "$BACKUP_DIR/" 2>/dev/null || true
            echo "  - Saved $path"
        fi
    done

    if [[ -d "$PROJECT_ROOT/models" ]]; then
        cp -R "$PROJECT_ROOT/models" "$BACKUP_DIR/" 2>/dev/null || true
        echo "  - Saved models/"
    fi

    echo -e "${GREEN}✓ Backup directory: $BACKUP_DIR${NC}"
}

ensure_env_file() {
    if [[ -f "$COMPOSE_ENV_FILE" ]]; then
        return
    fi

    if [[ ! -f "$ENV_TEMPLATE" ]]; then
        echo -e "${RED}Missing environment template (${ENV_TEMPLATE})${NC}"
        exit 1
    fi

    cp "$ENV_TEMPLATE" "$COMPOSE_ENV_FILE"
    echo -e "${YELLOW}Created $COMPOSE_ENV_FILE from template. Update secrets before running in production.${NC}"
}

build_images() {
    echo -e "${YELLOW}Building trading-bot image...${NC}"

    local targets=()
    if [[ "$DEPLOY_ENV" != "development" && "$BUILD_TARGET" != "testing" ]]; then
        targets+=("testing")
    fi
    targets+=("$BUILD_TARGET")

    for target in "${targets[@]}"; do
        echo "  - Building target: $target"
        BUILD_TARGET="$target" compose build trading-bot
    done

    echo -e "${GREEN}✓ Image build complete${NC}"
}

run_stack() {
    echo -e "${YELLOW}Starting services...${NC}"
    local profiles=()
    if [[ "$ENABLE_OBSERVABILITY" == "1" ]]; then
        profiles+=(--profile observability)
    fi
    if [[ "$DEPLOY_ENV" == "production" || "$WITH_PROXY" == "1" ]]; then
        profiles+=(--profile production)
    fi
    ENV="$RUNTIME_ENV" LOG_LEVEL="$LOG_LEVEL" compose "${profiles[@]}" up -d
    echo -e "${GREEN}✓ Services launched${NC}"
}

wait_for_health() {
    echo -e "${YELLOW}Waiting for trading-bot health endpoint...${NC}"
    local retries=30
    local sleep_seconds=5

    for ((i=1; i<=retries; i++)); do
        if ENV="$RUNTIME_ENV" compose exec -T trading-bot curl -fsS http://localhost:8080/health >/dev/null 2>&1; then
            echo -e "${GREEN}✓ trading-bot is healthy${NC}"
            return 0
        fi
        sleep "$sleep_seconds"
    done

    echo -e "${RED}trading-bot failed health check after $((retries * sleep_seconds)) seconds${NC}"
    ENV="$RUNTIME_ENV" compose logs trading-bot | tail -n 200 || true
    exit 1
}

display_info() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Deployment Complete${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Services:"
    echo "  - Trading bot: http://localhost:8080"
    if [[ "$DEPLOY_ENV" == "production" || "$WITH_PROXY" == "1" ]]; then
        echo "  - Secure API: https://localhost:8443 (if Nginx profile enabled)"
    fi
    if [[ "$ENABLE_OBSERVABILITY" == "1" ]]; then
        echo "  - Metrics (Prometheus): http://localhost:9090"
        echo "  - Grafana: http://localhost:3000"
    fi
    echo ""
    echo "Helpful commands (run inside $COMPOSE_DIR):"
    echo "  - docker compose ps"
    echo "  - docker compose logs trading-bot"
    echo "  - docker compose down"
    echo "  - docker compose restart trading-bot"
    echo ""
    echo "Backup location: $BACKUP_DIR"
    echo ""
}

main() {
    print_header
    check_prerequisites
    backup_artifacts
    ensure_env_file
    build_images
    run_stack
    wait_for_health
    display_info
}

trap 'echo -e "\n${RED}Deployment interrupted${NC}"; exit 1' INT TERM

main

exit 0
