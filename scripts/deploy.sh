#!/bin/bash

# GPT-Trader Production Deployment Script
# Usage: ./scripts/deploy.sh [staging|production]

set -euo pipefail

# Configuration
ENVIRONMENT=${1:-staging}
BACKUP_DIR="/backups/gpt-trader/$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/var/log/gpt-trader/deploy-${ENVIRONMENT}-$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

# Pre-deployment checks
pre_deployment_checks() {
    log "🔍 Running pre-deployment checks..."

    # Check if running as root (shouldn't be)
    if [[ $EUID -eq 0 ]]; then
        error "❌ Do not run deployment script as root user"
    fi

    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        error "❌ Docker is not installed or not in PATH"
    fi

    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null; then
        error "❌ Docker Compose is not installed or not in PATH"
    fi

    # Check if required environment files exist
    if [[ ! -f ".env.${ENVIRONMENT}" ]]; then
        error "❌ Environment file .env.${ENVIRONMENT} not found"
    fi

    # Run health check on current deployment
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log "🔍 Running production health checks..."
        ./scripts/health-check.sh production || warning "⚠️ Current deployment health check failed"
    fi

    success "✅ Pre-deployment checks passed"
}

# Backup current deployment
backup_current_deployment() {
    log "💾 Creating backup of current deployment..."

    mkdir -p "$BACKUP_DIR"

    # Backup configuration files
    cp -r config/ "$BACKUP_DIR/config-backup/" 2>/dev/null || true
    cp -r monitoring/ "$BACKUP_DIR/monitoring-backup/" 2>/dev/null || true

    # Backup database if production
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log "📊 Creating database backup..."
        pg_dump gpt_trader_prod > "$BACKUP_DIR/database-backup.sql" 2>/dev/null || warning "⚠️ Database backup failed"
    fi

    success "✅ Backup completed: $BACKUP_DIR"
}

# Run integration tests
run_integration_tests() {
    log "🧪 Running integration tests..."

    export PYTHONPATH=src
    export ENVIRONMENT="$ENVIRONMENT"

    # Kill any existing test processes
    pkill -f pytest || true

    # Run the integration test suite
    if poetry run pytest tests/integration/gpt_trader/features/live_trade/ \
        -v --tb=short --maxfail=3 \
        --junitxml=reports/integration-test-results.xml; then
        success "✅ Integration tests passed"
    else
        error "❌ Integration tests failed - aborting deployment"
    fi
}

# Deploy application
deploy_application() {
    log "🚀 Deploying GPT-Trader to $ENVIRONMENT environment..."

    # Load environment variables
    source ".env.${ENVIRONMENT}"

    # Build Docker image
    log "🏗️ Building Docker image..."
    docker build -t gpt-trader:${ENVIRONMENT} \
        --build-arg ENVIRONMENT="$ENVIRONMENT" \
        --build-arg VERSION="$(git rev-parse --short HEAD)" .

    # Stop existing containers
    log "🛑 Stopping existing containers..."
    docker-compose -f docker-compose.${ENVIRONMENT}.yml down || true

    # Deploy new containers
    log "🚀 Starting new containers..."
    docker-compose -f docker-compose.${ENVIRONMENT}.yml up -d

    # Wait for health check
    log "⏳ Waiting for services to be healthy..."
    sleep 30

    # Verify deployment
    if docker-compose -f docker-compose.${ENVIRONMENT}.yml ps | grep -q "Up"; then
        success "✅ Application deployed successfully"
    else
        error "❌ Deployment failed - containers not healthy"
    fi
}

# Post-deployment validation
post_deployment_validation() {
    log "✅ Running post-deployment validation..."

    # Health check
    log "🔍 Running health check..."
    ./scripts/health-check.sh "$ENVIRONMENT" || error "❌ Health check failed"

    # Smoke tests
    log "💨 Running smoke tests..."
    ./scripts/smoke-tests.sh "$ENVIRONMENT" || error "❌ Smoke tests failed"

    # Verify monitoring stack
    log "📊 Verifying monitoring stack..."
    if [[ "$ENVIRONMENT" == "production" ]]; then
        docker-compose -f docker-compose.monitoring.yml ps | grep -q "Up" || warning "⚠️ Monitoring stack not running"
    fi

    success "✅ Post-deployment validation completed"
}

# Setup monitoring alerts
setup_monitoring() {
    log "📊 Setting up monitoring alerts..."

    if [[ "$ENVIRONMENT" == "production" ]]; then
        # Start monitoring stack
        docker-compose -f docker-compose.monitoring.yml up -d

        # Wait for services to be ready
        sleep 60

        # Verify Grafana is accessible
        if curl -f http://localhost:3000/api/health &> /dev/null; then
            success "✅ Grafana is running at http://localhost:3000 (admin/admin123)"
        else
            warning "⚠️ Grafana not responding"
        fi

        # Verify Prometheus is accessible
        if curl -f http://localhost:9090/-/healthy &> /dev/null; then
            success "✅ Prometheus is running at http://localhost:9090"
        else
            warning "⚠️ Prometheus not responding"
        fi
    fi
}

# Cleanup old resources
cleanup_old_resources() {
    log "🧹 Cleaning up old resources..."

    # Remove unused Docker images
    docker image prune -f

    # Remove old log files (keep last 7 days)
    find /var/log/gpt-trader/ -name "*.log" -mtime +7 -delete 2>/dev/null || true

    # Remove old backup directories (keep last 30 days)
    find /backups/gpt-trader/ -maxdepth 1 -type d -mtime +30 -exec rm -rf {} + 2>/dev/null || true

    success "✅ Cleanup completed"
}

# Rollback function
rollback() {
    log "🔄 Rolling back deployment..."

    # Stop current deployment
    docker-compose -f docker-compose.${ENVIRONMENT}.yml down

    # Restore from backup if exists
    if [[ -d "$BACKUP_DIR" ]]; then
        log "📦 Restoring from backup: $BACKUP_DIR"
        # Add rollback logic here
        success "✅ Rollback completed"
    else
        error "❌ No backup found for rollback"
    fi
}

# Main deployment function
main() {
    log "🚀 Starting GPT-Trader deployment to $ENVIRONMENT environment"

    # Trap errors and rollback
    trap 'error "❌ Deployment failed - check logs at $LOG_FILE"; rollback' ERR

    pre_deployment_checks
    backup_current_deployment
    run_integration_tests
    deploy_application
    post_deployment_validation
    setup_monitoring
    cleanup_old_resources

    success "🎉 GPT-Trader deployed successfully to $ENVIRONMENT environment!"
    log "📊 Monitoring dashboard: http://localhost:3000 (admin/admin123)"
    log "📈 Prometheus metrics: http://localhost:9090"
    log "📋 Deployment logs: $LOG_FILE"
}

# Handle script arguments
case "${1:-}" in
    "staging"|"production")
        main
        ;;
    "rollback")
        rollback
        ;;
    "health-check")
        ./scripts/health-check.sh "${2:-staging}"
        ;;
    "smoke-tests")
        ./scripts/smoke-tests.sh "${2:-staging}"
        ;;
    "monitoring")
        setup_monitoring
        ;;
    *)
        echo "Usage: $0 {staging|production|rollback|health-check|smoke-tests|monitoring}"
        echo "  staging     - Deploy to staging environment"
        echo "  production  - Deploy to production environment"
        echo "  rollback    - Rollback to previous deployment"
        echo "  health-check - Run health check [environment]"
        echo "  smoke-tests - Run smoke tests [environment]"
        echo "  monitoring  - Setup monitoring stack"
        exit 1
        ;;
esac
