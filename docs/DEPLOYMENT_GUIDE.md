# GPT-Trader Production Deployment Guide

## Table of Contents
1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Database Configuration](#database-configuration)
4. [Application Deployment](#application-deployment)
5. [Model Deployment](#model-deployment)
6. [Monitoring Setup](#monitoring-setup)
7. [Security Configuration](#security-configuration)
8. [Rollback Procedures](#rollback-procedures)
9. [Maintenance Procedures](#maintenance-procedures)

## Pre-Deployment Checklist

### System Requirements
- [ ] Ubuntu 20.04+ or RHEL 8+
- [ ] Python 3.8+ installed
- [ ] Docker 20.10+ installed
- [ ] 16GB RAM minimum (32GB recommended)
- [ ] 8 CPU cores minimum
- [ ] 100GB SSD storage
- [ ] Network connectivity to trading APIs
- [ ] SSL certificates ready

### Dependencies Check
```bash
# Verify Python version
python3 --version  # Should be 3.8+

# Verify Docker
docker --version
docker-compose --version

# Check system resources
free -h
df -h
nproc
```

### API Keys & Credentials
- [ ] Trading API keys (Alpaca/Interactive Brokers)
- [ ] Market data subscriptions active
- [ ] Database credentials secure
- [ ] Monitoring service API keys
- [ ] Email/SMS alert credentials

## Infrastructure Setup

### 1. Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y \
    build-essential \
    python3-pip \
    python3-venv \
    git \
    htop \
    nginx \
    supervisor \
    redis-server

# Configure firewall
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 80/tcp  # HTTP
sudo ufw allow 443/tcp # HTTPS
sudo ufw allow 5432/tcp # PostgreSQL (restrict source)
sudo ufw enable
```

### 2. Create Application User

```bash
# Create dedicated user
sudo useradd -m -s /bin/bash gpttrader
sudo usermod -aG docker gpttrader

# Set up directories
sudo mkdir -p /opt/gpt-trader
sudo chown gpttrader:gpttrader /opt/gpt-trader

# Switch to app user
sudo su - gpttrader
```

### 3. Clone Repository

```bash
cd /opt/gpt-trader
git clone https://github.com/yourusername/GPT-Trader.git app
cd app

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn supervisor
```

## Database Configuration

### 1. Deploy PostgreSQL with Docker

```bash
# Navigate to deployment directory
cd /opt/gpt-trader/app/deploy/postgres

# Start database services
docker-compose up -d

# Verify services
docker-compose ps
```

### 2. Initialize Database

```bash
# Create database schema
docker exec -it postgres psql -U gpttrader -c "
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS ml;
CREATE SCHEMA IF NOT EXISTS performance;
CREATE SCHEMA IF NOT EXISTS analytics;
"

# Run migrations
cd /opt/gpt-trader/app
python scripts/migrate_to_postgres.py

# Create indexes for performance
docker exec -it postgres psql -U gpttrader -f deploy/postgres/create_indexes.sql
```

### 3. Configure TimescaleDB

```sql
-- Enable TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Convert tables to hypertables
SELECT create_hypertable('performance.predictions', 'timestamp');
SELECT create_hypertable('trading.orders', 'created_at');
SELECT create_hypertable('trading.positions', 'opened_at');

-- Set retention policy (keep 2 years)
SELECT add_retention_policy('performance.predictions', INTERVAL '2 years');
```

## Application Deployment

### 1. Environment Configuration

```bash
# Create production config
cp .env.template .env.production
```

Edit `.env.production`:
```env
# Database
DATABASE_URL=postgresql://gpttrader:password@localhost:5432/gpttrader
REDIS_URL=redis://localhost:6379/0

# Trading API
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# ML Configuration
ML_MODEL_PATH=/opt/gpt-trader/models
ML_FEATURE_COUNT=50
ML_CONFIDENCE_THRESHOLD=0.6

# Risk Management
MAX_POSITION_SIZE=0.1
MAX_DAILY_LOSS=0.05
MAX_DRAWDOWN=0.20

# Monitoring
ENABLE_MONITORING=true
MONITORING_INTERVAL=60
ALERT_EMAIL=alerts@yourcompany.com
```

### 2. Systemd Service Setup

Create `/etc/systemd/system/gpt-trader.service`:

```ini
[Unit]
Description=GPT-Trader Trading System
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=gpttrader
Group=gpttrader
WorkingDirectory=/opt/gpt-trader/app
Environment="PATH=/opt/gpt-trader/app/venv/bin"
ExecStart=/opt/gpt-trader/app/venv/bin/python -m src.bot.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable gpt-trader
sudo systemctl start gpt-trader
sudo systemctl status gpt-trader
```

### 3. Nginx Configuration

Create `/etc/nginx/sites-available/gpt-trader`:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/ssl/certs/your-cert.pem;
    ssl_certificate_key /etc/ssl/private/your-key.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /metrics {
        proxy_pass http://127.0.0.1:9090;
        auth_basic "Metrics";
        auth_basic_user_file /etc/nginx/.htpasswd;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/gpt-trader /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Model Deployment

### 1. Train Production Models

```python
# train_production_models.py
from src.bot.ml import IntegratedMLPipeline
from src.bot.ml import WalkForwardValidator
import joblib

# Load production data
data = load_production_data()

# Initialize pipeline
pipeline = IntegratedMLPipeline.from_config('config/production.yaml')

# Train model with walk-forward validation
validator = WalkForwardValidator()
results = validator.validate(pipeline.model, X, y, prices)

# Save if performance meets criteria
if results.mean_test_accuracy > 0.58 and results.mean_sharpe > 1.0:
    joblib.dump(pipeline.model, '/opt/gpt-trader/models/production_model.pkl')
    save_model_metadata(results)
else:
    raise ValueError("Model doesn't meet production criteria")
```

### 2. Model Versioning

```bash
# Create model directory structure
mkdir -p /opt/gpt-trader/models/{current,archive,staging}

# Version control for models
cd /opt/gpt-trader/models
git init
git add .
git commit -m "Initial production model v1.0"
```

### 3. A/B Testing Setup

```python
# ab_testing.py
class ABTestingEngine:
    def __init__(self):
        self.model_a = load_model('models/current/model_a.pkl')
        self.model_b = load_model('models/staging/model_b.pkl')
        self.traffic_split = 0.9  # 90% to A, 10% to B
    
    def predict(self, features):
        if random.random() < self.traffic_split:
            prediction = self.model_a.predict(features)
            log_prediction('model_a', prediction)
        else:
            prediction = self.model_b.predict(features)
            log_prediction('model_b', prediction)
        return prediction
```

## Monitoring Setup

### 1. Prometheus Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'gpt-trader'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']
```

### 2. Grafana Dashboards

Import dashboard configurations:

```bash
# Copy dashboard configs
cp deploy/monitoring/dashboards/*.json /var/lib/grafana/dashboards/

# Restart Grafana
sudo systemctl restart grafana-server
```

### 3. Alert Rules

Create `alerting_rules.yml`:

```yaml
groups:
  - name: trading_alerts
    rules:
      - alert: ModelAccuracyLow
        expr: model_accuracy < 0.55
        for: 5m
        annotations:
          summary: "Model accuracy below threshold"
          
      - alert: HighDrawdown
        expr: portfolio_drawdown > 0.15
        for: 1m
        annotations:
          summary: "Portfolio drawdown exceeds 15%"
          
      - alert: SystemDown
        expr: up{job="gpt-trader"} == 0
        for: 1m
        annotations:
          summary: "GPT-Trader system is down"
```

## Security Configuration

### 1. API Security

```python
# security.py
from cryptography.fernet import Fernet
import os

class SecureConfig:
    def __init__(self):
        self.key = os.environ.get('ENCRYPTION_KEY')
        self.cipher = Fernet(self.key)
    
    def encrypt_api_key(self, api_key):
        return self.cipher.encrypt(api_key.encode())
    
    def decrypt_api_key(self, encrypted_key):
        return self.cipher.decrypt(encrypted_key).decode()
```

### 2. Database Security

```sql
-- Create read-only user for monitoring
CREATE USER monitoring WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE gpttrader TO monitoring;
GRANT USAGE ON SCHEMA trading, ml, performance TO monitoring;
GRANT SELECT ON ALL TABLES IN SCHEMA trading, ml, performance TO monitoring;

-- Revoke unnecessary privileges
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
```

### 3. Network Security

```bash
# Configure fail2ban
sudo apt install fail2ban
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local

# Add custom jail for API
echo '[gpt-trader]
enabled = true
port = 8000
filter = gpt-trader
logpath = /opt/gpt-trader/logs/access.log
maxretry = 5
bantime = 3600' >> /etc/fail2ban/jail.local

sudo systemctl restart fail2ban
```

## Rollback Procedures

### 1. Model Rollback

```bash
#!/bin/bash
# rollback_model.sh

CURRENT_MODEL="/opt/gpt-trader/models/current/model.pkl"
BACKUP_MODEL="/opt/gpt-trader/models/archive/model_$(date -d yesterday +%Y%m%d).pkl"

# Backup current model
cp $CURRENT_MODEL "${CURRENT_MODEL}.failed"

# Restore previous model
cp $BACKUP_MODEL $CURRENT_MODEL

# Restart service
sudo systemctl restart gpt-trader

# Verify
curl -X POST http://localhost:8000/health
```

### 2. Database Rollback

```bash
# Restore from backup
pg_restore -U gpttrader -d gpttrader_backup < backup_20240101.dump

# Swap databases
psql -U postgres -c "
ALTER DATABASE gpttrader RENAME TO gpttrader_old;
ALTER DATABASE gpttrader_backup RENAME TO gpttrader;
"
```

### 3. Full System Rollback

```bash
# Stop services
sudo systemctl stop gpt-trader
sudo systemctl stop nginx

# Restore application
cd /opt/gpt-trader/app
git checkout stable
git pull

# Restore database
./scripts/restore_database.sh

# Restart services
sudo systemctl start nginx
sudo systemctl start gpt-trader
```

## Maintenance Procedures

### Daily Tasks

```bash
#!/bin/bash
# daily_maintenance.sh

# Check system health
curl http://localhost:8000/health

# Backup database
pg_dump -U gpttrader gpttrader > /backups/db_$(date +%Y%m%d).dump

# Rotate logs
logrotate -f /etc/logrotate.d/gpt-trader

# Check disk space
df -h | grep -E '^/dev/'

# Send daily report
python scripts/generate_daily_report.py
```

### Weekly Tasks

```bash
#!/bin/bash
# weekly_maintenance.sh

# Update model performance metrics
python scripts/evaluate_model_performance.py

# Clean old predictions
psql -U gpttrader -c "
DELETE FROM performance.predictions 
WHERE timestamp < NOW() - INTERVAL '90 days';
"

# Optimize database
psql -U gpttrader -c "VACUUM ANALYZE;"

# System updates (staging first)
apt update && apt list --upgradable
```

### Monthly Tasks

```bash
#!/bin/bash
# monthly_maintenance.sh

# Retrain models with new data
python scripts/retrain_models.py

# Full system backup
./scripts/full_backup.sh

# Security audit
./scripts/security_audit.sh

# Performance review
python scripts/generate_monthly_report.py
```

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
journalctl -u gpt-trader -n 50

# Check permissions
ls -la /opt/gpt-trader/

# Test configuration
python -m src.bot.main --test-config
```

#### Database Connection Issues
```bash
# Test connection
psql -U gpttrader -h localhost -c "SELECT 1;"

# Check PostgreSQL status
sudo systemctl status postgresql

# Review connection pool
psql -U gpttrader -c "SELECT * FROM pg_stat_activity;"
```

#### High Memory Usage
```bash
# Check memory consumers
ps aux --sort=-%mem | head

# Clear Redis cache
redis-cli FLUSHDB

# Restart with memory limit
systemd-run --uid=gpttrader --gid=gpttrader \
    --property=MemoryLimit=4G \
    /opt/gpt-trader/app/venv/bin/python -m src.bot.main
```

## Health Checks

### Automated Health Monitoring

```python
# health_check.py
import requests
import smtplib
from email.mime.text import MIMEText

def check_health():
    checks = {
        'api': 'http://localhost:8000/health',
        'database': 'http://localhost:8000/db/health',
        'redis': 'http://localhost:8000/cache/health',
        'model': 'http://localhost:8000/model/health'
    }
    
    failures = []
    for name, url in checks.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                failures.append(f"{name}: {response.status_code}")
        except Exception as e:
            failures.append(f"{name}: {str(e)}")
    
    if failures:
        send_alert("Health check failures: " + ", ".join(failures))
    
    return len(failures) == 0

if __name__ == "__main__":
    if not check_health():
        sys.exit(1)
```

### Monitoring Endpoints

```python
# monitoring_endpoints.py
from flask import Flask, jsonify
import psutil

app = Flask(__name__)

@app.route('/metrics')
def metrics():
    return jsonify({
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'active_connections': len(psutil.net_connections()),
        'model_accuracy': get_latest_model_accuracy(),
        'portfolio_value': get_portfolio_value(),
        'daily_trades': get_daily_trade_count()
    })

@app.route('/health')
def health():
    if all([
        check_database(),
        check_redis(),
        check_model(),
        check_risk_limits()
    ]):
        return jsonify({'status': 'healthy'}), 200
    return jsonify({'status': 'unhealthy'}), 503
```

## Production Readiness Checklist

### Final Verification
- [ ] All tests passing
- [ ] Performance benchmarks met
- [ ] Security scan completed
- [ ] Backup procedures tested
- [ ] Monitoring dashboards configured
- [ ] Alert rules active
- [ ] Documentation complete
- [ ] Team trained
- [ ] Rollback procedure tested
- [ ] Load testing completed

### Go-Live Steps
1. Enable maintenance mode
2. Final database backup
3. Deploy application
4. Run smoke tests
5. Monitor for 1 hour
6. Disable maintenance mode
7. Continue monitoring

---

*This deployment guide ensures a smooth, secure, and reliable production deployment of GPT-Trader.*