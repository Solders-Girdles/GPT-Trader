# GPT-Trader Kubernetes Deployment

## Overview

Production-ready Kubernetes manifests for deploying GPT-Trader in a cloud environment. These manifests provide a complete, scalable, and secure deployment of the GPT-Trader platform.

## Architecture Components

### Core Services
- **API Gateway**: REST/WebSocket API serving client requests
- **Trading Engine**: Core trading logic and strategy execution
- **Data Pipeline**: Real-time market data ingestion and processing
- **ML Inference**: GPU-accelerated machine learning model serving

### Data Storage
- **PostgreSQL**: Primary database for trades, positions, and metrics
- **Redis**: Cache and real-time data buffer

### Monitoring & Operations
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Market Autoscaler**: Dynamic resource scaling based on market conditions

### Supporting Services
- **Ingress Controller**: SSL termination and routing
- **Cert Manager**: Automatic SSL certificate management
- **Backup CronJobs**: Automated database and model backups

## Prerequisites

### Kubernetes Cluster
- Kubernetes 1.24+
- 3+ nodes (minimum)
- GPU nodes for ML inference (optional but recommended)

### Required Add-ons
```bash
# NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.5.1/deploy/static/provider/cloud/deploy.yaml

# Cert Manager for SSL
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.10.0/cert-manager.yaml

# Metrics Server for HPA
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

### Storage Classes
- Standard storage class for databases
- EFS/NFS storage class for shared model storage (multi-AZ recommended)

## Deployment Steps

### 1. Prepare Secrets

Create `.env.production` file:
```bash
# Trading API Credentials
ALPACA_API_KEY_ID=your-alpaca-key
ALPACA_API_SECRET=your-alpaca-secret
POLYGON_API_KEY=your-polygon-key

# Database Passwords
POSTGRES_PASSWORD=secure-postgres-password
REDIS_PASSWORD=secure-redis-password

# Security
JWT_SECRET_KEY=your-jwt-secret-key

# Monitoring
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK
GRAFANA_ADMIN_PASSWORD=secure-grafana-password

# AWS (for backups)
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
```

### 2. Deploy Using Kustomize

```bash
# Deploy everything
kubectl apply -k k8s/

# Or deploy step by step
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/postgresql.yaml
kubectl apply -f k8s/api-gateway.yaml
kubectl apply -f k8s/trading-engine.yaml
kubectl apply -f k8s/data-pipeline.yaml
kubectl apply -f k8s/ml-inference.yaml
kubectl apply -f k8s/monitoring.yaml
kubectl apply -f k8s/autoscaler.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/backup-cronjob.yaml
kubectl apply -f k8s/network-policies.yaml
```

### 3. Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n gpt-trader

# Check services
kubectl get svc -n gpt-trader

# Check ingress
kubectl get ingress -n gpt-trader

# View logs
kubectl logs -n gpt-trader deployment/api-gateway
kubectl logs -n gpt-trader deployment/trading-engine
```

### 4. Access Services

After deployment, services will be available at:
- API: https://api.gpt-trader.com
- WebSocket: wss://ws.gpt-trader.com
- Monitoring: https://monitoring.gpt-trader.com

Default credentials:
- Grafana: admin / [password from secret]
- API: Use `/api/v1/auth/login` endpoint

## Configuration

### Scaling Configuration

Edit `k8s/configmap.yaml` to adjust scaling policies:
```yaml
scaling:
  policies:
    - name: normal_market
      min_workers: 5
      max_workers: 10
      target_cpu: 60
```

### Resource Limits

Adjust resources in deployment manifests:
```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

### Trading Configuration

Modify trading parameters in `k8s/configmap.yaml`:
```yaml
portfolio:
  initial_capital: 100000
  max_position_size: 0.1
  risk_per_trade: 0.02
```

## Monitoring

### Prometheus Metrics

Available metrics endpoints:
- `/metrics` - Application metrics
- `/api/v1/system/metrics` - System metrics

### Grafana Dashboards

Pre-configured dashboards:
- System Overview
- Trading Performance
- ML Model Performance
- Market Data Pipeline
- Resource Utilization

### Alerts

Alert rules configured for:
- High drawdown (>10%)
- Low Sharpe ratio (<1.0)
- System errors
- Resource exhaustion
- Data pipeline failures

## Backup and Recovery

### Automated Backups

Configured CronJobs:
- Database: Daily at 2 AM UTC
- ML Models: Weekly on Sundays
- Cleanup: Daily old data removal

### Manual Backup

```bash
# Trigger manual database backup
kubectl create job --from=cronjob/database-backup manual-backup -n gpt-trader

# Trigger manual model backup
kubectl create job --from=cronjob/model-backup manual-model-backup -n gpt-trader
```

### Recovery

```bash
# Restore from S3 backup
kubectl exec -it postgresql-0 -n gpt-trader -- psql -U gpt_trader -d gpt_trader < backup.sql
```

## Security

### Network Policies

- Default deny all traffic
- Explicit allow rules for service communication
- Ingress only through NGINX controller

### Secrets Management

- All sensitive data in Kubernetes secrets
- Consider using:
  - HashiCorp Vault
  - AWS Secrets Manager
  - Azure Key Vault

### SSL/TLS

- Automatic certificate provisioning via Let's Encrypt
- Force SSL redirect enabled
- TLS 1.2+ only

## Troubleshooting

### Common Issues

1. **Pods not starting**
```bash
kubectl describe pod <pod-name> -n gpt-trader
kubectl logs <pod-name> -n gpt-trader
```

2. **Database connection issues**
```bash
kubectl exec -it <pod-name> -n gpt-trader -- nc -zv postgresql 5432
```

3. **Ingress not working**
```bash
kubectl get ingress -n gpt-trader
kubectl describe ingress gpt-trader-ingress -n gpt-trader
```

4. **Scaling issues**
```bash
kubectl get hpa -n gpt-trader
kubectl describe hpa <hpa-name> -n gpt-trader
```

### Debug Mode

Enable debug logging:
```bash
kubectl set env deployment/trading-engine LOG_LEVEL=DEBUG -n gpt-trader
```

## Performance Tuning

### Database Optimization

```sql
-- Add to postgres-init ConfigMap
ALTER SYSTEM SET shared_buffers = '1GB';
ALTER SYSTEM SET effective_cache_size = '3GB';
ALTER SYSTEM SET maintenance_work_mem = '256MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
```

### Redis Optimization

```yaml
# Add to redis deployment
args:
  - --maxmemory 2gb
  - --maxmemory-policy allkeys-lru
  - --tcp-backlog 511
  - --tcp-keepalive 60
```

## Production Checklist

- [ ] Update all secrets with production values
- [ ] Configure DNS for domains
- [ ] Set up monitoring alerts
- [ ] Configure backup S3 bucket
- [ ] Review and adjust resource limits
- [ ] Enable network policies
- [ ] Set up log aggregation
- [ ] Configure autoscaling policies
- [ ] Test disaster recovery procedures
- [ ] Document operational procedures

## Support

For issues or questions:
- GitHub Issues: https://github.com/gpt-trader/gpt-trader/issues
- Documentation: https://docs.gpt-trader.com
- Email: support@gpt-trader.com