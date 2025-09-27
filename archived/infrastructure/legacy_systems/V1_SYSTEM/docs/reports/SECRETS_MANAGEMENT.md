# GPT-Trader Secrets Management Guide

## Overview
This document outlines the secrets management practices for the GPT-Trader application, ensuring secure handling of sensitive credentials, API keys, and configuration data.

## Security Principles

1. **Never Hardcode Secrets**: All sensitive data must be externalized
2. **Environment Variables**: Primary method for secret injection
3. **Least Privilege**: Only grant minimum necessary permissions
4. **Rotation**: Regular rotation of secrets and keys
5. **Audit Trail**: Log access to secrets (not the secrets themselves)

## Required Environment Variables

### Core Security
```bash
# JWT Authentication
export JWT_SECRET_KEY="<32-128 character random string>"

# Admin Credentials
export ADMIN_PASSWORD="<strong password>"
export TRADER_PASSWORD="<strong password>"

# Database (if using external DB)
export DATABASE_URL="postgresql://user:pass@host:port/db"
export DATABASE_ENCRYPTION_KEY="<32-byte hex string>"
```

### Trading Platform APIs
```bash
# Alpaca Trading
export ALPACA_API_KEY="<your-alpaca-api-key>"
export ALPACA_SECRET_KEY="<your-alpaca-secret-key>"
export ALPACA_PAPER_TRADING="true"  # Use paper trading by default

# Interactive Brokers (if used)
export IB_ACCOUNT="<account-id>"
export IB_USERNAME="<username>"
export IB_PASSWORD="<password>"

# Data Providers
export POLYGON_API_KEY="<polygon-api-key>"
export ALPHA_VANTAGE_API_KEY="<alpha-vantage-key>"
export YAHOO_FINANCE_API_KEY="<yahoo-finance-key>"
```

### Monitoring & Alerting
```bash
# Monitoring Services
export DATADOG_API_KEY="<datadog-api-key>"
export SENTRY_DSN="<sentry-dsn-url>"

# Email Alerts
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USERNAME="<email>"
export SMTP_PASSWORD="<app-password>"
export ALERT_EMAIL_TO="<recipient-email>"
```

### Infrastructure
```bash
# API Configuration
export API_HOST="127.0.0.1"  # Production: Set to appropriate host
export API_PORT="8000"
export API_CORS_ORIGINS="http://localhost:3000,https://yourdomain.com"

# Redis (for caching/queuing)
export REDIS_URL="redis://localhost:6379/0"
export REDIS_PASSWORD="<redis-password>"

# Ray Cluster (if distributed)
export RAY_HEAD_NODE="ray://localhost:10001"
export RAY_REDIS_PASSWORD="<ray-redis-password>"
```

## Secret Generation

### Generate Secure Random Secrets
```bash
# Generate JWT Secret (64 characters)
openssl rand -hex 32

# Generate Database Encryption Key
openssl rand -hex 32

# Generate Strong Password
openssl rand -base64 32

# Generate API Key Format
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

### Password Requirements
- Minimum 16 characters
- Mix of uppercase, lowercase, numbers, and symbols
- No dictionary words or personal information
- Unique per service

## Development Setup

### Using .env Files (Development Only)
```bash
# Create .env file from template
cp .env.template .env

# Edit .env with your development secrets
nano .env

# Ensure .env is in .gitignore
echo ".env" >> .gitignore
```

### .env.template Example
```bash
# Copy this file to .env and fill in your values
# NEVER commit .env to version control

JWT_SECRET_KEY=change_me_in_production
ADMIN_PASSWORD=change_me_in_production
ALPACA_API_KEY=your_dev_api_key_here
ALPACA_SECRET_KEY=your_dev_secret_here
ALPACA_PAPER_TRADING=true
```

## Production Deployment

### Docker Secrets
```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    image: gpt-trader:latest
    secrets:
      - jwt_secret
      - db_password
      - api_keys
    environment:
      JWT_SECRET_KEY_FILE: /run/secrets/jwt_secret
      DATABASE_PASSWORD_FILE: /run/secrets/db_password

secrets:
  jwt_secret:
    external: true
  db_password:
    external: true
  api_keys:
    external: true
```

### Kubernetes Secrets
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: gpt-trader-secrets
type: Opaque
data:
  jwt-secret: <base64-encoded-secret>
  admin-password: <base64-encoded-password>
  alpaca-api-key: <base64-encoded-key>
```

### AWS Secrets Manager
```python
# src/bot/security/aws_secrets.py
import boto3
import json

def get_secret(secret_name: str) -> dict:
    """Retrieve secret from AWS Secrets Manager"""
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Usage
secrets = get_secret('gpt-trader/production')
os.environ['JWT_SECRET_KEY'] = secrets['jwt_secret']
```

### HashiCorp Vault
```python
# src/bot/security/vault_secrets.py
import hvac

def get_vault_secrets() -> dict:
    """Retrieve secrets from HashiCorp Vault"""
    client = hvac.Client(
        url='https://vault.example.com',
        token=os.environ['VAULT_TOKEN']
    )
    return client.secrets.kv.v2.read_secret_version(
        path='gpt-trader/production'
    )['data']['data']
```

## Security Best Practices

### 1. Secret Rotation Schedule
- **API Keys**: Every 90 days
- **Passwords**: Every 60 days
- **JWT Secrets**: Every 180 days
- **Database Keys**: Every 365 days

### 2. Access Control
- Use separate credentials for development/staging/production
- Implement role-based access control (RBAC)
- Audit secret access regularly
- Use service accounts for automated processes

### 3. Encryption at Rest
```python
# Example: Encrypting sensitive data before storage
from cryptography.fernet import Fernet

def encrypt_sensitive_data(data: str, key: bytes) -> str:
    """Encrypt sensitive data before storage"""
    f = Fernet(key)
    return f.encrypt(data.encode()).decode()

def decrypt_sensitive_data(encrypted: str, key: bytes) -> str:
    """Decrypt sensitive data after retrieval"""
    f = Fernet(key)
    return f.decrypt(encrypted.encode()).decode()
```

### 4. Secure Transmission
- Always use HTTPS/TLS for API communication
- Verify SSL certificates
- Use secure WebSocket connections (WSS)
- Implement mutual TLS for service-to-service communication

## Validation Checklist

### Pre-Deployment
- [ ] All hardcoded secrets removed from code
- [ ] Environment variables documented
- [ ] .env file not in version control
- [ ] Secrets rotation policy defined
- [ ] Access logs configured

### Runtime Validation
```python
# src/bot/security/secrets_validator.py
def validate_required_secrets() -> bool:
    """Validate all required secrets are present"""
    required = [
        'JWT_SECRET_KEY',
        'ADMIN_PASSWORD',
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY'
    ]

    missing = [key for key in required if not os.environ.get(key)]

    if missing:
        logger.error(f"Missing required secrets: {missing}")
        return False

    # Validate secret strength
    if len(os.environ.get('JWT_SECRET_KEY', '')) < 32:
        logger.error("JWT_SECRET_KEY too short (min 32 chars)")
        return False

    return True
```

## Emergency Procedures

### Secret Compromise Response
1. **Immediate Actions**:
   - Revoke compromised credentials
   - Generate new secrets
   - Update all affected systems
   - Review access logs

2. **Investigation**:
   - Identify scope of compromise
   - Review audit logs
   - Check for unauthorized access
   - Document incident

3. **Prevention**:
   - Update security policies
   - Enhance monitoring
   - Implement additional controls
   - Security training

### Backup Secret Recovery
```bash
# Backup encryption keys securely
gpg --encrypt --recipient security@company.com secrets-backup.json

# Restore from backup
gpg --decrypt secrets-backup.json.gpg > secrets-backup.json
```

## Monitoring & Auditing

### Secret Access Logging
```python
# Log secret access (not the secrets themselves)
logger.info("Secret accessed", extra={
    "secret_type": "api_key",
    "service": "alpaca",
    "accessed_by": user_id,
    "timestamp": datetime.now().isoformat()
})
```

### Alerting Rules
- Alert on failed authentication attempts
- Monitor for unusual secret access patterns
- Detect secrets in logs or error messages
- Track secret age and rotation compliance

## Testing Secrets Management

### Unit Tests
```python
def test_secrets_not_logged():
    """Ensure secrets are not logged"""
    with self.assertLogs() as logs:
        process_sensitive_operation()

    for log in logs.output:
        self.assertNotIn(os.environ['JWT_SECRET_KEY'], log)
        self.assertNotIn(os.environ['ADMIN_PASSWORD'], log)
```

### Integration Tests
```python
def test_secret_rotation():
    """Test secret rotation process"""
    old_secret = get_current_secret()
    rotate_secret('JWT_SECRET_KEY')
    new_secret = get_current_secret()

    self.assertNotEqual(old_secret, new_secret)
    self.assertTrue(validate_secret_strength(new_secret))
```

## Compliance & Documentation

### Regulatory Requirements
- PCI-DSS: Encryption of cardholder data
- GDPR: Protection of personal data
- SOC 2: Security controls documentation
- HIPAA: Healthcare data protection (if applicable)

### Documentation Requirements
- Document all secret types and purposes
- Maintain rotation schedule
- Track access control lists
- Record security incidents
- Update procedures regularly

## Tools & Resources

### Secret Scanning
```bash
# Use git-secrets to prevent committing secrets
git secrets --install
git secrets --register-aws

# Scan for secrets in codebase
trufflehog --regex --entropy=False .
```

### Secret Management Tools
- **Development**: dotenv, direnv
- **Cloud**: AWS Secrets Manager, Azure Key Vault, GCP Secret Manager
- **Enterprise**: HashiCorp Vault, CyberArk
- **Kubernetes**: Sealed Secrets, External Secrets Operator

## Contact & Support

- Security Team: security@company.com
- On-Call: Use PagerDuty for urgent issues
- Documentation: Internal wiki/security section

---

**Last Updated**: 2025-08-12
**Review Schedule**: Quarterly
**Owner**: Security Team
