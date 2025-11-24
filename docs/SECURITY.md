# Security Documentation

This document outlines the security measures implemented in GPT-Trader and provides guidance for secure deployment and operation.

## 🔒 Security Overview

### Hardcoded Secrets Elimination (SOT-PRE-002)

**Status: ✅ COMPLETED**

All hardcoded secrets have been successfully eliminated from the codebase and replaced with environment variable configuration.

#### Fixed Components:

1. **Database Manager** (`src/bot/database/manager.py`)
   - Removed hardcoded password `trader_password_dev`
   - Now requires `DATABASE_PASSWORD` environment variable
   - Added validation to ensure password is provided

2. **API Gateway** (`src/bot/api/gateway.py`)
   - Removed hardcoded passwords `change_admin_password` and `change_trader_password`
   - Now requires `ADMIN_PASSWORD` and `TRADER_PASSWORD` environment variables
   - Added production environment validation to reject weak passwords

3. **Core Security Module** (`src/bot/core/security.py`)
   - Removed hardcoded password `change_me_in_production`
   - Now requires `ADMIN_PASSWORD` environment variable
   - Added production environment validation

4. **Configuration Files**
   - `config/database.yaml`: Updated to use `${DATABASE_PASSWORD}`
   - `deploy/postgres/docker-compose.yml`: Updated to use environment variables
   - `monitoring/docker-compose.yml`: Updated to use environment variables

5. **Scripts**
   - Legacy helpers such as `scripts/migrate_to_postgres.py` and
     `scripts/test_postgres_connection.py` were removed in the 2025 cleanup. Use
     your infrastructure-as-code or database tooling to manage migrations and
     connection tests.

## 🔐 Required Environment Variables

### Core Security Variables

```bash
# Database access
DATABASE_PASSWORD=your-secure-database-password

# Application authentication
JWT_SECRET_KEY=your-secure-jwt-secret-key-minimum-32-characters
ADMIN_PASSWORD=your-secure-admin-password
TRADER_PASSWORD=your-secure-trader-password

# API credentials
COINBASE_API_KEY=your-coinbase-api-key
COINBASE_API_SECRET=your-coinbase-secret
```

### Optional Configuration Variables

```bash
# Database configuration
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=coinbase_trader
DATABASE_USER=trader_user

# Application configuration
ENVIRONMENT=development  # development, staging, production
LOG_LEVEL=INFO
DEBUG_MODE=false
API_HOST=127.0.0.1

# Email configuration (for alerts)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your-email@example.com
EMAIL_PASSWORD=your-app-specific-password
```

## 🛡️ Security Features

### 1. Environment Variable Validation

- All security-sensitive configuration is loaded from environment variables
- Missing required environment variables cause application startup to fail
- Production environment validation rejects common weak passwords

### 2. Production Password Validation

When `ENVIRONMENT=production`, the system validates that passwords are not:
- `change_admin_password`
- `change_trader_password`
- `change_me_in_production`
- `admin123`
- `trader123`
- `password`

### 3. Secure Defaults

- No fallback to hardcoded passwords
- Explicit validation for required secrets
- Clear error messages for missing configuration

## 🚀 Deployment Guide

### Development Environment

1. Copy the environment template:
   ```bash
   cp config/environments/.env.template .env.local
   ```

2. Fill in your actual values in `.env.local`

3. Load environment variables:
   ```bash
   export $(cat .env.local | xargs)
   ```

4. Run the application:
   ```bash
   python -m src.bot.cli
   ```

### Production Environment

1. **Never use the development template values in production**

2. Generate strong passwords:
   ```bash
   # Generate secure passwords
   DATABASE_PASSWORD=$(openssl rand -base64 32)
   JWT_SECRET_KEY=$(openssl rand -base64 64)
   ADMIN_PASSWORD=$(openssl rand -base64 24)
   TRADER_PASSWORD=$(openssl rand -base64 24)
   ```

3. Set environment variables securely:
   - Use your cloud provider's secret management service
   - Use Kubernetes secrets
   - Use HashiCorp Vault
   - Use Docker secrets

4. Set the environment:
   ```bash
   export ENVIRONMENT=production
   ```

### Docker Deployment

1. Create a `.env` file (not committed to git):
   ```bash
   DATABASE_PASSWORD=your-secure-password
   ADMIN_PASSWORD=your-secure-admin-password
   TRADER_PASSWORD=your-secure-trader-password
   JWT_SECRET_KEY=your-secure-jwt-secret
   ```

2. Run with environment file:
   ```bash
   docker-compose --env-file .env up
   ```

## 🔍 Security Validation

### Automated Security Scanning

Use the following checks during reviews:

```bash
# Security-focused unit tests
poetry run pytest tests/unit/gpt_trader/security -q

# Repository-wide secret scan (requires detect-secrets)
detect-secrets scan
```

### Manual Security Checklist

- [ ] All environment variables are set
- [ ] Production uses strong passwords
- [ ] No hardcoded secrets in codebase
- [ ] `.env` files are not committed to git
- [ ] Secrets are stored securely in production
- [ ] Database connections are encrypted
- [ ] API endpoints use proper authentication
- [ ] Logs don't contain sensitive information

## 🚨 Security Monitoring

### Audit Events

The system logs security events including:
- Authentication attempts
- Authorization failures
- Configuration changes
- Security violations

### Alerting

Configure alerts for:
- Failed authentication attempts
- Unauthorized access attempts
- Configuration changes
- System health issues

## 📞 Security Incident Response

### If Secrets are Compromised:

1. **Immediate Actions:**
   - Rotate all compromised secrets immediately
   - Revoke any exposed API keys
   - Change database passwords
   - Invalidate active JWT tokens

2. **Investigation:**
   - Review audit logs
   - Identify scope of compromise
   - Check for unauthorized access

3. **Recovery:**
   - Deploy new secrets
   - Restart affected services
   - Monitor for suspicious activity

### Emergency Contacts

- Security Team: [Configure as needed]
- Database Admin: [Configure as needed]
- Infrastructure Team: [Configure as needed]

## 🔄 Security Maintenance

### Regular Tasks

- [ ] Rotate secrets quarterly
- [ ] Update dependencies monthly
- [ ] Review access logs weekly
- [ ] Test backup/recovery procedures monthly
- [ ] Update security documentation as needed

### Security Updates

- Monitor for security advisories
- Apply security patches promptly
- Test updates in staging environment first
- Document all security changes

## 📚 Additional Resources

- [OWASP Security Guidelines](https://owasp.org/)
- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)
- [Kubernetes Security](https://kubernetes.io/docs/concepts/security/)
- [AWS Security Best Practices](https://aws.amazon.com/architecture/security-identity-compliance/)

## 📝 Change Log

### 2024-08-14 - SOT-PRE-002: Hardcoded Secrets Elimination

- ✅ Eliminated all hardcoded passwords from codebase
- ✅ Implemented environment variable configuration
- ✅ Added production password validation
- ✅ Updated all configuration files
- ✅ Created security validation scripts
- ✅ Updated deployment documentation

---

**Security Status: 🟢 SECURE**

All identified hardcoded secrets have been eliminated and replaced with secure environment variable configuration.
