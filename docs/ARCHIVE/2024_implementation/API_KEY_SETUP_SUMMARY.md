# API Key Setup Summary

## Overview
This document details the API key configuration for Coinbase Perpetuals trading integration.

## Key Architecture

### Production Keys

#### 1. Trading Key (Write Access)
```yaml
Name: perps-prod-trading-write-202508
Type: CDP API (JWT Authentication)
Auth: JWT with RS256
Scopes:
  - orders: CREATE, UPDATE, CANCEL
  - positions: READ, MODIFY
  - products: READ
  - fills: READ
  - accounts: READ (portfolio only)
Restrictions:
  - No transfers/withdrawals
  - Derivatives enabled (CFM)
  - IP allowlist enforced
```

#### 2. Monitor Key (Read Only)
```yaml
Name: perps-prod-monitor-read-202508
Type: CDP API (View Only)
Auth: JWT with RS256
Scopes:
  - orders: READ
  - positions: READ
  - products: READ
  - fills: READ
  - accounts: READ
  - analytics: READ
Restrictions:
  - Read-only access
  - No trading capabilities
  - Used for dashboards/monitoring
```

### Demo/Sandbox Keys

#### 1. Demo Trading Key
```yaml
Name: perps-demo-trading-write-202508
Type: CDP API (Sandbox)
Auth: JWT with RS256
Environment: api.sandbox.coinbase.com
Scopes: Same as production trading key
Purpose: Testing and validation
```

#### 2. Demo Monitor Key
```yaml
Name: perps-demo-monitor-read-202508
Type: CDP API (Sandbox)
Auth: JWT with RS256
Environment: api.sandbox.coinbase.com
Scopes: Same as production monitor key
Purpose: Test monitoring systems
```

## Configuration

### Environment Variables
```bash
# Core Configuration
COINBASE_API_MODE=advanced
COINBASE_AUTH_TYPE=JWT
COINBASE_ENABLE_DERIVATIVES=1

# CDP API Credentials
COINBASE_CDP_API_KEY=organizations/xxx/apiKeys/yyy
COINBASE_CDP_PRIVATE_KEY_PATH=/secure/path/to/private_key.pem

# Environment Selection
COINBASE_SANDBOX=0  # 0=production, 1=sandbox

# Safety Settings
COINBASE_MAX_POSITION_SIZE=0.01
COINBASE_DAILY_LOSS_LIMIT=0.02
COINBASE_MAX_IMPACT_BPS=15
```

### Private Key Storage
```bash
# Recommended: File-based with restricted permissions
chmod 400 /secure/path/to/private_key.pem
chown app:app /secure/path/to/private_key.pem

# Alternative: Environment variable (less secure)
COINBASE_CDP_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----..."
```

## Endpoints & Capabilities

### Derivatives (CFM) Endpoints
```python
# Verified endpoints with JWT auth
GET /api/v3/brokerage/cfm/positions         # List positions
GET /api/v3/brokerage/cfm/positions/{id}    # Get position
GET /api/v3/brokerage/cfm/sweeps            # Funding sweeps
GET /api/v3/brokerage/products               # List products (perps)
POST /api/v3/brokerage/orders                # Place orders
DELETE /api/v3/brokerage/orders              # Cancel orders
```

### WebSocket Channels
```python
# Authenticated user channels
{
    "type": "subscribe",
    "channel": "user",
    "jwt": "<jwt_token>",
    "product_ids": ["BTC-PERP", "ETH-PERP"]
}

# Subscribed events
- orders: Order updates
- fills: Trade executions  
- positions: Position changes
- funding: Funding payments
```

## Security Configuration

### IP Allowlist
```yaml
Production IPs:
  - Primary: xxx.xxx.xxx.xxx
  - Secondary: yyy.yyy.yyy.yyy
  - Failover: zzz.zzz.zzz.zzz

Rotation Procedure:
  1. Add new IP to allowlist
  2. Test connectivity
  3. Switch traffic
  4. Remove old IP after 24h
```

### Rate Limits
```yaml
Orders:
  - Create: 100/second
  - Cancel: 100/second
  - Query: 10/second

Market Data:
  - Products: 10/second
  - Positions: 10/second
  - Fills: 10/second

WebSocket:
  - Connections: 10 concurrent
  - Messages: 100/second
```

## Validation & Testing

### Capability Probe
```bash
# Run capability validation
python scripts/capability_probe.py

Expected Output:
✅ JWT Authentication successful
✅ Derivatives access confirmed
✅ Order placement capability verified
✅ WebSocket authentication working
✅ Funding rate endpoints accessible
```

### Preflight Checklist
```bash
# Run before deployment
python scripts/preflight_check.py

Validates:
- API credentials configured
- JWT token generation working
- Derivatives enabled
- IP allowlist active
- Rate limits respected
- Safety parameters set
```

## Key Rotation Schedule

### Quarterly Rotation
```yaml
Q1 2025: January 15
Q2 2025: April 15
Q3 2025: July 15
Q4 2025: October 15

Process:
1. Generate new key pair
2. Deploy to staging
3. Parallel run for 48h
4. Cutover to new key
5. Revoke old key after 7 days
```

### Emergency Rotation
```yaml
Triggers:
- Key compromise suspected
- Unauthorized access detected
- Security audit requirement

Response Time:
- Critical: < 15 minutes
- High: < 1 hour
- Medium: < 4 hours
```

## Monitoring & Alerts

### Key Metrics
- Authentication success rate
- API call latency
- Rate limit utilization
- Error rates by endpoint
- WebSocket connection stability

### Alert Thresholds
- Auth failures > 5 in 1 minute
- Latency > 500ms p95
- Rate limit > 80% utilized
- Error rate > 1%
- WebSocket disconnects > 3/hour

## Troubleshooting

### Common Issues

#### JWT Token Invalid
```bash
# Check token expiration
jwt decode $TOKEN

# Verify private key
openssl ec -in private_key.pem -check

# Test signature
curl -H "Authorization: Bearer $TOKEN" https://api.coinbase.com/api/v3/brokerage/accounts
```

#### Derivatives Not Accessible
```bash
# Verify CFM enabled
echo $COINBASE_ENABLE_DERIVATIVES  # Should be 1

# Check product list
curl https://api.coinbase.com/api/v3/brokerage/products?product_type=FUTURE
```

#### WebSocket Auth Failures
```python
# Ensure JWT in connection
ws_auth = {
    "type": "subscribe",
    "channel": "user",
    "jwt": generate_jwt(),  # Fresh token
    "timestamp": str(int(time.time()))
}
```

## Compliance & Audit

### Access Log
All API key usage is logged with:
- Timestamp
- Operation type
- Success/failure
- IP address
- User agent

### Audit Trail
- Key generation events
- Permission changes
- Rotation history
- Revocation records
- Access reviews

## Support Contacts

- **API Support**: api-support@coinbase.com
- **Security**: security@coinbase.com
- **Internal**: DevOps team
- **Escalation**: CTO office

---

*Document Version: 1.0*
*Last Updated: 2025-08-30*
*Next Review: 2025-09-30*