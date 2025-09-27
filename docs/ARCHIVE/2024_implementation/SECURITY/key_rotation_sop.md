# API Key Rotation Standard Operating Procedure

## Purpose
Ensure secure, zero-downtime rotation of Coinbase API keys for perpetuals trading.

## Rotation Schedule

### Regular Rotation (Quarterly)
- **Q1 2025**: January 15, 2025
- **Q2 2025**: April 15, 2025  
- **Q3 2025**: July 15, 2025
- **Q4 2025**: October 15, 2025

### Trigger Events
- Key compromise suspected
- Personnel changes
- Security audit findings
- Regulatory requirements
- 90-day maximum lifetime

## Pre-Rotation Checklist

- [ ] Schedule maintenance window (if required)
- [ ] Notify stakeholders 48 hours in advance
- [ ] Prepare new key configuration
- [ ] Verify backup keys are ready
- [ ] Test rollback procedure
- [ ] Document current key fingerprints

## Rotation Procedure

### Step 1: Generate New Keys (T-7 days)

```bash
# 1. Log into Coinbase CDP Console
# 2. Navigate to API Keys section
# 3. Create new key with naming convention:
#    perps-{env}-{purpose}-{access}-{YYYYMM}
#    Example: perps-prod-trading-write-202504

# 4. Configure permissions:
- Orders: Create, Read, Update, Cancel
- Positions: Read
- Products: Read
- Fills: Read
- Accounts: Read (portfolio only)

# 5. Enable derivatives (CFM)
# 6. Set IP allowlist
# 7. Download private key securely
```

### Step 2: Secure Key Storage (T-7 days)

```bash
# Store private key securely
chmod 400 /secure/keys/coinbase_cdp_private_key_new.pem
chown app:app /secure/keys/coinbase_cdp_private_key_new.pem

# Encrypt backup
openssl enc -aes-256-cbc -salt -in coinbase_cdp_private_key_new.pem \
  -out coinbase_cdp_private_key_new.pem.enc

# Store in secret manager
vault kv put secret/coinbase/keys/new \
  api_key=@api_key.txt \
  private_key=@coinbase_cdp_private_key_new.pem
```

### Step 3: Deploy to Staging (T-5 days)

```bash
# Update staging configuration
export COINBASE_CDP_API_KEY_NEW="organizations/xxx/apiKeys/new"
export COINBASE_CDP_PRIVATE_KEY_PATH_NEW="/secure/keys/coinbase_cdp_private_key_new.pem"

# Test new keys
python scripts/capability_probe.py
python scripts/preflight_check.py

# Verify functionality
python scripts/validate_perps_client_week1.py
```

### Step 4: Parallel Run (T-3 days)

```python
# Implement dual-key support
class DualKeyAuth:
    def __init__(self):
        self.primary_auth = JWTAuth(old_config)
        self.secondary_auth = JWTAuth(new_config)
        self.use_new = False
    
    def get_headers(self):
        if self.use_new:
            return self.secondary_auth.get_headers()
        return self.primary_auth.get_headers()
    
    def switch_to_new(self):
        self.use_new = True
```

### Step 5: Gradual Cutover (T-0)

```bash
# Phase 1: Route 10% traffic to new key (1 hour)
export COINBASE_NEW_KEY_PERCENTAGE=10

# Phase 2: Route 50% traffic to new key (2 hours)
export COINBASE_NEW_KEY_PERCENTAGE=50

# Phase 3: Route 100% traffic to new key (4 hours)
export COINBASE_NEW_KEY_PERCENTAGE=100

# Monitor metrics
watch -n 5 'curl http://metrics/api_key_usage'
```

### Step 6: Validation (T+1 day)

```bash
# Verify all traffic on new key
grep "api_key" /var/log/trading.log | tail -100

# Check error rates
python scripts/check_api_errors.py --since "1 day ago"

# Confirm old key no longer used
python scripts/verify_key_migration.py
```

### Step 7: Revoke Old Key (T+7 days)

```bash
# Final verification - no traffic on old key
python scripts/confirm_old_key_unused.py

# Revoke via Coinbase Console
# 1. Navigate to API Keys
# 2. Find old key
# 3. Click "Revoke"
# 4. Confirm revocation

# Remove from systems
rm /secure/keys/coinbase_cdp_private_key_old.pem
vault kv delete secret/coinbase/keys/old

# Update documentation
echo "Key rotation completed: $(date)" >> rotation_log.txt
```

## Emergency Rotation

### Immediate Actions (< 15 minutes)

1. **Isolate Compromised Key**:
   ```bash
   # Kill all active sessions
   pkill -f trading_bot
   
   # Block network access
   iptables -A OUTPUT -d api.coinbase.com -j DROP
   ```

2. **Deploy Standby Key**:
   ```bash
   # Switch to pre-generated standby key
   export COINBASE_CDP_API_KEY=$COINBASE_STANDBY_KEY
   export COINBASE_CDP_PRIVATE_KEY_PATH=$COINBASE_STANDBY_KEY_PATH
   
   # Restart services
   systemctl restart trading-bot
   ```

3. **Revoke Compromised Key**:
   ```bash
   # Via API if available
   curl -X DELETE https://api.coinbase.com/v2/api_keys/$KEY_ID \
     -H "Authorization: Bearer $ADMIN_TOKEN"
   
   # Or via Console immediately
   ```

### Post-Incident Actions

1. **Security Audit**:
   - Review access logs
   - Check for unauthorized trades
   - Verify fund security
   - Document timeline

2. **Root Cause Analysis**:
   - How was key compromised?
   - What controls failed?
   - What improvements needed?

3. **Report Generation**:
   - Incident report for management
   - Regulatory notifications if required
   - Lessons learned document

## Rollback Procedure

If issues occur during rotation:

```bash
# Step 1: Switch back to old key
export COINBASE_CDP_API_KEY=$COINBASE_OLD_KEY
export COINBASE_CDP_PRIVATE_KEY_PATH=$COINBASE_OLD_KEY_PATH

# Step 2: Restart services
systemctl restart trading-bot

# Step 3: Verify functionality
python scripts/preflight_check.py
python scripts/validate_perps_client_week1.py

# Step 4: Investigate issues
tail -f /var/log/trading.log
python scripts/diagnose_rotation_issue.py
```

## Validation Scripts

### verify_key_status.py
```python
#!/usr/bin/env python3
"""Verify API key status and permissions."""

def verify_key(api_key, private_key_path):
    # Generate JWT
    token = generate_jwt(api_key, private_key_path)
    
    # Test endpoints
    endpoints = [
        "/api/v3/brokerage/accounts",
        "/api/v3/brokerage/products",
        "/api/v3/brokerage/orders"
    ]
    
    for endpoint in endpoints:
        response = requests.get(
            f"https://api.coinbase.com{endpoint}",
            headers={"Authorization": f"Bearer {token}"}
        )
        print(f"{endpoint}: {response.status_code}")
    
    return all(r.status_code == 200 for r in responses)
```

## Documentation Requirements

### Rotation Log Entry
```yaml
Date: 2025-04-15
Type: Scheduled Quarterly Rotation
Old Key ID: perps-prod-trading-write-202501
New Key ID: perps-prod-trading-write-202504
Performed By: John Doe
Approved By: Jane Smith
Start Time: 14:00 UTC
Completion Time: 16:30 UTC
Issues: None
Validation: Passed all checks
```

### Approval Chain
1. **Requester**: DevOps Team
2. **Technical Approval**: Engineering Lead
3. **Risk Approval**: Risk Manager
4. **Final Approval**: CTO/Security Officer

## Key Naming Convention

```
perps-{environment}-{purpose}-{permission}-{YYYYMM}

environment: prod|demo|sandbox
purpose: trading|monitor|analytics
permission: write|read
YYYYMM: Year and month of creation

Examples:
- perps-prod-trading-write-202504
- perps-demo-monitor-read-202504
- perps-sandbox-analytics-read-202504
```

## Success Criteria

- [ ] Zero downtime during rotation
- [ ] No failed trades due to auth issues
- [ ] All monitoring continues uninterrupted
- [ ] Old key successfully revoked
- [ ] Audit trail complete
- [ ] Documentation updated

## Contacts

- **Primary**: DevOps Team Lead
- **Secondary**: Security Team
- **Escalation**: CTO
- **Coinbase Support**: api-support@coinbase.com

---

*SOP Version: 1.0*
*Last Updated: 2025-08-30*
*Next Review: 2025-03-31*