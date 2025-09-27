# IP Allowlist Configuration

## Overview
IP allowlisting provides an additional layer of security by restricting API access to specific IP addresses.

## Production Configuration

### Primary IPs
```yaml
Production Egress IPs:
  primary: xxx.xxx.xxx.xxx      # Main datacenter
  secondary: yyy.yyy.yyy.yyy    # Backup datacenter
  failover: zzz.zzz.zzz.zzz     # DR site
```

### Adding IPs to Allowlist

1. **Via Coinbase UI**:
   - Navigate to Settings → API → IP Allowlist
   - Add new IP with description
   - Save and confirm via email

2. **Via API** (if supported):
   ```bash
   curl -X POST https://api.coinbase.com/v2/user/api-keys/{key_id}/whitelist \
     -H "Authorization: Bearer $TOKEN" \
     -d '{"ip_address": "xxx.xxx.xxx.xxx"}'
   ```

## Rotation Procedure

### Planned Rotation
1. **Week -1**: Add new IP to allowlist
2. **Week 0**: Deploy application with dual IP support
3. **Week 0**: Test connectivity from new IP
4. **Week +1**: Switch primary traffic to new IP
5. **Week +2**: Monitor for issues
6. **Week +3**: Remove old IP from allowlist

### Emergency Rotation
1. Add emergency IP immediately
2. Update application configuration
3. Deploy with zero downtime
4. Verify connectivity
5. Remove compromised IP
6. Document incident

## IP Management Best Practices

### Documentation
- Maintain IP inventory with ownership
- Document purpose for each IP
- Track addition/removal dates
- Keep audit trail

### Security
- Use static IPs only
- Implement egress filtering
- Monitor for unauthorized IPs
- Regular security audits

## Monitoring

### Health Checks
```python
def verify_ip_allowlist():
    """Verify API accessible from allowlisted IP."""
    response = requests.get(
        "https://api.coinbase.com/v2/user",
        headers={"Authorization": f"Bearer {token}"}
    )
    return response.status_code == 200
```

### Alerts
- Connection failures from allowlisted IPs
- Successful connections from non-allowlisted IPs
- IP rotation reminders
- Certificate expiration warnings

## Troubleshooting

### Common Issues

#### API Returns 403 Forbidden
- Verify IP is allowlisted
- Check IP hasn't changed (dynamic IP issue)
- Confirm no proxy/NAT changes
- Validate API key still active

#### Intermittent Connectivity
- Check for multiple egress IPs
- Verify load balancer configuration
- Ensure all cluster nodes allowlisted
- Monitor for IP address changes

## Compliance

### Requirements
- Quarterly IP audit
- Annual penetration testing
- Change control documentation
- Security team approval for changes

### Audit Log
All IP changes must include:
- Date and time
- Requester
- Approver
- Business justification
- Rollback plan

---

*Last Updated: 2025-08-30*
*Review Schedule: Quarterly*