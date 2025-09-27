# Coinbase Troubleshooting Guide

---
status: current
last-updated: 2025-01-01
consolidates:
  - CDP_PERMISSION_ISSUE.md
  - COINBASE_FIXES_PR.md
  - COINBASE_REMEDIATION_PLAN.md
---

## Common Issues and Solutions

### Authentication Errors

#### CDP JWT Authentication Issues

**Problem**: 401 Unauthorized on private endpoints despite valid JWT
```
Error: {'error': 'UNAUTHENTICATED', 'message': 'unauthenticated'}
```

**Solutions**:
1. Verify API key format: `organizations/{org-id}/apiKeys/{key-id}`
2. Check private key includes EC PRIVATE KEY headers
3. Ensure all required permissions are granted in CDP dashboard
4. Verify `COINBASE_AUTH_TYPE=JWT` is set

#### Legacy HMAC Authentication

**Problem**: Invalid API key format
```
Error: Invalid API Key
```

**Solutions**:
1. Sandbox requires Exchange API keys (not CDP)
2. Ensure passphrase is included for Exchange API
3. Set `COINBASE_API_MODE=exchange` for sandbox

### WebSocket Connection Issues

#### Connection Drops

**Problem**: WebSocket disconnects frequently

**Solutions**:
1. Check heartbeat implementation
2. Verify subscription limits (100 channels max)
3. Monitor reconnection logic
4. Enable debug logging: `export WS_DEBUG=1`

#### Authentication Failed

**Problem**: User channel requires authentication

**Solutions**:
1. CDP JWT required for derivatives user events
2. Ensure `auth_type="JWT"` in config
3. Check JWT token generation in logs

### Order Placement Issues

#### Size Increment Errors

**Problem**: Order rejected due to invalid size
```
Error: Size must be a multiple of 0.001
```

**Solutions**:
1. Use product catalog to get size increments
2. Apply quantization before submission
3. Check `enforce_perp_rules()` implementation

#### Reduce-Only Rejection

**Problem**: Reduce-only order rejected

**Solutions**:
1. Verify position exists before placing reduce-only order
2. Check order side matches position reduction
3. Ensure quantity doesn't exceed position size

### Rate Limiting

#### 429 Too Many Requests

**Problem**: Rate limit exceeded

**Solutions**:
1. Default limit: 100 requests/minute
2. Enable throttling: `enable_throttle=True`
3. Implement exponential backoff
4. Use WebSocket for real-time data instead of polling

### Environment-Specific Issues

#### Sandbox Limitations

**Problem**: Perpetuals not available in sandbox

**Reality**: Sandbox only supports spot trading (BTC-USD, ETH-USD)

**Solution**: Use production with canary profile for safe perpetuals testing

#### Production vs Sandbox API Differences

| Feature | Production | Sandbox |
|---------|------------|---------|
| Products | Perpetuals | Spot only |
| API Version | Advanced v3 | Exchange v2 |
| Authentication | CDP JWT | HMAC |
| WebSocket | Real-time | Real-time |

### Debugging Tools

#### Check CDP Key
```bash
python scripts/diagnose_cdp_key.py
```

#### Test WebSocket
```bash
python scripts/ws_probe.py --sandbox
```

#### Verify Permissions
```bash
python scripts/test_cdp_permissions.py
```

#### Raw API Test
```bash
curl -X GET "https://api.coinbase.com/api/v3/brokerage/accounts" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Error Code Reference

| Code | Meaning | Solution |
|------|---------|----------|
| 401 | Unauthorized | Check API key permissions |
| 403 | Forbidden | Verify account eligibility |
| 429 | Rate Limited | Implement backoff |
| 500 | Server Error | Retry with backoff |
| 503 | Service Unavailable | Wait and retry |

### Connection Optimization

#### Disable Keep-Alive (for debugging)
```python
client = CoinbaseClient(
    enable_keep_alive=False  # Disable for proxy issues
)
```

#### Adjust Timeouts
```python
client = CoinbaseClient(
    timeout=30,  # Increase for slow connections
    max_retries=5  # More retries for unstable networks
)
```

### Getting Help

1. Check logs: `tail -f logs/coinbase.log`
2. Enable debug mode: `export COINBASE_DEBUG=1`
3. Test with minimal example
4. Review [API documentation](https://docs.cdp.coinbase.com/)