# Coinbase CB-VERSION Header Fix

## Problem Identified

The Coinbase CDP authentication was failing with 401 errors on account/trading endpoints despite successful JWT generation. The root cause was the missing **CB-VERSION** header, which is required by Coinbase for all API calls.

## The CB-VERSION Header

### What is CB-VERSION?

CB-VERSION is a required HTTP header for all Coinbase API calls that specifies which version of the API you want to use. It ensures API stability and prevents breaking changes from affecting your integration.

### Format

- **Format**: `YYYY-MM-DD` (ISO date format)
- **Example**: `2024-10-24`
- **Important**: Do NOT use the current date - use a stable, tested version

### Why It's Required

1. **API Versioning**: Coinbase uses date-based versioning to manage API changes
2. **Backward Compatibility**: Ensures your code continues working when Coinbase updates their API
3. **Permission Model**: Different API versions may have different permission requirements
4. **CDP Compatibility**: CDP keys specifically require the CB-VERSION header

## Implementation Details

### Files Modified

1. **client.py** - Added CB-VERSION to all requests
2. **models.py** - Added api_version field to APIConfig
3. **adapter.py** - Pass api_version to client
4. **.env.template** - Added COINBASE_API_VERSION configuration
5. **Test scripts** - Updated to use api_version

### Code Changes

#### 1. CoinbaseClient Constructor
```python
def __init__(self, base_url: str, auth: Optional[...] = None, 
             timeout: int = 30, api_version: str = "2024-10-24"):
    self.api_version = api_version
    # ...
```

#### 2. Request Headers
```python
headers: Dict[str, str] = {
    "Content-Type": "application/json",
    "CB-VERSION": self.api_version  # Required header
}
```

#### 3. APIConfig Model
```python
@dataclass
class APIConfig:
    # ... other fields ...
    api_version: str = "2024-10-24"  # CB-VERSION header value
```

## Configuration

### Environment Variable

Add to your `.env` file:
```bash
# Coinbase API Version (CB-VERSION header)
# Format: YYYY-MM-DD (use a stable date, not current date)
COINBASE_API_VERSION=2024-10-24
```

### Recommended Versions

- **2024-10-24** - Latest stable version (recommended)
- **2023-06-01** - Widely compatible version
- **2022-01-06** - Legacy support version

## Testing

### Test Script

Run the CB-VERSION test script:
```bash
# Test with multiple versions
python scripts/test_cb_version_fix.py --test-versions

# Test with specific version
python scripts/test_cb_version_fix.py --version 2023-06-01

# Test current configuration
python scripts/test_cb_version_fix.py
```

### Expected Output

When CB-VERSION is working correctly:
```
✅ Time endpoint works: 2024-10-24T12:00:00Z
✅ SUCCESS! Retrieved 49 accounts
🎉 CB-VERSION 2024-10-24 WORKS!
```

## Troubleshooting

### Still Getting 401 Errors?

1. **Try Different Versions**: Some CDP keys may require specific versions
   ```bash
   COINBASE_API_VERSION=2023-06-01  # Try older version
   ```

2. **Check CDP Key Provisioning**: Even with CB-VERSION, your CDP key must be properly provisioned by Coinbase

3. **Contact Support**: If no version works, contact Coinbase support to activate your CDP key for Advanced Trade API v3

### Common Issues

| Error | Cause | Solution |
|-------|-------|----------|
| 401 Unauthorized | Missing CB-VERSION | Ensure header is included |
| 401 with CB-VERSION | Wrong API version | Try different versions |
| 403 Forbidden | CDP key not provisioned | Contact Coinbase support |
| 400 Bad Request | Invalid version format | Use YYYY-MM-DD format |

## Best Practices

1. **Lock Your Version**: Once you find a working version, stick with it
2. **Don't Use Current Date**: This can break when API changes
3. **Test Before Upgrading**: When changing versions, test thoroughly
4. **Document Your Version**: Keep track of which version your code expects

## Migration Guide

### For Existing Code

1. Update your `.env` file:
   ```bash
   COINBASE_API_VERSION=2024-10-24
   ```

2. Update any manual API calls to include the header:
   ```python
   headers = {
       "CB-VERSION": "2024-10-24",
       # ... other headers
   }
   ```

3. Test your integration:
   ```bash
   python scripts/test_cb_version_fix.py
   ```

## Conclusion

The CB-VERSION header is mandatory for Coinbase API calls. This fix ensures the header is automatically included in all requests, resolving the authentication issues with CDP keys. Once your CDP key is properly provisioned by Coinbase and the CB-VERSION header is included, full trading functionality will be available.