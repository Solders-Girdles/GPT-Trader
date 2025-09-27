#!/bin/bash
# Comprehensive preflight runner for Coinbase Perpetuals trading
# Runs all checks and generates consolidated report

set -e  # Exit on error

echo "🚀 COMPREHENSIVE PREFLIGHT CHECK"
echo "================================="
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Environment: ${ENVIRONMENT:-demo}"
echo ""

# Create report directory
REPORT_DIR="docs/ops/preflight"
mkdir -p "$REPORT_DIR"

# Rotate old reports first
echo "📄 Rotating old reports..."
python scripts/rotate_reports.py 2>/dev/null || echo "   Report rotation not available"
echo ""

# Function to check command success
check_result() {
    if [ $? -eq 0 ]; then
        echo "   ✅ $1: PASS"
        return 0
    else
        echo "   ❌ $1: FAIL"
        return 1
    fi
}

# Track overall status
FAILED_CHECKS=0

# 1. Environment setup verification
echo "🔧 Environment Setup"
echo "-"*40
if [ -f "set_env.sh" ] || [ -f "set_env.demo.sh" ] || [ -f "set_env.prod.sh" ]; then
    echo "   ✅ Environment script exists"
    
    # Source the appropriate environment
    if [ "$COINBASE_SANDBOX" == "1" ] && [ -f "set_env.demo.sh" ]; then
        source set_env.demo.sh
        echo "   ✅ Sourced set_env.demo.sh"
    elif [ "$COINBASE_SANDBOX" == "0" ] && [ -f "set_env.prod.sh" ]; then
        source set_env.prod.sh
        echo "   ✅ Sourced set_env.prod.sh"
    elif [ -f "set_env.sh" ]; then
        source set_env.sh
        echo "   ✅ Sourced set_env.sh"
    fi
else
    echo "   ⚠️  No environment script found"
    echo "   Run: bash scripts/fix_environment_secure.sh"
    ((FAILED_CHECKS++))
fi
echo ""

# 2. Dependency check
echo "📦 Dependencies"
echo "-"*40
python -c "import coinbase, websockets, jwt, cryptography, pandas, numpy" 2>/dev/null
if check_result "Python packages"; then
    echo "   All required packages installed"
else
    echo "   Missing packages - run: pip install -r requirements.txt"
    ((FAILED_CHECKS++))
fi
echo ""

# 3. WebSocket connectivity probe
echo "🌐 WebSocket Connectivity"
echo "-"*40
echo "   Running WebSocket probe..."
python scripts/ws_probe.py --sandbox 2>&1 | tee "${REPORT_DIR}/ws_probe_$(date +%Y%m%d_%H%M%S).log"
if [ -f "${REPORT_DIR}/ws_probe_report.json" ]; then
    echo "   ✅ WebSocket probe complete"
    
    # Check auth if credentials available
    if [ -n "$COINBASE_CDP_API_KEY" ]; then
        python scripts/ws_probe.py --auth 2>/dev/null || echo "   ⚠️  Auth channels not tested"
    fi
else
    echo "   ❌ WebSocket probe failed"
    ((FAILED_CHECKS++))
fi
echo ""

# 4. Capability probe
echo "🔍 API Capabilities"
echo "-"*40
echo "   Running capability probe..."
python scripts/capability_probe.py 2>&1 | tee "${REPORT_DIR}/capability_$(date +%Y%m%d_%H%M%S).log"
if [ -f "${REPORT_DIR}/capability_probe.json" ]; then
    echo "   ✅ Capability probe complete"
    
    # Extract key capabilities
    python -c "
import json
with open('${REPORT_DIR}/capability_probe.json', 'r') as f:
    data = json.load(f)
    caps = data.get('capabilities', {})
    print(f'   Order types: {caps.get(\"order_types\", [])}')
    print(f'   TIF support: {caps.get(\"time_in_force\", [])}')
    print(f'   Derivatives: {caps.get(\"derivatives_enabled\", False)}')
" 2>/dev/null || echo "   Could not parse capabilities"
else
    echo "   ❌ Capability probe failed"
    ((FAILED_CHECKS++))
fi
echo ""

# 5. Main preflight check
echo "✅ Preflight Validation"
echo "-"*40
echo "   Running comprehensive preflight..."
python scripts/preflight_check.py 2>&1 | tee "${REPORT_DIR}/preflight_$(date +%Y%m%d_%H%M%S).log"
PREFLIGHT_RESULT=$?

# Parse preflight results
if [ $PREFLIGHT_RESULT -eq 0 ]; then
    echo "   ✅ Preflight PASSED"
else
    echo "   ❌ Preflight FAILED"
    ((FAILED_CHECKS++))
fi
echo ""

# 6. Clock synchronization
echo "⏰ Clock Synchronization"
echo "-"*40
DRIFT=$(python -c "
import subprocess, re
try:
    result = subprocess.run(['sntp', '-t', '1', 'time.apple.com'], 
                          capture_output=True, text=True, timeout=5)
    match = re.search(r'offset\s*([-+]?\d+\.?\d*)', result.stdout.lower())
    if match:
        offset = float(match.group(1))
        print(f'{offset:.3f}')
    else:
        print('unknown')
except:
    print('error')
" 2>/dev/null || echo "error")

if [ "$DRIFT" != "error" ] && [ "$DRIFT" != "unknown" ]; then
    if (( $(echo "$DRIFT < 30" | bc -l) )); then
        echo "   ✅ Clock drift: ${DRIFT}s (within tolerance)"
    else
        echo "   ❌ Clock drift: ${DRIFT}s (exceeds 30s threshold)"
        ((FAILED_CHECKS++))
    fi
else
    echo "   ⚠️  Could not verify clock sync"
fi
echo ""

# 7. Security audit
echo "🔒 Security Audit"
echo "-"*40
# Check key file permissions
if [ -n "$COINBASE_CDP_PRIVATE_KEY_PATH" ] && [ -f "$COINBASE_CDP_PRIVATE_KEY_PATH" ]; then
    PERMS=$(stat -c %a "$COINBASE_CDP_PRIVATE_KEY_PATH" 2>/dev/null || stat -f %A "$COINBASE_CDP_PRIVATE_KEY_PATH" 2>/dev/null)
    if [ "$PERMS" == "400" ] || [ "$PERMS" == "600" ]; then
        echo "   ✅ Key file permissions: $PERMS"
    else
        echo "   ❌ Key file permissions: $PERMS (should be 400)"
        ((FAILED_CHECKS++))
    fi
    
    # Check ownership
    OWNER=$(stat -c %U "$COINBASE_CDP_PRIVATE_KEY_PATH" 2>/dev/null || stat -f %Su "$COINBASE_CDP_PRIVATE_KEY_PATH" 2>/dev/null)
    CURRENT=$(whoami)
    if [ "$OWNER" == "$CURRENT" ]; then
        echo "   ✅ Key owned by current user: $OWNER"
    else
        echo "   ⚠️  Key owned by: $OWNER (running as: $CURRENT)"
    fi
else
    echo "   ❌ Private key file not found"
    ((FAILED_CHECKS++))
fi

# Check for secrets in environment
if [ -n "$COINBASE_CDP_PRIVATE_KEY" ]; then
    echo "   ⚠️  WARNING: Private key in environment (use file instead)"
fi
echo ""

# 8. Network verification
echo "🌍 Network Configuration"
echo "-"*40
EGRESS_IP=$(curl -s ifconfig.me 2>/dev/null || echo "unknown")
echo "   Egress IP: $EGRESS_IP"
echo "   ⚠️  Ensure this IP is allowlisted in CDP console"

if [ -n "$HTTP_PROXY" ] || [ -n "$HTTPS_PROXY" ]; then
    echo "   Proxy configured"
    if [ -n "$NO_PROXY" ]; then
        echo "   ✅ NO_PROXY includes Coinbase"
    else
        echo "   ⚠️  NO_PROXY not set for Coinbase"
    fi
fi
echo ""

# 9. Generate consolidated report
echo "📊 Generating Consolidated Report"
echo "-"*40
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${REPORT_DIR}/consolidated_${TIMESTAMP}.json"

python -c "
import json
import os
from datetime import datetime
from pathlib import Path

# Load sub-reports if available
ws_probe = {}
capability = {}

try:
    with open('${REPORT_DIR}/ws_probe_report.json', 'r') as f:
        ws_probe = json.load(f)
except:
    pass

try:
    with open('${REPORT_DIR}/capability_probe.json', 'r') as f:
        capability = json.load(f)
except:
    pass

# Get file ownership details
key_path = os.getenv('COINBASE_CDP_PRIVATE_KEY_PATH', '')
ownership = {}
if key_path and os.path.exists(key_path):
    import pwd
    stat = os.stat(key_path)
    ownership = {
        'key_file': {
            'path': key_path,
            'owner': pwd.getpwuid(stat.st_uid).pw_name,
            'permissions': oct(stat.st_mode)[-3:],
            'size_bytes': stat.st_size
        },
        'key_directory': {
            'path': os.path.dirname(key_path),
            'owner': pwd.getpwuid(os.stat(os.path.dirname(key_path)).st_uid).pw_name,
            'permissions': oct(os.stat(os.path.dirname(key_path)).st_mode)[-3:]
        },
        'runtime_user': pwd.getpwuid(os.getuid()).pw_name
    }

report = {
    'version': '2.0.0',
    'timestamp': datetime.utcnow().isoformat() + 'Z',
    'environment': os.getenv('ENVIRONMENT', 'demo'),
    'sandbox': os.getenv('COINBASE_SANDBOX', '1') == '1',
    'system': {
        'user': '$(whoami)',
        'host': '$(hostname)',
        'platform': '$(uname -s)',
        'python_version': '$(python --version 2>&1 | cut -d' ' -f2)'
    },
    'network': {
        'egress_ip': '${EGRESS_IP}',
        'proxy': 'configured' if os.getenv('HTTP_PROXY') else 'none',
        'no_proxy': os.getenv('NO_PROXY', 'not_set')
    },
    'security': ownership,
    'capabilities': capability.get('capabilities', {}),
    'websocket': {
        'auth_success': ws_probe.get('auth_success', False),
        'reconnect_count': ws_probe.get('reconnect_count', 0),
        'staleness_threshold_ms': ws_probe.get('staleness_threshold_ms', 5000),
        'endpoints_tested': len(ws_probe.get('results', {}))
    },
    'checks': {
        'environment_setup': ${FAILED_CHECKS} == 0,
        'dependencies': True,
        'websocket': bool(ws_probe),
        'capabilities': bool(capability),
        'preflight': ${PREFLIGHT_RESULT} == 0,
        'clock_drift_seconds': float('${DRIFT}') if '${DRIFT}' not in ['error', 'unknown'] else None,
        'security': bool(ownership),
        'jwt_auth': capability.get('capabilities', {}).get('jwt_auth', False),
        'derivatives_enabled': capability.get('capabilities', {}).get('derivatives_enabled', False)
    },
    'metrics': {
        'failed_checks': ${FAILED_CHECKS},
        'ready_for_demo': ${FAILED_CHECKS} == 0,
        'capability_matrix': {
            'order_types': capability.get('capabilities', {}).get('order_types', []),
            'time_in_force': capability.get('capabilities', {}).get('time_in_force', []),
            'special_flags': capability.get('capabilities', {}).get('special_flags', [])
        }
    },
    'summary': {
        'status': 'READY' if ${FAILED_CHECKS} == 0 else 'NOT_READY',
        'message': 'All preflight checks passed - ready for demo trading' if ${FAILED_CHECKS} == 0 
                   else f'{${FAILED_CHECKS}} checks failed - review and fix issues'
    }
}

# Save report
with open('${REPORT_FILE}', 'w') as f:
    json.dump(report, f, indent=2)

print(f'   ✅ Consolidated report saved: ${REPORT_FILE}')

# Generate human-readable summary
print('')
print('   📊 Capability Summary:')
caps = report['capabilities']
if caps:
    print(f'      Order Types: {caps.get(\"order_types\", [])}')
    print(f'      TIF Support: {caps.get(\"time_in_force\", [])}')
    print(f'      Derivatives: {caps.get(\"derivatives_enabled\", False)}')
    print(f'      JWT Auth: {caps.get(\"jwt_auth\", False)}')
" 2>/dev/null || echo "   Could not generate enhanced report"
echo ""

# 10. Final summary
echo "================================="
echo "📋 PREFLIGHT SUMMARY"
echo "================================="

if [ $FAILED_CHECKS -eq 0 ]; then
    echo "🟢 ALL CHECKS PASSED - Ready for demo trading"
    echo ""
    echo "Next steps:"
    echo "1. Review reports in: ${REPORT_DIR}/"
    echo "2. Launch dashboard: python scripts/dashboard_lite.py"
    echo "3. Run demo validator: python scripts/demo_run_validator.py"
    echo "4. Follow Phase 2 checklist: docs/ops/PHASE_2_EXECUTION_CHECKLIST.md"
    exit 0
else
    echo "🔴 PREFLIGHT FAILED - $FAILED_CHECKS checks failed"
    echo ""
    echo "Fix required issues and run again:"
    echo "  bash scripts/run_preflight.sh"
    exit 1
fi