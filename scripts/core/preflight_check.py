#!/usr/bin/env python3
"""
Preflight checklist for Coinbase Perpetuals trading.
Run before any deployment to validate configuration.
"""

import os
import sys
import time
import json
import subprocess
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
from decimal import Decimal

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class PreflightChecker:
    """Comprehensive pre-deployment validation."""
    
    def __init__(self):
        self.checks_passed = []
        self.checks_failed = []
        self.checks_warning = []
        
    def run_all_checks(self) -> bool:
        """Run all preflight checks."""
        print("🚀 PREFLIGHT CHECKLIST")
        print("=" * 60)
        print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
        print(f"Environment: {self._get_environment()}")
        print("=" * 60)
        
        checks = [
            ("Environment Variables", self.check_environment),
            ("API Authentication", self.check_authentication),
            ("Derivatives Configuration", self.check_derivatives),
            ("Safety Parameters", self.check_safety),
            ("Network Connectivity", self.check_connectivity),
            ("Clock Synchronization", self.check_clock),
            ("Disk Space", self.check_disk_space),
            ("Process Limits", self.check_process_limits),
            ("Dependencies", self.check_dependencies),
            ("Secrets Security", self.check_secrets),
        ]
        
        for check_name, check_func in checks:
            print(f"\n[{len(self.checks_passed) + len(self.checks_failed) + 1}/{len(checks)}] {check_name}")
            print("-" * 40)
            
            try:
                result, details = check_func()
                self._log_check_result(check_name, result, details)
            except Exception as e:
                self.checks_failed.append(check_name)
                print(f"❌ FAILED: {e}")
        
        return self._generate_summary()
    
    def check_environment(self) -> Tuple[str, Dict]:
        """Check environment variables."""
        required_vars = {
            "COINBASE_API_MODE": "advanced",
            "COINBASE_AUTH_TYPE": "JWT",
            "COINBASE_ENABLE_DERIVATIVES": "1"
        }
        
        optional_vars = [
            "COINBASE_CDP_API_KEY",
            "COINBASE_CDP_PRIVATE_KEY",
            "COINBASE_CDP_PRIVATE_KEY_PATH",
            "COINBASE_SANDBOX",
            "COINBASE_MAX_POSITION_SIZE",
            "COINBASE_DAILY_LOSS_LIMIT"
        ]
        
        missing = []
        incorrect = []
        configured = []
        
        # Check required
        for var, expected in required_vars.items():
            value = os.getenv(var)
            if not value:
                missing.append(var)
            elif value != expected:
                incorrect.append(f"{var}={value} (expected: {expected})")
            else:
                configured.append(f"{var}={value}")
        
        # Check optional
        for var in optional_vars:
            value = os.getenv(var)
            if value:
                # Mask sensitive values
                if "KEY" in var or "SECRET" in var or "PRIVATE" in var:
                    if len(value) > 20:
                        display = f"{var}={value[:6]}...{value[-4:]}"
                    else:
                        display = f"{var}=***configured***"
                elif "PATH" in var:
                    display = f"{var}={value}"  # Paths are OK to show
                else:
                    display = f"{var}={value}"
                configured.append(display)
        
        if missing or incorrect:
            return "FAIL", {
                "missing": missing,
                "incorrect": incorrect,
                "configured": configured
            }
        
        return "PASS", {"configured": configured}
    
    def check_authentication(self) -> Tuple[str, Dict]:
        """Check API authentication setup."""
        cdp_key = os.getenv("COINBASE_CDP_API_KEY")
        private_key = os.getenv("COINBASE_CDP_PRIVATE_KEY")
        private_key_path = os.getenv("COINBASE_CDP_PRIVATE_KEY_PATH")
        
        details = {}
        
        if not cdp_key:
            return "FAIL", {"error": "CDP API key not configured"}
        
        details["api_key"] = cdp_key[:30] + "..."
        
        # Check private key
        if private_key_path:
            if os.path.exists(private_key_path):
                # Check file permissions
                stat_info = os.stat(private_key_path)
                perms = oct(stat_info.st_mode)[-3:]
                if perms != "400":
                    details["warning"] = f"Private key permissions {perms} (should be 400)"
                    return "WARNING", details
                details["private_key"] = "file configured"
            else:
                return "FAIL", {"error": f"Private key file not found: {private_key_path}"}
        elif private_key:
            details["private_key"] = "environment variable"
        else:
            return "FAIL", {"error": "No private key configured"}
        
        # Try to generate a JWT token
        try:
            from bot_v2.features.brokerages.coinbase.cdp_auth_v2 import CDPAuthV2
            from bot_v2.features.brokerages.coinbase.models import APIConfig
            
            private_key_pem = private_key or (open(private_key_path).read() if private_key_path else None)
            if not private_key_pem:
                return "FAIL", {"error": "Private key content not available"}

            auth = CDPAuthV2(api_key_name=cdp_key, private_key_pem=private_key_pem)
            token = auth.generate_jwt(method="GET", path="/api/v3/brokerage/accounts")
            details["jwt_generation"] = "successful"
            
        except Exception as e:
            return "FAIL", {"error": f"JWT generation failed: {e}"}
        
        return "PASS", details
    
    def check_derivatives(self) -> Tuple[str, Dict]:
        """Check derivatives configuration."""
        enabled = os.getenv("COINBASE_ENABLE_DERIVATIVES") == "1"
        
        if not enabled:
            return "FAIL", {"error": "Derivatives not enabled"}
        
        # Check expected perpetuals
        try:
            from bot_v2.features.brokerages.coinbase.endpoints import get_perps_symbols
            expected = get_perps_symbols()
            
            return "PASS", {
                "derivatives_enabled": True,
                "expected_symbols": list(expected)
            }
        except Exception as e:
            return "WARNING", {
                "derivatives_enabled": True,
                "warning": f"Could not verify symbols: {e}"
            }
    
    def check_safety(self) -> Tuple[str, Dict]:
        """Check safety parameters."""
        params = {
            "max_position_size": float(os.getenv("COINBASE_MAX_POSITION_SIZE", "0.01")),
            "daily_loss_limit": float(os.getenv("COINBASE_DAILY_LOSS_LIMIT", "0.02")),
            "max_impact_bps": int(os.getenv("COINBASE_MAX_IMPACT_BPS", "15")),
            "kill_switch": os.getenv("COINBASE_KILL_SWITCH", "enabled"),
            "quiet_period_min": int(os.getenv("COINBASE_QUIET_MINUTES", "30"))
        }
        
        warnings = []
        
        # Check thresholds
        if params["max_position_size"] > 0.05:
            warnings.append("Position size > 5% of portfolio")
        
        if params["daily_loss_limit"] > 0.05:
            warnings.append("Daily loss limit > 5%")
        
        if params["max_impact_bps"] > 20:
            warnings.append("Market impact > 20 bps")
        
        if params["kill_switch"] != "enabled":
            warnings.append("Kill switch not enabled")
        
        if params["quiet_period_min"] < 15:
            warnings.append("Pre-funding quiet period < 15 minutes")
        
        if warnings:
            return "WARNING", {
                "parameters": params,
                "warnings": warnings
            }
        
        return "PASS", {"parameters": params}
    
    def check_connectivity(self) -> Tuple[str, Dict]:
        """Check network connectivity."""
        is_sandbox = os.getenv("COINBASE_SANDBOX") == "1"
        
        endpoints = {
            "api": "api.sandbox.coinbase.com" if is_sandbox else "api.coinbase.com",
            "websocket": "ws-direct.sandbox.coinbase.com" if is_sandbox else "ws-direct.coinbase.com"
        }
        
        results = {}
        
        for name, host in endpoints.items():
            try:
                # Use ping to check connectivity
                result = subprocess.run(
                    ["ping", "-c", "1", "-W", "2", host],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    # Extract latency
                    output = result.stdout
                    if "time=" in output:
                        latency = output.split("time=")[1].split()[0]
                        results[name] = f"✅ {latency}"
                    else:
                        results[name] = "✅ reachable"
                else:
                    results[name] = "❌ unreachable"
            except Exception as e:
                results[name] = f"❌ error: {e}"
        
        # Check if all endpoints are reachable
        all_ok = all("✅" in v for v in results.values())
        
        return "PASS" if all_ok else "FAIL", {
            "environment": "sandbox" if is_sandbox else "production",
            "endpoints": results
        }
    
    def check_clock(self) -> Tuple[str, Dict]:
        """Check system clock synchronization and JWT expiry window."""
        details = {}
        
        try:
            # Check system time
            system_time = datetime.now(timezone.utc)
            details["system_time"] = system_time.isoformat()
            
            # Get NTP offset (macOS/Linux)
            offset = None
            if sys.platform == "darwin":  # macOS
                result = subprocess.run(
                    ["sntp", "-t", "1", "time.apple.com"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            else:  # Linux
                result = subprocess.run(
                    ["ntpdate", "-q", "pool.ntp.org"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            
            output = result.stdout.lower()
            
            # Parse offset
            if "offset" in output:
                import re
                offset_match = re.search(r'offset\s*([-+]?\d+\.?\d*)', output)
                if offset_match:
                    offset = float(offset_match.group(1))
                    details["ntp_offset_seconds"] = offset
                    
                    # Check drift thresholds
                    if abs(offset) > 30:
                        details["error"] = "Clock drift > 30 seconds (JWT will fail)"
                        return "FAIL", details
                    elif abs(offset) > 5:
                        details["warning"] = "Clock drift > 5 seconds (may cause issues)"
                        return "WARNING", details
            
            # Check JWT expiry window (CDP tokens expire in 120 seconds)
            if offset is not None:
                jwt_window = 120  # seconds
                if abs(offset) > jwt_window / 2:
                    details["warning"] = f"Clock drift ({abs(offset):.1f}s) exceeds half JWT window ({jwt_window/2}s)"
                    return "WARNING", details
                else:
                    details["jwt_window_ok"] = True
            
            # Additional time checks
            import socket
            socket.setdefaulttimeout(2)
            try:
                import http.client
                conn = http.client.HTTPSConnection("worldtimeapi.org")
                conn.request("GET", "/api/timezone/UTC")
                response = conn.getresponse()
                if response.status == 200:
                    import json
                    data = json.loads(response.read())
                    api_time = datetime.fromisoformat(data['datetime'].replace('Z', '+00:00'))
                    local_drift = (system_time - api_time).total_seconds()
                    details["worldtime_drift_seconds"] = local_drift
                    
                    if abs(local_drift) > 30:
                        return "FAIL", details
            except:
                pass  # WorldTime API is optional
            
            if offset is not None:
                details["status"] = "synchronized"
                return "PASS", details
            else:
                details["note"] = "NTP check unavailable - ensure accurate time"
                return "PASS", details
            
        except subprocess.TimeoutExpired:
            details["warning"] = "NTP check timed out"
            details["system_time"] = datetime.now(timezone.utc).isoformat()
            return "WARNING", details
        except Exception as e:
            details["warning"] = f"Clock check error: {e}"
            details["system_time"] = datetime.now(timezone.utc).isoformat()
            return "WARNING", details
    
    def check_disk_space(self) -> Tuple[str, Dict]:
        """Check available disk space."""
        try:
            result = subprocess.run(
                ["df", "-h", "/"],
                capture_output=True,
                text=True
            )
            
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                # Parse the data line
                parts = lines[1].split()
                if len(parts) >= 4:
                    used = parts[2]
                    available = parts[3]
                    percent = parts[4].replace('%', '')
                    
                    percent_used = int(percent)
                    
                    if percent_used > 90:
                        return "FAIL", {
                            "used": used,
                            "available": available,
                            "percent_used": f"{percent_used}%",
                            "error": "Disk usage > 90%"
                        }
                    elif percent_used > 80:
                        return "WARNING", {
                            "used": used,
                            "available": available,
                            "percent_used": f"{percent_used}%",
                            "warning": "Disk usage > 80%"
                        }
                    else:
                        return "PASS", {
                            "used": used,
                            "available": available,
                            "percent_used": f"{percent_used}%"
                        }
            
            return "WARNING", {"warning": "Could not parse disk usage"}
            
        except Exception as e:
            return "WARNING", {"warning": f"Could not check disk space: {e}"}
    
    def check_process_limits(self) -> Tuple[str, Dict]:
        """Check process limits and resources."""
        try:
            import resource
            
            limits = {}
            
            # Check file descriptor limit
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            limits["file_descriptors"] = {"soft": soft, "hard": hard}
            
            if soft < 1024:
                return "WARNING", {
                    "limits": limits,
                    "warning": "File descriptor limit < 1024"
                }
            
            # Check process limit
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
                limits["processes"] = {"soft": soft, "hard": hard}
            except:
                pass
            
            return "PASS", {"limits": limits}
            
        except Exception as e:
            return "WARNING", {"warning": f"Could not check limits: {e}"}
    
    def check_dependencies(self) -> Tuple[str, Dict]:
        """Check required Python dependencies."""
        required = [
            "coinbase",
            "websockets",
            "pandas",
            "numpy",
            "dotenv"
        ]
        
        installed = []
        missing = []
        
        for package in required:
            try:
                __import__(package.replace("-", "_"))
                installed.append(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            return "FAIL", {
                "installed": installed,
                "missing": missing
            }
        
        return "PASS", {"installed": installed}
    
    def check_secrets(self) -> Tuple[str, Dict]:
        """Check secrets management security."""
        warnings = []
        details = {}
        
        # Check if secrets are in environment
        if os.getenv("COINBASE_CDP_PRIVATE_KEY"):
            warnings.append("Private key in environment (use file instead)")
        
        # Check key directory and file
        key_path = os.getenv("COINBASE_CDP_PRIVATE_KEY_PATH")
        if key_path and os.path.exists(key_path):
            key_stat = os.stat(key_path)
            key_perms = oct(key_stat.st_mode)[-3:]
            
            # Check ownership
            import pwd
            key_owner = pwd.getpwuid(key_stat.st_uid).pw_name
            current_user = pwd.getpwuid(os.getuid()).pw_name
            
            details["key_file"] = {
                "path": key_path,
                "permissions": key_perms,
                "owner": key_owner,
                "current_user": current_user
            }
            
            if key_perms not in ["400", "600"]:
                warnings.append(f"Key file permissions {key_perms} (should be 400)")
            
            if key_owner != current_user:
                warnings.append(f"Key file owned by {key_owner}, running as {current_user}")
            
            # Check directory
            key_dir = os.path.dirname(key_path)
            if os.path.exists(key_dir):
                dir_stat = os.stat(key_dir)
                dir_perms = oct(dir_stat.st_mode)[-3:]
                dir_owner = pwd.getpwuid(dir_stat.st_uid).pw_name
                
                details["key_directory"] = {
                    "path": key_dir,
                    "permissions": dir_perms,
                    "owner": dir_owner
                }
                
                if dir_perms != "700":
                    warnings.append(f"Key directory permissions {dir_perms} (should be 700)")
        
        # Check .env file permissions if it exists
        env_file = os.path.join(project_root, ".env")
        if os.path.exists(env_file):
            stat_info = os.stat(env_file)
            perms = oct(stat_info.st_mode)[-3:]
            if perms not in ["600", "400"]:
                warnings.append(f".env file permissions {perms} (should be 600 or 400)")
        
        # Check git
        gitignore = os.path.join(project_root, ".gitignore")
        if os.path.exists(gitignore):
            with open(gitignore, 'r') as f:
                content = f.read()
                if ".env" not in content:
                    warnings.append(".env not in .gitignore")
        
        if warnings:
            details["warnings"] = warnings
            return "WARNING", details
        
        details["status"] = "Secrets properly secured"
        return "PASS", details
    
    def _get_environment(self) -> str:
        """Get current environment."""
        if os.getenv("COINBASE_SANDBOX") == "1":
            return "SANDBOX"
        return "PRODUCTION"
    
    def _log_check_result(self, name: str, result: str, details: Dict):
        """Log individual check result."""
        symbols = {
            "PASS": "✅",
            "FAIL": "❌",
            "WARNING": "⚠️"
        }
        
        print(f"{symbols.get(result, '❓')} {result}")
        
        # Track results
        if result == "PASS":
            self.checks_passed.append(name)
        elif result == "FAIL":
            self.checks_failed.append(name)
        else:
            self.checks_warning.append(name)
        
        # Print details
        for key, value in details.items():
            if isinstance(value, list):
                print(f"  {key}:")
                for item in value:
                    print(f"    - {item}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
    
    def _generate_summary(self) -> bool:
        """Generate preflight summary."""
        print("\n" + "=" * 60)
        print("📊 PREFLIGHT SUMMARY")
        print("=" * 60)
        
        total = len(self.checks_passed) + len(self.checks_failed) + len(self.checks_warning)
        
        print(f"✅ Passed: {len(self.checks_passed)}/{total}")
        print(f"⚠️  Warnings: {len(self.checks_warning)}/{total}")
        print(f"❌ Failed: {len(self.checks_failed)}/{total}")
        
        if self.checks_failed:
            print("\n❌ FAILED CHECKS:")
            for check in self.checks_failed:
                print(f"  - {check}")
            print("\n🔴 PREFLIGHT FAILED - Do not proceed with deployment")
            return False
        
        elif self.checks_warning:
            print("\n⚠️  WARNINGS:")
            for check in self.checks_warning:
                print(f"  - {check}")
            print("\n🟡 PREFLIGHT PASSED WITH WARNINGS - Review before deployment")
            return True
        
        else:
            print("\n🟢 ALL PREFLIGHT CHECKS PASSED - Ready for deployment")
            return True


def main():
    """Run preflight checks."""
    checker = PreflightChecker()
    success = checker.run_all_checks()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
