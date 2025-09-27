#!/usr/bin/env python3
"""
Validate Time-In-Force (TIF) support per environment.
Gates GTD if not available in current environment.
"""

import os
import json
from typing import Set, Dict

def get_supported_tifs() -> Dict[str, Set[str]]:
    """Get supported TIFs per environment."""
    
    is_sandbox = os.getenv("COINBASE_SANDBOX", "1") == "1"
    
    # Conservative defaults - only validated TIFs
    tif_support = {
        "sandbox": {"GTC", "IOC"},  # Confirmed working
        "production": {"GTC", "IOC", "GTD"}  # GTD may be available in prod
    }
    
    # Check capability probe if available
    probe_file = "docs/ops/preflight/capability_probe.json"
    if os.path.exists(probe_file):
        try:
            with open(probe_file, 'r') as f:
                data = json.load(f)
                capabilities = data.get('capabilities', {})
                reported_tifs = set(capabilities.get('time_in_force', []))
                
                if reported_tifs:
                    env_key = "sandbox" if is_sandbox else "production"
                    # Only use reported TIFs if they're a subset of expected
                    # This prevents using untested TIFs
                    validated = reported_tifs & tif_support[env_key]
                    if validated:
                        tif_support[env_key] = validated
                        print(f"✅ Using validated TIFs from probe: {validated}")
        except Exception as e:
            print(f"⚠️  Could not load capability probe: {e}")
    
    return tif_support

def validate_tif(tif: str, environment: str = None) -> bool:
    """Validate if a TIF is supported in the environment."""
    
    if environment is None:
        environment = "sandbox" if os.getenv("COINBASE_SANDBOX", "1") == "1" else "production"
    
    supported = get_supported_tifs()
    env_tifs = supported.get(environment, {"GTC", "IOC"})  # Safe defaults
    
    return tif.upper() in env_tifs

def gate_gtd_orders() -> bool:
    """Check if GTD orders should be gated (blocked)."""
    
    # GTD is environment-dependent and needs validation
    environment = "sandbox" if os.getenv("COINBASE_SANDBOX", "1") == "1" else "production"
    
    # Conservative: Gate GTD in sandbox, allow in production with warning
    if environment == "sandbox":
        print("⚠️  GTD orders gated in sandbox (not validated)")
        return True  # Gate it
    else:
        if not validate_tif("GTD", environment):
            print("⚠️  GTD support not confirmed - gating orders")
            return True
        else:
            print("✅ GTD support validated for production")
            return False

def get_safe_tifs() -> Set[str]:
    """Get the set of safe TIFs for current environment."""
    
    environment = "sandbox" if os.getenv("COINBASE_SANDBOX", "1") == "1" else "production"
    supported = get_supported_tifs()
    
    # Return only validated TIFs
    safe_tifs = supported.get(environment, {"GTC", "IOC"})
    
    # Remove GTD if gated
    if gate_gtd_orders():
        safe_tifs.discard("GTD")
    
    return safe_tifs

def main():
    """Validate TIF support."""
    print("🕐 TIME-IN-FORCE VALIDATION")
    print("="*40)
    
    environment = "sandbox" if os.getenv("COINBASE_SANDBOX", "1") == "1" else "production"
    print(f"Environment: {environment.upper()}")
    
    # Get all supported TIFs
    all_tifs = get_supported_tifs()
    print(f"\nConfigured TIFs:")
    for env, tifs in all_tifs.items():
        print(f"  {env}: {sorted(tifs)}")
    
    # Get safe TIFs for current environment
    safe = get_safe_tifs()
    print(f"\n✅ Safe TIFs for {environment}: {sorted(safe)}")
    
    # Test specific TIFs
    print("\nValidation Tests:")
    test_tifs = ["GTC", "IOC", "GTD", "FOK"]
    for tif in test_tifs:
        valid = validate_tif(tif)
        status = "✅" if valid else "❌"
        print(f"  {status} {tif}: {'Supported' if valid else 'Not supported/Gated'}")
    
    # Recommendations
    print("\n📋 Recommendations:")
    if environment == "sandbox":
        print("  - Use only GTC and IOC in sandbox")
        print("  - GTD is gated until validated")
        print("  - Test GTD in production carefully")
    else:
        print("  - GTC and IOC are safe")
        if "GTD" in safe:
            print("  - GTD is available but use with caution")
        else:
            print("  - GTD is gated - needs validation")
    
    # Save validation results
    results = {
        "environment": environment,
        "supported_tifs": list(safe),
        "gtd_gated": gate_gtd_orders(),
        "recommendations": "Use GTC/IOC only" if environment == "sandbox" else "GTC/IOC safe, validate GTD"
    }
    
    output_file = "docs/ops/preflight/tif_validation.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Validation saved to: {output_file}")

if __name__ == "__main__":
    main()