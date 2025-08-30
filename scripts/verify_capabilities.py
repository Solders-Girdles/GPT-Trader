#!/usr/bin/env python3
"""
Verify all claimed system capabilities and update PROJECT_STATE.json with results.
This replaces scattered reports with actual executable verification.
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def run_test(command: str, timeout: int = 30) -> tuple[bool, str]:
    """Run a test command and return success status and output."""
    if not command:
        return False, "No test command provided"
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        success = result.returncode == 0
        output = result.stdout if success else result.stderr
        return success, output[:500]  # Limit output length
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout} seconds"
    except Exception as e:
        return False, str(e)

def verify_components():
    """Test each component and update verification status."""
    # Load current state
    state_file = Path("PROJECT_STATE.json")
    with open(state_file) as f:
        state = json.load(f)
    
    results = []
    print("🔍 Verifying System Capabilities...\n")
    
    for component, config in state["components"].items():
        print(f"Testing {component}...")
        
        if config.get("test_command"):
            success, output = run_test(config["test_command"])
            config["verified"] = True
            config["last_verified"] = datetime.now().isoformat()
            
            if success:
                print(f"  ✅ {component}: WORKING")
                if config["status"] != "working":
                    print(f"     ⚠️  Status mismatch! Was marked as: {config['status']}")
                    config["status"] = "working"
            else:
                print(f"  ❌ {component}: FAILED")
                if config["status"] == "working":
                    print(f"     ⚠️  Status mismatch! Was marked as working but failed!")
                    config["status"] = "failed"
                print(f"     Error: {output[:100]}...")
        else:
            print(f"  ⏭️  {component}: No test command")
        
        results.append((component, config["status"], config["verified"]))
    
    # Calculate real functional percentage
    working = sum(1 for _, status, _ in results if status == "working")
    total = len(results)
    functional_percentage = int((working / total) * 100)
    
    print(f"\n📊 Summary:")
    print(f"  - Components tested: {total}")
    print(f"  - Working: {working}")
    print(f"  - Failed: {total - working}")
    print(f"  - Functional: {functional_percentage}%")
    
    if state["system_status"]["functional_percentage"] != functional_percentage:
        print(f"  ⚠️  Updating functional percentage from {state['system_status']['functional_percentage']}% to {functional_percentage}%")
    
    state["system_status"]["functional_percentage"] = functional_percentage
    state["system_status"]["verified"] = True
    state["system_status"]["last_test_run"] = datetime.now().isoformat()
    
    # Save updated state
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)
    
    print(f"\n✅ PROJECT_STATE.json updated with verified results")
    
    return functional_percentage

def main():
    """Run verification and exit with appropriate code."""
    try:
        percentage = verify_components()
        if percentage < 50:
            print("\n⚠️  System is less than 50% functional!")
            sys.exit(1)
        sys.exit(0)
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()