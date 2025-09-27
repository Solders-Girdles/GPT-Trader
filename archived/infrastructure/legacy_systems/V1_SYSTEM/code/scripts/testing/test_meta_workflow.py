#!/usr/bin/env python3
"""
Test script for the complete meta-workflow system
"""

import asyncio
from claude_meta_workflow import ClaudeMetaWorkflow


async def test_workflow():
    """Test the meta-workflow with various scenarios"""
    
    print("="*60)
    print("🧪 TESTING META-WORKFLOW SYSTEM")
    print("="*60)
    
    # Initialize workflow
    workflow = ClaudeMetaWorkflow()
    
    # Test 1: Duplicate work prevention
    print("\n📝 TEST 1: Duplicate Work Prevention")
    print("-"*40)
    
    request1 = "Create a test for PaperTradingEngine"
    result1 = await workflow.handle_user_request(request1)
    
    # Try the same request again
    print("\n🔄 Trying same request again...")
    result2 = await workflow.handle_user_request(request1)
    
    if result2.get("notes") == "Task already completed in previous session":
        print("✅ Duplicate work successfully prevented!")
    
    # Test 2: Pattern learning
    print("\n📝 TEST 2: Pattern Learning")
    print("-"*40)
    
    request2 = "Find where RiskManager is implemented"
    result3 = await workflow.handle_user_request(request2)
    
    # Test 3: Failure mitigation
    print("\n📝 TEST 3: Failure Mitigation")
    print("-"*40)
    
    request3 = "Import a module that doesn't exist and handle the error"
    result4 = await workflow.handle_user_request(request3)
    
    # Show final metrics
    print("\n📊 FINAL METRICS:")
    print("-"*40)
    status = workflow.get_system_status()
    
    print(f"Components Verified: {status['components_verified']}")
    print(f"Patterns Learned: {status['patterns_learned']}")
    print(f"System Reliability: {status['system_reliability']:.1%}")
    print(f"Pending Tasks: {status['pending_tasks']}")
    
    print("\n✅ Meta-workflow test complete!")


if __name__ == "__main__":
    asyncio.run(test_workflow())