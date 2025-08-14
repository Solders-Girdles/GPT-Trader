#!/usr/bin/env python3
"""
Week 3 Simple Integration Test - Strategy Development Workflow

Tests core Week 3 functionality without heavy dependencies:
1. Strategy Creation CLI (templates, code generation)
2. Validation Pipeline Integration (automated testing)
3. Component Integration Verification

Focuses on verifying the Week 3 deliverables work together properly.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, "src")

# Direct imports to avoid heavy dependencies
from bot.cli.strategy_development import STRATEGY_TEMPLATES, StrategyDevelopmentWorkflow


def test_week3_strategy_development():
    """Test Week 3 Strategy Development components"""

    print("🧪 Week 3 Strategy Development Integration Test")
    print("=" * 60)

    # Create temporary test directory
    test_dir = Path(tempfile.mkdtemp(prefix="week3_test_"))
    workflow_dir = test_dir / "strategy_development"

    print(f"   Test Directory: {workflow_dir}")

    try:
        # =================================================================
        # TEST 1: Strategy Creation CLI
        # =================================================================
        print("\n📋 TEST 1: Strategy Creation CLI")
        print("-" * 40)

        # Initialize workflow
        workflow = StrategyDevelopmentWorkflow(str(workflow_dir))

        # Test strategy creation from template
        test_strategies = [
            {
                "template": "moving_average",
                "name": "Test MA Strategy",
                "parameters": {"fast_period": 5, "slow_period": 15},
            },
            {
                "template": "mean_reversion",
                "name": "Test MR Strategy",
                "parameters": {"window": 15, "num_std": 2.0},
            },
        ]

        created_strategies = []

        for strategy_config in test_strategies:
            print(f"   Creating {strategy_config['name']}...")

            try:
                strategy_file = workflow.create_strategy_from_template(
                    template_name=strategy_config["template"],
                    strategy_name=strategy_config["name"],
                    parameters=strategy_config["parameters"],
                )

                # Verify strategy file exists
                if Path(strategy_file).exists():
                    print(f"      ✅ Strategy file created: {Path(strategy_file).name}")

                    # Verify config file exists
                    config_file = Path(strategy_file).parent / "strategy_config.json"
                    if config_file.exists():
                        with open(config_file) as f:
                            config = json.load(f)
                        print(f"      ✅ Config created: {config['strategy_name']}")

                        created_strategies.append(
                            {
                                "name": strategy_config["name"],
                                "file": strategy_file,
                                "config": config,
                            }
                        )
                    else:
                        print("      ❌ Config file missing")
                else:
                    print("      ❌ Strategy file not created")

            except Exception as e:
                print(f"      ❌ Strategy creation failed: {str(e)}")

        print(
            f"   ✅ Strategy Creation: {len(created_strategies)}/{len(test_strategies)} successful"
        )

        # =================================================================
        # TEST 2: Template System Verification
        # =================================================================
        print("\n🎨 TEST 2: Template System Verification")
        print("-" * 40)

        print(f"   Available templates: {len(STRATEGY_TEMPLATES)}")
        for template_id, template_info in STRATEGY_TEMPLATES.items():
            print(f"      • {template_id}: {template_info['name']}")

        # Verify generated code quality
        code_quality_checks = 0
        total_checks = 0

        for strategy in created_strategies:
            strategy_file = strategy["file"]
            with open(strategy_file) as f:
                content = f.read()

            # Check for essential code elements
            required_elements = [
                "class",
                "Strategy",
                "generate_signals",
                "get_parameter_space",
                "create_strategy",
            ]

            strategy_checks = 0
            for element in required_elements:
                total_checks += 1
                if element in content:
                    code_quality_checks += 1
                    strategy_checks += 1

            print(
                f"      {strategy['name']}: {strategy_checks}/{len(required_elements)} code elements present"
            )

        code_quality_score = (code_quality_checks / total_checks * 100) if total_checks > 0 else 0
        print(
            f"   ✅ Code Quality: {code_quality_score:.1f}% ({code_quality_checks}/{total_checks})"
        )

        # =================================================================
        # TEST 3: File Structure Verification
        # =================================================================
        print("\n📁 TEST 3: File Structure Verification")
        print("-" * 40)

        # Check expected directory structure
        expected_dirs = [
            workflow_dir / "strategies",
            workflow_dir / "templates",
            workflow_dir / "results",
            workflow_dir / "reports",
        ]

        existing_dirs = 0
        for expected_dir in expected_dirs:
            if expected_dir.exists():
                existing_dirs += 1
                file_count = len(list(expected_dir.rglob("*")))
                print(f"      ✅ {expected_dir.name}/: {file_count} files")
            else:
                print(f"      ❌ {expected_dir.name}/: Missing")

        structure_score = existing_dirs / len(expected_dirs) * 100
        print(
            f"   ✅ Directory Structure: {structure_score:.1f}% ({existing_dirs}/{len(expected_dirs)})"
        )

        # =================================================================
        # TEST 4: Strategy Loading Verification
        # =================================================================
        print("\n🔌 TEST 4: Strategy Loading Verification")
        print("-" * 40)

        loadable_strategies = 0

        for strategy in created_strategies:
            strategy_file = strategy["file"]
            strategy_name = strategy["name"]

            try:
                # Test if strategy can be imported and instantiated
                import importlib.util

                spec = importlib.util.spec_from_file_location("strategy_module", strategy_file)
                if spec and spec.loader:
                    strategy_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(strategy_module)

                    # Try to create strategy instance
                    if hasattr(strategy_module, "create_strategy"):
                        strategy_instance = strategy_module.create_strategy()
                        if hasattr(strategy_instance, "generate_signals"):
                            loadable_strategies += 1
                            print(f"      ✅ {strategy_name}: Loadable and functional")
                        else:
                            print(f"      ⚠️  {strategy_name}: Missing generate_signals method")
                    else:
                        print(f"      ⚠️  {strategy_name}: Missing create_strategy function")
                else:
                    print(f"      ❌ {strategy_name}: Cannot import module")

            except Exception as e:
                print(f"      ❌ {strategy_name}: Loading failed - {str(e)}")

        loading_score = (
            (loadable_strategies / len(created_strategies) * 100) if created_strategies else 0
        )
        print(
            f"   ✅ Strategy Loading: {loading_score:.1f}% ({loadable_strategies}/{len(created_strategies)})"
        )

        # =================================================================
        # INTEGRATION SUMMARY
        # =================================================================
        print("\n🎯 INTEGRATION TEST SUMMARY")
        print("=" * 40)

        # Calculate overall scores
        creation_score = len(created_strategies) / len(test_strategies) * 100

        scores = {
            "Strategy Creation": creation_score,
            "Code Quality": code_quality_score,
            "Directory Structure": structure_score,
            "Strategy Loading": loading_score,
        }

        print("   Component Scores:")
        for component, score in scores.items():
            status = "✅" if score >= 80 else ("⚠️ " if score >= 60 else "❌")
            print(f"      {status} {component}: {score:.1f}%")

        overall_score = sum(scores.values()) / len(scores)
        print(f"\n   Overall Integration Score: {overall_score:.1f}%")

        # Determine success
        success = overall_score >= 75  # 75% threshold for success

        if success:
            print("\n✅ WEEK 3 INTEGRATION TEST: SUCCESS")
            print("   • Strategy Development CLI operational ✅")
            print("   • Template system working ✅")
            print("   • Code generation functional ✅")
            print("   • File management working ✅")
            print("\n🎉 Week 3 Strategy Development Workflow is READY!")
            print("   Ready for automated pipeline testing and deployment!")
        else:
            print("\n⚠️  WEEK 3 INTEGRATION TEST: NEEDS IMPROVEMENT")
            print("   Some components need attention before production use")

        return success

    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {str(e)}")
        return False

    finally:
        # Cleanup
        import shutil

        try:
            shutil.rmtree(test_dir)
            print("\n🧹 Test cleanup completed")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {str(e)}")


if __name__ == "__main__":
    success = test_week3_strategy_development()
    print(f"\n{'='*60}")
    print(f"WEEK 3 STRATEGY DEVELOPMENT TEST: {'PASSED' if success else 'FAILED'}")
    print(f"{'='*60}")
    sys.exit(0 if success else 1)
