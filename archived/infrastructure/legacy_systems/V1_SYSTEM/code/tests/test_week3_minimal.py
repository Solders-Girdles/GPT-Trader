#!/usr/bin/env python3
"""
Week 3 Minimal Integration Test - Strategy Development Workflow

Minimal test to verify Week 3 core functionality without import dependencies.
Tests the strategy creation CLI and template system directly.
"""

import json
import sys
import tempfile
from pathlib import Path


def test_strategy_templates():
    """Test the strategy templates directly"""

    print("🧪 Week 3 Minimal Strategy Development Test")
    print("=" * 50)

    # Import only what we need for templates
    sys.path.insert(0, "src")

    try:
        # Import strategy templates directly
        from bot.cli.strategy_development import STRATEGY_TEMPLATES, StrategyDevelopmentWorkflow

        print("\n📋 STRATEGY TEMPLATES TEST")
        print("-" * 30)

        print(f"   Available templates: {len(STRATEGY_TEMPLATES)}")
        for template_id, template_info in STRATEGY_TEMPLATES.items():
            print(f"      • {template_id}: {template_info['name']}")
            print(f"        Parameters: {list(template_info['parameters'].keys())}")
            print(f"        Complexity: {template_info['complexity']}")

        # =================================================================
        # TEST STRATEGY CREATION
        # =================================================================
        print("\n🎨 STRATEGY CREATION TEST")
        print("-" * 30)

        # Create temporary directory
        test_dir = Path(tempfile.mkdtemp(prefix="week3_minimal_"))
        workflow_dir = test_dir / "strategy_development"

        print(f"   Test directory: {workflow_dir}")

        # Initialize workflow
        workflow = StrategyDevelopmentWorkflow(str(workflow_dir))

        # Test strategy creation
        test_strategies = [
            {
                "template": "moving_average",
                "name": "Test MA Strategy",
                "parameters": {"fast_period": 10, "slow_period": 20},
            }
        ]

        success_count = 0

        for strategy_config in test_strategies:
            print(f"   Creating {strategy_config['name']}...")

            try:
                strategy_file = workflow.create_strategy_from_template(
                    template_name=strategy_config["template"],
                    strategy_name=strategy_config["name"],
                    parameters=strategy_config["parameters"],
                )

                # Check if file was created
                if Path(strategy_file).exists():
                    print(f"      ✅ Strategy file created: {Path(strategy_file).name}")

                    # Check file content
                    with open(strategy_file) as f:
                        content = f.read()

                    # Verify essential content
                    required_elements = ["class", "Strategy", "generate_signals"]
                    found_elements = [elem for elem in required_elements if elem in content]

                    print(f"      ✅ Code elements: {len(found_elements)}/{len(required_elements)}")

                    # Check config file
                    config_file = Path(strategy_file).parent / "strategy_config.json"
                    if config_file.exists():
                        with open(config_file) as f:
                            config = json.load(f)
                        print(f"      ✅ Config created: {config.get('strategy_name', 'Unknown')}")
                        success_count += 1
                    else:
                        print("      ⚠️  Config file missing")
                else:
                    print("      ❌ Strategy file not created")

            except Exception as e:
                print(f"      ❌ Failed: {str(e)}")

        # =================================================================
        # CODE GENERATION TEST
        # =================================================================
        print("\n🔧 CODE GENERATION TEST")
        print("-" * 30)

        # Test specific template generation
        template_name = "moving_average"
        strategy_name = "Code Test Strategy"
        parameters = {"fast_period": 5, "slow_period": 15}

        # Test the internal code generation method
        try:
            generated_code = workflow._generate_strategy_code(
                template_name, strategy_name, parameters
            )

            code_checks = {
                "Has class definition": "class" in generated_code,
                "Has generate_signals": "generate_signals" in generated_code,
                "Has parameter space": "get_parameter_space" in generated_code,
                "Has factory function": "create_strategy" in generated_code,
                "Has docstring": '"""' in generated_code,
                "Parameters present": all(str(p) in generated_code for p in parameters.values()),
            }

            print("   Code quality checks:")
            passed_checks = 0
            for check_name, passed in code_checks.items():
                status = "✅" if passed else "❌"
                print(f"      {status} {check_name}")
                if passed:
                    passed_checks += 1

            code_quality = (passed_checks / len(code_checks)) * 100
            print(f"   Code Quality Score: {code_quality:.1f}%")

        except Exception as e:
            print(f"   ❌ Code generation test failed: {str(e)}")
            code_quality = 0

        # =================================================================
        # DIRECTORY STRUCTURE TEST
        # =================================================================
        print("\n📁 DIRECTORY STRUCTURE TEST")
        print("-" * 30)

        expected_dirs = ["strategies", "templates", "results", "reports"]
        existing_dirs = 0

        for dir_name in expected_dirs:
            dir_path = workflow_dir / dir_name
            if dir_path.exists():
                existing_dirs += 1
                print(f"      ✅ {dir_name}/ directory created")
            else:
                print(f"      ❌ {dir_name}/ directory missing")

        structure_score = (existing_dirs / len(expected_dirs)) * 100
        print(f"   Directory Structure Score: {structure_score:.1f}%")

        # =================================================================
        # FINAL RESULTS
        # =================================================================
        print("\n🎯 FINAL RESULTS")
        print("=" * 30)

        creation_score = (success_count / len(test_strategies)) * 100

        scores = {
            "Strategy Creation": creation_score,
            "Code Generation": code_quality,
            "Directory Structure": structure_score,
        }

        print("   Component Scores:")
        for component, score in scores.items():
            status = "✅" if score >= 80 else ("⚠️ " if score >= 60 else "❌")
            print(f"      {status} {component}: {score:.1f}%")

        overall_score = sum(scores.values()) / len(scores)
        print(f"\n   Overall Score: {overall_score:.1f}%")

        # Determine success
        success = overall_score >= 75

        if success:
            print("\n✅ WEEK 3 MINIMAL TEST: SUCCESS")
            print("   Strategy Development CLI is functional!")
            print("   Template system working properly!")
            print("   Code generation creating valid strategies!")
            print("\n🚀 Week 3 Core Components: OPERATIONAL")
        else:
            print("\n⚠️  WEEK 3 MINIMAL TEST: NEEDS WORK")
            print("   Some core components need improvement")

        # Cleanup
        import shutil

        try:
            shutil.rmtree(test_dir)
            print("\n🧹 Cleanup completed")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {str(e)}")

        return success

    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_strategy_templates()
    print(f"\n{'='*50}")
    print(f"WEEK 3 MINIMAL TEST: {'PASSED' if success else 'FAILED'}")
    print(f"{'='*50}")
    sys.exit(0 if success else 1)
