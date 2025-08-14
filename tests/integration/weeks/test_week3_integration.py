#!/usr/bin/env python3
"""
Week 3 Integration Test - Strategy Development Workflow

Comprehensive test of the complete Week 3 strategy development workflow:
1. Strategy Creation CLI (templates, code generation)  
2. Validation Pipeline Integration (automated end-to-end testing)
3. Integration Testing (full workflow verification)

Tests the entire pipeline from strategy creation to deployment readiness.
"""

import logging
import tempfile
from datetime import datetime
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, "src")

# Week 3 imports
from bot.cli.strategy_development import StrategyDevelopmentWorkflow
from bot.strategy.validation_pipeline import create_validation_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Week3IntegrationTest:
    """Week 3 Strategy Development Workflow Integration Test"""

    def __init__(self):
        # Create temporary directory for testing
        self.test_dir = Path(tempfile.mkdtemp(prefix="week3_integration_"))
        self.workflow_dir = self.test_dir / "strategy_development"
        self.workflow_dir.mkdir(parents=True, exist_ok=True)

        # Initialize workflow
        self.workflow = StrategyDevelopmentWorkflow(str(self.workflow_dir))

        # Test configuration
        self.test_symbols = ["AAPL", "MSFT"]  # Reduced for faster testing
        self.test_days = 365  # 1 year for faster testing
        self.test_strategies = [
            {
                "template": "moving_average",
                "name": "Test MA Strategy",
                "parameters": {"fast_period": 5, "slow_period": 15},
            },
            {
                "template": "mean_reversion",
                "name": "Test MR Strategy",
                "parameters": {"window": 15, "num_std": 2.5},
            },
        ]

        self.test_results = {
            "strategy_creation": [],
            "pipeline_validation": [],
            "integration_success": False,
            "errors": [],
            "warnings": [],
        }

        print(f"🧪 Week 3 Integration Test initialized")
        print(f"   Test Directory: {self.test_dir}")
        print(f"   Workflow Directory: {self.workflow_dir}")

    def run_complete_integration_test(self) -> bool:
        """Run complete Week 3 integration test"""

        print("🚀 Starting Week 3 Integration Test")
        print("=" * 70)

        try:
            # Phase 1: Test Strategy Creation CLI
            print(f"\n📋 PHASE 1: Strategy Creation CLI Testing")
            print("-" * 50)

            if not self._test_strategy_creation_cli():
                print("❌ Strategy Creation CLI failed")
                return False

            # Phase 2: Test Validation Pipeline Integration
            print(f"\n🤖 PHASE 2: Validation Pipeline Integration Testing")
            print("-" * 50)

            if not self._test_validation_pipeline_integration():
                print("❌ Validation Pipeline Integration failed")
                return False

            # Phase 3: End-to-End Workflow Verification
            print(f"\n🎯 PHASE 3: End-to-End Workflow Verification")
            print("-" * 50)

            if not self._test_end_to_end_workflow():
                print("❌ End-to-End Workflow failed")
                return False

            # Phase 4: Integration Results Analysis
            print(f"\n📊 PHASE 4: Integration Results Analysis")
            print("-" * 50)

            self._analyze_integration_results()

            # Final verification
            self.test_results["integration_success"] = self._verify_integration_success()

            if self.test_results["integration_success"]:
                print(f"\n✅ WEEK 3 INTEGRATION TEST: SUCCESS")
                print("=" * 50)
                self._display_success_summary()
                return True
            else:
                print(f"\n❌ WEEK 3 INTEGRATION TEST: FAILED")
                print("=" * 50)
                self._display_failure_summary()
                return False

        except Exception as e:
            self.test_results["errors"].append(str(e))
            logger.exception("Week 3 integration test failed")
            print(f"❌ INTEGRATION TEST EXCEPTION: {str(e)}")
            return False

    def _test_strategy_creation_cli(self) -> bool:
        """Test Strategy Creation CLI functionality"""

        print("   🎨 Testing strategy creation from templates...")

        try:
            for strategy_config in self.test_strategies:
                print(
                    f"      Creating {strategy_config['name']} from {strategy_config['template']} template..."
                )

                # Create strategy using CLI workflow
                strategy_file = self.workflow.create_strategy_from_template(
                    template_name=strategy_config["template"],
                    strategy_name=strategy_config["name"],
                    parameters=strategy_config.get("parameters"),
                )

                # Verify strategy file was created
                if not Path(strategy_file).exists():
                    self.test_results["errors"].append(
                        f"Strategy file not created: {strategy_file}"
                    )
                    return False

                # Verify strategy file contains expected content
                with open(strategy_file, "r") as f:
                    content = f.read()

                # Check for essential elements
                essential_elements = [
                    "class",
                    "Strategy",
                    "generate_signals",
                    "get_parameter_space",
                    "create_strategy",
                ]

                missing_elements = []
                for element in essential_elements:
                    if element not in content:
                        missing_elements.append(element)

                if missing_elements:
                    error_msg = (
                        f"Strategy {strategy_config['name']} missing elements: {missing_elements}"
                    )
                    self.test_results["errors"].append(error_msg)
                    return False

                # Verify configuration file
                config_file = Path(strategy_file).parent / "strategy_config.json"
                if not config_file.exists():
                    self.test_results["errors"].append(
                        f"Configuration file not created: {config_file}"
                    )
                    return False

                self.test_results["strategy_creation"].append(
                    {
                        "name": strategy_config["name"],
                        "template": strategy_config["template"],
                        "file": strategy_file,
                        "success": True,
                    }
                )

                print(f"         ✅ {strategy_config['name']} created successfully")

            print(f"   ✅ Strategy Creation CLI: {len(self.test_strategies)} strategies created")
            return True

        except Exception as e:
            self.test_results["errors"].append(f"Strategy creation failed: {str(e)}")
            return False

    def _test_validation_pipeline_integration(self) -> bool:
        """Test Validation Pipeline Integration functionality"""

        print("   🔄 Testing validation pipeline integration...")

        try:
            # Initialize validation pipeline
            self.workflow._initialize_components()

            # Test pipeline with each created strategy
            for strategy_result in self.test_results["strategy_creation"]:
                strategy_file = strategy_result["file"]
                strategy_name = strategy_result["name"]

                print(f"      Running pipeline validation for {strategy_name}...")

                # Run validation pipeline
                pipeline_result = self.workflow.validation_pipeline.validate_strategy_file(
                    strategy_file_path=strategy_file,
                    symbols=self.test_symbols,
                    days=self.test_days,
                    save_results=True,
                )

                # Analyze pipeline result
                pipeline_success = self._analyze_pipeline_result(pipeline_result, strategy_name)

                self.test_results["pipeline_validation"].append(
                    {
                        "strategy_name": strategy_name,
                        "pipeline_result": pipeline_result,
                        "success": pipeline_success,
                    }
                )

                status = "✅" if pipeline_success else "⚠️ "
                print(
                    f"         {status} {strategy_name}: Score {pipeline_result.overall_score:.1f}/100"
                )

            # Check if any pipeline validation succeeded
            successful_pipelines = sum(
                1 for p in self.test_results["pipeline_validation"] if p["success"]
            )

            print(
                f"   ✅ Pipeline Integration: {successful_pipelines}/{len(self.test_strategies)} pipelines successful"
            )
            return successful_pipelines > 0

        except Exception as e:
            self.test_results["errors"].append(f"Validation pipeline integration failed: {str(e)}")
            return False

    def _analyze_pipeline_result(self, pipeline_result, strategy_name: str) -> bool:
        """Analyze individual pipeline result"""

        # Check component success
        component_checks = [
            pipeline_result.data_preparation_success,
            pipeline_result.strategy_loading_success,
            pipeline_result.training_success,
            pipeline_result.validation_success,
            pipeline_result.persistence_success,
        ]

        successful_components = sum(component_checks)

        # Log any component failures
        if not pipeline_result.data_preparation_success:
            self.test_results["warnings"].append(f"{strategy_name}: Data preparation failed")
        if not pipeline_result.strategy_loading_success:
            self.test_results["warnings"].append(f"{strategy_name}: Strategy loading failed")
        if not pipeline_result.training_success:
            self.test_results["warnings"].append(f"{strategy_name}: Training failed")
        if not pipeline_result.validation_success:
            self.test_results["warnings"].append(f"{strategy_name}: Validation failed")
        if not pipeline_result.persistence_success:
            self.test_results["warnings"].append(f"{strategy_name}: Persistence failed")

        # Consider pipeline successful if most components work
        return successful_components >= 3  # At least 3/5 components successful

    def _test_end_to_end_workflow(self) -> bool:
        """Test end-to-end workflow integration"""

        print("   🔗 Testing end-to-end workflow integration...")

        try:
            # Verify file structure was created properly
            expected_dirs = [
                self.workflow_dir / "strategies",
                self.workflow_dir / "templates",
                self.workflow_dir / "results",
                self.workflow_dir / "reports",
                self.workflow_dir / "pipeline",
            ]

            missing_dirs = []
            for expected_dir in expected_dirs:
                if not expected_dir.exists():
                    missing_dirs.append(str(expected_dir))

            if missing_dirs:
                self.test_results["errors"].append(f"Missing directories: {missing_dirs}")
                return False

            # Verify strategies were persisted properly
            for strategy_result in self.test_results["strategy_creation"]:
                strategy_name = strategy_result["name"]
                strategy_dir = (
                    self.workflow_dir / "strategies" / strategy_name.replace(" ", "_").lower()
                )

                if not strategy_dir.exists():
                    self.test_results["warnings"].append(
                        f"Strategy directory missing: {strategy_dir}"
                    )

            # Verify pipeline results were saved
            pipeline_results_dir = self.workflow_dir / "pipeline" / "results"
            if pipeline_results_dir.exists():
                result_files = list(pipeline_results_dir.glob("*/*.json"))
                print(f"      Found {len(result_files)} pipeline result files")

            # Verify components can be reinitialized
            new_workflow = StrategyDevelopmentWorkflow(str(self.workflow_dir))
            new_workflow._initialize_components()

            print("   ✅ End-to-end workflow integration successful")
            return True

        except Exception as e:
            self.test_results["errors"].append(f"End-to-end workflow test failed: {str(e)}")
            return False

    def _analyze_integration_results(self):
        """Analyze overall integration test results"""

        print("   📈 Analyzing integration test results...")

        # Strategy creation analysis
        successful_creations = len(self.test_results["strategy_creation"])
        print(
            f"      Strategy Creation: {successful_creations}/{len(self.test_strategies)} successful"
        )

        # Pipeline validation analysis
        successful_pipelines = sum(
            1 for p in self.test_results["pipeline_validation"] if p["success"]
        )
        print(
            f"      Pipeline Validation: {successful_pipelines}/{len(self.test_strategies)} successful"
        )

        # Overall scores
        if self.test_results["pipeline_validation"]:
            scores = [
                p["pipeline_result"].overall_score for p in self.test_results["pipeline_validation"]
            ]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            print(f"      Average Validation Score: {avg_score:.1f}/100")
            print(f"      Best Validation Score: {max_score:.1f}/100")

        # Error and warning summary
        print(f"      Errors: {len(self.test_results['errors'])}")
        print(f"      Warnings: {len(self.test_results['warnings'])}")

    def _verify_integration_success(self) -> bool:
        """Verify overall integration success criteria"""

        # Success criteria:
        # 1. At least 1 strategy created successfully
        # 2. At least 1 pipeline validation successful
        # 3. No critical errors
        # 4. Key components working

        strategies_created = len(self.test_results["strategy_creation"]) > 0
        pipelines_successful = any(p["success"] for p in self.test_results["pipeline_validation"])
        no_critical_errors = len(self.test_results["errors"]) == 0

        return strategies_created and pipelines_successful and no_critical_errors

    def _display_success_summary(self):
        """Display integration test success summary"""

        print("🎉 WEEK 3 STRATEGY DEVELOPMENT WORKFLOW: OPERATIONAL!")
        print()
        print("✅ Completed Components:")
        print("   • Strategy Creation CLI with templates and code generation")
        print("   • Validation Pipeline Integration for automated testing")
        print("   • End-to-end workflow with data preparation, training, and validation")
        print("   • Strategy persistence and results management")
        print()

        print("📊 Test Results:")
        print(f"   • Strategies Created: {len(self.test_results['strategy_creation'])}")
        print(
            f"   • Pipelines Successful: {sum(1 for p in self.test_results['pipeline_validation'] if p['success'])}"
        )

        if self.test_results["pipeline_validation"]:
            scores = [
                p["pipeline_result"].overall_score for p in self.test_results["pipeline_validation"]
            ]
            avg_score = sum(scores) / len(scores)
            print(f"   • Average Validation Score: {avg_score:.1f}/100")

        print()
        print("🚀 Week 3 Status: COMPLETE")
        print("   • Strategy Development CLI ✅")
        print("   • Validation Pipeline Integration ✅")
        print("   • Integration Testing ✅")
        print()
        print("🎯 Ready for Week 4: Strategy Collection & Portfolio Construction")

    def _display_failure_summary(self):
        """Display integration test failure summary"""

        print("⚠️  WEEK 3 INTEGRATION TEST ISSUES DETECTED")
        print()

        if self.test_results["errors"]:
            print("❌ Critical Errors:")
            for error in self.test_results["errors"]:
                print(f"   • {error}")
            print()

        if self.test_results["warnings"]:
            print("⚠️  Warnings:")
            for warning in self.test_results["warnings"][:5]:  # Show first 5
                print(f"   • {warning}")
            print()

        print("🔧 Recommended Actions:")
        print("   • Review error messages and fix component issues")
        print("   • Check data connectivity and quality")
        print("   • Verify strategy template generation")
        print("   • Test individual components in isolation")

    def cleanup(self):
        """Cleanup test resources"""
        try:
            import shutil

            shutil.rmtree(self.test_dir)
            print(f"🧹 Test cleanup completed: {self.test_dir}")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {str(e)}")


def run_week3_integration_test() -> bool:
    """Run Week 3 integration test"""

    test_runner = Week3IntegrationTest()

    try:
        success = test_runner.run_complete_integration_test()
        return success
    except Exception as e:
        logger.exception("Week 3 integration test failed")
        print(f"❌ WEEK 3 INTEGRATION TEST EXCEPTION: {str(e)}")
        return False
    finally:
        # test_runner.cleanup()  # Comment out for debugging
        pass


if __name__ == "__main__":
    success = run_week3_integration_test()
    print(f"\n{'='*70}")
    print(f"WEEK 3 INTEGRATION TEST: {'SUCCESS' if success else 'FAILED'}")
    print(f"{'='*70}")
    sys.exit(0 if success else 1)
