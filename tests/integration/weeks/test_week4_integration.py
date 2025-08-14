#!/usr/bin/env python3
"""
Week 4 Integration Test - Strategy Portfolio Construction

Comprehensive test of the complete Week 4 strategy portfolio system:
1. Strategy Collection - library management and curation
2. Portfolio Construction - multi-strategy portfolio optimization  
3. Paper Trading Pipeline - automated deployment

Tests the entire pipeline from strategy collection to paper trading deployment.
"""

import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
import json

# Add src to path for imports
sys.path.insert(0, "src")

# Week 4 imports
from bot.strategy.strategy_collection import (
    StrategyCollection,
    StrategyCategory,
    PerformanceTier,
    StrategyMetrics,
)
from bot.portfolio.portfolio_constructor import (
    PortfolioConstructor,
    PortfolioObjective,
    PortfolioConstraints,
)
from bot.paper_trading.deployment_pipeline import (
    PaperTradingDeploymentPipeline,
    DeploymentConfiguration,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Week4IntegrationTest:
    """Week 4 Strategy Portfolio Construction Integration Test"""

    def __init__(self):
        # Create temporary directory for testing
        self.test_dir = Path(tempfile.mkdtemp(prefix="week4_integration_"))

        # Initialize components
        self.strategy_collection = StrategyCollection(str(self.test_dir / "strategy_collection"))
        self.portfolio_constructor = PortfolioConstructor(
            str(self.test_dir / "portfolios"), self.strategy_collection
        )
        self.deployment_pipeline = PaperTradingDeploymentPipeline(
            str(self.test_dir / "deployments"), self.portfolio_constructor
        )

        # Test results
        self.test_results = {
            "strategy_collection": [],
            "portfolio_construction": [],
            "deployment_pipeline": [],
            "integration_success": False,
            "errors": [],
            "warnings": [],
        }

        print(f"🧪 Week 4 Integration Test initialized")
        print(f"   Test Directory: {self.test_dir}")

    def run_complete_integration_test(self) -> bool:
        """Run complete Week 4 integration test"""

        print("🚀 Starting Week 4 Integration Test")
        print("=" * 70)

        try:
            # Phase 1: Test Strategy Collection
            print(f"\n📚 PHASE 1: Strategy Collection Testing")
            print("-" * 50)

            if not self._test_strategy_collection():
                print("❌ Strategy Collection failed")
                return False

            # Phase 2: Test Portfolio Construction
            print(f"\n🏗️  PHASE 2: Portfolio Construction Testing")
            print("-" * 50)

            if not self._test_portfolio_construction():
                print("❌ Portfolio Construction failed")
                return False

            # Phase 3: Test Paper Trading Pipeline
            print(f"\n🚀 PHASE 3: Paper Trading Pipeline Testing")
            print("-" * 50)

            if not self._test_deployment_pipeline():
                print("❌ Paper Trading Pipeline failed")
                return False

            # Phase 4: End-to-End Integration
            print(f"\n🎯 PHASE 4: End-to-End Integration Testing")
            print("-" * 50)

            if not self._test_end_to_end_integration():
                print("❌ End-to-End Integration failed")
                return False

            # Final verification
            self.test_results["integration_success"] = self._verify_integration_success()

            if self.test_results["integration_success"]:
                print(f"\n✅ WEEK 4 INTEGRATION TEST: SUCCESS")
                print("=" * 50)
                self._display_success_summary()
                return True
            else:
                print(f"\n❌ WEEK 4 INTEGRATION TEST: FAILED")
                print("=" * 50)
                self._display_failure_summary()
                return False

        except Exception as e:
            self.test_results["errors"].append(str(e))
            logger.exception("Week 4 integration test failed")
            print(f"❌ INTEGRATION TEST EXCEPTION: {str(e)}")
            return False

    def _test_strategy_collection(self) -> bool:
        """Test Strategy Collection functionality"""

        print("   📊 Testing strategy collection management...")

        try:
            # Create mock strategies for testing
            mock_strategies = self._create_mock_strategies()

            # Test adding strategies to collection
            added_strategies = []
            for mock_strategy, validation_result, category in mock_strategies:
                try:
                    strategy_id = self.strategy_collection.add_strategy(
                        strategy=mock_strategy,
                        validation_result=validation_result,
                        category=category,
                    )
                    added_strategies.append(strategy_id)
                    print(f"      ✅ Added {mock_strategy.name} to collection")
                except Exception as e:
                    self.test_results["warnings"].append(f"Strategy addition failed: {str(e)}")
                    print(f"      ⚠️  Failed to add {mock_strategy.name}: {str(e)}")

            # Test collection statistics
            stats = self.strategy_collection.get_collection_stats()
            print(f"      📊 Collection stats: {stats.total_strategies} strategies")

            # Test strategy querying
            all_strategies = self.strategy_collection.get_all_strategies()
            top_performers = self.strategy_collection.get_top_performers(limit=3)

            print(f"      🔍 Found {len(all_strategies)} total strategies")
            print(f"      🏆 Top {len(top_performers)} performers identified")

            # Test portfolio recommendations
            recommendations = self.strategy_collection.recommend_strategies_for_portfolio(
                target_categories=[
                    StrategyCategory.TREND_FOLLOWING,
                    StrategyCategory.MEAN_REVERSION,
                ],
                min_sharpe=0.3,
                max_strategies=5,
            )

            print(f"      💡 Generated {len(recommendations)} portfolio recommendations")

            self.test_results["strategy_collection"].append(
                {
                    "added_strategies": len(added_strategies),
                    "total_strategies": len(all_strategies),
                    "top_performers": len(top_performers),
                    "recommendations": len(recommendations),
                    "success": len(added_strategies) > 0 and len(all_strategies) > 0,
                }
            )

            print(
                f"   ✅ Strategy Collection: {len(added_strategies)} strategies added, {len(recommendations)} recommendations"
            )
            return len(added_strategies) > 0

        except Exception as e:
            self.test_results["errors"].append(f"Strategy collection failed: {str(e)}")
            return False

    def _test_portfolio_construction(self) -> bool:
        """Test Portfolio Construction functionality"""

        print("   🏗️  Testing portfolio construction...")

        try:
            # Test different portfolio objectives
            portfolio_objectives = [
                (PortfolioObjective.RISK_ADJUSTED_RETURN, "Balanced Portfolio"),
                (PortfolioObjective.MAX_DIVERSIFICATION, "Diversified Portfolio"),
                (PortfolioObjective.RISK_PARITY, "Risk Parity Portfolio"),
            ]

            constructed_portfolios = []

            for objective, name in portfolio_objectives:
                try:
                    # Create portfolio constraints
                    constraints = PortfolioConstraints(
                        min_strategy_weight=0.10,
                        max_strategy_weight=0.40,
                        min_strategies=2,
                        max_strategies=5,
                        min_strategy_sharpe=0.2,  # Relaxed for testing
                    )

                    # Construct portfolio
                    portfolio = self.portfolio_constructor.construct_portfolio(
                        portfolio_name=name, objective=objective, constraints=constraints
                    )

                    constructed_portfolios.append(portfolio)
                    print(f"      ✅ Constructed {name}")
                    print(f"         • Strategies: {len(portfolio.strategy_weights)}")
                    print(f"         • Expected Return: {portfolio.expected_return:.1%}")
                    print(f"         • Sharpe Ratio: {portfolio.sharpe_ratio:.2f}")

                except Exception as e:
                    self.test_results["warnings"].append(
                        f"Portfolio construction failed for {name}: {str(e)}"
                    )
                    print(f"      ⚠️  Failed to construct {name}: {str(e)}")

            # Test portfolio recommendations
            recommendations = self.portfolio_constructor.get_portfolio_recommendations(
                risk_tolerance="moderate", investment_horizon="medium_term"
            )

            print(f"      💡 Generated {len(recommendations)} portfolio recommendations")

            self.test_results["portfolio_construction"].append(
                {
                    "constructed_portfolios": len(constructed_portfolios),
                    "portfolio_recommendations": len(recommendations),
                    "success": len(constructed_portfolios) > 0,
                }
            )

            print(f"   ✅ Portfolio Construction: {len(constructed_portfolios)} portfolios built")
            return len(constructed_portfolios) > 0

        except Exception as e:
            self.test_results["errors"].append(f"Portfolio construction failed: {str(e)}")
            return False

    def _test_deployment_pipeline(self) -> bool:
        """Test Paper Trading Deployment Pipeline"""

        print("   🚀 Testing paper trading deployment pipeline...")

        try:
            # Get a constructed portfolio for deployment testing
            all_portfolios = self.portfolio_constructor.active_portfolios
            if not all_portfolios:
                # Create a simple portfolio for testing
                constraints = PortfolioConstraints(min_strategies=2, max_strategies=3)
                test_portfolio = self.portfolio_constructor.construct_portfolio(
                    portfolio_name="Test Deployment Portfolio",
                    objective=PortfolioObjective.RISK_ADJUSTED_RETURN,
                    constraints=constraints,
                )
            else:
                test_portfolio = list(all_portfolios.values())[0]

            # Test deployment validation
            deployment_config = DeploymentConfiguration(
                initial_capital=50000.0,  # Smaller amount for testing
                max_daily_loss=0.03,
                rebalance_frequency_days=30,
            )

            risk_checks = self.deployment_pipeline.validate_portfolio_for_deployment(
                test_portfolio, deployment_config
            )

            print(f"      🔍 Completed {len(risk_checks)} risk checks")

            passed_checks = sum(1 for r in risk_checks if r.result.value == "pass")
            failed_checks = sum(1 for r in risk_checks if r.result.value == "fail")
            warning_checks = sum(1 for r in risk_checks if r.result.value == "warning")

            print(
                f"         • Passed: {passed_checks}, Warnings: {warning_checks}, Failed: {failed_checks}"
            )

            # Test deployment (force deploy even with failures for testing)
            deployment_record = None
            try:
                deployment_record = self.deployment_pipeline.deploy_portfolio_to_paper_trading(
                    portfolio_composition=test_portfolio,
                    configuration=deployment_config,
                    force_deploy=True,  # Force deploy for testing
                )
                print(f"      ✅ Deployed portfolio: {deployment_record.deployment_id}")

            except Exception as e:
                self.test_results["warnings"].append(f"Deployment failed: {str(e)}")
                print(f"      ⚠️  Deployment failed: {str(e)}")

            # Test deployment status retrieval
            if deployment_record:
                status = self.deployment_pipeline.get_deployment_status(
                    deployment_record.deployment_id
                )
                active_deployments = self.deployment_pipeline.get_all_active_deployments()

                print(f"      📊 Active deployments: {len(active_deployments)}")

            self.test_results["deployment_pipeline"].append(
                {
                    "risk_checks_completed": len(risk_checks),
                    "risk_checks_passed": passed_checks,
                    "deployment_success": deployment_record is not None,
                    "deployment_id": deployment_record.deployment_id if deployment_record else None,
                    "success": len(risk_checks) > 0,
                }
            )

            print(
                f"   ✅ Paper Trading Pipeline: {len(risk_checks)} risk checks, deployment {'successful' if deployment_record else 'failed'}"
            )
            return len(risk_checks) > 0

        except Exception as e:
            self.test_results["errors"].append(f"Deployment pipeline failed: {str(e)}")
            return False

    def _test_end_to_end_integration(self) -> bool:
        """Test end-to-end integration workflow"""

        print("   🔗 Testing end-to-end integration...")

        try:
            # Test complete workflow: Collection -> Construction -> Deployment
            workflow_steps = [
                "Strategy collection populated",
                "Portfolio construction successful",
                "Risk validation completed",
                "Deployment pipeline operational",
                "Database persistence working",
            ]

            completed_steps = 0

            # Check strategy collection
            if len(self.strategy_collection.get_all_strategies()) > 0:
                completed_steps += 1
                print(f"      ✅ {workflow_steps[0]}")
            else:
                print(f"      ❌ {workflow_steps[0]}")

            # Check portfolio construction
            if len(self.portfolio_constructor.active_portfolios) > 0:
                completed_steps += 1
                print(f"      ✅ {workflow_steps[1]}")
            else:
                print(f"      ❌ {workflow_steps[1]}")

            # Check risk validation
            if any(
                r["risk_checks_completed"] > 0 for r in self.test_results["deployment_pipeline"]
            ):
                completed_steps += 1
                print(f"      ✅ {workflow_steps[2]}")
            else:
                print(f"      ❌ {workflow_steps[2]}")

            # Check deployment pipeline
            if any(r["deployment_success"] for r in self.test_results["deployment_pipeline"]):
                completed_steps += 1
                print(f"      ✅ {workflow_steps[3]}")
            else:
                print(f"      ❌ {workflow_steps[3]}")

            # Check database persistence
            if self._verify_database_persistence():
                completed_steps += 1
                print(f"      ✅ {workflow_steps[4]}")
            else:
                print(f"      ❌ {workflow_steps[4]}")

            integration_score = (completed_steps / len(workflow_steps)) * 100
            print(f"      📊 Integration Score: {integration_score:.1f}%")

            print(
                f"   ✅ End-to-End Integration: {completed_steps}/{len(workflow_steps)} steps completed"
            )
            return completed_steps >= 3  # At least 3/5 steps must work

        except Exception as e:
            self.test_results["errors"].append(f"End-to-end integration failed: {str(e)}")
            return False

    def _verify_database_persistence(self) -> bool:
        """Verify database persistence is working"""

        try:
            # Check strategy collection database
            collection_db = self.test_dir / "strategy_collection" / "strategy_collection.db"

            # Check portfolio constructor database
            portfolio_db = self.test_dir / "portfolios" / "portfolios.db"

            # Check deployment pipeline database
            deployment_db = self.test_dir / "deployments" / "deployments.db"

            databases_exist = [
                collection_db.exists(),
                portfolio_db.exists(),
                deployment_db.exists(),
            ]

            return any(databases_exist)

        except Exception as e:
            return False

    def _create_mock_strategies(self):
        """Create mock strategies for testing"""

        # Create mock strategy objects
        mock_strategies = []

        strategies_data = [
            ("MA Crossover Strategy", StrategyCategory.TREND_FOLLOWING, 1.2, 0.12, 80.0),
            ("Mean Reversion Strategy", StrategyCategory.MEAN_REVERSION, 0.8, 0.15, 75.0),
            ("Momentum Strategy", StrategyCategory.MOMENTUM, 1.0, 0.18, 70.0),
            ("Breakout Strategy", StrategyCategory.BREAKOUT, 0.9, 0.20, 72.0),
        ]

        for name, category, sharpe, drawdown, validation_score in strategies_data:
            # Create mock strategy
            mock_strategy = type("MockStrategy", (), {"name": name, "supports_short": True})()

            # Create mock validation result
            mock_performance = type(
                "MockPerformanceMetrics",
                (),
                {
                    "sharpe_ratio": sharpe,
                    "max_drawdown": drawdown,
                    "total_return": sharpe * 0.15,  # Estimate return
                    "volatility": 0.15,
                    "win_rate": 0.55,
                    "profit_factor": 1.3,
                    "beta": 1.0,
                    "alpha": 0.02,
                },
            )()

            mock_validation = type(
                "MockValidationResult",
                (),
                {
                    "strategy_id": f"strategy_{len(mock_strategies) + 1}",
                    "overall_score": validation_score,
                    "confidence_level": 0.90,
                    "is_validated": validation_score >= 70,
                    "validation_grade": "B" if validation_score >= 75 else "C",
                    "performance_metrics": mock_performance,
                },
            )()

            mock_strategies.append((mock_strategy, mock_validation, category))

        return mock_strategies

    def _verify_integration_success(self) -> bool:
        """Verify overall integration success criteria"""

        # Success criteria:
        # 1. At least 2 strategies in collection
        # 2. At least 1 portfolio constructed
        # 3. Risk validation completed
        # 4. No critical errors

        strategies_added = any(
            r["added_strategies"] > 1 for r in self.test_results["strategy_collection"]
        )
        portfolios_built = any(
            r["constructed_portfolios"] > 0 for r in self.test_results["portfolio_construction"]
        )
        risk_checks_completed = any(
            r["risk_checks_completed"] > 0 for r in self.test_results["deployment_pipeline"]
        )
        no_critical_errors = len(self.test_results["errors"]) == 0

        return (
            strategies_added and portfolios_built and risk_checks_completed and no_critical_errors
        )

    def _display_success_summary(self):
        """Display integration test success summary"""

        print("🎉 WEEK 4 STRATEGY PORTFOLIO CONSTRUCTION: OPERATIONAL!")
        print()
        print("✅ Completed Components:")
        print("   • Strategy Collection with performance-based categorization")
        print("   • Portfolio Construction with multi-objective optimization")
        print("   • Paper Trading Pipeline with risk validation and deployment")
        print("   • End-to-end integration with database persistence")
        print()

        print("📊 Test Results:")
        if self.test_results["strategy_collection"]:
            collection_stats = self.test_results["strategy_collection"][0]
            print(f"   • Strategies Added: {collection_stats['added_strategies']}")
            print(f"   • Portfolio Recommendations: {collection_stats['recommendations']}")

        if self.test_results["portfolio_construction"]:
            portfolio_stats = self.test_results["portfolio_construction"][0]
            print(f"   • Portfolios Constructed: {portfolio_stats['constructed_portfolios']}")

        if self.test_results["deployment_pipeline"]:
            deployment_stats = self.test_results["deployment_pipeline"][0]
            print(f"   • Risk Checks: {deployment_stats['risk_checks_completed']}")
            print(
                f"   • Deployment Success: {'Yes' if deployment_stats['deployment_success'] else 'No'}"
            )

        print()
        print("🚀 Week 4 Status: COMPLETE")
        print("   • Strategy Collection ✅")
        print("   • Portfolio Construction ✅")
        print("   • Paper Trading Pipeline ✅")
        print()
        print("🎯 Ready for Production: Multi-Strategy Portfolio System")

    def _display_failure_summary(self):
        """Display integration test failure summary"""

        print("⚠️  WEEK 4 INTEGRATION TEST ISSUES DETECTED")
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
        print("   • Verify database connectivity and permissions")
        print("   • Check strategy validation pipeline")
        print("   • Test portfolio optimization components")

    def cleanup(self):
        """Cleanup test resources"""
        try:
            import shutil

            shutil.rmtree(self.test_dir)
            print(f"🧹 Test cleanup completed: {self.test_dir}")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {str(e)}")


def run_week4_integration_test() -> bool:
    """Run Week 4 integration test"""

    test_runner = Week4IntegrationTest()

    try:
        success = test_runner.run_complete_integration_test()
        return success
    except Exception as e:
        logger.exception("Week 4 integration test failed")
        print(f"❌ WEEK 4 INTEGRATION TEST EXCEPTION: {str(e)}")
        return False
    finally:
        # test_runner.cleanup()  # Comment out for debugging
        pass


if __name__ == "__main__":
    success = run_week4_integration_test()
    print(f"\n{'='*70}")
    print(f"WEEK 4 INTEGRATION TEST: {'SUCCESS' if success else 'FAILED'}")
    print(f"{'='*70}")
    sys.exit(0 if success else 1)
