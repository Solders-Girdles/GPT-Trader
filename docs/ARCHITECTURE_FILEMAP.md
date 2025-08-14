## Architecture File Map

This document is auto-generated. Run `python scripts/generate_filemap.py` to refresh it.


### src/bot/

- (root)
  - __init__.py
  - cli.py
  - cli_rapid_evolution.py
  - config.py
  - exceptions.py
  - health.py
  - logging.py
  - performance.py
  - startup_validation.py

- analytics/
  - __init__.py
  - alpha_analysis.py
  - attribution.py
  - correlation_modeling.py
  - decomposition.py
  - risk_decomposition.py

- api/
  - gateway.py

- backtest/
  - __init__.py
  - engine.py
  - engine_portfolio.py
  - realistic_backtester.py

- cli/
  - __init__.py
  - __main__.py
  - base.py
  - cli.py
  - commands.py
  - ml_commands.py
  - utils.py

- config/
  - __init__.py
  - demo_mode.py
  - financial_config.py
  - unified_config.py

- core/
  - __init__.py
  - analytics.py
  - base.py
  - caching.py
  - concurrency.py
  - config.py
  - container.py
  - database.py
  - deployment.py
  - disaster_recovery.py
  - error_handling.py
  - exceptions.py
  - metrics.py
  - migration.py
  - observability.py
  - performance.py
  - security.py

- dashboard/
  - app.py
  - ml_dashboard.py
  - ml_pages.py

- data/
  - unified_pipeline.py

- database/
  - manager.py
  - models.py

- dataflow/
  - __init__.py
  - alternative_data.py
  - base.py
  - data_quality_framework.py
  - data_source_manager.py
  - historical_data_manager.py
  - realtime_feed.py
  - sources/__init__.py
  - sources/enhanced_yfinance_source.py
  - sources/yfinance_source.py
  - streaming_data.py
  - validate.py

- exceptions/
  - __init__.py
  - decorators.py
  - enhanced_exceptions.py
  - examples.py
  - user_friendly.py

- exec/
  - __init__.py
  - alpaca_paper.py
  - base.py
  - ledger.py
  - order_management.py

- indicators/
  - __init__.py
  - atr.py
  - donchian.py
  - enhanced.py
  - optimized.py
  - sma.py

- live/
  - __init__.py
  - audit.py
  - cycles/__init__.py
  - cycles/performance.py
  - cycles/risk.py
  - cycles/selection.py
  - data_manager.py
  - event_driven_architecture.py
  - events.py
  - market_data_pipeline.py
  - order_management.py
  - performance_tracker.py
  - phase4_integration.py
  - portfolio_manager.py
  - production_orchestrator.py
  - risk_monitor.py
  - strategy_selector.py
  - trading_engine.py

- logging/
  - __init__.py
  - log_aggregator.py
  - structured_logger.py

- metrics/
  - __init__.py
  - report.py

- ml/
  - __init__.py
  - ab_testing_framework.py
  - advanced_degradation_detector.py
  - auto_retraining.py
  - base.py
  - baseline_models.py
  - deep_learning/__init__.py
  - deep_learning/attention_mechanisms.py
  - deep_learning/deep_ensemble.py
  - deep_learning/distributed_training.py
  - deep_learning/gpu_optimization.py
  - deep_learning/integrated_lstm_pipeline.py
  - deep_learning/lstm_architecture.py
  - deep_learning/lstm_data_pipeline.py
  - deep_learning/lstm_training.py
  - deep_learning/model_compression.py
  - deep_learning/positional_encoding.py
  - deep_learning/simple_test.py
  - deep_learning/test_deep_learning.py
  - deep_learning/transfer_learning.py
  - deep_learning/transformer_architecture.py
  - deep_learning/transformer_models.py
  - degradation_dashboard.py
  - degradation_integration.py
  - drift_detector.py
  - efficiency_analyzer.py
  - ensemble_manager.py
  - feature_engineering_v2.py
  - feature_evolution.py
  - feature_selector.py
  - features/__init__.py
  - features/engineering.py
  - features/market_regime.py
  - features/technical.py
  - integrated_pipeline.py
  - learning_scheduler.py
  - model_calibrator.py
  - model_comparison_report.py
  - model_degradation_monitor.py
  - model_promotion.py
  - model_validation.py
  - model_versioning.py
  - models/__init__.py
  - models/portfolio_optimizer.py
  - models/regime_detector.py
  - models/strategy_selector.py
  - models/training_utils.py
  - online_learning.py
  - online_learning_simple.py
  - performance_benchmark.py
  - performance_metrics_collector.py
  - performance_targets.py
  - performance_tracker.py
  - portfolio/__init__.py
  - portfolio/allocator.py
  - portfolio/optimizer.py
  - reinforcement_learning/actor_critic.py
  - reinforcement_learning/deep_q_network.py
  - reinforcement_learning/multi_agent.py
  - reinforcement_learning/policy_gradient.py
  - reinforcement_learning/ppo.py
  - reinforcement_learning/q_learning.py
  - retraining_scheduler.py
  - shadow_mode.py
  - statistical_analyzer.py
  - threshold_optimizer.py
  - validation_reporter.py
  - walk_forward_validator.py

- monitoring/
  - __init__.py
  - intelligent_alerts.py
  - metrics_exporter.py
  - ml_logging_integration.py
  - monitor.py
  - ops_dashboard.py
  - ops_runbooks.py
  - structured_logger.py

- optimization/
  - __init__.py
  - analyzer.py
  - cache_eviction_policies.py
  - cli.py
  - config.py
  - deployment_pipeline.py
  - engine.py
  - enhanced_evolution.py
  - enhanced_evolution_with_knowledge.py
  - evolutionary.py
  - grid.py
  - hierarchical_evolution.py
  - intelligent_cache.py
  - memory_profiler.py
  - multi_objective.py
  - multi_objective_visualizer.py
  - parallel_evaluator.py
  - parallel_optimizer.py
  - rapid_evolution.py
  - report.py
  - strategy_diversity.py
  - visualizer.py
  - walk_forward_validator.py

- paper_trading/
  - deployment_pipeline.py
  - ml_paper_trader.py

- portfolio/
  - __init__.py
  - allocator.py
  - dynamic_allocation.py
  - optimizer.py
  - portfolio_constructor.py
  - portfolio_optimization.py

- rebalancing/
  - __init__.py
  - costs.py
  - engine.py
  - triggers.py

- risk/
  - __init__.py
  - advanced_optimization.py
  - anomaly_alert_system.py
  - anomaly_detector.py
  - basic.py
  - circuit_breakers.py
  - correlation_monitor.py
  - greeks_calculator.py
  - live_risk_monitor.py
  - lstm_anomaly_detector.py
  - manager.py
  - realtime_websocket.py
  - risk_limit_monitor.py
  - risk_metrics_engine.py
  - stress_testing.py

- security/
  - __init__.py
  - config.py
  - input_validation.py
  - secrets_manager.py

- strategy/
  - __init__.py
  - base.py
  - components.py
  - demo_ma.py
  - enhanced_trend_breakout.py
  - ml_enhanced.py
  - multi_instrument.py
  - optimized_ma.py
  - persistence.py
  - strategy_collection.py
  - talib_optimized_ma.py
  - training_pipeline.py
  - trend_breakout.py
  - validation_engine.py
  - validation_pipeline.py

- utils/
  - __init__.py
  - base.py
  - config.py
  - paths.py
  - settings.py
  - validation.py

- validation/
  - __init__.py
  - framework.py

### tests/

- (root)
  - README.md
  - conftest.py
  - conftest_full.py
  - factories.py
  - test_degradation_integration.py
  - test_paper_trading_setup.py
  - test_phase3_monitoring.py
  - test_phase3_monitoring_standalone.py
  - test_sizing_price.py

- acceptance/
  - __init__.py
  - test_paper_trading_validation.py
  - test_real_world_scenarios.py

- integration/
  - __init__.py
  - phases/test_phase1_integration.py
  - pipelines/test_strategy_pipeline_integration.py
  - pipelines/test_strategy_pipeline_simplified.py
  - test_backtest_integration.py
  - test_cli_integration.py
  - test_component_integration.py
  - test_consolidated_architecture.py
  - test_data_flow.py
  - test_data_pipeline_integration.py
  - test_error_handling.py
  - test_strategy_verification.py
  - weeks/test_week3_integration.py
  - weeks/test_week3_integration_simple.py
  - weeks/test_week3_minimal.py
  - weeks/test_week4_integration.py
  - workflow/comprehensive_workflow_test.py
  - workflow/minimal_workflow_test.py
  - workflow/test_workflow.py

- ml/
  - test_feature_engineering.py
  - test_models.py
  - test_portfolio_optimization.py

- performance/
  - __init__.py
  - benchmark_consolidated.py
  - test_load_performance.py
  - test_memory_usage.py
  - test_optimizer_sla.py
  - test_stress_performance.py

- production/
  - __init__.py
  - test_deployment.py
  - test_monitoring_observability.py
  - test_production_readiness.py

- system/
  - __init__.py
  - test_data_preparation.py
  - test_e2e_workflows.py
  - test_joblib_migration.py
  - test_multiprocessing.py
  - test_system_startup_shutdown.py
  - test_talib_integration.py
  - test_trading_cycles.py
  - test_user_interfaces.py

- unit/
  - backtest/test_engine_portfolio.py
  - backtest/test_ledger.py
  - ml/test_ensemble_and_evolution.py
  - ml/test_online_learning.py
  - ml/test_online_learning_standalone.py
  - monitoring/test_structured_logger.py
  - portfolio/test_allocator.py
  - portfolio/test_portfolio_manager.py
  - risk/test_anomaly_basic.py
  - risk/test_anomaly_detection.py
  - risk/test_greeks_calculator.py
  - risk/test_risk_components.py
  - risk/test_risk_limit_monitor.py
  - risk/test_risk_manager.py
  - risk/test_risk_metrics_engine.py
  - risk/test_week4_components.py
  - strategy/test_base_strategy.py
  - strategy/test_base_strategy_comprehensive.py
  - strategy/test_demo_ma.py
  - strategy/test_trend_breakout.py
  - test_alert_manager.py
  - test_audit_records.py
  - test_audit_summary.py
  - test_auto_retraining.py
  - test_config.py
  - test_enhanced_exceptions.py
  - test_performance_monitor.py
  - test_portfolio_optimizer.py
  - test_secrets_manager.py
  - test_strategy_selector.py
  - test_structured_logging.py
  - test_unified_config.py
  - test_validation_framework.py

### scripts/

- (root)
  - analyze_grid.py
  - consolidate_databases.py
  - dashboard.py
  - generate_filemap.py
  - migrate_financial_constants.py
  - migrate_secrets.py
  - migrate_to_postgres.py
  - run_grid.sh
  - run_is_oos.py
  - run_tests.py
  - run_tests_parallel.py
  - simple_claude_update.sh
  - test_auto_retraining_integration.py
  - test_database_stress.py
  - test_ml_integration.py
  - test_model_calibration.py
  - test_performance_benchmark.py
  - test_pipeline_performance.py
  - test_postgres_connection.py
  - test_realtime_data.py
  - test_retraining_standalone.py
  - test_structured_logging_performance.py
  - test_walk_forward_integration.py
  - update_claude_md.py

### examples/

- (root)
  - architecture_migration_demo.py
  - complete_pipeline_example.py
  - component_based_strategy_example.py
  - demo_enhanced_cli.py
  - enhanced_evolution_example.py
  - hierarchical_evolution_example.py
  - knowledge_enhanced_evolution_example.py
  - multi_objective_optimization_example.py
  - optimization_example.py
  - paper_trading_example.py
  - phase2_integration_demo.py
  - phase3_meta_learning_example.py
  - phase3_observability_demo.py
  - phase4_advanced_analytics_example.py
  - phase4_operational_excellence_demo.py
  - phase5_production_integration_example.py
  - rapid_evolution_real.py
  - rapid_evolution_test.py
  - structured_logging_demo.py

---

Note: Only key file types are listed (py/md/sh). Hidden and cache dirs are skipped.
