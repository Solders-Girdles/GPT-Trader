"""
ML-Optimized Trading Workflows

This module defines comprehensive machine learning workflows for trading operations,
including feature engineering, model training, prediction generation, and model monitoring.
"""

from .engine import WorkflowStep

# ML Feature Engineering Workflow
ML_FEATURE_ENGINEERING = [
    WorkflowStep(
        name="Fetch Raw Data",
        function="fetch_ml_data",
        description="Get raw market data for features",
        required_context=['symbol', 'lookback_days'],
        outputs=['raw_data']
    ),
    WorkflowStep(
        name="Calculate Technical Features",
        function="calculate_technical_features",
        description="Generate technical indicators",
        required_context=['raw_data'],
        outputs=['technical_features']
    ),
    WorkflowStep(
        name="Calculate Market Features",
        function="calculate_market_features",
        description="Generate market microstructure features",
        required_context=['raw_data'],
        outputs=['market_features']
    ),
    WorkflowStep(
        name="Engineer Combined Features",
        function="engineer_features",
        description="Create interaction and derived features",
        required_context=['technical_features', 'market_features'],
        outputs=['feature_matrix', 'feature_names']
    ),
    WorkflowStep(
        name="Select Features",
        function="select_features",
        description="Feature selection and importance ranking",
        required_context=['feature_matrix', 'feature_names'],
        outputs=['selected_features', 'feature_importance']
    )
]

# ML Model Training Workflow
ML_MODEL_TRAINING = [
    WorkflowStep(
        name="Prepare Training Data",
        function="prepare_training_data",
        description="Prepare features and labels",
        required_context=['symbol', 'training_period'],
        outputs=['X_train', 'y_train', 'X_val', 'y_val']
    ),
    WorkflowStep(
        name="Train Models",
        function="train_ml_models",
        description="Train ensemble of models",
        required_context=['X_train', 'y_train'],
        outputs=['models', 'training_metrics']
    ),
    WorkflowStep(
        name="Validate Models",
        function="validate_models",
        description="Validate on holdout data",
        required_context=['models', 'X_val', 'y_val'],
        outputs=['validation_metrics', 'best_model']
    ),
    WorkflowStep(
        name="Optimize Hyperparameters",
        function="optimize_hyperparameters",
        description="Hyperparameter tuning",
        required_context=['best_model', 'X_train', 'y_train'],
        outputs=['optimized_model', 'best_params']
    ),
    WorkflowStep(
        name="Save Model",
        function="save_ml_model",
        description="Persist trained model",
        required_context=['optimized_model', 'validation_metrics'],
        outputs=['model_path', 'model_version']
    )
]

# ML Prediction and Trading Workflow
ML_PREDICTION_TRADING = [
    WorkflowStep(
        name="Load Model",
        function="load_ml_model",
        description="Load latest trained model",
        required_context=['symbol'],
        outputs=['model', 'model_metadata']
    ),
    WorkflowStep(
        name="Prepare Live Features",
        function="prepare_live_features",
        description="Generate features for prediction",
        required_context=['symbol'],
        outputs=['live_features']
    ),
    WorkflowStep(
        name="Generate Predictions",
        function="generate_predictions",
        description="Make ML predictions",
        required_context=['model', 'live_features'],
        outputs=['predictions', 'confidence_scores']
    ),
    WorkflowStep(
        name="Apply Trading Logic",
        function="apply_ml_trading_logic",
        description="Convert predictions to signals",
        required_context=['predictions', 'confidence_scores'],
        outputs=['trading_signal', 'position_size']
    ),
    WorkflowStep(
        name="Execute ML Trade",
        function="execute_ml_trade",
        description="Execute based on ML signal",
        required_context=['symbol', 'trading_signal', 'position_size'],
        outputs=['execution_result']
    ),
    WorkflowStep(
        name="Log ML Performance",
        function="log_ml_performance",
        description="Track model performance",
        required_context=['predictions', 'execution_result'],
        outputs=['performance_logged']
    )
]

# ML Model Monitoring Workflow
ML_MODEL_MONITORING = [
    WorkflowStep(
        name="Collect Recent Predictions",
        function="collect_predictions",
        description="Get recent model predictions",
        required_context=['model_version'],
        outputs=['recent_predictions', 'actual_outcomes']
    ),
    WorkflowStep(
        name="Calculate Performance Metrics",
        function="calculate_ml_metrics",
        description="Calculate accuracy, precision, recall",
        required_context=['recent_predictions', 'actual_outcomes'],
        outputs=['performance_metrics']
    ),
    WorkflowStep(
        name="Detect Model Drift",
        function="detect_model_drift",
        description="Check for concept drift",
        required_context=['performance_metrics', 'baseline_metrics'],
        outputs=['drift_detected', 'drift_severity']
    ),
    WorkflowStep(
        name="Retrain If Needed",
        function="trigger_retraining",
        description="Trigger model retraining",
        required_context=['drift_detected', 'drift_severity'],
        outputs=['retraining_triggered'],
        continue_on_failure=True
    )
]

# Advanced ML Workflow for Real-time Strategy Selection
ML_STRATEGY_SELECTION = [
    WorkflowStep(
        name="Analyze Market Conditions",
        function="analyze_market_conditions",
        description="Assess current market regime and volatility",
        required_context=['symbol'],
        outputs=['market_regime', 'volatility_state', 'trend_strength']
    ),
    WorkflowStep(
        name="Evaluate Strategy Performance",
        function="evaluate_strategy_performance",
        description="Assess recent performance of available strategies",
        required_context=['market_regime', 'volatility_state'],
        outputs=['strategy_scores', 'performance_rankings']
    ),
    WorkflowStep(
        name="Select Optimal Strategy",
        function="select_optimal_strategy",
        description="Choose best strategy based on ML predictions",
        required_context=['strategy_scores', 'performance_rankings', 'trend_strength'],
        outputs=['selected_strategy', 'confidence_level']
    ),
    WorkflowStep(
        name="Configure Strategy Parameters",
        function="configure_strategy_params",
        description="Optimize parameters for selected strategy",
        required_context=['selected_strategy', 'market_regime'],
        outputs=['optimized_params']
    ),
    WorkflowStep(
        name="Deploy Strategy",
        function="deploy_ml_strategy",
        description="Activate selected strategy with optimal parameters",
        required_context=['selected_strategy', 'optimized_params', 'confidence_level'],
        outputs=['deployment_status']
    )
]

# ML Portfolio Optimization Workflow
ML_PORTFOLIO_OPTIMIZATION = [
    WorkflowStep(
        name="Gather Portfolio Data",
        function="gather_portfolio_data",
        description="Collect current positions and market data",
        required_context=['portfolio_id'],
        outputs=['current_positions', 'market_data', 'correlation_matrix']
    ),
    WorkflowStep(
        name="Calculate Risk Metrics",
        function="calculate_portfolio_risk",
        description="Compute VaR, CVaR, and other risk measures",
        required_context=['current_positions', 'market_data', 'correlation_matrix'],
        outputs=['risk_metrics', 'exposure_breakdown']
    ),
    WorkflowStep(
        name="Optimize Allocation",
        function="optimize_portfolio_allocation",
        description="ML-driven portfolio optimization",
        required_context=['current_positions', 'risk_metrics', 'market_data'],
        outputs=['optimal_weights', 'expected_return', 'portfolio_risk']
    ),
    WorkflowStep(
        name="Generate Rebalancing Orders",
        function="generate_rebalancing_orders",
        description="Create orders to achieve optimal allocation",
        required_context=['current_positions', 'optimal_weights'],
        outputs=['rebalancing_orders', 'transaction_costs']
    ),
    WorkflowStep(
        name="Execute Rebalancing",
        function="execute_portfolio_rebalancing",
        description="Execute rebalancing trades",
        required_context=['rebalancing_orders', 'transaction_costs'],
        outputs=['execution_results', 'new_portfolio_state'],
        continue_on_failure=True
    )
]

# Export all ML workflows
ML_WORKFLOWS = {
    'ml_feature_engineering': ML_FEATURE_ENGINEERING,
    'ml_model_training': ML_MODEL_TRAINING,
    'ml_prediction_trading': ML_PREDICTION_TRADING,
    'ml_model_monitoring': ML_MODEL_MONITORING,
    'ml_strategy_selection': ML_STRATEGY_SELECTION,
    'ml_portfolio_optimization': ML_PORTFOLIO_OPTIMIZATION
}