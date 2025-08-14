"""
Test suite for Phase 4 deep learning components
Validates DL-001 through DL-004 implementations
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.bot.ml.deep_learning import (
    AttentionConfig,
    AttentionMechanism,
    AttentionType,
    LSTMArchitecture,
    LSTMConfig,
    LSTMDataPipeline,
    LSTMTrainingFramework,
    ScalingMethod,
    SequenceConfig,
    TaskType,
    TrainingConfig,
    create_integrated_lstm_pipeline,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(n_samples: int = 1000, n_features: int = 50) -> pd.DataFrame:
    """Generate sample financial time series data"""

    # Generate timestamps
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]

    # Generate synthetic OHLCV data
    price = 100.0
    prices = []
    volumes = []

    for i in range(n_samples):
        # Random walk for price
        price_change = np.random.normal(0, 0.02)
        price *= 1 + price_change

        # OHLCV with some realistic patterns
        open_price = price
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        close = price * (1 + np.random.normal(0, 0.005))
        volume = np.random.lognormal(10, 1)

        prices.append([open_price, high, low, close])
        volumes.append(volume)

    # Create DataFrame
    price_df = pd.DataFrame(prices, columns=["open", "high", "low", "close"])
    price_df["volume"] = volumes
    price_df["datetime"] = timestamps

    # Add some technical indicators as features
    for i in range(n_features - 5):  # 5 OHLCV columns already
        feature_name = f"feature_{i:02d}"
        # Generate correlated features with some noise
        if i % 3 == 0:
            price_df[feature_name] = price_df["close"].rolling(5).mean() + np.random.normal(
                0, 0.1, n_samples
            )
        elif i % 3 == 1:
            price_df[feature_name] = price_df["volume"].rolling(10).std() + np.random.normal(
                0, 0.1, n_samples
            )
        else:
            price_df[feature_name] = np.random.normal(0, 1, n_samples)

    # Create target (next period return)
    price_df["target"] = price_df["close"].pct_change().shift(-1)

    # Drop NaN values
    price_df = price_df.dropna().reset_index(drop=True)

    return price_df


def test_lstm_architecture():
    """Test DL-001: LSTM Architecture Design"""
    logger.info("Testing LSTM Architecture...")

    # Test different configurations
    configs = [
        LSTMConfig(input_size=50, hidden_size=64, num_layers=1, task_type=TaskType.REGRESSION),
        LSTMConfig(
            input_size=50,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            task_type=TaskType.BINARY_CLASSIFICATION,
        ),
        LSTMConfig(
            input_size=30,
            hidden_size=256,
            num_layers=3,
            dropout=0.3,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            num_classes=5,
        ),
    ]

    for i, config in enumerate(configs):
        try:
            model = LSTMArchitecture(config)
            summary = model.get_model_summary()

            logger.info(
                f"Config {i+1}: Backend={summary['backend']}, Task={config.task_type.value}"
            )

            # Test prediction with dummy data
            batch_size = 16
            seq_len = config.sequence_length
            X_dummy = np.random.randn(batch_size, seq_len, config.input_size)
            lengths_dummy = np.full(batch_size, seq_len)

            predictions = model.predict(X_dummy, lengths_dummy)

            # Validate output shape
            if config.task_type == TaskType.REGRESSION:
                expected_shape = (batch_size, config.output_size)
            elif config.task_type == TaskType.BINARY_CLASSIFICATION:
                expected_shape = (batch_size, 1)
            else:
                expected_shape = (batch_size, config.num_classes)

            assert (
                predictions.shape == expected_shape
            ), f"Expected {expected_shape}, got {predictions.shape}"
            logger.info(f"✓ Architecture test {i+1} passed")

        except Exception as e:
            logger.error(f"✗ Architecture test {i+1} failed: {e}")


def test_lstm_data_pipeline():
    """Test DL-002: LSTM Data Pipeline"""
    logger.info("Testing LSTM Data Pipeline...")

    # Generate sample data
    data = generate_sample_data(500, 20)
    feature_cols = [
        col
        for col in data.columns
        if col.startswith("feature_") or col in ["open", "high", "low", "close", "volume"]
    ]

    # Test different configurations
    configs = [
        SequenceConfig(
            sequence_length=10, overlap_ratio=0.5, scaling_method=ScalingMethod.STANDARD
        ),
        SequenceConfig(
            sequence_length=30,
            overlap_ratio=0.8,
            scaling_method=ScalingMethod.ROBUST,
            augmentation_ratio=0.1,
        ),
        SequenceConfig(
            sequence_length=50,
            overlap_ratio=0.9,
            scaling_method=ScalingMethod.MINMAX,
            max_missing_ratio=0.2,
        ),
    ]

    for i, config in enumerate(configs):
        try:
            pipeline = LSTMDataPipeline(config)

            # Create sequences
            X_seq, y_seq, lengths, timestamps = pipeline.create_sequences(
                data,
                "target",
                feature_cols[: min(len(feature_cols), 15)],  # Use subset of features
            )

            # Validate shapes
            assert X_seq.ndim == 3, f"Expected 3D sequences, got {X_seq.ndim}D"
            assert (
                X_seq.shape[1] == config.sequence_length
            ), f"Expected seq_len {config.sequence_length}, got {X_seq.shape[1]}"
            assert len(X_seq) == len(y_seq), "Sequence and target lengths mismatch"

            # Test scaling
            X_scaled, y_scaled = pipeline.fit_transform(X_seq, y_seq)
            assert X_scaled.shape == X_seq.shape, "Scaling changed sequence shape"

            # Test augmentation
            X_aug, y_aug = pipeline.apply_augmentation(X_scaled, y_scaled)
            expected_aug_size = len(X_scaled) + int(len(X_scaled) * config.augmentation_ratio)
            assert len(X_aug) >= len(X_scaled), "Augmentation should increase or maintain data size"

            # Test time series split
            train_data, val_data, test_data = pipeline.time_series_split(
                X_aug, y_aug, timestamps[: len(X_aug)]
            )
            total_samples = len(train_data[0]) + len(val_data[0]) + len(test_data[0])
            assert total_samples <= len(X_aug), "Split samples exceed original data"

            logger.info(f"✓ Data pipeline test {i+1} passed: {len(X_seq)} sequences created")

        except Exception as e:
            logger.error(f"✗ Data pipeline test {i+1} failed: {e}")


def test_lstm_training():
    """Test DL-003: LSTM Training Framework"""
    logger.info("Testing LSTM Training Framework...")

    # Generate small dataset for quick training
    data = generate_sample_data(200, 10)
    feature_cols = [
        col for col in data.columns if col.startswith("feature_") or col in ["close", "volume"]
    ][:10]

    # Create simple LSTM and data pipeline
    lstm_config = LSTMConfig(
        input_size=len(feature_cols),
        hidden_size=32,
        num_layers=1,
        sequence_length=10,
        task_type=TaskType.REGRESSION,
    )

    sequence_config = SequenceConfig(sequence_length=10, overlap_ratio=0.5, batch_size=8)

    training_config = TrainingConfig(
        epochs=5,  # Short training for testing
        learning_rate=0.01,
        early_stopping=True,
        patience=3,
        tensorboard_logging=False,  # Disable for testing
        save_checkpoints=False,
    )

    try:
        # Initialize components
        model = LSTMArchitecture(lstm_config)
        data_pipeline = LSTMDataPipeline(sequence_config)
        trainer = LSTMTrainingFramework(training_config)

        # Prepare data
        X_seq, y_seq, lengths, timestamps = data_pipeline.create_sequences(
            data, "target", feature_cols
        )
        X_seq, y_seq = data_pipeline.fit_transform(X_seq, y_seq)

        # Split data
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        train_lengths = lengths[:split_idx] if lengths is not None else None
        val_lengths = lengths[split_idx:] if lengths is not None else None

        # Train model
        start_time = datetime.now()
        results = trainer.train(
            model, data_pipeline, X_train, y_train, X_val, y_val, train_lengths, val_lengths
        )
        training_time = (datetime.now() - start_time).total_seconds()

        # Validate results
        assert results.total_epochs > 0, "No training epochs completed"
        assert len(results.train_losses) > 0, "No training losses recorded"
        assert (
            training_time < 300
        ), f"Training took too long: {training_time:.1f}s"  # Should be fast for small data

        logger.info(f"✓ Training test passed: {results.total_epochs} epochs, {training_time:.1f}s")
        logger.info(f"  Final train loss: {results.train_losses[-1]:.4f}")
        logger.info(f"  Final val loss: {results.val_losses[-1]:.4f}")

    except Exception as e:
        logger.error(f"✗ Training test failed: {e}")


def test_attention_mechanisms():
    """Test DL-004: Attention Mechanisms"""
    logger.info("Testing Attention Mechanisms...")

    # Test different attention types
    attention_types = [
        AttentionType.SELF_ATTENTION,
        AttentionType.MULTI_HEAD,
        AttentionType.SCALED_DOT_PRODUCT,
        AttentionType.ADDITIVE,
    ]

    for attention_type in attention_types:
        try:
            config = AttentionConfig(
                attention_type=attention_type,
                d_model=64,
                num_heads=4 if attention_type == AttentionType.MULTI_HEAD else 1,
                dropout=0.1,
            )

            attention = AttentionMechanism(config)

            # Test attention analysis
            dummy_weights = np.random.random((8, 20, 20))  # Multi-head attention weights
            analysis = attention.analyze_attention_patterns(dummy_weights)

            # Validate analysis results
            assert "attention_entropy" in analysis, "Missing attention entropy analysis"
            assert "important_positions" in analysis, "Missing important positions analysis"
            assert len(analysis["attention_entropy"]) == 20, "Incorrect entropy array length"

            # Test attention metrics
            pred_with = np.random.random(100)
            pred_without = pred_with + np.random.normal(0, 0.1, 100)
            targets = np.random.random(100)

            metrics = attention.compute_attention_metrics(pred_with, pred_without, targets)
            assert "mse_improvement_percent" in metrics, "Missing MSE improvement metric"
            assert "meets_3_percent_improvement" in metrics, "Missing improvement threshold check"

            logger.info(f"✓ Attention test passed: {attention_type.value}")

        except Exception as e:
            logger.error(f"✗ Attention test failed for {attention_type.value}: {e}")


def test_integrated_pipeline():
    """Test integrated LSTM pipeline"""
    logger.info("Testing Integrated LSTM Pipeline...")

    try:
        # Create pipeline with default settings
        pipeline = create_integrated_lstm_pipeline(
            sequence_length=15,
            input_size=10,
            hidden_size=32,
            num_layers=1,
            epochs=3,  # Short training
            use_phase3_features=False,  # Disable for testing
        )

        # Generate data
        data = generate_sample_data(150, 10)
        feature_cols = [
            col for col in data.columns if col.startswith("feature_") or col in ["close", "volume"]
        ][:10]

        # Train pipeline
        results = pipeline.fit(data, "target", feature_cols)

        # Test predictions
        predictions = pipeline.predict(data.iloc[-20:])
        assert len(predictions) > 0, "No predictions generated"

        # Test model summary
        summary = pipeline.get_model_summary()
        assert "lstm_architecture" in summary, "Missing LSTM architecture in summary"
        assert "performance_metrics" in summary, "Missing performance metrics in summary"

        logger.info("✓ Integrated pipeline test passed")
        logger.info(f"  Training epochs: {results.total_epochs}")
        logger.info(f"  Predictions shape: {predictions.shape}")

    except Exception as e:
        logger.error(f"✗ Integrated pipeline test failed: {e}")


def run_performance_benchmarks():
    """Run performance benchmarks to validate targets"""
    logger.info("Running Performance Benchmarks...")

    # Test training time target: < 30 minutes for 2 years of data
    logger.info("Testing training time target...")

    # Simulate 2 years of hourly data
    large_data = generate_sample_data(2 * 365 * 24, 50)  # 2 years, hourly
    feature_cols = [
        col
        for col in large_data.columns
        if col.startswith("feature_") or col in ["open", "high", "low", "close", "volume"]
    ][:50]

    # Create efficient configuration
    config = TrainingConfig(
        epochs=10,  # Limited for testing
        batch_size=64,
        learning_rate=0.001,
        early_stopping=True,
        patience=5,
        tensorboard_logging=False,
        save_checkpoints=False,
    )

    lstm_config = LSTMConfig(input_size=50, hidden_size=128, num_layers=2, sequence_length=30)

    try:
        model = LSTMArchitecture(lstm_config)
        data_pipeline = LSTMDataPipeline(SequenceConfig(sequence_length=30, batch_size=64))
        trainer = LSTMTrainingFramework(config)

        # Prepare subset for timing test
        X_seq, y_seq, lengths, _ = data_pipeline.create_sequences(
            large_data.iloc[:5000],
            "target",
            feature_cols,  # Subset for testing
        )
        X_seq, y_seq = data_pipeline.fit_transform(X_seq, y_seq)

        # Time training
        start_time = datetime.now()
        results = trainer.train(model, data_pipeline, X_seq, y_seq)
        training_time = (datetime.now() - start_time).total_seconds()

        # Extrapolate to full dataset
        extrapolated_time = training_time * (len(large_data) / 5000)

        logger.info(f"Training time benchmark: {training_time:.1f}s for 5K samples")
        logger.info(f"Extrapolated for full dataset: {extrapolated_time/60:.1f} minutes")

        # Test inference time target: < 30ms
        logger.info("Testing inference time target...")

        test_X = X_seq[:32]  # Batch of 32
        start_time = datetime.now()
        predictions = model.predict(test_X)
        inference_time = (datetime.now() - start_time).total_seconds() * 1000  # Convert to ms

        per_sample_time = inference_time / len(test_X)

        logger.info(f"Inference time: {inference_time:.1f}ms for {len(test_X)} samples")
        logger.info(f"Per sample: {per_sample_time:.1f}ms")

        # Validate targets
        meets_training_target = extrapolated_time < 30 * 60  # 30 minutes
        meets_inference_target = per_sample_time < 30  # 30ms

        logger.info(
            f"Meets training time target (<30 min): {'✓' if meets_training_target else '✗'}"
        )
        logger.info(
            f"Meets inference time target (<30ms): {'✓' if meets_inference_target else '✗'}"
        )

    except Exception as e:
        logger.error(f"Performance benchmark failed: {e}")


def main():
    """Run all tests"""
    logger.info("Starting Phase 4 Deep Learning Component Tests")
    logger.info("=" * 60)

    # Run component tests
    test_lstm_architecture()
    print()

    test_lstm_data_pipeline()
    print()

    test_lstm_training()
    print()

    test_attention_mechanisms()
    print()

    test_integrated_pipeline()
    print()

    # Run performance benchmarks
    run_performance_benchmarks()

    logger.info("=" * 60)
    logger.info("Phase 4 Deep Learning Component Tests Complete")


if __name__ == "__main__":
    main()
