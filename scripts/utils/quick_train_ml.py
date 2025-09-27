#!/usr/bin/env python3
"""
Quick ML Training Script - Minimal setup for testing

This script provides a quick way to train ML models with minimal data
for testing and development purposes.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_train():
    """Quick training with minimal data for testing"""
    print("\n🚀 Quick ML Model Training (Test Mode)")
    print("="*50)
    
    try:
        # Import training functions
        from bot_v2.features.ml_strategy.ml_strategy import (
            train_strategy_selector,
            predict_best_strategy,
            get_strategy_recommendation
        )
        
        # Use small dataset for quick training
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6 months
        
        print(f"📊 Training on {len(symbols)} symbols")
        print(f"📅 Period: {start_date.date()} to {end_date.date()}")
        
        # Train model
        print("\n⏳ Training strategy selector...")
        result = train_strategy_selector(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            validation_split=0.2
        )
        
        print(f"✅ Training complete!")
        print(f"   Validation score: {result.validation_score:.2%}")
        
        # Test predictions
        print("\n🔮 Testing predictions:")
        for symbol in symbols[:3]:
            recommendation = get_strategy_recommendation(symbol, min_confidence=0.5)
            if recommendation:
                print(f"   {symbol}: {recommendation.value}")
        
        print("\n✅ Quick training successful!")
        print("   Models are now ready for use.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logger.error(f"Quick training failed: {e}", exc_info=True)
        return False


def test_ml_integration():
    """Test the ML integration after training"""
    print("\n🧪 Testing ML Integration")
    print("="*50)
    
    try:
        from bot_v2.orchestration.ml_integration import create_ml_integrator
        
        integrator = create_ml_integrator({
            'min_confidence': 0.5,  # Lower threshold for testing
            'max_position_size': 0.2
        })
        
        # Test decision making
        decision = integrator.make_trading_decision(
            symbol='AAPL',
            portfolio_value=10000,
            current_positions={}
        )
        
        print(f"\n📊 Test Decision for AAPL:")
        print(f"   Strategy: {decision.strategy}")
        print(f"   Confidence: {decision.confidence:.1%}")
        print(f"   Action: {decision.decision}")
        print(f"   Position Size: {decision.risk_adjusted_size:.1%}")
        
        if decision.decision != 'hold':
            print("\n✅ ML integration working - generating trade signals!")
        else:
            print("\n⚠️ ML integration working but confidence may be low")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("="*50)
    print("Quick ML Training for GPT-Trader")
    print("="*50)
    print("\nThis will train models with minimal data for testing.")
    print("Full training recommended for production use.\n")
    
    # Run quick training
    if quick_train():
        # Test integration
        test_ml_integration()
        
        print("\n" + "="*50)
        print("Quick training complete!")
        print("="*50)
        print("\nNext steps:")
        print("1. Run full training: python scripts/train_ml_models.py")
        print("2. Test ML pipeline: python demos/ml_pipeline_demo.py")
    else:
        print("\nTraining failed. Check logs for details.")
        sys.exit(1)