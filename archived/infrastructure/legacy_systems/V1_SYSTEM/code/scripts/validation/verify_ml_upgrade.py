
# Verification: Check if predictive features are working
def verify_predictive_features():
    try:
        from ml_features_predictive_integration import create_enhanced_predictive_features
        import pandas as pd
        import numpy as np
        
        # Create test data
        test_data = pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 102,
            'low': np.random.randn(100) + 98,
            'close': np.random.randn(100) + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        features = create_enhanced_predictive_features(test_data)
        if len(features.columns) > 0:
            print(f"✅ Predictive features verified: {len(features.columns)} features generated")
            return True
        else:
            print("❌ Predictive features verification failed: No features generated")
            return False
    except Exception as e:
        print(f"❌ Predictive features verification failed: {e}")
        return False

# Run verification
if __name__ == "__main__":
    print("Verifying predictive features integration...")
    verify_predictive_features()
