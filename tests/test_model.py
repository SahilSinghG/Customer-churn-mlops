import pytest
import pandas as pd
import joblib
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.predict import ChurnPredictor
    HAS_SRC = True
except ImportError:
    HAS_SRC = False
    print("⚠️ Could not import from src, some tests will be skipped")

class TestChurnModel:
    """Test suite for Churn Prediction Model"""
    
    @pytest.fixture
    def predictor(self):
        """Initialize predictor for testing"""
        if not HAS_SRC:
            pytest.skip("src module not available")
        return ChurnPredictor()
    
    @pytest.fixture
    def sample_data(self):
        """Sample test data"""
        return pd.read_csv("data/processed/test_data.csv")
    
    def test_model_loading(self, predictor):
        """Test that model loads correctly"""
        assert predictor.model is not None
        assert hasattr(predictor.model, "predict")
        assert hasattr(predictor.model, "predict_proba")
        assert len(predictor.feature_columns) > 0
    
    @pytest.mark.skipif(not HAS_SRC, reason="src module not available")
    def test_preprocessor_loading(self, predictor):
        """Test that preprocessor loads correctly"""
        assert predictor.scaler is not None
        assert isinstance(predictor.label_encoders, dict)
        assert len(predictor.feature_columns) > 0
    
    def test_data_files_exist(self):
        """Test that data files exist"""
        assert os.path.exists("data/processed/test_data.csv")
        assert os.path.exists("models/best_model.joblib")
    
    def test_model_file_loads(self):
        """Test that model file can be loaded"""
        model = joblib.load("models/best_model.joblib")
        assert model is not None
        assert hasattr(model, "predict")
    
    def test_prediction_works(self):
        """Test basic prediction with loaded model"""
        model = joblib.load("models/best_model.joblib")
        # Create simple test data
        test_features = np.zeros((1, 19))  # 19 features
        prediction = model.predict(test_features)
        assert prediction.shape == (1,)
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(test_features)
            assert proba.shape[1] == 2  # Two classes

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
