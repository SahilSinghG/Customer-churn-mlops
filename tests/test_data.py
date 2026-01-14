import pytest
import pandas as pd
import numpy as np
import os

class TestDataQuality:
    """Test suite for data quality"""
    
    def test_data_files_exist(self):
        """Test that required data files exist"""
        required_files = [
            "data/raw/telco_churn.csv",
            "data/processed/train_data.csv",
            "data/processed/test_data.csv",
            "models/best_model.joblib",
            "models/preprocessor.joblib"
        ]
        
        for file_path in required_files:
            assert os.path.exists(file_path), f"Missing file: {file_path}"
    
    def test_raw_data_quality(self):
        """Test raw data quality"""
        df = pd.read_csv("data/raw/telco_churn.csv")
        
        # Check basic structure
        assert len(df) > 0, "Raw data is empty"
        assert len(df.columns) > 0, "Raw data has no columns"
        
        # Check required columns
        required_cols = ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_processed_data_quality(self):
        """Test processed data quality"""
        train_data = pd.read_csv("data/processed/train_data.csv")
        test_data = pd.read_csv("data/processed/test_data.csv")
        
        # Check no missing values
        assert train_data.isnull().sum().sum() == 0, "Train data has missing values"
        assert test_data.isnull().sum().sum() == 0, "Test data has missing values"
        
        # Check all numeric (except maybe target)
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) >= len(train_data.columns) - 1, "Non-numeric columns found"
        
        # Check train/test split ratio
        total_samples = len(train_data) + len(test_data)
        test_ratio = len(test_data) / total_samples
        assert 0.15 <= test_ratio <= 0.25, f"Test ratio {test_ratio:.2%} outside expected range"
    
    def test_class_balance(self):
        """Test that classes are reasonably balanced"""
        train_data = pd.read_csv("data/processed/train_data.csv")
        
        if 'Churn' in train_data.columns:
            churn_rate = train_data['Churn'].mean()
            # Churn rate should be between 20-40% for telecom
            assert 0.2 <= churn_rate <= 0.4, f"Unusual churn rate: {churn_rate:.2%}"
    
    def test_feature_consistency(self):
        """Test feature consistency between train and test"""
        train_data = pd.read_csv("data/processed/train_data.csv")
        test_data = pd.read_csv("data/processed/test_data.csv")
        
        # Same columns
        assert set(train_data.columns) == set(test_data.columns), "Train/test columns mismatch"
        
        # Similar distributions (basic check)
        for col in train_data.columns:
            if col != 'Churn' and train_data[col].dtype in [np.float64, np.int64]:
                train_mean = train_data[col].mean()
                test_mean = test_data[col].mean()
                
                # Means should be within 20% of each other
                ratio = abs(train_mean - test_mean) / (abs(train_mean) + 1e-10)
                assert ratio < 0.2, f"Large difference in {col}: train={train_mean:.2f}, test={test_mean:.2f}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
