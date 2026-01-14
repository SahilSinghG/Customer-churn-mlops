import pytest
import requests
import json

BASE_URL = "http://localhost:8000"

class TestChurnAPI:
    """Test suite for Churn Prediction API"""
    
    @pytest.fixture
    def sample_customer(self):
        """Sample customer data for testing"""
        return {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 79.99,
            "TotalCharges": 959.88
        }
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] == True
        assert "XGBClassifier" in data["model_type"]
    
    def test_predict_endpoint(self, sample_customer):
        """Test prediction endpoint"""
        response = requests.post(
            f"{BASE_URL}/predict",
            json=sample_customer,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "churn_prediction" in data
        assert "churn_probability" in data
        assert "risk_level" in data
        assert "recommendation" in data
        assert "model_type" in data
        
        # Check data types
        assert isinstance(data["churn_prediction"], bool)
        assert 0 <= data["churn_probability"] <= 1
    
    def test_batch_predict(self, sample_customer):
        """Test batch prediction endpoint"""
        batch_data = {
            "customers": [sample_customer, sample_customer]
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=batch_data,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
    
    def test_invalid_input(self):
        """Test API with invalid input"""
        invalid_data = {"tenure": "not_a_number"}
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_data,
            headers={"Content-Type": "application/json"}
        )
        # Should return 422 for validation error
        assert response.status_code == 422
    
    def test_features_endpoint(self):
        """Test features endpoint"""
        response = requests.get(f"{BASE_URL}/features")
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert "count" in data
        assert data["count"] > 0

if __name__ == "__main__":
    # Run tests directly
    import sys
    pytest.main([__file__, "-v"])
