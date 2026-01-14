import pytest
from fastapi.testclient import TestClient
from src.api import app  # Import your FastAPI app

client = TestClient(app)

class TestChurnAPI:
    """Test cases for Churn Prediction API"""
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    
    def test_predict_endpoint(self):
        """Test prediction endpoint"""
        sample_customer = {
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
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 79.99,
            "TotalCharges": 959.88
        }
        
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
    
    def test_batch_predict(self):
        """Test batch prediction endpoint"""
        sample_customer = {
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
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 79.99,
            "TotalCharges": 959.88
        }
        
        batch_data = {"customers": [sample_customer, sample_customer]}
        
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
    
    def test_invalid_input(self):
        """Test API with invalid input"""
        invalid_data = {"tenure": "not_a_number"}
        
        response = client.post("/predict", json=invalid_data)
        # Should return 422 or 400 error
        assert response.status_code in [400, 422]
    
    def test_features_endpoint(self):
        """Test features endpoint"""
        response = client.get("/features")
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert len(data["features"]) > 0