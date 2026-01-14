from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from src.predict import predictor
import json
from datetime import datetime

# Initialize FastAPI
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ===== DATA MODELS =====
class CustomerFeatures(BaseModel):
    """Input features for churn prediction"""

    gender: str = Field(..., description="Gender: Male or Female")
    SeniorCitizen: int = Field(..., description="Senior citizen: 0 or 1")
    Partner: str = Field(..., description="Has partner: Yes or No")
    Dependents: str = Field(..., description="Has dependents: Yes or No")
    tenure: int = Field(..., description="Months with company")
    PhoneService: str = Field(..., description="Has phone service: Yes or No")
    MultipleLines: Optional[str] = Field(
        "No", description="Multiple lines: Yes, No, or No phone service"
    )
    InternetService: str = Field(
        ..., description="Internet service: DSL, Fiber optic, or No"
    )
    OnlineSecurity: Optional[str] = Field(
        "No", description="Online security: Yes, No, or No internet service"
    )
    OnlineBackup: Optional[str] = Field(
        "No", description="Online backup: Yes, No, or No internet service"
    )
    DeviceProtection: Optional[str] = Field(
        "No", description="Device protection: Yes, No, or No internet service"
    )
    TechSupport: Optional[str] = Field(
        "No", description="Tech support: Yes, No, or No internet service"
    )
    StreamingTV: Optional[str] = Field(
        "No", description="Streaming TV: Yes, No, or No internet service"
    )
    StreamingMovies: Optional[str] = Field(
        "No", description="Streaming movies: Yes, No, or No internet service"
    )
    Contract: str = Field(
        ..., description="Contract type: Month-to-month, One year, or Two year"
    )
    PaperlessBilling: str = Field(..., description="Paperless billing: Yes or No")
    PaymentMethod: str = Field(
        ...,
        description="Payment method: Electronic check, Mailed check, Bank transfer, or Credit card",
    )
    MonthlyCharges: float = Field(..., description="Monthly charges")
    TotalCharges: Optional[float] = Field(
        None, description="Total charges (calculated if not provided)"
    )

    class Config:
        schema_extra = {
            "example": {
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
                "TotalCharges": 959.88,
            }
        }


class PredictionResponse(BaseModel):
    """Response model for churn prediction"""

    churn_prediction: bool
    churn_probability: float
    confidence: float
    risk_level: str
    recommendation: str
    features_used: int
    model_type: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    features_count: int
    uptime: str


# ===== API ENDPOINTS =====
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": type(predictor.model).__name__,
        "features_count": len(predictor.feature_columns),
        "uptime": datetime.now().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_churn(customer: CustomerFeatures):
    """
    Predict churn for a single customer

    - **customer**: Customer features in JSON format
    - Returns: Churn prediction with probability and recommendations
    """
    try:
        # Convert Pydantic model to dict
        input_data = customer.dict()

        # Calculate TotalCharges if not provided
        if input_data.get("TotalCharges") is None:
            input_data["TotalCharges"] = (
                input_data["tenure"] * input_data["MonthlyCharges"]
            )

        # Make prediction
        result = predictor.predict(input_data)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Add timestamp
        result["timestamp"] = datetime.now().isoformat()

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(batch_request: BatchPredictionRequest):
    """
    Predict churn for multiple customers

    - **customers**: List of customer features
    - Returns: List of predictions
    """
    try:
        customers_data = []
        for customer in batch_request.customers:
            customer_dict = customer.dict()
            # Calculate TotalCharges if not provided
            if customer_dict.get("TotalCharges") is None:
                customer_dict["TotalCharges"] = (
                    customer_dict["tenure"] * customer_dict["MonthlyCharges"]
                )
            customers_data.append(customer_dict)

        results = predictor.predict_batch(customers_data)

        # Add timestamps
        for result in results:
            result["timestamp"] = datetime.now().isoformat()

        return {
            "predictions": results,
            "count": len(results),
            "batch_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features", tags=["Information"])
async def get_features():
    """Get list of features used by the model"""
    return {
        "features": predictor.feature_columns,
        "count": len(predictor.feature_columns),
        "feature_types": {
            "categorical": ["gender", "Contract", "PaymentMethod", "InternetService"],
            "binary": ["Partner", "Dependents", "PhoneService", "PaperlessBilling"],
            "numerical": ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"],
        },
    }


# ===== RUN SERVER =====
if __name__ == "__main__":
    print("=" * 60)
    print("Starting Customer Churn Prediction API")
    print("=" * 60)
    print(f"Model: {type(predictor.model).__name__}")
    print(f"Features: {len(predictor.feature_columns)}")
    print(f"API Docs: http://localhost:8000/docs")
    print("=" * 60)

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
