import pandas as pd
import joblib
from typing import Dict, Any


class ChurnPredictor:
    """Loads model and makes churn predictions"""

    def __init__(
        self,
        model_path="models/best_model.joblib",
        preprocessor_path="models/preprocessor.joblib",
    ):
        """Initialize predictor with model and preprocessor"""
        print("Loading model and preprocessor...")
        self.model = joblib.load(model_path)
        preprocessor_data = joblib.load(preprocessor_path)

        self.label_encoders = preprocessor_data["label_encoders"]
        self.scaler = preprocessor_data["scaler"]
        self.feature_columns = preprocessor_data["feature_columns"]

        print(f"Model: {type(self.model).__name__}")
        print(f"Features: {len(self.feature_columns)} columns")

    def prepare_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert API input to model-ready format"""

        # Create DataFrame from input
        df = pd.DataFrame([input_data])

        # ===== ENCODE CATEGORICAL VARIABLES =====
        # Gender
        if "gender" in df.columns:
            df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

        # Binary columns
        binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({"Yes": 1, "No": 0})

        # Internet service columns
        internet_cols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        for col in internet_cols:
            if col in df.columns:
                df[col] = df[col].map({"Yes": 1, "No": 0, "No internet service": 0})

        # MultipleLines
        if "MultipleLines" in df.columns:
            df["MultipleLines"] = df["MultipleLines"].map(
                {"Yes": 1, "No": 0, "No phone service": 0}
            )

        # Label encode other categoricals using saved encoders
        categorical_cols = ["InternetService", "Contract", "PaymentMethod"]
        for col in categorical_cols:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen labels
                df[col] = df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # Ensure numeric columns are float
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Fill missing numeric values with 0
        df = df.fillna(0)

        # ===== SCALE NUMERICAL FEATURES =====
        scale_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        existing_scale_cols = [col for col in scale_cols if col in df.columns]

        if existing_scale_cols:
            df[existing_scale_cols] = self.scaler.transform(df[existing_scale_cols])

        # ===== ENSURE ALL FEATURES EXIST =====
        # Add missing columns with 0
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns to match training
        df = df[self.feature_columns]

        return df

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make churn prediction"""
        try:
            # Prepare input
            features_df = self.prepare_input(input_data)

            # Make prediction
            prediction = self.model.predict(features_df)[0]
            prediction_proba = self.model.predict_proba(features_df)[0]

            # Prepare response
            churn_probability = float(
                prediction_proba[1]
            )  # Probability of churn (class 1)
            will_churn = bool(prediction == 1)

            # Risk categories
            if churn_probability < 0.3:
                risk_level = "Low"
                recommendation = "Maintain current service"
            elif churn_probability < 0.7:
                risk_level = "Medium"
                recommendation = "Offer loyalty discount"
            else:
                risk_level = "High"
                recommendation = "Immediate retention offer needed"

            return {
                "churn_prediction": will_churn,
                "churn_probability": round(churn_probability, 4),
                "confidence": round(max(prediction_proba), 4),
                "risk_level": risk_level,
                "recommendation": recommendation,
                "features_used": len(self.feature_columns),
                "model_type": type(self.model).__name__,
            }

        except Exception as e:
            return {
                "error": str(e),
                "churn_prediction": None,
                "churn_probability": None,
            }

    def predict_batch(self, input_list: list) -> list:
        """Predict for multiple customers"""
        results = []
        for input_data in input_list:
            results.append(self.predict(input_data))
        return results


# Singleton instance
predictor = ChurnPredictor()

if __name__ == "__main__":
    # Test the predictor
    test_input = {
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

    result = predictor.predict(test_input)
    print("Test prediction:")
    print(f"Will churn: {result['churn_prediction']}")
    print(f"Probability: {result['churn_probability']:.1%}")
    print(f"Risk: {result['risk_level']}")
    print(f"Recommendation: {result['recommendation']}")
