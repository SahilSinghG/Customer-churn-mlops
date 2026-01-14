import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


class DataPreprocessor:
    """Handles all data preprocessing for churn prediction - COMPLETE FIXED VERSION"""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = "Churn"

    def load_data(self, filepath):
        """Load and prepare data"""
        df = pd.read_csv(filepath)
        print(f"Loaded data: {df.shape}")
        return df

    def clean_data(self, df):
        """Clean and prepare raw data"""
        df_clean = df.copy()

        # Convert TotalCharges to numeric
        df_clean["TotalCharges"] = pd.to_numeric(
            df_clean["TotalCharges"], errors="coerce"
        )

        # Fill missing values
        df_clean["TotalCharges"].fillna(0, inplace=True)

        # Drop customerID (not useful for modeling)
        if "customerID" in df_clean.columns:
            df_clean = df_clean.drop("customerID", axis=1)

        return df_clean

    def encode_categorical(self, df, fit=True):
        """Encode ALL categorical variables - COMPLETE FIXED VERSION"""
        df_encoded = df.copy()

        # ===== 1. BINARY COLUMNS (Yes/No) =====
        binary_cols = [
            "Partner",
            "Dependents",
            "PhoneService",
            "PaperlessBilling",
            "Churn",
        ]

        for col in binary_cols:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].map({"Yes": 1, "No": 0})

        # ===== 2. COLUMNS WITH "No internet service" =====
        internet_service_cols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

        for col in internet_service_cols:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].map(
                    {"No": 0, "Yes": 1, "No internet service": 0}
                )

        # ===== 3. MULTIPLE LINES =====
        if "MultipleLines" in df_encoded.columns:
            df_encoded["MultipleLines"] = df_encoded["MultipleLines"].map(
                {"No": 0, "Yes": 1, "No phone service": 0}
            )

        # ===== 4. INTERNET SERVICE =====
        if "InternetService" in df_encoded.columns:
            if fit:
                le = LabelEncoder()
                df_encoded["InternetService"] = le.fit_transform(
                    df_encoded["InternetService"]
                )
                self.label_encoders["InternetService"] = le
            else:
                if "InternetService" in self.label_encoders:
                    le = self.label_encoders["InternetService"]
                    df_encoded["InternetService"] = df_encoded["InternetService"].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        # ===== 5. CONTRACT =====
        if "Contract" in df_encoded.columns:
            if fit:
                le = LabelEncoder()
                df_encoded["Contract"] = le.fit_transform(df_encoded["Contract"])
                self.label_encoders["Contract"] = le
            else:
                if "Contract" in self.label_encoders:
                    le = self.label_encoders["Contract"]
                    df_encoded["Contract"] = df_encoded["Contract"].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        # ===== 6. PAYMENT METHOD =====
        if "PaymentMethod" in df_encoded.columns:
            if fit:
                le = LabelEncoder()
                df_encoded["PaymentMethod"] = le.fit_transform(
                    df_encoded["PaymentMethod"]
                )
                self.label_encoders["PaymentMethod"] = le
            else:
                if "PaymentMethod" in self.label_encoders:
                    le = self.label_encoders["PaymentMethod"]
                    df_encoded["PaymentMethod"] = df_encoded["PaymentMethod"].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        # ===== 7. GENDER =====
        if "gender" in df_encoded.columns:
            if fit:
                le = LabelEncoder()
                df_encoded["gender"] = le.fit_transform(df_encoded["gender"])
                self.label_encoders["gender"] = le
            else:
                if "gender" in self.label_encoders:
                    le = self.label_encoders["gender"]
                    df_encoded["gender"] = df_encoded["gender"].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        # ===== 8. SENIOR CITIZEN (already numeric but ensure int) =====
        if "SeniorCitizen" in df_encoded.columns:
            df_encoded["SeniorCitizen"] = df_encoded["SeniorCitizen"].astype(int)

        # ===== 9. TENURE, MONTHLY CHARGES, TOTAL CHARGES (already numeric) =====
        # Just ensure they're float
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        for col in numeric_cols:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].astype(float)

        return df_encoded

    def scale_features(self, df, fit=True):
        """Scale numerical features"""
        numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

        # Check which numeric columns exist
        existing_numeric_cols = [col for col in numerical_cols if col in df.columns]

        if existing_numeric_cols:
            if fit:
                scaled_features = self.scaler.fit_transform(df[existing_numeric_cols])
            else:
                scaled_features = self.scaler.transform(df[existing_numeric_cols])

            # Create new DataFrame with scaled features
            df_scaled = df.copy()
            df_scaled[existing_numeric_cols] = scaled_features

            return df_scaled
        else:
            return df

    def prepare_features_target(self, df):
        """Split into features and target"""
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]

        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()

        return X, y

    def train_test_split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Train set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Train churn rate: {y_train.mean():.2%}")
        print(f"Test churn rate: {y_test.mean():.2%}")

        return X_train, X_test, y_train, y_test

    def save_preprocessor(self, filepath):
        """Save preprocessor artifacts"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        artifacts = {
            "label_encoders": self.label_encoders,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
        }
        joblib.dump(artifacts, filepath)
        print(f"Preprocessor saved to: {filepath}")

    def load_preprocessor(self, filepath):
        """Load preprocessor artifacts"""
        artifacts = joblib.load(filepath)
        self.label_encoders = artifacts["label_encoders"]
        self.scaler = artifacts["scaler"]
        self.feature_columns = artifacts["feature_columns"]
        self.target_column = artifacts.get("target_column", "Churn")
        print(f"Preprocessor loaded from: {filepath}")


def main():
    """Main preprocessing pipeline"""
    print("=" * 50)
    print("DATA PREPROCESSING PIPELINE - COMPLETE VERSION")
    print("=" * 50)

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Load raw data
    raw_data_path = "data/raw/telco_churn.csv"
    df_raw = preprocessor.load_data(raw_data_path)

    # Clean data
    df_clean = preprocessor.clean_data(df_raw)

    # Encode categorical variables
    df_encoded = preprocessor.encode_categorical(df_clean, fit=True)

    # Verify all columns are numeric
    print("\n" + "=" * 50)
    print("VERIFYING DATA TYPES")
    print("=" * 50)
    print("Data types after encoding:")
    print(df_encoded.dtypes)

    non_numeric_cols = df_encoded.select_dtypes(include=["object"]).columns
    if len(non_numeric_cols) > 0:
        print(f"\nWARNING: Non-numeric columns found: {list(non_numeric_cols)}")
        for col in non_numeric_cols:
            print(f"  {col}: {df_encoded[col].unique()[:10]}")
    else:
        print("\nSUCCESS: All columns are numeric!")

    # Scale features
    df_scaled = preprocessor.scale_features(df_encoded, fit=True)

    # Prepare features and target
    X, y = preprocessor.prepare_features_target(df_scaled)

    # Split data
    X_train, X_test, y_train, y_test = preprocessor.train_test_split_data(X, y)

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)

    # Save train/test sets
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data.to_csv("data/processed/train_data.csv", index=False)
    test_data.to_csv("data/processed/test_data.csv", index=False)

    # Save preprocessor
    preprocessor.save_preprocessor("models/preprocessor.joblib")

    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE - ALL COLUMNS NUMERIC")
    print("=" * 50)
    print(f"Training data saved: data/processed/train_data.csv")
    print(f"Testing data saved: data/processed/test_data.csv")
    print(f"Preprocessor saved: models/preprocessor.joblib")

    # Final verification
    print("\n" + "=" * 50)
    print("FINAL VERIFICATION")
    print("=" * 50)
    print("Training data sample (first 3 rows):")
    print(train_data.head(3))
    print("\nTraining data info:")
    print(f"Shape: {train_data.shape}")
    print(f"Columns: {list(train_data.columns)}")
    print(f"Data types: \n{train_data.dtypes}")

    return preprocessor, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocessor, X_train, X_test, y_train, y_test = main()
