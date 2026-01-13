import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    """Handles all data preprocessing for churn prediction"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'Churn'
        
    def load_data(self, filepath):
        """Load and prepare data"""
        df = pd.read_csv(filepath)
        print(f"Loaded data: {df.shape}")
        return df
    
    def clean_data(self, df):
        """Clean and prepare raw data"""
        df_clean = df.copy()
        
        # Convert TotalCharges to numeric
        df_clean['TotalCharges'] = pd.to_numeric(
            df_clean['TotalCharges'], errors='coerce'
        )
        
        # Fill missing values
        df_clean['TotalCharges'].fillna(0, inplace=True)
        
        # Drop customerID (not useful for modeling)
        if 'customerID' in df_clean.columns:
            df_clean = df_clean.drop('customerID', axis=1)
            
        return df_clean
    
    def encode_categorical(self, df, fit=True):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        # Binary encode yes/no columns
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 
                      'PaperlessBilling', 'Churn']
        
        for col in binary_cols:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})
        
        # Label encode other categorical columns (simplified for Windows)
        categorical_cols = ['gender', 'Contract', 'PaymentMethod']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if fit:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Transform with handling unseen labels
                        df_encoded[col] = df_encoded[col].apply(
                            lambda x: le.transform([str(x)])[0] 
                            if str(x) in le.classes_ 
                            else -1
                        )
        
        return df_encoded
    
    def scale_features(self, df, fit=True):
        """Scale numerical features"""
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        if fit:
            scaled_features = self.scaler.fit_transform(df[numerical_cols])
        else:
            scaled_features = self.scaler.transform(df[numerical_cols])
        
        # Create new DataFrame with scaled features
        df_scaled = df.copy()
        df_scaled[numerical_cols] = scaled_features
        
        return df_scaled
    
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
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
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
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(artifacts, filepath)
        print(f"Preprocessor saved to: {filepath}")
    
    def load_preprocessor(self, filepath):
        """Load preprocessor artifacts"""
        artifacts = joblib.load(filepath)
        self.label_encoders = artifacts['label_encoders']
        self.scaler = artifacts['scaler']
        self.feature_columns = artifacts['feature_columns']
        print(f"Preprocessor loaded from: {filepath}")

def main():
    """Main preprocessing pipeline"""
    print("="*50)
    print("DATA PREPROCESSING PIPELINE")
    print("="*50)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load raw data
    raw_data_path = "data/raw/telco_churn.csv"
    df_raw = preprocessor.load_data(raw_data_path)
    
    # Clean data
    df_clean = preprocessor.clean_data(df_raw)
    
    # Encode categorical variables
    df_encoded = preprocessor.encode_categorical(df_clean, fit=True)
    
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
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print("="*50)
    print(f"Training data saved: data/processed/train_data.csv")
    print(f"Testing data saved: data/processed/test_data.csv")
    print(f"Preprocessor saved: models/preprocessor.joblib")
    
    return preprocessor, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocessor, X_train, X_test, y_train, y_test = main()