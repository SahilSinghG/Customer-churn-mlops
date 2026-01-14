import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings("ignore")
import joblib
import os
import json
from datetime import datetime


class ModelTrainer:
    """Trains and tracks multiple ML models for churn prediction"""

    def __init__(self, experiment_name="Customer_Churn_Prediction"):
        self.experiment_name = experiment_name
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0

        # Setup MLflow
        mlflow.set_experiment(experiment_name)

    def load_data(self):
        """Load preprocessed train/test data"""
        print("=" * 60)
        print("LOADING PREPROCESSED DATA")
        print("=" * 60)

        # Load train/test data
        train_data = pd.read_csv("data/processed/train_data.csv")
        test_data = pd.read_csv("data/processed/test_data.csv")

        # Separate features and target
        X_train = train_data.drop("Churn", axis=1)
        y_train = train_data["Churn"]
        X_test = test_data.drop("Churn", axis=1)
        y_test = test_data["Churn"]

        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        print(f"Train churn rate: {y_train.mean():.2%}")
        print(f"Test churn rate: {y_test.mean():.2%}")

        return X_train, X_test, y_train, y_test

    def handle_class_imbalance(self, X_train, y_train):
        """Handle imbalanced dataset using SMOTE"""
        print("\n" + "=" * 60)
        print("HANDLING CLASS IMBALANCE WITH SMOTE")
        print("=" * 60)

        # Before SMOTE
        original_counts = y_train.value_counts()
        print(
            f"Before SMOTE - Class 0: {original_counts[0]}, Class 1: {original_counts[1]}"
        )
        print(f"Imbalance ratio: {original_counts[0]/original_counts[1]:.2f}:1")

        # Apply SMOTE
        smote = SMOTE(random_state=42, sampling_strategy=0.5)  # 1:2 ratio
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # After SMOTE
        balanced_counts = pd.Series(y_train_balanced).value_counts()
        print(
            f"After SMOTE - Class 0: {balanced_counts[0]}, Class 1: {balanced_counts[1]}"
        )
        print(f"New ratio: {balanced_counts[0]/balanced_counts[1]:.2f}:1")

        return X_train_balanced, y_train_balanced

    def define_models(self):
        """Define model configurations to train"""
        models = {
            "Logistic_Regression": {
                "model": LogisticRegression(random_state=42, max_iter=1000),
                "params": {"C": 0.1, "solver": "liblinear"},
            },
            "Random_Forest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5,
                },
            },
            "Gradient_Boosting": {
                "model": GradientBoostingClassifier(random_state=42),
                "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
            },
            "XGBoost": {
                "model": XGBClassifier(random_state=42, eval_metric="logloss"),
                "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
            },
            "SVM": {
                "model": SVC(random_state=42, probability=True),
                "params": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
            },
        }
        return models

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation"""
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": (
                roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
            ),
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Additional metrics
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Business metrics (assuming cost/benefit)
        # Assume: FP cost = $10 (wrong retention offer), FN cost = $100 (lost customer)
        fp_cost, fn_cost = 10, 100
        metrics["expected_cost"] = (fp * fp_cost) + (fn * fn_cost)

        return metrics, cm, classification_report(y_test, y_pred, output_dict=True)

    def train_models(self, X_train, X_test, y_train, y_test, use_smote=True):
        """Train multiple models with MLflow tracking"""
        print("\n" + "=" * 60)
        print("TRAINING MODELS WITH MLFLOW TRACKING")
        print("=" * 60)

        # Handle class imbalance if requested
        if use_smote:
            X_train_processed, y_train_processed = self.handle_class_imbalance(
                X_train, y_train
            )
        else:
            X_train_processed, y_train_processed = X_train.copy(), y_train.copy()

        # Get model definitions
        models_config = self.define_models()

        # Train each model
        for model_name, config in models_config.items():
            print(f"\n{'='*40}")
            print(f"Training: {model_name}")
            print(f"{'='*40}")

            # Start MLflow run
            with mlflow.start_run(run_name=model_name):
                try:
                    # Create model with parameters
                    model = config["model"]
                    model.set_params(**config["params"])

                    # Log parameters
                    mlflow.log_params(config["params"])
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("use_smote", use_smote)

                    # Train model
                    model.fit(X_train_processed, y_train_processed)

                    # Evaluate
                    metrics, cm, clf_report = self.evaluate_model(
                        model, X_test, y_test, model_name
                    )

                    # Log metrics
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)

                    # Log model
                    mlflow.sklearn.log_model(model, f"model_{model_name}")

                    # Log confusion matrix as artifact
                    cm_df = pd.DataFrame(
                        cm,
                        columns=["Predicted No", "Predicted Yes"],
                        index=["Actual No", "Actual Yes"],
                    )

                    # Create temp directory if it doesn't exist
                    os.makedirs("temp", exist_ok=True)

                    cm_path = f"temp/cm_{model_name}.csv"
                    cm_df.to_csv(cm_path)
                    mlflow.log_artifact(cm_path)

                    # Store results
                    self.models[model_name] = model
                    self.results[model_name] = {
                        "metrics": metrics,
                        "confusion_matrix": cm,
                        "classification_report": clf_report,
                    }

                    # Update best model
                    if metrics["f1_score"] > self.best_score:
                        self.best_score = metrics["f1_score"]
                        self.best_model = (model_name, model)

                    # Print results
                    print(f"Accuracy: {metrics['accuracy']:.3f}")
                    print(f"F1-Score: {metrics['f1_score']:.3f}")
                    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
                    print(f"Expected Cost: ${metrics['expected_cost']:.2f}")

                    # Cleanup temp file
                    if os.path.exists(cm_path):
                        os.remove(cm_path)

                except Exception as e:
                    print(f"Error training {model_name}: {str(e)}")
                    continue

        return self.models, self.results

    def save_best_model(self):
        """Save the best performing model"""
        if self.best_model:
            model_name, model = self.best_model

            print("\n" + "=" * 60)
            print(f"SAVING BEST MODEL: {model_name}")
            print("=" * 60)

            # Create models directory
            os.makedirs("models", exist_ok=True)

            # Save model
            model_path = f"models/best_model.joblib"
            joblib.dump(model, model_path)
            print(f"Model saved: {model_path}")

            # Save model metadata
            # Convert numpy types to Python native types for JSON serialization
            metrics = self.results[model_name]["metrics"]
            # Convert numpy types to Python types
            metrics_serializable = {}
            for key, value in metrics.items():
                if hasattr(value, "item"):  # Check if it's a numpy type
                    metrics_serializable[key] = value.item()
                else:
                    metrics_serializable[key] = value

            metadata = {
                "model_name": model_name,
                "training_date": datetime.now().isoformat(),
                "metrics": metrics_serializable,
                "features_count": (
                    int(model.n_features_in_)
                    if hasattr(model, "n_features_in_")
                    else "unknown"
                ),
            }

            metadata_path = "models/model_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"Metadata saved: {metadata_path}")

            return model_path
        else:
            print("No model trained successfully!")
            return None

    def compare_models(self):
        """Create comparison of all trained models"""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        if not self.results:
            print("No results to compare!")
            return

        # Create comparison DataFrame
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result["metrics"]
            row = {
                "Model": model_name,
                "Accuracy": f"{metrics['accuracy']:.3f}",
                "Precision": f"{metrics['precision']:.3f}",
                "Recall": f"{metrics['recall']:.3f}",
                "F1-Score": f"{metrics['f1_score']:.3f}",
                "ROC-AUC": f"{metrics['roc_auc']:.3f}",
                "Expected Cost": f"${metrics['expected_cost']:.2f}",
            }
            comparison_data.append(row)

        # Create and display DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        print("\nPerformance Comparison:")
        print(comparison_df.to_string(index=False))

        # Save comparison
        comparison_df.to_csv("models/model_comparison.csv", index=False)
        print(f"\nComparison saved: models/model_comparison.csv")

        return comparison_df


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("CUSTOMER CHURN MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Initialize trainer
    trainer = ModelTrainer()

    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data()

    # Train models
    models, results = trainer.train_models(
        X_train, X_test, y_train, y_test, use_smote=True
    )

    # Compare models
    comparison_df = trainer.compare_models()

    # Save best model
    best_model_path = trainer.save_best_model()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    # Instructions for MLflow UI
    print("\nTo view experiment results:")
    print("1. Open new terminal")
    print("2. Run: mlflow ui --backend-store-uri sqlite:///mlruns.db")
    print("3. Open browser: http://localhost:5000")

    return trainer, comparison_df


if __name__ == "__main__":
    trainer, comparison_df = main()
