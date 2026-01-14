import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

print("=" * 60)
print("SIMPLE MODEL EVALUATION")
print("=" * 60)

# 1. Load the model
print("1. Loading model...")
model = joblib.load("models/best_model.joblib")
print(f"   Model loaded: {type(model).__name__}")

# 2. Load test data
print("2. Loading test data...")
test_data = pd.read_csv("data/processed/test_data.csv")
X_test = test_data.drop("Churn", axis=1)
y_test = test_data["Churn"]
print(f"   Test data: {X_test.shape}")

# 3. Make predictions
print("3. Making predictions...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 4. Create single chart (simplified)
print("4. Creating visualization...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Chart 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
axes[0, 0].set_title("Confusion Matrix")
axes[0, 0].set_ylabel("Actual")
axes[0, 0].set_xlabel("Predicted")

# Chart 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
axes[0, 1].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
axes[0, 1].plot([0, 1], [0, 1], "k--")
axes[0, 1].set_title("ROC Curve")
axes[0, 1].legend()

# Chart 3: Simple metrics
metrics = {
    "Accuracy": f"{(y_pred == y_test).mean():.1%}",
    "Churn Rate": f"{y_test.mean():.1%}",
    "Model": type(model).__name__,
}
axes[1, 0].axis("off")
axes[1, 0].text(
    0.1,
    0.5,
    "\n".join([f"{k}: {v}" for k, v in metrics.items()]),
    fontsize=12,
    verticalalignment="center",
)

# Chart 4: Save instruction
axes[1, 1].axis("off")
axes[1, 1].text(
    0.1,
    0.5,
    "File saved as:\nmodels/evaluation.png\n\nClose this window to continue to Day 3!",
    fontsize=12,
    verticalalignment="center",
)

plt.tight_layout()
plt.savefig("models/evaluation.png", dpi=150)
plt.show()

print("5. Visualization saved: models/evaluation.png")

# 5. Print simple results
print("\n" + "=" * 60)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 60)
print(f"Model: {type(model).__name__}")
print(f"Accuracy: {(y_pred == y_test).mean():.1%}")
print(f"Churn customers correctly identified: {cm[1,1]} out of {cm[1,0]+cm[1,1]}")
print(f"ROC-AUC Score: {roc_auc:.3f}")

print("\n" + "=" * 60)
print("READY FOR DAY 3: MODEL SERVING!")
print("=" * 60)
