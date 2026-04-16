#compare_models.py
import joblib
import os
from sklearn.metrics import classification_report
import pandas as pd

MODELS_DIR = os.path.join("outputs", "models")
DATASET_PATH = os.path.join("data_insight", "Diabetes_scaled_for_modeling.csv")

# Load dataset
df = pd.read_csv(DATASET_PATH)
target_col = "diabetes_stage"
X = df.drop(columns=[target_col])
y = df[target_col]

# Load and evaluate each model
model_files = [
    "decision_tree_model.pkl",
    "random_forest_model.pkl",
    "xgboost_model.pkl"
]

results = []
for file in model_files:
    path = os.path.join(MODELS_DIR, file)
    bundle = joblib.load(path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]
    feature_names = bundle["feature_names"]

    # Prepare data
    X_scaled = scaler.transform(df[feature_names])
    y_enc = label_encoder.transform(y)

    y_pred = model.predict(X_scaled)
    report = classification_report(y_enc, y_pred, output_dict=True, zero_division=0)

    results.append({
        "Model": file.replace("_model.pkl", "").title(),
        "Accuracy": report["accuracy"],
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1": report["weighted avg"]["f1-score"]
    })

# Display comparison
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"), index=False)
