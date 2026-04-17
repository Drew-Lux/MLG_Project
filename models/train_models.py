import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

#Setup of paths for access of dataset and storage of .pkl files
DATASET_PATH = os.path.join("data_insight", "Diabetes_scaled_for_modeling.csv")
MODELS_DIR   = os.path.join("outputs", "models")
METRICS_DIR  = os.path.join("outputs", "metrics")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

#Load dataset into dataframe 
df = pd.read_csv(DATASET_PATH)
print(f"\nDataset shape: {df.shape}")

target_col = "diabetes_stage"

#Leage columns that when included will make the model follow hard-coded rules, not learning
leakage_features = [
    "hba1c",
    "diabetes_risk_score",
    "diagnosed_diabetes"
]

df = df.drop(columns=[col for col in leakage_features if col in df.columns])

print(f"\nRemoved leakage features: {leakage_features}")

#Target encoding
le_target = LabelEncoder()
df["diabetes_stage_enc"] = le_target.fit_transform(df[target_col])

print(f"Target classes: {dict(enumerate(le_target.classes_))}")

#Ensures that clean features are used for most efficient learning
feature_cols = [
    "bmi",
    "systolic_bp",
    "physical_activity_minutes_per_week",
    "diet_score",
    "Age"
]

# Ensure all exist
feature_cols = [col for col in feature_cols if col in df.columns]

X = df[feature_cols]
y = df["diabetes_stage_enc"]

print(f"\nFinal features used: {feature_cols}")

#Initiating a train/split test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Scaling to ensure functionality across different environments
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#Structure of each model that needs to be trained
models = {
    "xgboost_model.pkl": XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    ),

    "random_forest_model.pkl": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),

    "decision_tree_model.pkl": DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    )
}

#Training and evaluation of each model 
metrics_results = []

for filename, model in models.items():

    print(f"\nTraining: {filename}")

    # Train
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Save classification report
    report = classification_report(y_test, y_pred, zero_division=0)

    report_path = os.path.join(
        METRICS_DIR, filename.replace(".pkl", "_report.txt")
    )

    with open(report_path, "w") as f:
        f.write(report)

    # Store metrics
    metrics_results.append({
        "model": filename.replace(".pkl", ""),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })

    # Save model bundle
    model_bundle = {
        "model": model,
        "scaler": scaler,
        "label_encoder": le_target,
        "feature_names": feature_cols
    }

    model_path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model_bundle, model_path)

    print(f"Model saved → {model_path}")
    print(f"Report saved → {report_path}")

#Saves the metrics of each model's performance into a csv
metrics_df = pd.DataFrame(metrics_results)
metrics_df["rank"] = metrics_df["accuracy"].rank(ascending=False)

csv_path = os.path.join(METRICS_DIR, "model_comparison.csv")
metrics_df.to_csv(csv_path, index=False)

print(f"\n Metrics saved → {csv_path}")

#Showcases which model performed the best with dataset based of accuracy
best_model = metrics_df.sort_values("accuracy", ascending=False).iloc[0]

print("\n")
print(f"BEST MODEL: {best_model['model']}")
print(f"Accuracy: {best_model['accuracy']:.4f}")
print("\n")