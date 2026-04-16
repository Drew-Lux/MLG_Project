#train_decision_tree_model.py
import numpy as np
import pandas as pd
import joblib
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

# Paths
DATASET_PATH = os.path.join("data_insight", "Diabetes_scaled_for_modeling.csv")
OUTPUT_DIR   = os.path.join("outputs", "models")
MODEL_NAME   = "decision_tree_model.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATASET_PATH)
target_col = "diabetes_stage"

# Encode target
le_target = LabelEncoder()
df["diabetes_stage_enc"] = le_target.fit_transform(df[target_col])

drop_cols = [target_col, "diabetes_stage_enc"]
cat_cols = df.select_dtypes(include="object").columns.difference(drop_cols)
num_cols = df.select_dtypes(include=[np.number]).columns.difference(drop_cols)

df_enc = df.copy()
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    le_dict[col] = le

feature_cols = list(num_cols) + list(cat_cols)
X = df_enc[feature_cols]
y = df["diabetes_stage_enc"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save bundle
model_bundle = {
    "model": model,
    "scaler": scaler,
    "label_encoder": le_target,
    "feature_names": feature_cols,
    "categorical_encoders": le_dict
}
joblib.dump(model_bundle, os.path.join(OUTPUT_DIR, MODEL_NAME))