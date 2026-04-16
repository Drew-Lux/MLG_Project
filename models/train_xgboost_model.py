import numpy as np
import pandas as pd
import joblib
import os

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

#Paths for accessign dataset, and knowing where to store model file
DATASET_PATH = os.path.join("data_insight", "Diabetes_scaled_for_modeling.csv")
OUTPUT_DIR   = os.path.join("outputs", "models")
MODEL_NAME   = "xgboost_model.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

#Store dataset into a datafram used for modeling
df = pd.read_csv(DATASET_PATH)

#Set target column that needs to be predicted
target_col = "diabetes_stage"

#Target encoding that matches pipeline
le_target = LabelEncoder()
df["diabetes_stage_enc"] = le_target.fit_transform(df[target_col])

#Dropped columns to make predictions
drop_cols = [target_col, "diabetes_stage_enc"]

cat_cols = df.select_dtypes(include="object").columns.difference(drop_cols)
num_cols = df.select_dtypes(include=[np.number]).columns.difference(drop_cols)

df_enc = df.copy()

# Encode categorical features
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    le_dict[col] = le

feature_cols = list(num_cols) + list(cat_cols)

#Setup training data
X = df_enc[feature_cols]
y = df["diabetes_stage_enc"]


#Train set data and split for modeling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Scaling in order to match pipeline structure
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#Base model structure
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)

#Use function to train model
model.fit(X_train_scaled, y_train)

#Evaluation of model
y_pred = model.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

#Save model for use in dash app matching pipeline in Unsupervised_interpretability file
model_bundle = {
    "model": model,                 
    "scaler": scaler,               
    "label_encoder": le_target,     
    "feature_names": feature_cols,  
    "categorical_encoders": le_dict 
}

#Set path for where to save .pkl file
save_path = os.path.join(OUTPUT_DIR, MODEL_NAME)

#Save model as set bundle in path established
joblib.dump(model_bundle, save_path)






