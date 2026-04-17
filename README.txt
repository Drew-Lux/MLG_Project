Insulin Insight Portal – Diabetes Risk & Lifestyle Clustering
---------------------------------------------------------------------------------
Project Overview
---------------------------------------------------------------------------------
This project develops a decision support system for diabetes care. It combines classification models to identify diabetes risk stages with unsupervised clustering to group patients by lifestyle factors. The system integrates interpretability (via SHAP) and generates actionable recommendations for healthcare providers.

Key goals:

 - Classify patients into diabetes risk categories.

 - Identify lifestyle and behavioural drivers of risk.

 - Group patients into lifestyle clusters for tailored interventions.

 - Provide interpretable outputs and recommendations for clinicians.

---------------------------------------------------------------------------------
Repository Structure
---------------------------------------------------------------------------------
Code
MLG_Project/
│
├── data_insight/                  # Input dataset(s)
│   └── Diabetes_scaled_for_modeling.csv
│
├── outputs/
│   ├── clustering/                # Clustering outputs
│   │   ├── kmeans_validation.png
│   │   ├── cluster_profile_heatmap.png
│   │   ├── cluster_pca.png
│   │   ├── shap_classification_bar.png
│   │   ├── shap_classification_beeswarm.png
│   │   ├── shap_cluster_bar.png
│   │   ├── shap_cluster_0_beeswarm.png
│   │   ├── shap_cluster_1_beeswarm.png
│   │   ├── shap_cluster_2_beeswarm.png
│   │   ├── shap_classification_importance.csv
│   │   ├── shap_cluster_importance.csv
│   │   ├── actionable_recommendations.csv
│   │   ├── cluster_profiles.csv
│   │   ├── cluster_stage_distribution.csv
│   │   └── patient_cluster_assignments.csv
│   └── models/                    # Saved models
│       ├── decision_tree_model.pkl
│       └── xgboost_model.pkl
│
├── train_models.py                 # Script to train classification models
├── Unsupervised_interpretability.py # Script for clustering, SHAP, recommendations
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation

---------------------------------------------------------------------------------
Data Summary
---------------------------------------------------------------------------------
 - Dataset size: 39,692 rows, 9 columns (scaled subset for modelling).
 - Features used for clustering: HbA1c, BMI, systolic blood pressure, physical activity minutes per week, diet score, age.
 - Target column: diabetes_stage (encoded as 0.0 = No Diabetes, 2.0 = Type 2 Diabetes).
 - Leakage features excluded: diabetes_risk_score, diagnosed_diabetes.
 - Preparation: Null checks, duplicate removal, categorical encoding, scaling with StandardScaler.

---------------------------------------------------------------------------------
Requirements
---------------------------------------------------------------------------------
Install dependencies with:

bash
pip install -r requirements.txt

Key packages:
 - numpy, pandas, scikit-learn
 - matplotlib, seaborn
 - shap
 - joblib

---------------------------------------------------------------------------------
How to Run
---------------------------------------------------------------------------------
1. Train Classification Models
Run:
bash
python train_models.py
This trains Decision Tree, Random Forest, and XGBoost models, evaluates them, and saves the chosen model (decision_tree_model.pkl) into outputs/models/.

2. Run Clustering + Interpretability
Run:
bash
python Unsupervised_interpretability.py
This script: 
 - Loads the dataset and preprocesses features.
 - Performs K‑Means clustering (k=3).
 - Profiles clusters and generates heatmaps.
 - Applies PCA for visualization.
 - Loads the trained classification model and computes SHAP values.
 - Trains a surrogate model for cluster interpretability.
 - Generates actionable recommendations per cluster.
 - Saves all plots and CSV outputs into outputs/clustering/.

---------------------------------------------------------------------------------
Evaluation
---------------------------------------------------------------------------------
Classification (Decision Tree)
 - Accuracy: 88.1%
 - Precision: 83.9%
 - Recall: 88.1%
 - F1-score: 82.9%

Class-specific performance:
 - No Diabetes → Precision: 0.52, Recall: 0.02, F1: 0.03
 - Diabetes → Precision: 0.88, Recall: 1.00, F1: 0.94

Interpretation: The model excels at identifying diabetic patients, with lower sensitivity for non-diabetic cases. This trade-off is acceptable for early risk identification.

---------------------------------------------------------------------------------
Results
---------------------------------------------------------------------------------
Classification: Decision Tree chosen for interpretability. Accuracy ~88%, strong recall for diabetic patients.
Clustering: 
Three lifestyle clusters identified:
 - Low Risk Lifestyle
 - Moderate Risk Lifestyle
 - High Risk Lifestyle

Interpretability: SHAP values highlight age, physical activity, BMI, diet score, and blood pressure as key drivers.

Recommendations: CSV outputs provide tailored lifestyle guidance per cluster.

---------------------------------------------------------------------------------
Deployment
---------------------------------------------------------------------------------
Model artifacts: decision_tree_model.pkl, kmeans_pipeline.pkl.

REST API integration planned for EHR systems.

Monitoring schedule: weekly performance checks, monthly drift detection, quarterly retraining.

---------------------------------------------------------------------------------
Links
---------------------------------------------------------------------------------
Github: https://github.com/Drew-Lux/MLG_Project

Web App: https://insulin-insight-portal.onrender.com