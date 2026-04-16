#app.py
import os
import joblib
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# ─── DYNAMIC PATH LOGIC ──────────────────────────────────────────────────────
# This ensures app.py finds the data/outputs folders even when inside the 'app' subfolder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data_insight", "Diabetes_scaled_for_modeling.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "clustering")

# ─── CLASSIFICATION MODEL LOADING ────────────────────────────────────────────
CLASS_MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "xgboost_model.pkl")

clf_bundle = joblib.load(CLASS_MODEL_PATH) if os.path.exists(CLASS_MODEL_PATH) else None
clf_model = clf_bundle["model"] if clf_bundle else None
clf_scaler = clf_bundle["scaler"] if clf_bundle else None
clf_label_encoder = clf_bundle["label_encoder"] if clf_bundle else None
clf_features = clf_bundle["feature_names"] if clf_bundle else None

# ─── BRANDING & COLORS ───────────────────────────────────────────────────────
COLORS = {
    "primary": "#2A9D8F",   # Medical Mint
    "secondary": "#264653",  # Deep Slate
    "danger": "#E63946",    # Alert Red
    "warning": "#F4A261",   # Amber
    "bg": "#F8F9FA"         # Light Grey
}

# ─── DATA & ASSET LOADING ────────────────────────────────────────────────────


def load_assets():
    try:
        df = pd.read_csv(DATASET_PATH)
        km_path = os.path.join(OUTPUT_DIR, "kmeans_pipeline.pkl")
        shap_path = os.path.join(OUTPUT_DIR, "shap_cluster_importance.csv")
        recs_path = os.path.join(OUTPUT_DIR, "actionable_recommendations.csv")

        km = joblib.load(km_path) if os.path.exists(km_path) else None
        shap = pd.read_csv(shap_path) if os.path.exists(shap_path) else None
        recs = pd.read_csv(recs_path) if os.path.exists(recs_path) else None
        return df, km, shap, recs
    except Exception as e:
        print(f"Pathing Error: {e}")
        return pd.DataFrame(), None, None, None


df, km_bundle, shap_df, recs_df = load_assets()

# ─── UI HELPERS ──────────────────────────────────────────────────────────────


def info_card(title, value, subtitle, color):
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="text-muted text-uppercase small"),
            html.H2(value, style={"color": color}, className="fw-bold mb-0"),
            html.P(subtitle, className="text-muted small mb-0")
        ])
    ], className="shadow-sm border-0 mb-3")

# ─── APP START ───────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

app.layout = dbc.Container([
    # Professional Medical Header
    dbc.Row([
        dbc.Col(html.Div([
            html.H1("BC ANALYTICS | PRECISION RISK ENGINE",
                    className="fw-bold text-white mb-0", style={"letter-spacing": "1.5px"}),
            html.P("AI-Driven Clinical Risk Stratification & Population Insights",
                   className="text-white-50 mb-0")
        ], className="bg-primary p-4 rounded-bottom shadow mb-4"), width=12)
    ]),

    dbc.Tabs([
        # TAB 1: Patient Assessment
        dbc.Tab(label="PATIENT ASSESSMENT", children=[
            dbc.Row([
    # Left column: grouped input cards
    dbc.Col([
        html.H5("Clinical Entry", className="mt-4 fw-bold text-secondary"),

        # Demographics
        dbc.Card([
            dbc.CardHeader("Demographics"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.Label("Age", className="fw-bold"), width="auto"),
                    dbc.Col(html.Span("🛈", id="info-age", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                ]),
                dbc.Input(id="input-age", type="number", placeholder="Enter Age", className="mb-3"),
                dbc.Tooltip("Age is a key risk factor. No tools needed — just your date of birth.", target="info-age", placement="right"),
            ])
        ], className="mb-3"),

        # Vitals
        dbc.Card([
            dbc.CardHeader("Vitals"),
            dbc.CardBody([
                # BMI
                dbc.Row([
                    dbc.Col(html.Label("BMI (Body Mass Index)", className="fw-bold"), width="auto"),
                    dbc.Col(html.Span("🛈", id="info-bmi", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                ]),
                dbc.Input(id="input-bmi", type="number", placeholder="e.g. 28.5", className="mb-3"),
                dbc.Tooltip("BMI measures body fat based on height and weight.", target="info-bmi", placement="right"),

                # Systolic BP
                dbc.Row([
                    dbc.Col(html.Label("Systolic Blood Pressure (mmHg)", className="fw-bold"), width="auto"),
                    dbc.Col(html.Span("🛈", id="info-sbp", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                ]),
                dbc.Input(id="input-sbp", type="number", placeholder="e.g. 120", className="mb-3"),
                dbc.Tooltip("Systolic BP is the pressure when the heart beats. Normal is 90–120 mmHg.", target="info-sbp", placement="right"),

                # Diastolic BP
                dbc.Row([
                    dbc.Col(html.Label("Diastolic Blood Pressure (mmHg)", className="fw-bold"), width="auto"),
                    dbc.Col(html.Span("🛈", id="info-dbp", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                ]),
                dbc.Input(id="input-dbp", type="number", placeholder="e.g. 80", className="mb-3"),
                dbc.Tooltip("Diastolic BP is the pressure when the heart rests between beats. Normal is 60–80 mmHg.", target="info-dbp", placement="right"),

                # Heart Rate
                dbc.Row([
                    dbc.Col(html.Label("Heart Rate (beats/min)", className="fw-bold"), width="auto"),
                    dbc.Col(html.Span("🛈", id="info-hr", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                ]),
                dbc.Input(id="input-hr", type="number", placeholder="e.g. 72", className="mb-3"),
                dbc.Tooltip("Normal resting heart rate: 60–100 bpm.", target="info-hr", placement="right"),
            ])
        ], className="mb-3"),

        # Lab Results
        dbc.Card([
            dbc.CardHeader("Lab Results"),
            dbc.CardBody([
                # Glucose
                dbc.Row([
                    dbc.Col(html.Label("Fasting Glucose (mg/dL)", className="fw-bold"), width="auto"),
                    dbc.Col(html.Span("🛈", id="info-glucose", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                ]),
                dbc.Input(id="input-glucose", type="number", placeholder="e.g. 95", className="mb-3"),
                dbc.Tooltip("Fasting glucose normal range: 70–99 mg/dL.", target="info-glucose", placement="right"),

                # HbA1c
                dbc.Row([
                    dbc.Col(html.Label("HbA1c (%)", className="fw-bold"), width="auto"),
                    dbc.Col(html.Span("🛈", id="info-hba1c", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                ]),
                dbc.Input(id="input-hba1c", type="number", placeholder="e.g. 5.6", className="mb-3"),
                dbc.Tooltip("HbA1c reflects 2–3 month average glucose.", target="info-hba1c", placement="right"),

                # Cholesterol
                dbc.Row([
                    dbc.Col(html.Label("Total Cholesterol (mg/dL)", className="fw-bold"), width="auto"),
                    dbc.Col(html.Span("🛈", id="info-chol", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                ]),
                dbc.Input(id="input-chol", type="number", placeholder="e.g. 180", className="mb-3"),
                dbc.Tooltip("Normal total cholesterol <200 mg/dL.", target="info-chol", placement="right"),

                # Triglycerides
                dbc.Row([
                    dbc.Col(html.Label("Triglycerides (mg/dL)", className="fw-bold"), width="auto"),
                    dbc.Col(html.Span("🛈", id="info-trig", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                ]),
                dbc.Input(id="input-trig", type="number", placeholder="e.g. 140", className="mb-3"),
                dbc.Tooltip("Normal triglycerides <150 mg/dL.", target="info-trig", placement="right"),
            ])
        ], className="mb-3"),

        # Lifestyle
        dbc.Card([
            dbc.CardHeader("Lifestyle"),
            dbc.CardBody([
                # Activity
                dbc.Row([
                    dbc.Col(html.Label("Physical Activity (Minutes/Week)", className="fw-bold"), width="auto"),
                    dbc.Col(html.Span("🛈", id="info-act", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                ]),
                dbc.Input(id="input-act", type="number", placeholder="e.g. 150", className="mb-3"),
                dbc.Tooltip("Weekly minutes of moderate exercise. ≥150 minutes is recommended.", target="info-act", placement="right"),

                # Diet
                dbc.Row([
                    dbc.Col(html.Label("Diet Quality Score (1–10)", className="fw-bold"), width="auto"),
                    dbc.Col(html.Span("🛈", id="info-diet", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                ]),
                dcc.Slider(1, 10, 1, value=5, id="input-diet", marks={1: 'Poor', 5: 'Avg', 10: 'Great'}),
                dbc.Tooltip("Self‑rated diet quality. Higher scores indicate healthier eating habits.", target="info-diet", placement="right"),

                # Sleep
                dbc.Row([
                    dbc.Col(html.Label("Sleep Hours per Day", className="fw-bold"), width="auto"),
                    dbc.Col(html.Span("🛈", id="info-sleep", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                ]),
                dbc.Input(id="input-sleep", type="number", placeholder="e.g. 7", className="mb-3"),
                dbc.Tooltip("Average nightly sleep duration. Adults typically need 7–9 hours.", target="info-sleep", placement="right"),

                # Stress
                dbc.Row([
                    dbc.Col(html.Label("Stress Score (1–10)", className="fw-bold"), width="auto"),
                    dbc.Col(html.Span("🛈", id="info-stress", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                ]),
                dcc.Slider(1, 10, 1, value=5, id="input-stress", marks={1: 'Low', 5: 'Moderate', 10: 'High'}),
                dbc.Tooltip("Self-rated stress. 1–3 low, 4–6 moderate, 7–10 high.", target="info-stress", placement="right"),

                 # Smoking
            dbc.Row([
                dbc.Col(html.Label("Smoking Status", className="fw-bold"), width="auto"),
                dbc.Col(html.Span("🛈", id="info-smoking", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
            ]),
            dcc.Dropdown(id="input-smoking", options=[{"label": "Non-Smoker", "value": 0}, {"label": "Smoker", "value": 1}], placeholder="Select status", className="mb-3"),
            dbc.Tooltip("Smoking increases cardiovascular and diabetes risk. Select your current status.", 
                        target="info-smoking", placement="right"),

            # Alcohol
            dbc.Row([
                dbc.Col(html.Label("Alcohol Use", className="fw-bold"), width="auto"),
                dbc.Col(html.Span("🛈", id="info-alcohol", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
            ]),
            dcc.Dropdown(
                id="input-alcohol",
                options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                placeholder="Select",
                className="mb-3"
            ),
            dbc.Tooltip("Alcohol use affects metabolic health. Select if you currently consume alcohol.", 
                        target="info-alcohol", placement="right"),
        ])
    ], className="mb-3"),

    # Generate Button
    dbc.Button(
        "GENERATE DIAGNOSTIC",
        id="predict-btn",
        color="primary",
        className="mt-4 w-100 fw-bold shadow-sm"
    )
], width=4),

                dbc.Col([
                    html.H5("Diagnostic Intelligence Profile",
                            className="mt-4 fw-bold text-secondary"),
                    html.Div(id="prediction-output", children=[
                        dbc.Alert("Analysis Ready: Enter patient clinical data to generate profile.",
                                  color="light", className="text-center border shadow-sm")
                    ])
                ], width=8)
            ])
        ]),

        # TAB 2: Population Insights
        dbc.Tab(label="LIFESTYLE CLUSTERING", children=[
            dbc.Row([
                dbc.Col([
                    html.H5("Population Similarity Map",
                            className="mt-4 fw-bold text-secondary"),
                    html.P("Groups patients by lifestyle habits. The Star represents your current patient.",
                           className="text-muted small"),
                    dcc.Graph(id='cluster-graph', style={"height": "500px"})
                ], width=7),
                dbc.Col([
                    html.H5("Primary Risk Drivers",
                            className="mt-4 fw-bold text-secondary"),
                    html.P("Top 10 features influencing global risk scores.",
                           className="text-muted small"),
                    dcc.Graph(id='shap-graph', style={"height": "500px"})
                ], width=5)
            ])
        ])
    ]),

    # SYSTEM STATUS FOOTER
    dbc.Row([
        dbc.Col(html.Hr(), width=12),
        dbc.Col(html.P(f"System Status: {'✅ Model Connected' if km_bundle else '❌ Model File Missing - Check Path'}",
                       className=f"text-center small {'text-success' if km_bundle else 'text-danger'}"), width=12)
    ])
], fluid=True, style={"backgroundColor": COLORS["bg"], "minHeight": "100vh"})

# ─── DASHBOARD CALLBACK ──────────────────────────────────────────────────────


@app.callback(
    [Output("prediction-output", "children"),
     Output("cluster-graph", "figure"),
     Output("shap-graph", "figure")],
    Input("predict-btn", "n_clicks"),
    [
        State("input-age", "value"),
        State("input-bmi", "value"),
        State("input-act", "value"),
        State("input-diet", "value"),
        State("input-sbp", "value"),
        State("input-dbp", "value"),
        State("input-glucose", "value"),
        State("input-hba1c", "value"),
        State("input-chol", "value"),
        State("input-trig", "value"),
        State("input-hr", "value"),
        State("input-sleep", "value"),
        State("input-stress", "value"),
        State("input-smoking", "value"),
        State("input-alcohol", "value")
    ],
    prevent_initial_call=False
)

def update_dashboard(n_clicks, age, bmi, act, diet, sbp, dbp, glucose, hba1c, chol, trig, hr, sleep, stress, smoking, alcohol):
    # Default Visuals
    fig_cluster = go.Figure().update_layout(title="Awaiting Model Assets...", template="plotly_white")
    fig_shap = go.Figure().update_layout(title="Awaiting SHAP Data...", template="plotly_white")

    # 1. VISUALIZATION LOGIC
    if not df.empty and km_bundle:
        scaler_features = km_bundle["scaler"].feature_names_in_
        df_enc = df.copy()
        for col in df_enc.select_dtypes(include=['object']).columns:
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

        # PCA Projection
        X_scaled = km_bundle["scaler"].transform(df_enc.reindex(columns=scaler_features, fill_value=0))
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)

        fig_cluster = px.scatter(
            x=coords[:, 0], y=coords[:, 1],
            color=df.get("diabetes_stage", "Population"),
            labels={'x': 'Lifestyle Metric (PC1)', 'y': 'Clinical Metric (PC2)'},
            template="plotly_white", opacity=0.4, title="Population Lifestyle Map"
        )

        # Highlight User
        if age and bmi:
            patient_row = pd.DataFrame([{col: 0 for col in scaler_features}])
            patient_row['Age'] = age or 30
            patient_row['bmi'] = bmi or 28.5
            patient_row['physical_activity_minutes_per_week'] = act or 150
            patient_row['diet_score'] = diet or 5
            patient_row['systolic_bp'] = sbp or 120
            patient_row['diastolic_bp'] = dbp or 80
            patient_row['fasting_glucose'] = glucose or 95
            patient_row['hba1c'] = hba1c or 5.6
            patient_row['cholesterol_total'] = chol or 180
            patient_row['triglycerides'] = trig or 140
            patient_row['heart_rate'] = hr or 72
            patient_row['sleep_hours_per_day'] = sleep or 7
            patient_row['stress_score'] = stress or 5
            patient_row['smoking_status'] = smoking or 0
            patient_row['alcohol_use'] = alcohol or 0
            p_scaled = km_bundle["scaler"].transform(patient_row.reindex(columns=scaler_features, fill_value=0))
            p_pca = pca.transform(p_scaled)
            fig_cluster.add_trace(go.Scatter(
                x=[p_pca[0, 0]], y=[p_pca[0, 1]], mode="markers",
                marker=dict(size=22, color="black", symbol="star", line=dict(width=2, color="white")),
                name="Current Patient"
            ))

    if shap_df is not None:
        fig_shap = px.bar(shap_df.head(10)[::-1], x="mean_|SHAP|", y="feature", orientation="h",
                          title="Top Risk Drivers", color_discrete_sequence=[COLORS["primary"]])

    # 2. PREDICTION LOGIC
    if n_clicks is None:
        return dash.no_update, fig_cluster, fig_shap

    if clf_model and clf_scaler and clf_features:
        # Build patient row with all required features
        patient_row = pd.DataFrame([{col: 0 for col in clf_features}])
        patient_row['Age'] = age or 30
        patient_row['bmi'] = bmi or 28.5
        patient_row['physical_activity_minutes_per_week'] = act or 150
        patient_row['diet_score'] = diet or 5
        patient_row['systolic_bp'] = sbp or 120
        patient_row['diastolic_bp'] = dbp or 80
        patient_row['fasting_glucose'] = glucose or 95
        patient_row['hba1c'] = hba1c or 5.6
        patient_row['cholesterol_total'] = chol or 180
        patient_row['triglycerides'] = trig or 140
        patient_row['heart_rate'] = hr or 72
        patient_row['sleep_hours_per_day'] = sleep or 7
        patient_row['stress_score'] = stress or 5
        patient_row['smoking_status'] = smoking or 0
        patient_row['alcohol_use'] = alcohol or 0

        # Scale features safely
        p_scaled = clf_scaler.transform(patient_row.reindex(columns=clf_features, fill_value=0))

        # Predict risk class
        pred_class = clf_model.predict(p_scaled)[0]

        # Map to meaningful labels
        stage_map = {"0": "Stable", "1": "Moderate Risk", "2": "Elevated Risk"}
        level = stage_map.get(str(pred_class), "Unknown")

        # Confidence (probability)
        prob = clf_model.predict_proba(p_scaled).max()
        risk_score = int(prob * 100)
    else:
        level = "UNKNOWN"
        risk_score = 0
        
    res_color = COLORS["danger"] if level == "ELEVATED" else COLORS["warning"] if level == "MODERATE" else COLORS["primary"]

    result_content = html.Div([
        dbc.Row([
            dbc.Col(info_card("Risk Level", level, "Clinical Classification", res_color)),
            dbc.Col(info_card("Model Confidence", f"{int(risk_score)}%", "Risk Probability", "#264653")),
            dbc.Col(info_card("Lifestyle Segment", "Segment B", "Patient Grouping", COLORS["secondary"])),
        ]),
        dbc.Alert([
            html.H5("Intervention Recommended", className="alert-heading fw-bold"),
            html.P(f"Patient falls within the {level.lower()} risk bracket. Recommended actions include metabolic testing and lifestyle counseling.")
        ], color="light", className="border-start border-primary border-4 shadow-sm mt-3")
    ])

    # 3. Recommendations Section
    if recs_df is not None and not recs_df.empty:
        cluster_recs = recs_df[recs_df["cluster_name"].str.contains(level, case=False)]
        rec_cards = [
            dbc.Alert(r, color="light", className="border-start border-primary border-4 shadow-sm mt-2")
            for r in cluster_recs["recommendation"].head(3)
        ]
        result_content.children.append(html.Div([
            html.H5("Actionable Recommendations", className="fw-bold text-secondary mt-4"),
            html.Div(rec_cards)
        ]))

    return result_content, fig_cluster, fig_shap

if __name__ == '__main__':
    app.run(debug=True)
