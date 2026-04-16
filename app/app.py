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
DATASET_PATH = os.path.join(
    BASE_DIR, "data", "Diabetes_and_LifeStyle_Dataset_.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "clustering")

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

        km = joblib.load(km_path) if os.path.exists(km_path) else None
        shap = pd.read_csv(shap_path) if os.path.exists(shap_path) else None
        return df, km, shap
    except Exception as e:
        print(f"Pathing Error: {e}")
        return pd.DataFrame(), None, None


df, km_bundle, shap_df = load_assets()

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
                dbc.Col([
                    html.H5("Clinical Entry",
                            className="mt-4 fw-bold text-secondary"),
                    dbc.Card([
                        dbc.CardBody([
                            # Age
                            dbc.Row([
                                dbc.Col(html.Label("Age", className="fw-bold"), width="auto"),
                                dbc.Col(html.Span("🛈", id="info-age", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                            ]),
                            dbc.Input(id="input-age", type="number", placeholder="Enter Age", className="mb-3"),
                            dbc.Tooltip("Age is a key risk factor for diabetes and cardiovascular disease.", target="info-age", placement="right"),

                            # BMI
                            dbc.Row([
                                dbc.Col(html.Label("BMI (Body Mass Index)", className="fw-bold"), width="auto"),
                                dbc.Col(html.Span("🛈", id="info-bmi", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                            ]),
                            dbc.Input(id="input-bmi", type="number", placeholder="e.g. 28.5", className="mb-3"),
                            dbc.Tooltip("BMI measures body fat based on height and weight. Normal range is 18.5–24.9.", target="info-bmi", placement="right"),

                            # Physical Activity
                            dbc.Row([
                                dbc.Col(html.Label("Physical Activity (Minutes/Week)", className="fw-bold"), width="auto"),
                                dbc.Col(html.Span("🛈", id="info-act", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                            ]),
                            dbc.Input(id="input-act", type="number", placeholder="e.g. 150", className="mb-3"),
                            dbc.Tooltip("Weekly minutes of moderate exercise. ≥150 minutes is recommended.", target="info-act", placement="right"),

                            # Diet Quality
                            dbc.Row([
                                dbc.Col(html.Label("Diet Quality Score (1–10)", className="fw-bold"), width="auto"),
                                dbc.Col(html.Span("🛈", id="info-diet", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                            ]),
                            dcc.Slider(1, 10, 1, value=5, id="input-diet", marks={1: 'Poor', 5: 'Avg', 10: 'Great'}),
                            dbc.Tooltip("Self‑rated diet quality. Higher scores indicate healthier eating habits.", target="info-diet", placement="right"),

                            # Systolic BP
                            dbc.Row([
                                dbc.Col(html.Label("Systolic Blood Pressure (mmHg)", className="fw-bold"), width="auto"),
                                dbc.Col(html.Span("🛈", id="info-sbp", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                            ]),
                            dbc.Input(id="input-sbp", type="number", placeholder="e.g. 120", className="mb-3"),
                            dbc.Tooltip("Systolic BP is the pressure when the heart beats. Normal is ~120 mmHg.", target="info-sbp", placement="right"),

                            # Diastolic BP
                            dbc.Row([
                                dbc.Col(html.Label("Diastolic Blood Pressure (mmHg)", className="fw-bold"), width="auto"),
                                dbc.Col(html.Span("🛈", id="info-dbp", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                            ]),
                            dbc.Input(id="input-dbp", type="number", placeholder="e.g. 80", className="mb-3"),
                            dbc.Tooltip("Diastolic BP is the pressure when the heart rests between beats. Normal is ~80 mmHg.", target="info-dbp", placement="right"),

                            # Sleep Hours
                            dbc.Row([
                                dbc.Col(html.Label("Sleep Hours per Day", className="fw-bold"), width="auto"),
                                dbc.Col(html.Span("🛈", id="info-sleep", style={"cursor": "pointer", "color": "#2A9D8F"}), width="auto")
                            ]),
                            dbc.Input(id="input-sleep", type="number", placeholder="e.g. 7", className="mb-3"),
                            dbc.Tooltip("Average nightly sleep duration. Adults typically need 7–9 hours.", target="info-sleep", placement="right"),

                            dbc.Button("GENERATE DIAGNOSTIC", id="predict-btn",
                            color="primary", className="mt-4 w-100 fw-bold shadow-sm")
                        ])
                    ], className="border-0 shadow-sm p-2")
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
    [State("input-age", "value"),
     State("input-bmi", "value"),
     State("input-act", "value"),
    State("input-diet", "value"),
    State("input-sbp", "value"),
    State("input-dbp", "value"),
    State("input-sleep", "value")],
    prevent_initial_call=False
)
def update_dashboard(n_clicks, age, bmi, act, diet, sbp, dbp, sleep):
    # Default Visuals
    fig_cluster = go.Figure().update_layout(
        title="Awaiting Model Assets...", template="plotly_white")
    fig_shap = go.Figure().update_layout(
        title="Awaiting SHAP Data...", template="plotly_white")

    # 1. VISUALIZATION LOGIC
    if not df.empty and km_bundle:
        scaler_features = km_bundle["scaler"].feature_names_in_
        df_enc = df.copy()
        for col in df_enc.select_dtypes(include=['object']).columns:
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

        # PCA Projection
        X_scaled = km_bundle["scaler"].transform(
            df_enc.reindex(columns=scaler_features, fill_value=0))
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)

        fig_cluster = px.scatter(x=coords[:, 0], y=coords[:, 1],
                                 color=df.get("diabetes_stage", "Population"),
                                 labels={
                                     'x': 'Lifestyle Metric (PC1)', 'y': 'Clinical Metric (PC2)'},
                                 template="plotly_white", opacity=0.4, title="Population Lifestyle Map")

        # Highlight User
        if age and bmi:
            patient_row = pd.DataFrame([{col: 0 for col in scaler_features}])
            patient_row['age'] = age
            patient_row['bmi'] = bmi
            patient_row['physical_activity_minutes_per_week'] = act or 0
            patient_row['diet_score'] = diet or 5
            patient_row['systolic_bp'] = sbp or 120
            patient_row['diastolic_bp'] = dbp or 80
            patient_row['sleep_hours_per_day'] = sleep or 7
            p_scaled = km_bundle["scaler"].transform(
                patient_row.reindex(columns=scaler_features, fill_value=0))
            p_pca = pca.transform(p_scaled)
            fig_cluster.add_trace(go.Scatter(x=[p_pca[0, 0]], y=[p_pca[0, 1]], mode="markers",
                                             marker=dict(size=22, color="black", symbol="star", line=dict(
                                                 width=2, color="white")),
                                             name="Current Patient"))

    if shap_df is not None:
        fig_shap = px.bar(shap_df.head(10)[::-1], x="mean_|SHAP|", y="feature", orientation="h",
                          title="Top Risk Drivers", color_discrete_sequence=[COLORS["primary"]])

    # 2. PREDICTION LOGIC
    if n_clicks is None:
        return dash.no_update, fig_cluster, fig_shap

    # Mock Decision Logic (Update with Role 3 XGBoost later)
    risk_score = min(100,(age or 0) * 0.4 + (bmi or 0) * 1.6 + (sbp or 0) * 0.05 + (dbp or 0) * 0.05 + (sleep or 0) * -0.5)
    level = "ELEVATED" if risk_score > 65 else "MODERATE" if risk_score > 40 else "STABLE"
    res_color = COLORS["danger"] if level == "ELEVATED" else COLORS["warning"] if level == "MODERATE" else COLORS["primary"]

    result_content = html.Div([
        dbc.Row([
            dbc.Col(info_card("Risk Level", level,
                    "Clinical Classification", res_color)),
            dbc.Col(info_card("Model Confidence",
                    f"{int(risk_score)}%", "Risk Probability", "#264653")),
            dbc.Col(info_card("Lifestyle Segment", "Segment B",
                    "Patient Grouping", COLORS["secondary"])),
        ]),
        dbc.Alert([
            html.H5("Intervention Recommended",
                    className="alert-heading fw-bold"),
            html.P(
                f"Patient falls within the {level.lower()} risk bracket. Recommended actions include metabolic testing and lifestyle counseling.")
        ], color="light", className="border-start border-primary border-4 shadow-sm mt-3")
    ])

    return result_content, fig_cluster, fig_shap


if __name__ == '__main__':
    app.run(debug=True)
