# app.py
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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(
    BASE_DIR, "data_insight", "Diabetes_scaled_for_modeling.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "clustering")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")

# ─── ASSET LOADING ───────────────────────────────────────────────────────────
CLASS_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_model.pkl")
clf_bundle = joblib.load(CLASS_MODEL_PATH) if os.path.exists(
    CLASS_MODEL_PATH) else None
clf_model = clf_bundle["model"] if clf_bundle else None
clf_scaler = clf_bundle["scaler"] if clf_bundle else None
clf_label_encoder = clf_bundle["label_encoder"] if clf_bundle else None
clf_features = clf_bundle["feature_names"] if clf_bundle else None

KM_PATH = os.path.join(OUTPUT_DIR, "kmeans_pipeline.pkl")
km_bundle = joblib.load(KM_PATH) if os.path.exists(KM_PATH) else None

REC_PATH = os.path.join(OUTPUT_DIR, "actionable_recommendations.csv")
recs_df = pd.read_csv(REC_PATH) if os.path.exists(REC_PATH) else None

ASSIGN_PATH = os.path.join(OUTPUT_DIR, "patient_cluster_assignments.csv")
assignments_df = pd.read_csv(
    ASSIGN_PATH) if os.path.exists(ASSIGN_PATH) else None

# ─── DATA PREP ───────────────────────────────────────────────────────────────


def load_base_data():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        if assignments_df is not None:
            df["cluster_name"] = assignments_df["cluster_name"].values
        return df
    return pd.DataFrame()


df_base = load_base_data()
COLORS = {"primary": "#2A9D8F", "secondary": "#264653",
          "danger": "#E63946", "warning": "#F4A261", "bg": "#F8F9FA"}

# ─── UI COMPONENTS ───────────────────────────────────────────────────────────


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
    dbc.Row([
        dbc.Col(html.Div([
            html.H1("BC ANALYTICS | PRECISION RISK ENGINE",
                    className="fw-bold text-white mb-0"),
            html.P("Validated Clinical Intelligence Dashboard",
                   className="text-white-50 mb-0")
        ], className="bg-primary p-4 rounded-bottom shadow mb-4"), width=12)
    ]),

    dbc.Tabs([
        dbc.Tab(label="PATIENT ASSESSMENT", children=[
            dbc.Row([
                dbc.Col([
                    html.H5("Comprehensive Clinical Entry",
                            className="mt-4 fw-bold text-secondary"),
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                # Column 1
                                dbc.Col([
                                    html.Label("Age", className="fw-bold"),
                                    dbc.Input(
                                        id="input-age", type="number", placeholder="e.g., 45", className="mb-2"),
                                    html.Label("BMI", className="fw-bold"),
                                    dbc.Input(
                                        id="input-bmi", type="number", placeholder="e.g., 28.5", className="mb-2"),
                                    html.Label("HbA1c Level",
                                               className="fw-bold"),
                                    dbc.Input(
                                        id="input-hba1c", type="number", placeholder="e.g., 5.7", className="mb-2"),
                                    html.Label("Systolic BP",
                                               className="fw-bold"),
                                    dbc.Input(
                                        id="input-bp", type="number", placeholder="e.g., 120", className="mb-2"),
                                ], width=6),
                                # Column 2
                                dbc.Col([
                                    html.Label("Activity (Mins/Week)",
                                               className="fw-bold"),
                                    dbc.Input(
                                        id="input-act", type="number", placeholder="e.g., 150", className="mb-2"),
                                    html.Label("Diet Quality Score",
                                               className="fw-bold"),
                                    dcc.Slider(1, 10, 1, value=5, id="input-diet", marks={i: str(
                                        i) for i in range(1, 11)}, tooltip={"always_visible": True}),
                                    html.Label("Smoking Status",
                                               className="fw-bold mt-3"),
                                    dcc.Dropdown(id="input-smoke", options=[{'label': 'Never', 'value': 0}, {
                                                 'label': 'Former', 'value': 1}, {'label': 'Current', 'value': 2}], placeholder="Select..."),
                                    html.Label("Alcohol Consumption",
                                               className="fw-bold mt-2"),
                                    dcc.Dropdown(id="input-alc", options=[{'label': 'None', 'value': 0}, {
                                                 'label': 'Moderate', 'value': 1}, {'label': 'Heavy', 'value': 2}], placeholder="Select..."),
                                ], width=6)
                            ]),
                            dbc.Button("GENERATE FULL DIAGNOSTIC", id="predict-btn",
                                       color="primary", className="mt-4 w-100 fw-bold")
                        ])
                    ], className="border-0 shadow-sm p-3")
                ], width=5),
                dbc.Col([
                    html.H5("Diagnostic Intelligence Profile",
                            className="mt-4 fw-bold text-secondary"),
                    html.Div(id="prediction-output")
                ], width=7)
            ])
        ]),
        dbc.Tab(label="LIFESTYLE CLUSTERING", children=[
            dcc.Graph(id='cluster-graph', style={"height": "600px"})
        ])
    ])
], fluid=True, style={"backgroundColor": COLORS["bg"], "minHeight": "100vh"})

# ─── CALLBACK ────────────────────────────────────────────────────────────────


@app.callback(
    [Output("prediction-output", "children"),
     Output("cluster-graph", "figure")],
    Input("predict-btn", "n_clicks"),
    [State("input-age", "value"), State("input-bmi", "value"), State("input-hba1c", "value"),
     State("input-bp", "value"), State("input-act",
                                       "value"), State("input-diet", "value"),
     State("input-smoke", "value"), State("input-alc", "value")],
    prevent_initial_call=False
)
def update_dashboard(n_clicks, age, bmi, hba1c, bp, act, diet, smoke, alc):
    # Setup Figure
    fig_cluster = go.Figure().update_layout(template="plotly_white")
    if not df_base.empty and km_bundle:
        sf = km_bundle["scaler"].feature_names_in_
        X_pop = km_bundle["scaler"].transform(
            df_base.reindex(columns=sf, fill_value=0))
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_pop)
        fig_cluster = px.scatter(x=coords[:, 0], y=coords[:, 1], color=df_base.get("cluster_name"),
                                 labels={'x': 'Lifestyle Intensity', 'y': 'Clinical Index'}, opacity=0.3)
        if age and bmi:
            p_df = pd.DataFrame([{f: 0 for f in sf}])
            p_df['Age'], p_df['bmi'] = age, bmi
            p_scaled = km_bundle["scaler"].transform(
                p_df.reindex(columns=sf, fill_value=0))
            p_pca = pca.transform(p_scaled)
            fig_cluster.add_trace(go.Scatter(x=[p_pca[0, 0]], y=[
                                  p_pca[0, 1]], mode="markers", marker=dict(size=20, color="black", symbol="star")))

    if n_clicks is None or age is None or bmi is None:
        return html.Div([dbc.Alert("Analysis Ready: Enter patient data.", color="info")]), fig_cluster

    # ── THE REAL-TIME PROCESSING ──
    input_df = pd.DataFrame([{f: 0 for f in clf_features}])

    # Map Inputs
    mapping = {'Age': age, 'bmi': bmi, 'hba1c': hba1c or 0, 'systolic_bp': bp or 0,
               'physical_activity_minutes_per_week': act or 0, 'diet_score': diet or 5,
               'smoking_history': smoke or 0, 'alcohol_consumption': alc or 0}

    for k, v in mapping.items():
        if k in clf_features:
            input_df[k] = v

    # ── DEBUG LOGGING (Changes should be visible in terminal) ──
    print(
        f"DEBUG: Processing patient - Age: {age}, BMI: {bmi}, HbA1c: {hba1c}")

    # Scale and Predict
    X_scaled = clf_scaler.transform(input_df[clf_features])
    probs = clf_model.predict_proba(X_scaled)[0]
    p_idx = np.argmax(probs)

    # Dynamic values
    conf, label = probs[p_idx] * \
        100, clf_label_encoder.inverse_transform([p_idx])[0]

    # Re-Cluster
    seg = "Unknown"
    if km_bundle:
        cid = km_bundle["kmeans"].predict(km_bundle["scaler"].transform(
            input_df.reindex(columns=km_bundle["scaler"].feature_names_in_, fill_value=0)))[0]
        if assignments_df is not None:
            seg = assignments_df.set_index(
                "cluster")["cluster_name"].to_dict().get(cid, f"Cluster {cid}")

    # Fetch dynamic recommendations
    recs = [html.Li(r, className="mb-2") for r in recs_df[recs_df["cluster_name"]
                                                          == seg]["recommendation"].tolist()] if recs_df is not None else []

    # UI Construction
    res_color = COLORS["danger"] if "Diabetes" in str(
        label) else COLORS["primary"]

    return html.Div([
        dbc.Row([
            dbc.Col(info_card("Risk Status", label,
                    "XGBoost Prediction", res_color)),
            dbc.Col(info_card("Confidence",
                    f"{int(conf)}%", "Model Reliability", "#264653")),
            dbc.Col(info_card("Lifestyle Group", seg,
                    "Population Segment", COLORS["warning"])),
        ]),
        dbc.Alert([
            html.H5(
                f"Evidence-Based Strategy for {seg}:", className="fw-bold"),
            html.Ul(recs)
        ], color="light", className="shadow-sm border-start border-primary border-4 mt-3")
    ]), fig_cluster


if __name__ == '__main__':
    app.run(debug=True)
