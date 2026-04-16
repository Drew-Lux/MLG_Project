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

# ─── CLASSIFICATION MODEL LOADING ────────────────────────────────────────────
CLASS_MODEL_PATH = os.path.join(
    BASE_DIR, "outputs", "models", "xgboost_model.pkl")
clf_bundle = joblib.load(CLASS_MODEL_PATH) if os.path.exists(
    CLASS_MODEL_PATH) else None
clf_model = clf_bundle["model"] if clf_bundle else None
clf_scaler = clf_bundle["scaler"] if clf_bundle else None
clf_label_encoder = clf_bundle["label_encoder"] if clf_bundle else None
clf_features = clf_bundle["feature_names"] if clf_bundle else None

# ─── CLUSTERING ASSETS ───────────────────────────────────────────────────────
KM_PATH = os.path.join(OUTPUT_DIR, "kmeans_pipeline.pkl")
km_bundle = joblib.load(KM_PATH) if os.path.exists(KM_PATH) else None

REC_PATH = os.path.join(OUTPUT_DIR, "actionable_recommendations.csv")
recs_df = pd.read_csv(REC_PATH) if os.path.exists(REC_PATH) else None

ASSIGN_PATH = os.path.join(OUTPUT_DIR, "patient_cluster_assignments.csv")
assignments_df = pd.read_csv(
    ASSIGN_PATH) if os.path.exists(ASSIGN_PATH) else None

# ─── DATA LOAD ───────────────────────────────────────────────────────────────


def load_base_data():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        if assignments_df is not None:
            df["cluster_name"] = assignments_df["cluster_name"].values
        return df
    return pd.DataFrame()


df = load_base_data()
COLORS = {"primary": "#2A9D8F", "secondary": "#264653",
          "danger": "#E63946", "warning": "#F4A261", "bg": "#F8F9FA"}

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
                                # Column 1: Clinical
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
                                # Column 2: Lifestyle
                                dbc.Col([
                                    html.Label("Activity (Mins/Week)",
                                               className="fw-bold"),
                                    dbc.Input(
                                        id="input-act", type="number", placeholder="e.g., 150", className="mb-2"),

                                    html.Label("Diet Quality Score",
                                               className="fw-bold"),
                                    dcc.Slider(1, 10, 1, value=5, id="input-diet",
                                               marks={i: str(i) for i in range(1, 11)}, tooltip={"placement": "bottom", "always_visible": True}),

                                    html.Label("Smoking Status",
                                               className="fw-bold mt-3"),
                                    dcc.Dropdown(id="input-smoke", options=[
                                        {'label': 'Never Smoked', 'value': 0},
                                        {'label': 'Former Smoker', 'value': 1},
                                        {'label': 'Current Smoker', 'value': 2}
                                    ], placeholder="Select status...", className="mb-2"),

                                    html.Label("Alcohol Consumption",
                                               className="fw-bold"),
                                    dcc.Dropdown(id="input-alc", options=[
                                        {'label': 'None / Occasional', 'value': 0},
                                        {'label': 'Moderate', 'value': 1},
                                        {'label': 'Frequent / Heavy', 'value': 2}
                                    ], placeholder="Select level...", className="mb-2"),
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
            dbc.Row([
                dbc.Col([
                    html.H5("Lifestyle Similarity Mapping",
                            className="mt-4 fw-bold text-secondary"),
                    html.P("This map groups patients by lifestyle habits. Red = Higher Clinical Risk, Green = Optimal Lifestyle.",
                           className="text-muted small"),
                    dcc.Graph(id='cluster-graph', style={"height": "550px"})
                ], width=12)
            ])
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
    # 1. GRAPH LOGIC (Now with descriptive axes)
    fig_cluster = go.Figure().update_layout(template="plotly_white")
    if not df.empty and km_bundle:
        scaler_features = km_bundle["scaler"].feature_names_in_
        X_scaled = km_bundle["scaler"].transform(
            df.reindex(columns=scaler_features, fill_value=0))
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)

        fig_cluster = px.scatter(x=coords[:, 0], y=coords[:, 1],
                                 color=df.get("cluster_name", None),
                                 labels={'x': 'Lifestyle Intensity Index',
                                         'y': 'Clinical Variance Index'},
                                 template="plotly_white", opacity=0.3, title="Population Similarity Map")

        if age and bmi:
            p_row = pd.DataFrame([{f: 0 for f in scaler_features}])
            p_row['Age'], p_row['bmi'] = age, bmi
            p_pca = pca.transform(km_bundle["scaler"].transform(
                p_row.reindex(columns=scaler_features, fill_value=0)))
            fig_cluster.add_trace(go.Scatter(x=[p_pca[0, 0]], y=[p_pca[0, 1]], mode="markers",
                                             marker=dict(size=22, color="black", symbol="star", line=dict(
                                                 width=2, color="white")),
                                             name="Current Patient Position"))

    if n_clicks is None or not age or not bmi or not clf_model:
        return dash.no_update, fig_cluster

    # 2. PREDICTION LOGIC
    input_df = pd.DataFrame([{f: 0 for f in clf_features}])
    mapping = {
        'Age': age, 'bmi': bmi, 'hba1c': hba1c, 'systolic_bp': bp,
        'physical_activity_minutes_per_week': act, 'diet_score': diet,
        'smoking_history': smoke if smoke is not None else 0,
        'alcohol_consumption': alc if alc is not None else 0
    }
    for col, val in mapping.items():
        if col in clf_features:
            input_df[col] = val if val is not None else 0

    # Confidence Fix (Probability of predicted class)
    X_clf = clf_scaler.transform(input_df[clf_features])
    probs = clf_model.predict_proba(X_clf)[0]
    p_idx = np.argmax(probs)
    conf = probs[p_idx] * 100
    label = clf_label_encoder.inverse_transform([p_idx])[0]

    # Dynamic Segment Fix (No hardcoded B)
    segment = "Analysis Pending"
    if km_bundle:
        cid = km_bundle["kmeans"].predict(km_bundle["scaler"].transform(
            input_df.reindex(columns=km_bundle["scaler"].feature_names_in_, fill_value=0)))[0]
        if assignments_df is not None:
            segment = assignments_df.set_index(
                "cluster")["cluster_name"].to_dict().get(cid, f"Cluster {cid}")

    # Recommendations Filtered by Segment
    recs = [html.Li(r) for r in recs_df[recs_df["cluster_name"] == segment]
            ["recommendation"].tolist()] if recs_df is not None else []

    res_color = COLORS["danger"] if "Diabetes" in str(
        label) else COLORS["primary"]

    return html.Div([
        dbc.Row([
            dbc.Col(info_card("Risk Assessment", label,
                    "XGBoost Diagnosis", res_color)),
            dbc.Col(info_card("Confidence",
                    f"{int(conf)}%", "Model Probability", "#264653")),
            dbc.Col(info_card("Lifestyle Group", segment,
                    "Clinical Clustering", COLORS["warning"])),
        ]),
        dbc.Alert([
            html.H5(
                f"Evidence-Based Strategy for {segment}:", className="alert-heading fw-bold"),
            html.Ul(recs) if recs else "Maintain current clinical tracking."
        ], color="light", className="border-start border-primary border-4 shadow-sm mt-3")
    ]), fig_cluster


if __name__ == '__main__':
    app.run(debug=True)
