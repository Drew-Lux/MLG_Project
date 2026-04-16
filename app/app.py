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

# ─── LOAD ALL ASSETS ─────────────────────────────────────────────────────────


def load_all_assets():
    try:
        df = pd.read_csv(DATASET_PATH)
        recs = pd.read_csv(os.path.join(OUTPUT_DIR, "actionable_recommendations.csv")) if os.path.exists(
            os.path.join(OUTPUT_DIR, "actionable_recommendations.csv")) else None
        km = joblib.load(os.path.join(OUTPUT_DIR, "kmeans_pipeline.pkl")) if os.path.exists(
            os.path.join(OUTPUT_DIR, "kmeans_pipeline.pkl")) else None
        clf = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl")) if os.path.exists(
            os.path.join(MODELS_DIR, "xgboost_model.pkl")) else None
        assign = pd.read_csv(os.path.join(OUTPUT_DIR, "patient_cluster_assignments.csv")) if os.path.exists(
            os.path.join(OUTPUT_DIR, "patient_cluster_assignments.csv")) else None
        return df, km, clf, recs, assign
    except Exception as e:
        print(f"Asset Load Error: {e}")
        return pd.DataFrame(), None, None, None, None


df, km_bundle, clf_bundle, recs_df, assignments_df = load_all_assets()

# ─── BRANDING ────────────────────────────────────────────────────────────────
COLORS = {"primary": "#2A9D8F", "secondary": "#264653",
          "danger": "#E63946", "warning": "#F4A261", "bg": "#F8F9FA"}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# UI Helper for Metric Cards


def info_card(title, value, subtitle, color):
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="text-muted text-uppercase small"),
            html.H2(value, style={"color": color}, className="fw-bold mb-0"),
            html.P(subtitle, className="text-muted small mb-0")
        ])
    ], className="shadow-sm border-0 mb-3")


app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.Div([
            html.H1("BC ANALYTICS | RISK ENGINE", className="fw-bold text-white mb-0",
                    style={"letter-spacing": "1.5px"}),
            html.P("Validated Clinical Intelligence & Population Insights",
                   className="text-white-50 mb-0")
        ], className="bg-primary p-4 rounded-bottom shadow mb-4"), width=12)
    ]),

    dbc.Tabs([
        # TAB 1: Patient Assessment
        dbc.Tab(label="PATIENT ASSESSMENT", children=[
            dbc.Row([
                dbc.Col([
                    html.H5("Comprehensive Clinical Entry",
                            className="mt-4 fw-bold text-secondary"),
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                # Column 1: Clinical Metrics
                                dbc.Col([
                                    html.Label("Age", className="fw-bold"),
                                    dbc.Input(id="input-age",
                                              type="number", className="mb-2"),
                                    html.Label("BMI", className="fw-bold"),
                                    dbc.Input(id="input-bmi",
                                              type="number", className="mb-2"),
                                    html.Label("HbA1c Level",
                                               className="fw-bold"),
                                    dbc.Input(id="input-hba1c",
                                              type="number", className="mb-2"),
                                    html.Label(
                                        "Systolic Blood Pressure", className="fw-bold"),
                                    dbc.Input(id="input-bp",
                                              type="number", className="mb-2"),
                                ], width=6),
                                # Column 2: Lifestyle Metrics
                                dbc.Col([
                                    html.Label("Activity (Mins/Week)",
                                               className="fw-bold"),
                                    dbc.Input(id="input-act",
                                              type="number", className="mb-2"),
                                    html.Label("Diet Quality (1-10)",
                                               className="fw-bold"),
                                    dcc.Slider(1, 10, 1, value=5,
                                               id="input-diet"),
                                    html.Label("Smoking Status",
                                               className="fw-bold mt-3"),
                                    dcc.Dropdown(id="input-smoke", options=[{'label': 'Never', 'value': 0}, {
                                                 'label': 'Former', 'value': 1}, {'label': 'Current', 'value': 2}], placeholder="Select..."),
                                    html.Label("Alcohol Consumption",
                                               className="fw-bold mt-2"),
                                    dcc.Dropdown(id="input-alcohol", options=[{'label': 'None', 'value': 0}, {
                                                 'label': 'Moderate', 'value': 1}, {'label': 'Heavy', 'value': 2}], placeholder="Select..."),
                                ], width=6)
                            ]),
                            dbc.Button("GENERATE DIAGNOSTIC", id="predict-btn",
                                       color="primary", className="mt-4 w-100 fw-bold shadow-sm")
                        ])
                    ], className="border-0 shadow-sm p-3")
                ], width=5),

                dbc.Col([
                    html.H5("Diagnostic Profile",
                            className="mt-4 fw-bold text-secondary"),
                    html.Div(id="prediction-output")
                ], width=7)
            ])
        ]),

        # TAB 2: Population Insights
        dbc.Tab(label="LIFESTYLE CLUSTERING", children=[
            dcc.Graph(id='cluster-graph', style={"height": "600px"})
        ])
    ])
], fluid=True, style={"backgroundColor": COLORS["bg"], "minHeight": "100vh"})


@app.callback(
    [Output("prediction-output", "children"),
     Output("cluster-graph", "figure")],
    Input("predict-btn", "n_clicks"),
    [State("input-age", "value"), State("input-bmi", "value"), State("input-hba1c", "value"),
     State("input-bp", "value"), State("input-act",
                                       "value"), State("input-diet", "value"),
     State("input-smoke", "value"), State("input-alcohol", "value")],
    prevent_initial_call=False
)
def update_dashboard(n_clicks, age, bmi, hba1c, bp, act, diet, smoke, alcohol):
    # Setup Figure
    fig_cluster = go.Figure().update_layout(template="plotly_white")
    if not df.empty and km_bundle:
        scaler_features = km_bundle["scaler"].feature_names_in_
        X_scaled = km_bundle["scaler"].transform(
            df.reindex(columns=scaler_features, fill_value=0))
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        fig_cluster = px.scatter(x=coords[:, 0], y=coords[:, 1], color=df["cluster_name"]
                                 if "cluster_name" in df.columns else None, opacity=0.4)

        if age and bmi:
            p_row = pd.DataFrame([{f: 0 for f in scaler_features}])
            p_row['Age'], p_row['bmi'], p_row['hba1c'], p_row['systolic_bp'] = age, bmi, hba1c or 0, bp or 0
            p_pca = pca.transform(km_bundle["scaler"].transform(
                p_row.reindex(columns=scaler_features, fill_value=0)))
            fig_cluster.add_trace(go.Scatter(x=[p_pca[0, 0]], y=[
                                  p_pca[0, 1]], mode="markers", marker=dict(size=20, color="black", symbol="star")))

    if n_clicks is None or not age or not bmi or not clf_bundle:
        return dash.no_update, fig_cluster

    # PREDICTION & CONFIDENCE FIX
    features = clf_bundle["feature_names"]
    p_data = pd.DataFrame([{f: 0 for f in features}])
    p_data['Age'], p_data['bmi'] = age, bmi
    if 'hba1c' in features:
        p_data['hba1c'] = hba1c or 0
    if 'systolic_bp' in features:
        p_data['systolic_bp'] = bp or 0
    if 'physical_activity_minutes_per_week' in features:
        p_data['physical_activity_minutes_per_week'] = act or 0
    if 'diet_score' in features:
        p_data['diet_score'] = diet or 5

    X_clf = clf_bundle["scaler"].transform(p_data[features])
    probs = clf_bundle["model"].predict_proba(X_clf)[0]
    idx = np.argmax(probs)
    conf, label = probs[idx] * \
        100, clf_bundle["label_encoder"].inverse_transform([idx])[0]

    # DYNAMIC CLUSTER & RECS FIX
    seg = "Unknown"
    if km_bundle:
        cid = km_bundle["kmeans"].predict(km_bundle["scaler"].transform(
            p_data.reindex(columns=km_bundle["scaler"].feature_names_in_, fill_value=0)))[0]
        if assignments_df is not None:
            seg = assignments_df.set_index(
                "cluster")["cluster_name"].to_dict().get(cid, f"Cluster {cid}")

    recs = [html.Li(r) for r in recs_df[recs_df["cluster_name"] == seg]
            ["recommendation"].tolist()] if recs_df is not None else []

    return html.Div([
        dbc.Row([
            dbc.Col(info_card("Risk Status", label, "Prediction",
                    COLORS["danger"] if "Diabetes" in str(label) else COLORS["primary"])),
            dbc.Col(info_card("Confidence",
                    f"{int(conf)}%", "Specific Class Prob", "#264653")),
            dbc.Col(info_card("Lifestyle Segment", seg,
                    "Dynamic Result", COLORS["warning"])),
        ]),
        dbc.Alert([html.H5(f"Advice for {seg}:"), html.Ul(
            recs)], color="light", className="shadow-sm mt-3")
    ]), fig_cluster


if __name__ == '__main__':
    app.run(debug=True)
