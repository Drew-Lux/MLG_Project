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

        # Load Cluster Recommendations (Role 4)
        rec_path = os.path.join(OUTPUT_DIR, "actionable_recommendations.csv")
        recs = pd.read_csv(rec_path) if os.path.exists(rec_path) else None

        # Load K-Means Bundle (Role 4)
        km_path = os.path.join(OUTPUT_DIR, "kmeans_pipeline.pkl")
        km = joblib.load(km_path) if os.path.exists(km_path) else None

        # Load Classification Bundle (Role 3)
        clf_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
        clf = joblib.load(clf_path) if os.path.exists(clf_path) else None

        # Load Cluster Assignments (Mapping IDs to names)
        assign_path = os.path.join(
            OUTPUT_DIR, "patient_cluster_assignments.csv")
        assignments = pd.read_csv(
            assign_path) if os.path.exists(assign_path) else None

        return df, km, clf, recs, assignments
    except Exception as e:
        print(f"Asset Load Error: {e}")
        return pd.DataFrame(), None, None, None, None


df, km_bundle, clf_bundle, recs_df, assignments_df = load_all_assets()

# ─── BRANDING ────────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#2A9D8F",   # Medical Mint
    "secondary": "#264653",  # Deep Slate
    "danger": "#E63946",    # Alert Red
    "warning": "#F4A261",   # Amber
    "bg": "#F8F9FA"         # Light Grey
}

# UI Helper


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
                    html.H5("Diagnostic Input",
                            className="mt-4 fw-bold text-secondary"),
                    dbc.Card([
                        dbc.CardBody([
                            html.Label("Age", className="fw-bold"),
                            dbc.Input(id="input-age", type="number",
                                      placeholder="Enter Age", className="mb-3"),

                            html.Label("BMI", className="fw-bold"),
                            dbc.Input(id="input-bmi", type="number",
                                      placeholder="e.g. 28.5", className="mb-3"),

                            html.Label("Physical Activity (Mins/Week)",
                                       className="fw-bold"),
                            dbc.Input(id="input-act", type="number",
                                      placeholder="e.g. 150", className="mb-3"),

                            html.Label("Diet Quality Score (1-10)",
                                       className="fw-bold"),
                            dcc.Slider(1, 10, 1, value=5, id="input-diet",
                                       marks={1: '1', 5: '5', 10: '10'}),

                            dbc.Button("GENERATE DIAGNOSTIC", id="predict-btn",
                                       color="primary", className="mt-4 w-100 fw-bold shadow-sm")
                        ])
                    ], className="border-0 shadow-sm p-2")
                ], width=4),

                dbc.Col([
                    html.H5("Diagnostic Profile",
                            className="mt-4 fw-bold text-secondary"),
                    html.Div(id="prediction-output", children=[
                        dbc.Alert("Analysis Ready: Please enter patient clinical data.",
                                  color="light", className="text-center border shadow-sm")
                    ])
                ], width=8)
            ])
        ]),

        # TAB 2: Population Insights
        dbc.Tab(label="LIFESTYLE CLUSTERING", children=[
            dbc.Row([
                dbc.Col([
                    html.H5("Population Lifestyle Map",
                            className="mt-4 fw-bold text-secondary"),
                    dcc.Graph(id='cluster-graph', style={"height": "550px"})
                ], width=12)
            ])
        ])
    ]),

    # Status Footer
    dbc.Row([
        dbc.Col(html.Hr(), width=12),
        dbc.Col(html.P(f"System Status: {'✅ Model Connected' if clf_bundle else '❌ Missing xgboost_model.pkl'}",
                       className=f"text-center small {'text-success' if clf_bundle else 'text-danger'}"), width=12)
    ])
], fluid=True, style={"backgroundColor": COLORS["bg"], "minHeight": "100vh"})

# ─── DASHBOARD CALLBACK ──────────────────────────────────────────────────────


@app.callback(
    [Output("prediction-output", "children"),
     Output("cluster-graph", "figure")],
    Input("predict-btn", "n_clicks"),
    [State("input-age", "value"), State("input-bmi", "value"),
     State("input-act", "value"), State("input-diet", "value")],
    prevent_initial_call=False
)
def update_dashboard(n_clicks, age, bmi, act, diet):
    # Initial Figure Logic
    fig_cluster = go.Figure().update_layout(
        title="Awaiting Input...", template="plotly_white")

    # 1. VISUALIZATION (PCA Map)
    if not df.empty and km_bundle:
        scaler_features = km_bundle["scaler"].feature_names_in_
        X_scaled = km_bundle["scaler"].transform(
            df.reindex(columns=scaler_features, fill_value=0))
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)

        fig_cluster = px.scatter(x=coords[:, 0], y=coords[:, 1],
                                 color=df["cluster_name"] if "cluster_name" in df.columns else None,
                                 template="plotly_white", opacity=0.4, title="Lifestyle Segments")

        # Add Patient Highlight
        if age and bmi:
            p_row = pd.DataFrame([{f: 0 for f in scaler_features}])
            p_row['Age'] = age
            p_row['bmi'] = bmi
            p_scaled = km_bundle["scaler"].transform(
                p_row.reindex(columns=scaler_features, fill_value=0))
            p_pca = pca.transform(p_scaled)
            fig_cluster.add_trace(go.Scatter(x=[p_pca[0, 0]], y=[p_pca[0, 1]], mode="markers",
                                             marker=dict(size=22, color="black", symbol="star", line=dict(
                                                 width=2, color="white")),
                                             name="Current Patient"))

    # 2. PREDICTION LOGIC
    if n_clicks is None or not age or not bmi or not clf_bundle:
        return dash.no_update, fig_cluster

    # A. Predict Risk & REAL Confidence
    feature_names = clf_bundle["feature_names"]
    patient_data = pd.DataFrame([{f: 0 for f in feature_names}])
    patient_data['Age'] = age
    patient_data['bmi'] = bmi
    patient_data['physical_activity_minutes_per_week'] = act if act else 0
    patient_data['diet_score'] = diet

    X_clf_scaled = clf_bundle["scaler"].transform(patient_data[feature_names])
    probs = clf_bundle["model"].predict_proba(X_clf_scaled)[0]
    pred_idx = np.argmax(probs)

    # FIX: confidence is probability of the SPECIFIC predicted class
    confidence_val = probs[pred_idx] * 100
    pred_label = clf_bundle["label_encoder"].inverse_transform([pred_idx])[0]

    # B. DYNAMIC CLUSTER (No more Segment B)
    segment_name = "Analysis Pending"
    if km_bundle:
        km_features = km_bundle["scaler"].feature_names_in_
        km_scaled = km_bundle["scaler"].transform(
            patient_data.reindex(columns=km_features, fill_value=0))
        cid = km_bundle["kmeans"].predict(km_scaled)[0]

        # Map ID to name using assignments file or dataframe
        if assignments_df is not None:
            name_map = assignments_df.set_index(
                "cluster")["cluster_name"].to_dict()
            segment_name = name_map.get(cid, f"Cluster {cid}")
        elif "cluster_name" in df.columns:
            # Fallback if assignments file is missing but base df has it
            segment_name = df[df['cluster'] ==
                              cid]['cluster_name'].iloc[0] if cid in df['cluster'].values else f"Cluster {cid}"

    # C. FILTERED RECOMMENDATIONS
    recommendations = []
    if recs_df is not None:
        # Filter advice specifically for the patient's cluster
        filtered_recs = recs_df[recs_df["cluster_name"]
                                == segment_name]["recommendation"].tolist()
        recommendations = [html.Li(r) for r in filtered_recs]

    res_color = COLORS["danger"] if "Diabetes" in str(
        pred_label) else COLORS["primary"]

    result_content = html.Div([
        dbc.Row([
            dbc.Col(info_card("Risk Assessment", pred_label,
                    "Model Prediction", res_color)),
            dbc.Col(info_card(
                "Confidence", f"{int(confidence_val)}%", "Probability for Result", "#264653")),
            dbc.Col(info_card("Lifestyle Segment", segment_name,
                    "Dynamic Clustering", COLORS["warning"])),
        ]),
        dbc.Alert([
            html.H5(f"Recommendations for {segment_name}:",
                    className="alert-heading fw-bold"),
            html.Ul(
                recommendations) if recommendations else "Continue current health management."
        ], color="light", className="border-start border-primary border-4 shadow-sm mt-3")
    ])

    return result_content, fig_cluster


if __name__ == '__main__':
    app.run_server(debug=True)
