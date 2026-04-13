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
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─── Data & model paths ───────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "Diabetes_and_Lifestyle_Dataset_Dummy.csv")
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs", "clustering")
MODELS_DIR   = os.path.join(BASE_DIR, "outputs", "models")

# ─── Load pipeline outputs ────────────────────────────────────────────────────
def load_base_data():
    df = pd.read_csv(DATASET_PATH)
    assignments_path = os.path.join(OUTPUT_DIR, "patient_cluster_assignments.csv")
    if os.path.exists(assignments_path):
        assignments = pd.read_csv(assignments_path)
        df["cluster"]      = assignments["cluster"].values
        df["cluster_name"] = assignments["cluster_name"].values
    else:
        df["cluster"]      = 0
        df["cluster_name"] = "Unknown"
    return df
 
def load_shap_importance():
    path = os.path.join(OUTPUT_DIR, "shap_cluster_importance.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    path2 = os.path.join(OUTPUT_DIR, "shap_classification_importance.csv")
    if os.path.exists(path2):
        return pd.read_csv(path2)
    return None
 
def load_recommendations():
    path = os.path.join(OUTPUT_DIR, "actionable_recommendations.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()
 
def load_kmeans_pipeline():
    path = os.path.join(OUTPUT_DIR, "kmeans_pipeline.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None
 
def load_classification_model():
    path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None
 
df        = load_base_data()
shap_df   = load_shap_importance()
rec_df    = load_recommendations()
km_bundle = load_kmeans_pipeline()
clf_model = load_classification_model()
 
# Encode for PCA
df_enc = df.copy()
for col in df_enc.select_dtypes(include=["object", "str"]).columns:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
 
feature_cols = [c for c in df_enc.columns
                if c not in ["cluster", "cluster_name", "diabetes_stage",
                              "diabetes_stage_enc"]]
 
CLUSTER_COLORS = {
    "Low Risk Lifestyle":      "#2A9D8F",
    "Moderate Risk Lifestyle": "#F4A261",
    "High Risk Lifestyle":     "#E63946",
}
 
# ─── Build real figures ───────────────────────────────────────────────────────
def build_cluster_figure():
    # Use the exact feature order the scaler was fitted with
    if km_bundle:
        scaler_features = km_bundle["scaler"].feature_names_in_
        X_df    = pd.DataFrame(df_enc[feature_cols].values, columns=feature_cols)
        X_df    = X_df[scaler_features]   # reorder to match scaler
        X_scaled = km_bundle["scaler"].transform(X_df)
    else:
        X_df     = pd.DataFrame(df_enc[feature_cols].values, columns=feature_cols)
        X_scaled = StandardScaler().fit_transform(X_df)

    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    ev     = pca.explained_variance_ratio_
    plot_df = pd.DataFrame({
        "PC1":     coords[:, 0],
        "PC2":     coords[:, 1],
        "Segment": df["cluster_name"].values,
    })
    fig = px.scatter(
        plot_df, x="PC1", y="PC2", color="Segment",
        color_discrete_map=CLUSTER_COLORS,
        opacity=0.65,
        title="Lifestyle Segments (k=3)",
        labels={"PC1": f"PC1 ({ev[0]*100:.1f}% var)",
                "PC2": f"PC2 ({ev[1]*100:.1f}% var)"},
    )
    fig.update_traces(marker_size=6)
    fig.update_layout(legend_title="Lifestyle Segment", template="plotly_white")
    return fig
 
def build_shap_figure():
    if shap_df is None:
        fig = go.Figure()
        fig.add_annotation(text="Run pipeline to generate SHAP values.",
                           showarrow=False, font_size=14)
        return fig
    top = shap_df.head(12)
    fig = px.bar(
        top[::-1], x="mean_|SHAP|", y="feature", orientation="h",
        title="Patient-Specific Risk Drivers (SHAP)",
        labels={"mean_|SHAP|": "Influence on Risk Score", "feature": "Feature"},
        color="mean_|SHAP|",
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_layout(coloraxis_showscale=False, template="plotly_white")
    return fig
 
cluster_fig = build_cluster_figure()
shap_fig    = build_shap_figure()

# Initialize the app with a professional medical theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server  # Required for Render deployment

# --- UI COMPONENTS ---

# 1. Header Section
header = dbc.Row([
    dbc.Col(html.H1("BC Analytics | Precision Risk Engine",
            className="text-center text-primary fw-bold mb-4"), width=12)
], className="mt-4")

# 2. Patient Assessment Form (Phase 2)
assessment_form = dbc.Card([
    dbc.CardHeader(html.H4("Patient Clinical & Lifestyle Metrics",
                   className="text-white bg-primary m-0")),
    dbc.CardBody([
        dbc.Row([
            # Column: Clinical Metrics
            dbc.Col([
                html.Label("Age", className="fw-bold"),
                dbc.Input(id="input-age", type="number",
                          placeholder="Enter age", className="mb-3"),

                html.Label("Body Mass Index (BMI)", className="fw-bold"),
                dbc.Input(id="input-bmi", type="number",
                          placeholder="e.g., 25.4", className="mb-3"),

                html.Label("Blood Pressure (Systolic)", className="fw-bold"),
                dbc.Input(id="input-bp", type="number",
                          placeholder="e.g., 120", className="mb-3"),
            ], width=6),

            # Column: Lifestyle Factors
            dbc.Col([
                html.Label("Smoking History", className="fw-bold"),
                dcc.Dropdown(
                    id="input-smoking",
                    options=[
                        {'label': 'Never Smoked', 'value': 0},
                        {'label': 'Former Smoker', 'value': 1},
                        {'label': 'Current Smoker', 'value': 2}
                    ], placeholder="Select status", className="mb-3"
                ),

                html.Label("Physical Activity (Hours/Week)",
                           className="fw-bold"),
                dcc.Slider(0, 20, 1, value=5, id='input-activity',
                           marks={0: '0', 10: '10', 20: '20+'}),

                html.Label("Diet Quality (1-10)", className="fw-bold mt-3"),
                dcc.Slider(1, 10, 1, value=5, id='input-diet',
                           marks={1: 'Poor', 10: 'Excellent'}),
            ], width=6),
        ]),
        dbc.Button("Generate Risk Profile", id="predict-btn",
                   color="success", className="mt-4 w-100 fw-bold"),
    ])
], className="shadow")

# 3. Segmentation & SHAP Content (Phase 3)
segmentation_tab_content = html.Div([
    dbc.Row([
        dbc.Col([
            html.H4("Lifestyle Cluster Analysis (k=3)",
                    className="mt-4 fw-bold"),
            html.P("Visualizing patient position within lifestyle-based segments.",
                   className="text-muted"),
            # Placeholder for K-Means plot [cite: 33]
            dcc.Graph(id='cluster-graph', figure = cluster_fig),
        ], width=12),
        dbc.Col([
            html.H4("Key Risk Drivers (SHAP Analysis)",
                    className="mt-4 fw-bold"),
            html.P("Factors contributing most significantly to the predicted risk.",
                   className="text-muted"),
            # Placeholder for SHAP analysis [cite: 36]
            dcc.Graph(id='shap-graph', figure = shap_fig),
        ], width=12)
    ])
])

# --- MAIN LAYOUT ---

app.layout = dbc.Container([
    header,
    dbc.Tabs([
        dbc.Tab(assessment_form, label="1. Patient Assessment",
                tab_id="tab-risk", className="mt-4"),
        dbc.Tab(segmentation_tab_content,
                label="2. Population Segments", tab_id="tab-segments"),
    ], id="tabs", active_tab="tab-risk"),

    # Results Output Area
    html.Div(id="prediction-output", className="mt-4 mb-5")
], fluid=True)

# --- CALLBACKS (Logic) ---


@app.callback(
    [Output("prediction-output", "children"),
     Output("cluster-graph", "figure"),
     Output("shap-graph", "figure")],
    Input("predict-btn", "n_clicks"),
    [State("input-age", "value"), State("input-bmi", "value")],
    prevent_initial_call=True
)
def run_full_analysis(n_clicks, age, bmi):
    # 1. Validation check
    if age is None or bmi is None:
        return dbc.Alert("Error: Please provide at least Age and BMI.", color="danger"), dash.no_update, dash.no_update

    # 2. Risk Prediction Mock Logic (To be updated with Role 3's model)
    is_high_risk = age > 50 or bmi > 30
    risk_text = "HIGH RISK" if is_high_risk else "STABLE / LOW RISK"
    risk_color = "danger" if is_high_risk else "success"

    prediction_card = dbc.Card([
        dbc.CardBody([
            html.H2(f"Assessment Result: {risk_text}",
                    className=f"text-center text-{risk_color} fw-bold"),
            html.Hr(),
            html.P("Switch to the 'Population Segments' tab to see driver and cluster details.",
                   className="text-center text-muted mb-0")
        ])
    ], className=f"border-{risk_color} shadow-lg")

    # 3. Graph Mock Logic
    if km_bundle:
        scaler_features = km_bundle["scaler"].feature_names_in_

        input_vec = pd.DataFrame([{c: 0 for c in scaler_features}])
        if "age" in scaler_features:
            input_vec["age"] = age
        if "bmi" in scaler_features:
            input_vec["bmi"] = bmi
        input_vec = input_vec[scaler_features]

        X_in       = km_bundle["scaler"].transform(input_vec)
        cluster_id = km_bundle["kmeans"].predict(X_in)[0]
        id_to_name = (df[["cluster", "cluster_name"]]
                      .drop_duplicates()
                      .set_index("cluster")["cluster_name"]
                      .to_dict())
        segment = id_to_name.get(cluster_id, f"Cluster {cluster_id}")

        # Rebuild PCA with patient star marker
        X_all    = pd.DataFrame(df_enc[feature_cols].values, columns=feature_cols)
        X_all    = X_all[scaler_features]
        X_scaled = km_bundle["scaler"].transform(X_all)
        pca      = PCA(n_components=2, random_state=42)
        coords   = pca.fit_transform(X_scaled)
        ev       = pca.explained_variance_ratio_

        plot_df = pd.DataFrame({
            "PC1":     coords[:, 0],
            "PC2":     coords[:, 1],
            "Segment": df["cluster_name"].values,
        })
        updated_cluster_fig = px.scatter(
            plot_df, x="PC1", y="PC2", color="Segment",
            color_discrete_map=CLUSTER_COLORS, opacity=0.65,
            title=f"Lifestyle Segments (k=3) — Patient assigned: {segment}",
            labels={"PC1": f"PC1 ({ev[0]*100:.1f}% var)",
                    "PC2": f"PC2 ({ev[1]*100:.1f}% var)"},
        )
        patient_pca = pca.transform(km_bundle["scaler"].transform(input_vec))
        updated_cluster_fig.add_trace(go.Scatter(
            x=[patient_pca[0, 0]], y=[patient_pca[0, 1]],
            mode="markers",
            marker=dict(size=18, color="black", symbol="star"),
            name="This Patient",
        ))
        updated_cluster_fig.update_layout(template="plotly_white")
    else:
        updated_cluster_fig = cluster_fig
 
    return prediction_card, updated_cluster_fig, shap_fig


if __name__ == '__main__':
    app.run(debug=True)
