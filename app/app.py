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
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")

# ─── LOAD PIPELINE ASSETS ─────────────────────────────────────────────────────


def load_base_data():
    try:
        df = pd.read_csv(DATASET_PATH)
        assignments_path = os.path.join(
            OUTPUT_DIR, "patient_cluster_assignments.csv")
        if os.path.exists(assignments_path):
            assignments = pd.read_csv(assignments_path)
            # Match assignments to the dataset rows
            df["cluster_name"] = assignments["cluster_name"].values
        else:
            df["cluster_name"] = "Unknown"
        return df
    except Exception as e:
        print(f"Data Load Error: {e}")
        return pd.DataFrame()


def load_kmeans_pipeline():
    path = os.path.join(OUTPUT_DIR, "kmeans_pipeline.pkl")
    return joblib.load(path) if os.path.exists(path) else None


def load_shap_importance():
    path = os.path.join(OUTPUT_DIR, "shap_cluster_importance.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


# Global Data Assets
df = load_base_data()
km_bundle = load_kmeans_pipeline()
shap_df = load_shap_importance()

# ─── VISUALIZATIONS ───────────────────────────────────────────────────────────


def build_cluster_figure():
    if km_bundle and not df.empty:
        scaler_features = km_bundle["scaler"].feature_names_in_

        # Prepare data for PCA visualization
        df_vis = df.copy()
        for col in df_vis.select_dtypes(include=['object']).columns:
            df_vis[col] = LabelEncoder().fit_transform(df_vis[col].astype(str))

        # Align features and scale
        X_df = df_vis.reindex(columns=scaler_features, fill_value=0)
        X_scaled = km_bundle["scaler"].transform(X_df)

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)

        plot_df = pd.DataFrame({
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "Segment": df["cluster_name"].values
        })

        fig = px.scatter(plot_df, x="PC1", y="PC2", color="Segment",
                         title="Population Lifestyle Segments (k=3)",
                         template="plotly_white", opacity=0.6)
        fig.update_layout(legend_title_text='Lifestyle Risk')
        return fig
    return px.scatter(title="Clustering Visualization Pending Assets...")


def build_shap_figure():
    if shap_df is not None:
        top = shap_df.head(10)
        fig = px.bar(top[::-1], x="mean_|SHAP|", y="feature", orientation="h",
                     title="Key Risk Drivers (Population Level)",
                     template="plotly_white", color_discrete_sequence=["#2A9D8F"])
        return fig
    return go.Figure().add_annotation(text="SHAP analysis pending...", showarrow=False)


# ─── APP LAYOUT ───────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("BC Analytics | Precision Risk Engine",
                className="text-center text-primary fw-bold mt-4 mb-4"), width=12)
    ]),

    dbc.Tabs([
        dbc.Tab(label="1. Patient Assessment", tab_id="tab-risk", children=[
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Age", className="fw-bold"),
                            dbc.Input(id="input-age", type="number",
                                      placeholder="e.g. 45", className="mb-3"),
                            html.Label("BMI", className="fw-bold"),
                            dbc.Input(id="input-bmi", type="number",
                                      placeholder="e.g. 28.5", className="mb-3"),
                            html.Label("Systolic Blood Pressure",
                                       className="fw-bold"),
                            dbc.Input(id="input-bp", type="number",
                                      placeholder="e.g. 130", className="mb-3"),
                        ], width=6),
                        dbc.Col([
                            html.Label("Physical Activity (Mins/Week)",
                                       className="fw-bold"),
                            dbc.Input(id="input-act", type="number",
                                      placeholder="e.g. 150", className="mb-3"),
                            html.Label("Diet Quality Score (1-10)",
                                       className="fw-bold"),
                            dcc.Slider(1, 10, 1, value=5, id="input-diet",
                                       marks={1: '1', 5: '5', 10: '10'}),
                            html.Label("Smoking Status",
                                       className="fw-bold mt-3"),
                            dcc.Dropdown(
                                id="input-smoke",
                                options=[{'label': 'Never', 'value': 0}, {
                                    'label': 'Former', 'value': 1}, {'label': 'Current', 'value': 2}],
                                placeholder="Select Status"
                            )
                        ], width=6)
                    ]),
                    dbc.Button("Run Diagnostic", id="predict-btn",
                               color="success", className="mt-4 w-100 fw-bold")
                ])
            ], className="shadow mt-3")
        ]),
        dbc.Tab(label="2. Lifestyle & Population Segments", tab_id="tab-segments", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id='cluster-graph',
                        figure=build_cluster_figure()), width=12),
                dbc.Col(dcc.Graph(id='shap-graph',
                        figure=build_shap_figure()), width=12)
            ])
        ]),
    ], id="tabs", active_tab="tab-risk"),

    html.Div(id="prediction-output", className="mt-4 mb-5")
], fluid=True)

# ─── CALLBACKS ────────────────────────────────────────────────────────────────


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    [State("input-age", "value"), State("input-bmi", "value")],
    prevent_initial_call=True
)
def run_prediction(n_clicks, age, bmi):
    if not age or not bmi:
        return dbc.Alert("Please enter core clinical data (Age and BMI).", color="warning")

    # Placeholder for XGBoost Prediction (Role 3)
    # Once xgboost_model.pkl is ready, we load and predict here.
    risk_status = "Elevated Risk" if age > 55 or bmi > 30 else "Normal Range"
    color = "danger" if risk_status == "Elevated Risk" else "success"

    return dbc.Card([
        dbc.CardBody([
            html.H3(f"Risk Assessment: {risk_status}",
                    className=f"text-center text-{color} fw-bold"),
            html.P("Insights based on k-means lifestyle segmentation.",
                   className="text-center text-muted")
        ])
    ], className=f"border-{color} shadow")


if __name__ == '__main__':
    app.run_server(debug=True)
