# app/app.py
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


def load_all_assets():
    try:
        # Load the base dataset
        df = pd.read_csv(DATASET_PATH)

        # Load Classification Model Bundle (XGBoost)
        clf_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
        clf = joblib.load(clf_path) if os.path.exists(clf_path) else None

        # Load K-Means Model Bundle
        km_path = os.path.join(OUTPUT_DIR, "kmeans_pipeline.pkl")
        km = joblib.load(km_path) if os.path.exists(km_path) else None

        # Load Actionable Recommendations
        rec_path = os.path.join(OUTPUT_DIR, "actionable_recommendations.csv")
        recs = pd.read_csv(rec_path) if os.path.exists(rec_path) else None

        # Load Cluster Assignment Mapping
        assign_path = os.path.join(
            OUTPUT_DIR, "patient_cluster_assignments.csv")
        assign = pd.read_csv(assign_path) if os.path.exists(
            assign_path) else None

        return df, km, clf, recs, assign
    except Exception as e:
        print(f"Asset Load Error: {e}")
        return pd.DataFrame(), None, None, None, None


df_base, km_bundle, clf_bundle, recs_df, assignments_df = load_all_assets()

# ─── BRANDING & UI STYLING ───────────────────────────────────────────────────
COLORS = {
    "primary": "#2A9D8F",
    "secondary": "#264653",
    "danger": "#E63946",
    "warning": "#F4A261",
    "bg": "#F8F9FA"
}


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
    # Header Section
    dbc.Row([
        dbc.Col(html.Div([
            html.H1("BC ANALYTICS | RISK ENGINE", className="fw-bold text-white mb-0",
                    style={"letter-spacing": "1.5px"}),
            html.P("Validated Precision Diagnostic & Population Intelligence (5-Feature Profile)",
                   className="text-white-50 mb-0")
        ], className="bg-primary p-4 rounded-bottom shadow mb-4"), width=12)
    ]),

    dbc.Tabs([
        # TAB 1: Patient Assessment
        dbc.Tab(label="PATIENT ASSESSMENT", children=[
            dbc.Row([
                dbc.Col([
                    html.H5("Clinical Data Entry",
                            className="mt-4 fw-bold text-secondary"),
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                # Column 1: Clinical Markers
                                dbc.Col([
                                    html.Label("Age", className="fw-bold"),
                                    dbc.Input(
                                        id="input-age", type="number", placeholder="Years (e.g. 45)", className="mb-2"),

                                    html.Label("BMI", className="fw-bold"),
                                    dbc.Input(
                                        id="input-bmi", type="number", placeholder="e.g. 28.5", className="mb-2"),

                                    html.Label("Systolic BP",
                                               className="fw-bold"),
                                    dbc.Input(
                                        id="input-bp", type="number", placeholder="mmHg (e.g. 120)", className="mb-2"),
                                ], width=6),

                                # Column 2: Lifestyle Markers
                                dbc.Col([
                                    html.Label("Physical Activity",
                                               className="fw-bold"),
                                    dbc.Input(
                                        id="input-act", type="number", placeholder="Mins/Week (e.g. 150)", className="mb-2"),

                                    html.Label("Diet Quality Score",
                                               className="fw-bold"),
                                    dcc.Slider(1, 10, 1, value=5, id="input-diet",
                                               marks={i: str(i)
                                                      for i in range(1, 11)},
                                               tooltip={"placement": "bottom", "always_visible": True}),
                                ], width=6)
                            ]),
                            dbc.Button("GENERATE FULL DIAGNOSTIC", id="predict-btn",
                                       color="primary", className="mt-4 w-100 fw-bold shadow-sm")
                        ])
                    ], className="border-0 shadow-sm p-3")
                ], width=5),

                dbc.Col([
                    html.H5("Diagnostic Intelligence Profile",
                            className="mt-4 fw-bold text-secondary"),
                    html.Div(id="prediction-output", children=[
                        dbc.Alert("Analysis Ready: Enter patient data to generate result.",
                                  color="light", className="text-center border shadow-sm")
                    ])
                ], width=7)
            ])
        ]),

        # TAB 2: Population Insights
        dbc.Tab(label="LIFESTYLE CLUSTERING", children=[
            dbc.Row([
                dbc.Col([
                    html.H5("Population Similarity Mapping",
                            className="mt-4 fw-bold text-secondary"),
                    html.P("Mapping patient habits relative to the clinical population using core features.",
                           className="text-muted small"),
                    dcc.Graph(id='cluster-graph', style={"height": "550px"})
                ], width=12)
            ])
        ])
    ]),

    # Status Bar
    dbc.Row([
        dbc.Col(html.Hr(), width=12),
        dbc.Col(html.P(f"System Status: {'✅ Clinical Assets Loaded' if clf_bundle else '❌ Assets Missing'}",
                       className=f"text-center small {'text-success' if clf_bundle else 'text-danger'}"), width=12)
    ])
], fluid=True, style={"backgroundColor": COLORS["bg"], "minHeight": "100vh"})

# ─── DASHBOARD CALLBACK ──────────────────────────────────────────────────────


@app.callback(
    [Output("prediction-output", "children"),
     Output("cluster-graph", "figure")],
    Input("predict-btn", "n_clicks"),
    [State("input-age", "value"), State("input-bmi", "value"),
     State("input-bp", "value"), State("input-act", "value"),
     State("input-diet", "value")],
    prevent_initial_call=False
)
def update_dashboard(n_clicks, age, bmi, bp, act, diet):
    # 1. VISUALIZATION LOGIC
    fig_cluster = go.Figure().update_layout(template="plotly_white")
    if not df_base.empty and km_bundle:
        sf = km_bundle["scaler"].feature_names_in_
        pca = PCA(n_components=2, random_state=42)
        X_pop_scaled = km_bundle["scaler"].transform(
            df_base.reindex(columns=sf, fill_value=0))
        coords = pca.fit_transform(X_pop_scaled)

        fig_cluster = px.scatter(x=coords[:, 0], y=coords[:, 1],
                                 color=df_base.get("cluster_name"),
                                 labels={
                                     'x': 'Lifestyle Intensity (PC1)', 'y': 'Clinical Variance (PC2)'},
                                 template="plotly_white", opacity=0.3)

        if age and bmi:
            p_df = pd.DataFrame([{f: 0 for f in sf}])
            p_df['Age'], p_df['bmi'] = age, bmi
            p_df['systolic_bp'] = bp or 120
            p_df['physical_activity_minutes_per_week'] = act or 150
            p_df['diet_score'] = diet or 5

            p_scaled = km_bundle["scaler"].transform(
                p_df.reindex(columns=sf, fill_value=0))
            p_pca = pca.transform(p_scaled)
            fig_cluster.add_trace(go.Scatter(x=[p_pca[0, 0]], y=[p_pca[0, 1]], mode="markers",
                                             marker=dict(size=22, color="black", symbol="star", line=dict(
                                                 width=2, color="white")),
                                             name="Current Patient Position"))

    if n_clicks is None or age is None or bmi is None:
        return dash.no_update, fig_cluster

    # 2. PREDICTION LOGIC (XGBOOST)
    f_names = clf_bundle["feature_names"]
    input_df = pd.DataFrame([{f: 0 for f in f_names}])

    # Map Inputs ONLY (HbA1c is omitted from the mapping logic)
    mapping = {
        'Age': age,
        'bmi': bmi,
        'systolic_bp': bp if bp is not None else 120,
        'physical_activity_minutes_per_week': act if act is not None else 150,
        'diet_score': diet if diet is not None else 5
    }

    for k, v in mapping.items():
        if k in f_names:
            input_df[k] = v

    # SCALE DATA USING PRE-TRAINED SCALER
    X_input_scaled = clf_bundle["scaler"].transform(input_df[f_names])
    probs = clf_bundle["model"].predict_proba(X_input_scaled)[0]
    p_idx = np.argmax(probs)

    # Prediction results
    conf, label = probs[p_idx] * \
        100, clf_bundle["label_encoder"].inverse_transform([p_idx])[0]

    # 3. DYNAMIC CLUSTERING (K-MEANS)
    segment = "Analysis Pending"
    if km_bundle:
        km_scaled = km_bundle["scaler"].transform(input_df.reindex(
            columns=km_bundle["scaler"].feature_names_in_, fill_value=0))
        cid = km_bundle["kmeans"].predict(km_scaled)[0]
        if assignments_df is not None:
            segment = assignments_df.set_index(
                "cluster")["cluster_name"].to_dict().get(cid, f"Cluster {cid}")

    # 4. ACTIONABLE RECOMMENDATIONS
    recs = []
    if recs_df is not None:
        filtered = recs_df[recs_df["cluster_name"]
                           == segment]["recommendation"].tolist()
        recs = [html.Li(r, className="mb-2") for r in filtered]

    # UI Construction
    res_color = COLORS["danger"] if "Diabetes" in str(
        label) else COLORS["primary"]

    dynamic_profile = html.Div([
        dbc.Row([
            dbc.Col(info_card("Risk Status", label,
                    "XGBoost Classification", res_color)),
            dbc.Col(info_card("Confidence",
                    f"{int(conf)}%", "Model Probability", "#264653")),
            dbc.Col(info_card("Lifestyle Group", segment,
                    "Clinical Clustering", COLORS["warning"])),
        ]),
        dbc.Alert([
            html.H5(
                f"Evidence-Based Strategy for {segment}:", className="alert-heading fw-bold"),
            html.Hr(),
            html.Ul(recs) if recs else html.P(
                "Continue current clinical monitoring protocols.")
        ], color="light", className="border-start border-primary border-4 shadow-sm mt-3")
    ])

    return dynamic_profile, fig_cluster


if __name__ == '__main__':
    app.run(debug=True)
