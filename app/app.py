import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px

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
            dcc.Graph(id='cluster-graph'),
        ], width=12),
        dbc.Col([
            html.H4("Key Risk Drivers (SHAP Analysis)",
                    className="mt-4 fw-bold"),
            html.P("Factors contributing most significantly to the predicted risk.",
                   className="text-muted"),
            # Placeholder for SHAP analysis [cite: 36]
            dcc.Graph(id='shap-graph'),
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

    # 3. Graph Mock Logic (To be updated with Role 4's visuals)
    # Cluster Mock
    dummy_df = px.data.iris()
    cluster_fig = px.scatter(dummy_df, x="sepal_width", y="sepal_length", color="species",
                             title="Lifestyle Segments (k=3 Mockup)")

    # SHAP Mock
    shap_fig = px.bar(x=[5, 2, -1], y=["Age", "BMI", "Diet"], orientation='h',
                      title="Patient-Specific Risk Drivers (SHAP Mockup)")
    shap_fig.update_layout(
        xaxis_title="Influence on Risk Score", yaxis_title="Feature")

    return prediction_card, cluster_fig, shap_fig


if __name__ == '__main__':
    app.run_server(debug=True)
