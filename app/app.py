import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Initialize the app with a clean medical-style theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server  # Crucial for Render deployment

# --- UI COMPONENTS ---

# 1. Header
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

            # Column: Lifestyle Factors [cite: 15]
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

# --- LAYOUT ---

app.layout = dbc.Container([
    header,
    dbc.Tabs([
        dbc.Tab(assessment_form, label="1. Patient Assessment",
                tab_id="tab-risk", className="mt-4"),
        dbc.Tab(html.Div([
            html.H3("Lifestyle Segmentation (k=3)", className="mt-4"),
            html.P("Waiting for cluster data from the Unsupervised Learning Specialist...",
                   className="text-muted")
        ]), label="2. Population Segments", tab_id="tab-segments"),
    ], id="tabs", active_tab="tab-risk"),

    # Results Output Area
    html.Div(id="prediction-output", className="mt-4 mb-5")
], fluid=True)

# --- CALLBACKS (Logic) ---


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    [State("input-age", "value"), State("input-bmi", "value")],
    prevent_initial_call=True
)
def run_prediction(n_clicks, age, bmi):
    # Basic validation
    if age is None or bmi is None:
        return dbc.Alert("Error: Please provide at least Age and BMI to generate a profile.", color="danger")

    # Placeholder Logic: Replace this once Role 3 provides the .pkl model
    is_high_risk = age > 50 or bmi > 30
    risk_text = "HIGH RISK" if is_high_risk else "STABLE / LOW RISK"
    risk_color = "danger" if is_high_risk else "success"

    return dbc.Card([
        dbc.CardBody([
            html.H2(f"Assessment Result: {risk_text}",
                    className=f"text-center text-{risk_color} fw-bold"),
            html.Hr(),
            html.P("Next Steps: Review SHAP drivers and patient lifestyle segment positioning below.",
                   className="text-center text-muted mb-0")
        ])
    ], className=f"border-{risk_color} shadow-lg")


if __name__ == '__main__':
    app.run_server(debug=True)
