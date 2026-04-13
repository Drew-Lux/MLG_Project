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

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(
    BASE_DIR, "data", "Diabetes_and_LifeStyle_Dataset_.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "clustering")

# --- DATA LOAD ---


def load_assets():
    try:
        df = pd.read_csv(DATASET_PATH)
        path = os.path.join(OUTPUT_DIR, "kmeans_pipeline.pkl")
        km = joblib.load(path) if os.path.exists(path) else None
        return df, km
    except:
        return pd.DataFrame(), None


df, km_bundle = load_assets()

# --- STYLING ---
COLORS = {"primary": "#2A9D8F", "secondary": "#264653",
          "danger": "#E63946", "warning": "#F4A261"}

# --- UI HELPERS ---


def make_metric_card(title, value, color):
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="text-muted"),
            html.H3(value, style={"color": color}, className="fw-bold")
        ])
    ], className="shadow-sm border-0")


# --- APP START ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

app.layout = dbc.Container([
    # Professional Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("BC ANALYTICS", style={
                        "letter-spacing": "3px"}, className="fw-bold text-white mb-0"),
                html.P("Precision Diabetes Risk Management System",
                       className="text-white-50")
            ], className="bg-primary p-4 rounded-bottom shadow mb-4")
        ], width=12)
    ]),

    # Tab System
    dbc.Tabs([
        # TAB 1: CLINICAL ENTRY
        dbc.Tab(label="PATIENT ASSESSMENT", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Patient Input", className="mt-4 mb-3 fw-bold"),
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Age"), dbc.Input(
                                        id="input-age", type="number", className="mb-3"),
                                    html.Label("BMI"), dbc.Input(
                                        id="input-bmi", type="number", className="mb-3"),
                                    html.Label("Systolic BP"), dbc.Input(
                                        id="input-bp", type="number", className="mb-3"),
                                ]),
                                dbc.Col([
                                    html.Label(
                                        "Activity (Mins/Week)"), dbc.Input(id="input-act", type="number", className="mb-3"),
                                    html.Label(
                                        "Diet (1-10)"), dcc.Slider(1, 10, 1, value=5, id="input-diet"),
                                    dbc.Button("GENERATE DIAGNOSTIC", id="predict-btn",
                                               color="primary", className="mt-4 w-100 fw-bold")
                                ])
                            ])
                        ])
                    ], className="shadow border-0")
                ], width=5),

                # Dynamic Results Side
                dbc.Col([
                    html.H4("Risk Profile", className="mt-4 mb-3 fw-bold"),
                    html.Div(id="prediction-output")
                ], width=7)
            ])
        ]),

        # TAB 2: POPULATION INSIGHTS
        dbc.Tab(label="LIFESTYLE SEGMENTS", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Population Clustering", className="mt-4 fw-bold"),
                    html.P("Each dot represents a patient. Your current patient is highlighted with a star.",
                           className="text-muted small"),
                    dcc.Graph(id='cluster-graph')
                ], width=8),
                dbc.Col([
                    html.H4("Risk Drivers", className="mt-4 fw-bold"),
                    html.P("What causes the most risk in this population?",
                           className="text-muted small"),
                    dcc.Graph(id='shap-graph')
                ], width=4)
            ])
        ])
    ], className="mt-2"),

], fluid=True)


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    [State("input-age", "value"), State("input-bmi", "value")],
    prevent_initial_call=True
)
def run_prediction(n_clicks, age, bmi):
    if not age or not bmi:
        return dbc.Alert("Please enter core data to begin.", color="warning")

    risk_score = (age * 0.5) + (bmi * 1.2)  # Dummy math for now
    risk_level = "High" if risk_score > 60 else "Stable"
    color = COLORS["danger"] if risk_level == "High" else COLORS["primary"]

    return html.Div([
        dbc.Row([
            dbc.Col(make_metric_card("Risk Level", risk_level, color)),
            dbc.Col(make_metric_card("Risk Score",
                    f"{int(risk_score)}%", color)),
            dbc.Col(make_metric_card(
                "Segment", "High Risk Lifestyle", COLORS["warning"])),
        ], className="mb-4"),
        dbc.Alert([
            html.H5("Clinical Note:", className="alert-heading"),
            html.P(
                f"This patient exhibits a {risk_level.lower()} risk profile. We recommend focusing on BMI reduction and lifestyle intervention.")
        ], color="light", className="border-start border-4 shadow-sm")
    ])


if __name__ == '__main__':
    app.run_server(debug=True)
