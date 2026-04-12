import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

# Initialize the app with a professional theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server  # Required for Render deployment

app.layout = dbc.Container([
    # Header Section
    dbc.Row([
        dbc.Col(html.H1("BC Analytics: Diabetes Decision Support",
                className="text-center text-primary mb-4"), width=12)
    ], className="mt-4"),

    # Tabs for different CRISP-DM outputs
    dbc.Tabs([
        dbc.Tab(label="Patient Risk Assessment", tab_id="tab-risk"),
        dbc.Tab(label="Lifestyle Segmentation (k=3)", tab_id="tab-segments"),
        dbc.Tab(label="Key Driver Analysis (SHAP)", tab_id="tab-shap"),
    ], id="tabs", active_tab="tab-risk"),

    html.Div(id="tab-content", className="p-4")
], fluid=True)

if __name__ == '__main__':
    app.run_server(debug=True)
