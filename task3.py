"""
Internship Task - 3: Dashboard Development
Tool: Plotly Dash
Deliverable: Interactive dashboard with actionable insights
Dataset: Iris (classification dataset from seaborn)
"""

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import seaborn as sns
import pandas as pd

# --------------------------
# Load dataset (Iris)
# --------------------------
iris = sns.load_dataset("iris")

# --------------------------
# Initialize Dash app
# --------------------------
app = dash.Dash(__name__)
app.title = "Iris Dashboard"

# --------------------------
# App Layout
# --------------------------
app.layout = html.Div([
    html.H1("🌸 Iris Dataset Dashboard", style={"textAlign": "center"}),

    # Dropdown filter
    html.Div([
        html.Label("Select Species:"),
        dcc.Dropdown(
            id="species_filter",
            options=[{"label": s.title(), "value": s} for s in iris["species"].unique()] + [{"label": "All", "value": "all"}],
            value="all",
            clearable=False,
            style={"width": "40%"}
        ),
    ], style={"textAlign": "center"}),

    # Charts row
    html.Div([
        dcc.Graph(id="scatter_plot", style={"width": "48%", "display": "inline-block"}),
        dcc.Graph(id="box_plot", style={"width": "48%", "display": "inline-block"})
    ]),

    # Bar chart + insights
    html.Div([
        dcc.Graph(id="bar_chart", style={"width": "70%", "display": "inline-block"}),
        html.Div(id="insight_box", style={"width": "28%", "display": "inline-block", "verticalAlign": "top", "padding": "20px"})
    ])
])

# --------------------------
# Callbacks
# --------------------------
@app.callback(
    [Output("scatter_plot", "figure"),
     Output("box_plot", "figure"),
     Output("bar_chart", "figure"),
     Output("insight_box", "children")],
    [Input("species_filter", "value")]
)
def update_dashboard(species):
    # Filter dataset
    df = iris if species == "all" else iris[iris["species"] == species]

    # Scatter plot
    scatter_fig = px.scatter(
        df, x="sepal_length", y="sepal_width",
        color="species", size="petal_length", hover_data=["petal_width"],
        title="Sepal Length vs Width (Bubble = Petal Length)"
    )

    # Box plot
    box_fig = px.box(
        df, x="species", y="petal_length", color="species",
        title="Distribution of Petal Length"
    )

    # Bar chart (mean values)
    bar_fig = df.groupby("species").mean(numeric_only=True).reset_index()
    bar_fig = px.bar(
        bar_fig, x="species", y="petal_width", color="species",
        title="Average Petal Width by Species"
    )

    # Generate insight text
    insights = []
    if not df.empty:
        max_species = df.groupby("species")["petal_length"].mean().idxmax()
        insights.append(f"🌟 Species with longest petals on average: **{max_species.title()}**")
        widest = df.groupby("species")["sepal_width"].mean().idxmax()
        insights.append(f"🌟 Species with widest sepals: **{widest.title()}**")
        count = df['species'].value_counts().to_dict()
        insights.append(f"🌟 Sample counts: {count}")

    return scatter_fig, box_fig, bar_fig, html.Div([
        html.H3("Insights"),
        html.Ul([html.Li(i) for i in insights])
    ])

# --------------------------
# Run app
# --------------------------
if __name__ == "__main__":
    app.run_server(debug=True)
