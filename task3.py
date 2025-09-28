# --------------------------
# Install Required Packages
# --------------------------
!pip install dash jupyter-dash plotly seaborn pandas --quiet

# --------------------------
# Imports
# --------------------------
# from jupyter_dash import JupyterDash
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import seaborn as sns
import pandas as pd

# --------------------------
# Load Dataset (Iris)
# --------------------------
iris = sns.load_dataset("iris")

# --------------------------
# Initialize Dash app
# --------------------------
# app = JupyterDash(__name__)
app = Dash(__name__)
app.title = "Iris Dashboard"

# --------------------------
# App Layout
# --------------------------
app.layout = html.Div([
    html.H1("🌸 Iris Dataset Dashboard", style={"textAlign": "center"}),

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

    html.Div([
        dcc.Graph(id="scatter_plot", style={"width": "48%", "display": "inline-block"}),
        dcc.Graph(id="box_plot", style={"width": "48%", "display": "inline-block"})
    ]),

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
    df = iris if species == "all" else iris[iris["species"] == species]

    scatter_fig = px.scatter(
        df, x="sepal_length", y="sepal_width",
        color="species", size="petal_length", hover_data=["petal_width"],
        title="Sepal Length vs Width (Bubble = Petal Length)"
    )

    box_fig = px.box(
        df, x="species", y="petal_length", color="species",
        title="Distribution of Petal Length"
    )

    bar_df = df.groupby("species").mean(numeric_only=True).reset_index()
    bar_fig = px.bar(
        bar_df, x="species", y="petal_width", color="species",
        title="Average Petal Width by Species"
    )

    insights = []
    if not df.empty:
        max_species = df.groupby("species")["petal_length"].mean().idxmax()
        insights.append(f"🌟 Species with longest petals on average: {max_species.title()}")
        widest = df.groupby("species")["sepal_width"].mean().idxmax()
        insights.append(f"🌟 Species with widest sepals: {widest.title()}")
        count = df['species'].value_counts().to_dict()
        insights.append(f"🌟 Sample counts: {count}")


    return scatter_fig, box_fig, bar_fig, html.Div([
        html.H3("Insights"),
        html.Ul([html.Li(i) for i in insights])
    ])

# --------------------------
# Run app in browser tab from Colab
# --------------------------
# Use mode="inline" for displaying in the notebook or remove mode for default behavior
app.run(debug=True)
