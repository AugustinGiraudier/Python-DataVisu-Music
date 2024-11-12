import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import statsmodels.api as sm

# Load and prepare the dataset
data = pd.read_csv("dataset/dataset_filtered.csv")

# Keep only relevant columns and drop rows with non-numeric or NaN values in the required columns
data_filtered = data[['track', 'spotify_playlists', 'streams', 'artist_name', 'released_year']].dropna()
data_filtered = data_filtered[data_filtered['streams'].apply(lambda x: str(x).isdigit())]
data_filtered = data_filtered[data_filtered['spotify_playlists'].apply(lambda x: str(x).isdigit())]
data_filtered = data_filtered[data_filtered['released_year'].apply(lambda x: str(x).isdigit())]

# Convert columns to appropriate data types
data_filtered['spotify_playlists'] = data_filtered['spotify_playlists'].astype(int)
data_filtered['streams'] = data_filtered['streams'].astype(int)
data_filtered['released_year'] = data_filtered['released_year'].astype(int)

# Perform a linear regression to add a regression line
X = sm.add_constant(data_filtered['spotify_playlists'])
model = sm.OLS(data_filtered['streams'], X).fit()
data_filtered['regression_line'] = model.predict(X)

# Create a Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Spotify Songs: Playlists vs Streams by Release Year"),
    dcc.Graph(id='scatter-plot')
])

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('scatter-plot', 'id')
)
def update_graph(_):
    fig = px.scatter(
        data_filtered,
        x='spotify_playlists',
        y='streams',
        color='released_year',
        color_continuous_scale=["yellow", "orange", "purple"],
        hover_data={
            'track': True,
            'artist_name': True,
            'spotify_playlists': True,
            'streams': True,
            'released_year': True
        },
        labels={
            'released_year': 'Released Year',
            'spotify_playlists': 'Number of Playlists',
            'streams': 'Number of Streams'
        },
        title="Relation between Spotify Playlists and Streams by Release Year"
    )

    # Add regression line
    fig.add_scatter(
        x=data_filtered['spotify_playlists'],
        y=data_filtered['regression_line'],
        mode='lines',
        name='Regression Line',
        line=dict(color='red', dash='dash')
    )
    
    fig.update_layout(
        xaxis_title="Number of Playlists",
        yaxis_title="Number of Streams",
        hovermode="closest"
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
