import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import statsmodels.api as sm
import colorsys
import networkx as nx
import re
from itertools import combinations

# Initialize the Dash app
app = dash.Dash(__name__)

# Load and prepare data
def load_and_prepare_data(file_path='dataset/dataset_filtered.csv'):
    df = pd.read_csv(file_path)
    df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
    df = df.dropna(subset=['streams'])
    
    def extract_artists(artist_string):
        artist_string = re.sub(r'\s*(feat\.|ft\.|&|,|and)\s*', ',', str(artist_string), flags=re.IGNORECASE)
        return [a.strip() for a in artist_string.split(',')]
    
    df['artists_list'] = df['artist_name'].apply(extract_artists)
    return df

def build_graph(df, top_n_artists=500):
    artist_streams_total = df.explode('artists_list').groupby('artists_list')['streams'].sum()
    stream_threshold = artist_streams_total.quantile(0.5)
    top_artists = artist_streams_total[artist_streams_total >= stream_threshold].index
    
    df_filtered = df[df['artists_list'].apply(lambda artists: sum(artist in top_artists for artist in artists) >= 1)]
    
    edges = []
    for idx, row in df_filtered.iterrows():
        artists = row['artists_list']
        streams = row['streams']
        track = row['track']
        if len(artists) >= 1:
            for pair in combinations(artists, 2):
                if pair[0] in top_artists and pair[1] in top_artists:
                    edges.append({
                        'artist1': pair[0],
                        'artist2': pair[1],
                        'streams': streams,
                        'track_name': track,
                        'weight': streams
                    })
    
    edges_df = pd.DataFrame(edges)
    edges_df['streams'] = pd.to_numeric(edges_df['streams'], errors='coerce')
    edges_df = edges_df.dropna(subset=['streams'])
    
    G = nx.Graph()
    for idx, row in edges_df.iterrows():
        artist1, artist2 = row['artist1'], row['artist2']
        if G.has_edge(artist1, artist2):
            G[artist1][artist2]['streams'] += row['streams']
            G[artist1][artist2]['weight'] += row['weight']
            G[artist1][artist2]['tracks'].append(row['track_name'])
        else:
            G.add_edge(artist1, artist2, 
                      streams=row['streams'],
                      weight=row['weight'],
                      tracks=[row['track_name']])
    
    return G, artist_streams_total

# Load data and build graph
df = load_and_prepare_data()
G, artist_streams_total = build_graph(df)

def create_musical_characteristics():
    # Load and prepare data
    df = pd.read_csv('dataset/dataset_filtered.csv')
    df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
    df = df.dropna(subset=['streams'])
    
    # Aggregate data by year for musical characteristics
    df_agg = df.groupby(['released_year', 'released_month'])[['danceability', 'energy', 'acousticness', 'valence']].mean().reset_index()
    
    return html.Div([
        html.Div([
            html.H1("Caractéristiques Musicales des Titres", style={'textAlign': 'center', 'color': '#2c3e50'}),
            html.P("Explorez les caractéristiques musicales des titres et leur popularité.",
                   style={'textAlign': 'center', 'color': '#7f8c8d'})
        ], style=styles['header']),
        
        # Scatter plot
        dcc.Graph(
            figure=px.scatter(
                df,
                x='danceability',
                y='energy',
                size='streams',
                color='streams',
                hover_name='track',
                hover_data=['artist_name', 'streams'],
                color_continuous_scale='Viridis',
                title='Distribution des Caractéristiques Musicales',
                labels={'danceability': 'Dansabilité', 'energy': 'Énergie'},
                template='plotly_white'
            ).update_layout(
                coloraxis_colorbar=dict(title='Streams', tickformat=',.0f'),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
        ),
        
        html.Div([
            html.H2("Évolution des Caractéristiques Musicales", 
                    style={'textAlign': 'center', 'marginTop': '40px', 'color': '#2c3e50'}),
            html.P("Analysez l'évolution temporelle des caractéristiques musicales.",
                   style={'textAlign': 'center', 'color': '#7f8c8d'})
        ]),
        
        # Year range slider
        html.Div([
            html.Label("Sélectionnez une période :", style={'marginTop': '20px'}),
            dcc.RangeSlider(
                id='year-slider',
                min=df_agg['released_year'].min(),
                max=df_agg['released_year'].max(),
                step=1,
                marks={str(year): str(year) for year in range(
                    df_agg['released_year'].min(), 
                    df_agg['released_year'].max() + 1, 
                    5
                )},
                value=[df_agg['released_year'].min(), df_agg['released_year'].max()]
            )
        ], style={'margin': '40px 20px'}),
        
        # Stacked area chart
        dcc.Graph(id='stacked-area-chart')
    ])

# Add new callback for the stacked area chart
@app.callback(
    Output('stacked-area-chart', 'figure'),
    Input('year-slider', 'value')
)

def update_stacked_area_chart(selected_years):
    df = pd.read_csv('dataset/dataset_filtered.csv')
    df_agg = df.groupby(['released_year', 'released_month'])[['danceability', 'energy', 'acousticness', 'valence']].mean().reset_index()
    
    # Filter data for selected years
    filtered_df = df_agg[
        (df_agg['released_year'] >= selected_years[0]) & 
        (df_agg['released_year'] <= selected_years[1])
    ]

    fig = go.Figure()

    # Color scheme
    colors = {
        'danceability': 'rgb(99, 110, 250)',
        'energy': 'rgb(239, 85, 59)',
        'acousticness': 'rgb(0, 204, 150)',
        'valence': 'rgb(255, 161, 90)'
    }
    
    # Add stacked area traces
    for feature in ['danceability', 'energy', 'acousticness', 'valence']:
        # Translate feature names
        feature_names = {
            'danceability': 'Dansabilité',
            'energy': 'Énergie',
            'acousticness': 'Acoustique',
            'valence': 'Valence'
        }
        
        fig.add_trace(go.Scatter(
            x=filtered_df['released_year'],
            y=filtered_df[feature],
            mode='lines',
            stackgroup='one',
            name=feature_names[feature],
            line=dict(color=colors[feature])
        ))

        # Add monthly average points
        monthly_avg = df.groupby(['released_year', 'released_month'])[feature].mean().reset_index()
        monthly_avg_filtered = monthly_avg[
            (monthly_avg['released_year'] >= selected_years[0]) & 
            (monthly_avg['released_year'] <= selected_years[1])
        ]

        fig.add_trace(go.Scatter(
            x=monthly_avg_filtered['released_year'] + monthly_avg_filtered['released_month'] / 12,
            y=monthly_avg_filtered[feature],
            mode='markers',
            name=f'{feature_names[feature]} (Moyenne Mensuelle)',
            marker=dict(size=8, color=colors[feature], symbol='circle'),
            showlegend=False
        ))

    fig.update_layout(
        title="Évolution des Caractéristiques Musicales au Fil du Temps",
        xaxis_title="Année",
        yaxis_title="Moyenne des attributs",
        template="plotly_white",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

def create_playlist_analysis():
    # Load and prepare data
    df = pd.read_csv('dataset/dataset_filtered.csv')
    
    # Filter data
    df_filtered = df[['track', 'spotify_playlists', 'streams', 'artist_name', 'released_year']].dropna()
    df_filtered = df_filtered[df_filtered['streams'].apply(lambda x: str(x).isdigit())]
    df_filtered = df_filtered[df_filtered['spotify_playlists'].apply(lambda x: str(x).isdigit())]
    df_filtered = df_filtered[df_filtered['released_year'].apply(lambda x: str(x).isdigit())]
    
    # Convert columns to appropriate types
    df_filtered['spotify_playlists'] = df_filtered['spotify_playlists'].astype(int)
    df_filtered['streams'] = df_filtered['streams'].astype('int64')
    df_filtered['released_year'] = df_filtered['released_year'].astype(int)
    
    # Calculate regression line
    X = sm.add_constant(df_filtered['spotify_playlists'])
    model = sm.OLS(df_filtered['streams'], X).fit()
    df_filtered['regression_line'] = model.predict(X)
    
    return html.Div([
        html.Div([
            html.H1("Analyse des Playlists", style={'textAlign': 'center', 'color': '#2c3e50'}),
            html.P("Analysez la relation entre le nombre de playlists Spotify et le nombre de streams par année de sortie.",
                   style={'textAlign': 'center', 'color': '#7f8c8d'})
        ], style=styles['header']),
        
        html.Div([
            dcc.Graph(id='playlist-scatter-plot')
        ], style={'marginTop': '20px'})
    ], style=styles['container'])

# Add this callback after the other callbacks
@app.callback(
    Output('playlist-scatter-plot', 'figure'),
    Input('playlist-scatter-plot', 'id')
)
def update_playlist_graph(_):
    # Load and prepare data
    df = pd.read_csv('dataset/dataset_filtered.csv')
    
    # Filter data
    df_filtered = df[['track', 'spotify_playlists', 'streams', 'artist_name', 'released_year']].dropna()
    df_filtered = df_filtered[df_filtered['streams'].apply(lambda x: str(x).isdigit())]
    df_filtered = df_filtered[df_filtered['spotify_playlists'].apply(lambda x: str(x).isdigit())]
    df_filtered = df_filtered[df_filtered['released_year'].apply(lambda x: str(x).isdigit())]
    
    # Convert columns to appropriate types
    df_filtered['spotify_playlists'] = df_filtered['spotify_playlists'].astype(int)
    df_filtered['streams'] = df_filtered['streams'].astype('int64')
    df_filtered['released_year'] = df_filtered['released_year'].astype(int)
    
    # Calculate regression line
    X = sm.add_constant(df_filtered['spotify_playlists'])
    model = sm.OLS(df_filtered['streams'], X).fit()
    df_filtered['regression_line'] = model.predict(X)
    
    fig = px.scatter(
        df_filtered,
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
            'released_year': 'Année de sortie',
            'spotify_playlists': 'Nombre de playlists',
            'streams': 'Nombre de streams'
        },
        title="Relation entre le nombre de playlists Spotify et le nombre de streams par année de sortie"
    )

    # Add regression line
    fig.add_scatter(
        x=df_filtered['spotify_playlists'],
        y=df_filtered['regression_line'],
        mode='lines',
        name='Ligne de régression',
        line=dict(color='red', dash='dash')
    )

    fig.update_layout(
        xaxis_title="Nombre de Playlists",
        yaxis_title="Nombre de Streams",
        hovermode="closest",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black')
    )

    return fig
def get_season_color(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'rgb(135,206,235)'  # Winter
    elif month in [3, 4, 5]:
        return 'rgb(143,206,0)'    # Spring
    elif month in [6, 7, 8]:
        return 'rgb(255,206,58)'   # Summer
    return 'rgb(206,126,0)'        # Autumn

def create_temporal_analysis():
    return html.Div([
        html.Div([
            html.H1("Analyse Temporelle", style={'textAlign': 'center', 'color': '#2c3e50'}),
            html.P("Analysez l'évolution des streams en fonction du temps et des saisons.",
                   style={'textAlign': 'center', 'color': '#7f8c8d'})
        ], style=styles['header']),
        
        html.Div([
            html.Div([
                html.Label("Seuil minimum de streams (en millions) :"),
                dcc.Slider(
                    id='stream-threshold',
                    min=1,
                    max=100,
                    value=10,
                    marks={i: f'{i}M' for i in range(0, 101, 10)},
                    step=1
                )
            ], style={'marginBottom': '20px'}),
            
            html.Div([
                html.Label("Plage de dates :"),
                dcc.RangeSlider(
                    id='date-range',
                    min=2020,
                    max=2024,
                    value=[2020, 2024],
                    marks={i: str(i) for i in range(2020, 2025)},
                    step=1
                )
            ], style={'marginBottom': '20px'}),
            
            dcc.Graph(id='temporal-streams-graph')
        ])
    ], style=styles['container'])

@app.callback(
    Output('temporal-streams-graph', 'figure'),
    [
        Input('stream-threshold', 'value'),
        Input('date-range', 'value')
    ]
)
def update_temporal_graph(stream_threshold, date_range):
    # Load and prepare data
    df = pd.read_csv('dataset/dataset_filtered.csv')
    df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
    df['date'] = pd.to_datetime(dict(
        year=df['released_year'],
        month=df['released_month'],
        day=df['released_day']
    ))
    
    # Filter based on streams threshold and date range
    min_streams = stream_threshold * 1_000_000
    df_filtered = df[df['streams'] >= min_streams]
    df_filtered = df_filtered[
        (df_filtered['released_year'] >= date_range[0]) & 
        (df_filtered['released_year'] <= date_range[1])
    ]
    
    # Aggregate monthly streams
    monthly_streams = df_filtered.groupby(
        ['date', 'track', 'artist_name', 'artist_count']
    )['streams'].sum().reset_index()
    
    # Calculate visualization parameters
    monthly_streams['color'] = monthly_streams['date'].apply(get_season_color)
    artist_count_range = monthly_streams['artist_count'].agg(['min', 'max'])
    marker_sizes = 10 + (monthly_streams['artist_count'] - artist_count_range['min']) / (
        artist_count_range['max'] - artist_count_range['min']
    ) * 30
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=monthly_streams['date'],
        y=monthly_streams['streams'],
        mode='markers',
        marker=dict(
            color=monthly_streams['color'],
            size=marker_sizes,
            opacity=0.7,
            line=dict(width=1, color='#2c3e50')
        ),
        hovertemplate=(
            "Date: %{x|%d/%m/%Y}<br>"
            "Streams: %{y:,.0f}<br>"
            "Titre: %{customdata[0]}<br>"
            "Artiste: %{customdata[1]}<br>"
            "Nombre d'artistes: %{customdata[2]}<extra></extra>"
        ),
        customdata=monthly_streams[['track', 'artist_name', 'artist_count']].values
    ))
    
    # Add season legend
    seasons = [
        ("Hiver", 'rgb(135,206,235)'),
        ("Printemps", 'rgb(143,206,0)'),
        ("Été", 'rgb(255,206,58)'),
        ("Automne", 'rgb(206,126,0)')
    ]
    
    for i, (name, color) in enumerate(seasons):
        fig.add_shape(
            type="rect",
            x0=-0.24, x1=-0.2,
            y0=1.05 - i * 0.12,
            y1=1.0 - i * 0.12,
            fillcolor=color,
            line=dict(width=0),
            xref="paper", yref="paper"
        )
        fig.add_annotation(
            text=name,
            x=-0.18, y=1.025 - i * 0.12,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(color='black', size=12),
            align="left"
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Dynamique des streams: Influence des saisons et des artistes",
            font=dict(size=20),
            x=0.5, y=0.95
        ),
        xaxis=dict(
            title="Date",
            gridcolor='lightgray',
            showgrid=True,
            tickformat='%d/%m/%Y',
        ),
        yaxis=dict(
            title="Nombre de streams",
            gridcolor='lightgray',
            showgrid=True,
            type='log',
            range=[np.log10(min_streams), np.log10(df_filtered['streams'].max())],
        ),
        template='plotly_white',
        hovermode='closest',
        height=800,
        margin=dict(t=100, l=100, r=50, b=50)
    )
    
    return fig

def create_color_gradient(value, max_value=100):
    percentage = value / max_value
    h, s = 210, 100  # Constant hue (blue) and saturation
    l = 90 - (percentage * 60)  # Light to dark gradient
    rgb = [int(x * 255) for x in colorsys.hls_to_rgb(h/360, l/100, s/100)]
    return f'rgb({rgb[0]},{rgb[1]},{rgb[2]})'

def create_bpm_liveness():
    # Load and prepare data for BPM and Liveness analysis
    df = pd.read_csv('dataset/dataset_filtered.csv')
    for col in ['liveness', 'bpm', 'streams']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['date'] = pd.to_datetime(dict(
        year=df['released_year'],
        month=df['released_month'],
        day=df['released_day']
    ))

    # Filter for popular tracks
    popular_tracks = df.groupby('track')['streams'].sum() >= 10_000_000
    df_filtered = df[df['track'].isin(popular_tracks[popular_tracks].index)].copy()

    grid_style = {
        'display': 'grid',
        'gridTemplateColumns': 'repeat(10, 1fr)',
        'gridTemplateRows': 'repeat(10, 1fr)',
        'gap': '3px',
        'width': '100%',
        'height': '300px',
        'margin': 'auto',
        'backgroundColor': '#f0f0f0',
        'padding': '10px',
        'borderRadius': '10px',
        'boxSizing': 'border-box'
    }

    grid_cell_style = {
        'width': '100%',
        'height': '100%',
        'position': 'relative',
        'border': '1px solid white'
    }

    return html.Div([
        html.Div([
            html.H1("BPM et Liveness : Clés de la Popularité Musicale", 
                    style={'textAlign': 'center', 'color': '#2c3e50'}),
            html.P("Analysez l'impact des BPM et de la liveness sur le succès des hits musicaux.",
                   style={'textAlign': 'center', 'color': '#7f8c8d'})
        ], style=styles['header']),
        
        html.Div([
            # BPM Section
            html.Div([
                html.H3("Filtre BPM (min-max)"),
                dcc.RangeSlider(
                    id='bpm-slider',
                    min=0, max=210, step=1, value=[0, 210],
                    marks={i: str(i) for i in [0, 70, 140, 210]}
                ),
                dcc.Graph(id='bpm-gauge', style={'height': '300px'}),
                html.Div(id='bpm-text', style={'textAlign': 'center', 'fontSize': '1.2em', 'marginTop': '10px'})
            ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
            
            # Liveness Section
            html.Div([
                html.H3("Filtre Liveness (min-max)"),
                dcc.RangeSlider(
                    id='liveness-slider',
                    min=0, max=100, step=1, value=[0, 100],
                    marks={i: f'{i}%' for i in range(0, 101, 25)}
                ),
                html.Div(id='liveness-grid', style=grid_style),
                html.Div(id='liveness-text', style={'marginTop': '20px', 'fontSize': '1.2em'})
            ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'})
        ], style={'display': 'flex', 'alignItems': 'center'}),
        
        dcc.Graph(id='streams-chart'),
        html.Div(id='filter-info', style={
            'marginTop': '20px',
            'padding': '10px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '5px',
            'textAlign': 'center'
        })
    ], style=styles['container'])

# Add new callback for BPM and Liveness visualization
@app.callback(
    [Output('streams-chart', 'figure'),
     Output('bpm-gauge', 'figure'),
     Output('liveness-grid', 'children'),
     Output('bpm-text', 'children'),
     Output('liveness-text', 'children'),
     Output('filter-info', 'children')],
    [Input('bpm-slider', 'value'),
     Input('liveness-slider', 'value')]
)
def update_bpm_liveness_visuals(bpm_range, liveness_range):
    # Load and filter data
    df = pd.read_csv('dataset/dataset_filtered.csv')
    for col in ['liveness', 'bpm', 'streams']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['date'] = pd.to_datetime(dict(
        year=df['released_year'],
        month=df['released_month'],
        day=df['released_day']
    ))

    popular_tracks = df.groupby('track')['streams'].sum() >= 10_000_000
    df_filtered = df[df['track'].isin(popular_tracks[popular_tracks].index)].copy()
    
    # Filter data based on selected ranges
    current_filter = df_filtered[
        (df_filtered['bpm'].between(bpm_range[0], bpm_range[1])) &
        (df_filtered['liveness'].between(liveness_range[0], liveness_range[1]))
    ]
    
    monthly_streams = current_filter.groupby(
        ['date', 'track', 'artist_name', 'bpm', 'liveness']
    )['streams'].sum().reset_index()

    # Create streams chart
    fig_streams = go.Figure()
    if not monthly_streams.empty:
        fig_streams.add_trace(go.Scatter(
            x=monthly_streams['date'],
            y=monthly_streams['streams'],
            mode='markers',
            marker=dict(color='#2E86C1', size=8, opacity=0.7),
            hovertemplate=(
                "Date: %{x|%d/%m/%Y}<br>"
                "Streams: %{y:,.0f}<br>"
                "Titre: %{customdata[0]}<br>"
                "Artiste: %{customdata[1]}<extra></extra>"
            ),
            customdata=monthly_streams[['track', 'artist_name']].values
        ))
    
    fig_streams.update_layout(
        title="Dynamique des streams (filtrés)",
        xaxis_title="Date",
        yaxis_title="Nombre de streams (échelle logarithmique)",
        yaxis_type='log',
        xaxis_range=['2020-01-01', monthly_streams['date'].max().strftime('%Y-%m-%d')] 
            if not monthly_streams.empty else None,
        height=400,
        template='plotly_white'
    )

    # Create BPM gauge
    bpm_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sum(bpm_range) / 2,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 210]},
            'bar': {'color': "#FFFFFF"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 70], 'color': '#87ceeb'},
                {'range': [70, 140], 'color': '#2E86C1'},
                {'range': [140, 210], 'color': '#1B4F72'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': bpm_range[1]
            }
        }
    ))
    bpm_fig.update_layout(font={'size': 16}, margin=dict(l=20, r=20, t=50, b=20), height=300)

    # Create liveness grid
    grid_squares = [
        html.Div(style={
            'width': '100%',
            'height': '100%',
            'position': 'relative',
            'border': '1px solid white',
            'backgroundColor': create_color_gradient(i) if liveness_range[0] <= i <= liveness_range[1] else '#E0E0E0'
        }) for i in range(100)
    ]

    return (
        fig_streams,
        bpm_fig,
        grid_squares,
        f"BPM Range: {bpm_range[0]} - {bpm_range[1]}",
        f"Liveness Range: {liveness_range[0]}% - {liveness_range[1]}%",
        f"Affichage de {len(current_filter['track'].unique())} morceaux correspondant aux critères sélectionnés"
    )

def adjust_positions(pos, min_distance):
    adjusted_pos = pos.copy()
    nodes = list(pos.keys())
    
    for _ in range(25): 
        moved = False
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                dx = adjusted_pos[node1][0] - adjusted_pos[node2][0]
                dy = adjusted_pos[node1][1] - adjusted_pos[node2][1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < min_distance:
                    force = (min_distance - distance) / distance
                    move_x = dx * force * 0.5
                    move_y = dy * force * 0.5
                    
                    adjusted_pos[node1][0] += move_x
                    adjusted_pos[node1][1] += move_y
                    adjusted_pos[node2][0] -= move_x
                    adjusted_pos[node2][1] -= move_y
                    moved = True
        
        if not moved:
            break
    
    return adjusted_pos

# Styles
styles = {
    'container': {
        'max-width': '1200px',
        'margin': '0 auto',
        'padding': '20px'
    },
    'header': {
        'backgroundColor': '#f8f9fa',
        'padding': '20px',
        'borderRadius': '10px',
        'marginBottom': '20px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    },
    'controls': {
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'marginBottom': '20px'
    }
}

# Main app layout
app.layout = html.Div([
    html.H1("Spotify Analytics Dashboard", style={'textAlign': 'center'}),
    
    dcc.Tabs([
        dcc.Tab(label='Réseau des Collaborations', children=[
            html.Div([
                html.Div([
                    html.H1("Réseau des Collaborations entre Artistes", 
                            style={'textAlign': 'center', 'color': '#2c3e50'}),
                    html.P("Explorez les collaborations musicales entre artistes et découvrez leurs connexions.",
                           style={'textAlign': 'center', 'color': '#7f8c8d'})
                ], style=styles['header']),
                
                html.Div([
                    html.Div([
                        html.Label("Rechercher un artiste:", 
                                  style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='artist-search',
                            options=[{'label': artist, 'value': artist} for artist in sorted(G.nodes())],
                            value=None,
                            placeholder='Sélectionnez un artiste...',
                            style={'width': '300px'}
                        )
                    ], style={'marginBottom': '20px'})
                ], style=styles['controls']),
                
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='network-graph',
                            style={'height': '700px'},
                            config={
                                'scrollZoom': True,
                                'displayModeBar': True,
                                'modeBarButtonsToAdd': ['select2d', 'lasso2d']
                            }
                        )
                    ], style={'flex': '3'}),
                    
                    html.Div([
                        html.H3("Statistiques", style={'textAlign': 'center', 'color': '#2c3e50'}),
                        html.Div(id='stats-panel', children=[
                            html.Div(id='selected-artist-stats'),
                            html.Hr(style={'margin': '20px 0'}),
                            html.Div(id='general-stats')
                        ], style={'padding': '20px'})
                    ], style={'flex': '1', 'backgroundColor': '#f8f9fa', 'margin': '0 0 0 20px', 'borderRadius': '10px'})
                ], style={'display': 'flex', 'marginBottom': '20px'}),
                
                html.Div([
                    html.Label(
                        'Seuil minimum de streams :',
                        style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}
                    ),
                    dcc.Slider(
                        id='stream-threshold-slider',
                        min=0,
                        max=int(max(artist_streams_total)),
                        value=int(max(artist_streams_total) * 0.1),
                        marks={
                            int(i): f"{int(i/1e6)}M" 
                            for i in np.linspace(0, int(max(artist_streams_total)), 6)
                        },
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    )
                ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
            ], style=styles['container'])
        ]),
        dcc.Tab(label='Caractéristiques Musicales', children=[
            create_musical_characteristics()
        ]),
        dcc.Tab(label='Analyse des Playlists', children=[
            create_playlist_analysis()
        ]),
        dcc.Tab(label='Analyse Temporelle', children=[
            create_temporal_analysis()
        ]),
        dcc.Tab(label='BPM & Liveness', children=[
            create_bpm_liveness()
        ])
    ])
])

@app.callback(
    [
        Output('network-graph', 'figure'),
        Output('selected-artist-stats', 'children'),
        Output('general-stats', 'children')
    ],
    [
        Input('stream-threshold-slider', 'value'),
        Input('artist-search', 'value')
    ],
)
def update_visualization(threshold, selected_artist):
    # Filtrage du graphe
    G_filtered = G.copy()
    nodes_to_remove = [
        node for node in G_filtered.nodes()
        if artist_streams_total.get(node, 0) < threshold
    ]
    G_filtered.remove_nodes_from(nodes_to_remove)
    
    # Création du sous-graphe pour l'artiste sélectionné
    if selected_artist and selected_artist in G_filtered:
        neighbors = list(G_filtered.neighbors(selected_artist))
        G_filtered = G_filtered.subgraph([selected_artist] + neighbors)
    
    # Vérification si le graphe n'est pas vide
    if len(G_filtered.nodes()) > 0:
        pos = nx.spring_layout(
            G_filtered,
            k=2.0 if not selected_artist else 1.5,
            iterations=100 if not selected_artist else 50,
            weight='weight'
        )
        
        pos = adjust_positions(pos, min_distance=0.2 if selected_artist else 0.1)
        
        edge_traces = []
        streams_list = [G_filtered[u][v]['streams'] for u, v in G_filtered.edges()]
        
        if streams_list:
            max_streams = max(streams_list)
            min_streams = min(streams_list)
            stream_range = max_streams - min_streams if max_streams != min_streams else 1
            
            for edge in G_filtered.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                streams = edge[2]['streams']
                
                width = 0.5 + (streams - min_streams) / stream_range * 2
                color_intensity = 0.1 + (streams - min_streams) / stream_range * 0.4
                
                if selected_artist and (edge[0] == selected_artist or edge[1] == selected_artist):
                    edge_color = f'rgba(255, 0, 0, {color_intensity * 2})'
                else:
                    edge_color = f'rgba(0, 0, 200, {color_intensity})'
                
                x_hover = np.linspace(x0, x1, 50)
                y_hover = np.linspace(y0, y1, 50)
                
                tracks_info = edge[2]['tracks']
                hover_text = (
                    f"{edge[0]} & {edge[1]}<br>"
                    f"Streams: {streams:,.0f}<br>"
                    f"Collaborations: {len(tracks_info)}<br>"
                    f"Titres: {', '.join(tracks_info[:3])}"
                    f"{'...' if len(tracks_info) > 3 else ''}"
                )
                
                edge_traces.extend([
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        line=dict(width=width, color=edge_color),
                        hoverinfo='skip',
                        mode='lines'
                    ),
                    go.Scatter(
                        x=x_hover,
                        y=y_hover,
                        mode='lines',
                        line=dict(width=10, color='rgba(0,0,0,0)'),
                        hoverinfo='text',
                        text=hover_text,
                        showlegend=False
                    )
                ])
        
        node_x, node_y, node_size, node_color, node_text, node_hovertext = [], [], [], [], [], []
        
        streams_values = [max(artist_streams_total.get(node, 1), 1) for node in G_filtered.nodes()]
        log_streams = np.log10(streams_values)
        min_log = min(log_streams)
        max_log = max(log_streams)
        size_range = 45
        min_size = 5
        
        for node in G_filtered.nodes():
            x = pos[node][0] + np.random.uniform(-0.02, 0.02)
            y = pos[node][1] + np.random.uniform(-0.02, 0.02)
            node_x.append(x)
            node_y.append(y)
            
            streams = max(artist_streams_total.get(node, 1), 1)
            log_size = np.log10(streams)
            
            if max_log > min_log:
                normalized_size = min_size + size_range * ((log_size - min_log) / (max_log - min_log)) ** 2
            else:
                normalized_size = min_size
            
            if node == selected_artist:
                normalized_size *= 1.5
            
            node_size.append(normalized_size)
            node_color.append(streams)
            
            show_this_label = node == selected_artist or (
                selected_artist and 
                selected_artist in G_filtered and 
                node in G_filtered.neighbors(selected_artist)
            )
            node_text.append(node if show_this_label else '')
            
            hover_text = (
                f"{'🎤 ' if node == selected_artist else ''}{node}<br>"
                f"Streams totaux: {streams:,.0f}<br>"
                f"Collaborateurs: {len(list(G_filtered.neighbors(node)))}"
            )
            node_hovertext.append(hover_text)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            hoverinfo='text',
            hovertext=node_hovertext,
            marker=dict(
                colorscale='Viridis',
                color=node_color,
                size=node_size,
                line=dict(color='black', width=1),
                showscale=True,
                colorbar=dict(
                    title='Streams totaux',
                    titleside='right',
                    tickformat=',.0f',
                    ticksuffix=' streams',
                    thickness=20
                )
            )
        )
        
        selected_stats = []
        if selected_artist:
            artist_streams = artist_streams_total.get(selected_artist, 0)
            if artist_streams >= threshold:
                neighbors = list(G_filtered.neighbors(selected_artist))
                total_collabs = sum(len(G_filtered[selected_artist][neighbor]['tracks']) 
                                  for neighbor in neighbors)
                
                selected_stats = html.Div([
                    html.H4(f"Statistiques de {selected_artist}"),
                    html.Ul([
                        html.Li(f"Streams totaux: {artist_streams:,.0f}"),
                        html.Li(f"Nombre de collaborateurs: {len(neighbors)}"),
                        html.Li(f"Nombre total de collaborations: {total_collabs}"),
                        html.Li(
                            f"Collaborateur principal: {max(neighbors, key=lambda x: G_filtered[selected_artist][x]['streams']) if neighbors else 'Aucun'}"
                        )
                    ], style={'listStyleType': 'none', 'padding': '0'})
                ])
            else:
                selected_stats = html.Div([
                    html.H4(f"Statistiques de {selected_artist}"),
                    html.P(f"Cet artiste a {artist_streams:,} streams. Réglez le seuil en dessous de cette valeur pour le voir.",
                          style={'color': 'red'})
                ])
        
        general_stats = html.Div([
            html.H4("Statistiques globales"),
            html.Ul([
                html.Li(f"Nombre d'artistes visibles: {len(G_filtered.nodes())}"),
                html.Li(f"Nombre de collaborations: {len(G_filtered.edges())}"),
                html.Li(
                    f"Moyenne de streams: {np.mean([artist_streams_total.get(node, 0) for node in G_filtered.nodes()]):,.0f}"
                )
            ], style={'listStyleType': 'none', 'padding': '0'})
        ])
        
        layout = go.Layout(
            title=dict(
                text=f"{'Réseau de ' + selected_artist if selected_artist else 'Réseau complet des collaborations'}",
                x=0.5,
                font=dict(color='black')
            ),
            paper_bgcolor='white',
            plot_bgcolor='white',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, color='black'),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, color='black'),
            annotations=[
                dict(
                    text="Utilisez la molette pour zoomer",
                    showarrow=False,
                    x=0.5,
                    y=1.1,
                    xref='paper',
                    yref='paper',
                    font=dict(color='black', size=12)
                )
            ]
        )
        
        return go.Figure(data=edge_traces + [node_trace], layout=layout), selected_stats, general_stats
    
    return go.Figure(), "Aucun artiste sélectionné", "Aucune donnée à afficher"

if __name__ == '__main__':
    app.run_server(debug=True)
