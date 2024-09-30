import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
from plotly.subplots import make_subplots

# Load the data
df = pd.read_csv('cereal_spain_cleaned_v0.csv')

# Filtering necessary columns
data = df[['brands', 'sugars_value', 'fat_value', 'off:nutriscore_grade']]
data.dropna(inplace=True)

# Calculate average sugar content per brand
avg_sugar_by_brand = data.groupby('brands')['sugars_value'].mean().reset_index()

# Sort by sugar content
avg_sugar_by_brand = avg_sugar_by_brand.sort_values(by='sugars_value', ascending=False)

# Extract units
fat_unit = df['fat_unit'].dropna().unique()[0]
sugars_unit = df['sugars_unit'].dropna().unique()[0]

# Mapping NutriScore to numeric values
nutriscore_to_numeric = {'a': 5, 'b': 4, 'c': 3, 'd': 2, 'e': 1}
data['nutriscore_numeric'] = data['off:nutriscore_grade'].map(nutriscore_to_numeric)

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define layout with multiple tabs
app.layout = dbc.Container([
    dbc.Tabs([
        dbc.Tab(label='Dashboard and Data Description', children=[
            html.Div([
                html.H1('Dashboard and Data Description'),
                html.P('This section will contain the dashboard and data description.')
            ])
        ]),
        dbc.Tab(label='Nutritional Analysis Dashboard', children=[
            html.Div([
                dcc.Graph(id='nutritional-analysis'),
                html.Label('Select the rank of the brand by sugar content, this brand and the 9 ranked directly after it will be displayed on the bar chart', style={'text-align': 'center'}),
                dcc.Slider(
                    id='brand-slider',
                    min=0,
                    max=len(avg_sugar_by_brand) - 10,
                    step=1,
                    value=0,
                    marks={i: str(i) for i in range(0, len(avg_sugar_by_brand), 50)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ])
    ])
])

@app.callback(
    Output('nutritional-analysis', 'figure'),
    Input('brand-slider', 'value'),
    Input('nutritional-analysis', 'clickData')
)
def update_dashboard(slider_value, click_data):
    # Update bar chart data
    bar_data = avg_sugar_by_brand.iloc[slider_value:slider_value + 10]

    # Create bar chart
    bar_chart = go.Bar(
        x=bar_data['brands'],
        y=bar_data['sugars_value'],
        name=f'Average Sugar Content ({sugars_unit})',
        marker=dict(color='blue')
    )

    # Create empty box plot
    box_plot = go.Box(
        y=[],
        name=f'Fat Content Distribution ({fat_unit})',
        marker=dict(color='red')
    )

    # Create empty scatter plot with color scale for NutriScore
    scatter_plot = go.Scatter(
        x=[],
        y=[],
        mode='markers',
        name=f'Sugar vs Fat Content ({sugars_unit} vs {fat_unit})',
        marker=dict(
            color=[],
            colorbar=dict(
                title='NutriScore',
                tickvals=[1, 2, 3, 4, 5],
                ticktext=['E', 'D', 'C', 'B', 'A'],
                len=0.5,
                y=0.2
            ),
            colorscale='RdYlGn',
            showscale=True,
            cmin=1,
            cmax=5
        ),
        text=[],
        showlegend=False
    )

    # Combine plots into a single interactive layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Average Sugar Content by Brand", "Fat Content Distribution", "", "Sugar vs Fat Content"),
        specs=[[{"type": "bar"}, {"type": "box"}], [{"colspan": 2}, None]],
        vertical_spacing=0.3
    )

    # Add traces to the figure
    fig.add_trace(bar_chart, row=1, col=1)
    fig.add_trace(box_plot, row=1, col=2)
    fig.add_trace(scatter_plot, row=2, col=1)

    # If a bar is clicked, update box plot and scatter plot
    if click_data:
        brand = click_data['points'][0]['x']
        filtered_data = data[data['brands'] == brand]
        colors = filtered_data['nutriscore_numeric']

        fig.data[1].y = filtered_data['fat_value']
        fig.data[2].x = filtered_data['sugars_value']
        fig.data[2].y = filtered_data['fat_value']
        fig.data[2].marker.color = colors
        fig.data[2].text = filtered_data['off:nutriscore_grade']

    # Update layout with legends and axis labels
    fig.update_layout(
        title_text='Nutritional Analysis Dashboard',
        height=800,
        showlegend=False,
        legend=dict(x=0.5, y=-0.1, orientation='h', xanchor='center', yanchor='top')
    )

    # Update axes titles for each subplot
    fig.update_xaxes(title_text="Brands", row=1, col=1)
    fig.update_yaxes(title_text=f"Average Sugar Content ({sugars_unit})", row=1, col=1)
    fig.update_yaxes(title_text="", row=1, col=2)  # Remove axis title for the box plot
    fig.update_xaxes(title_text=f"Sugar Content ({sugars_unit})", row=2, col=1)
    fig.update_yaxes(title_text=f"Fat Content ({fat_unit})", row=2, col=1)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
