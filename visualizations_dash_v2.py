import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
from plotly.subplots import make_subplots
####################################################################################################
# LOAD AND PREPARE THE DATA
####################################################################################################
df = pd.read_csv('cereal_spain_cleaned_v0.csv')

# Filter out rows where units are not present and unify units for specified columns
unit_conversion = {'g': 1, 'mg': 0.001, 'µg': 0.000001, 'kcal': 4.184, 'kJ': 1}

# Filter rows where units are not present
unit_columns = [
    'saturated-fat_unit', 'carbohydrates_unit', 'sugars_unit',
    'proteins_unit', 'salt_unit', 'energy_unit', 'sodium_unit',
    'energy-kcal_unit', 'fat_unit'
]

for column in unit_columns:
    df = df[~df[column].isna()]

# Unify units for specified columns
df['salt_value'] = df['salt_value'] * df['salt_unit'].map(unit_conversion).fillna(1)
df['energy_value'] = df['energy_value'] * df['energy_unit'].map(unit_conversion).fillna(1)
df['sodium_value'] = df['sodium_value'] * df['sodium_unit'].map(unit_conversion).fillna(1)

# Drop NaNs from other columns
nutrient_columns = [
    'sugars_value', 'energy_value', 'fat_value', 'saturated-fat_value',
    'carbohydrates_value', 'fiber_value', 'proteins_value', 'salt_value',
    'sodium_value', 'energy_value', 'off:nutriscore_score', 'off:nutriscore_grade'
]

df = df.dropna(subset=nutrient_columns)
####################################################################################################
# DATA PREPROCESSING FOR THE FIRST TAB
####################################################################################################
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
####################################################################################################

####################################################################################################
# DATA PREPROCESSING FOR THE SECOND TAB
####################################################################################################

# Define the country mapping
country_mapping = {
    "fr:belgien": "belgium",
    "fr:deutschland": "germany",
    "fr:frankreich": "france",
    "fr:schweiz": "switzerland",
    "es:cantabria": "spain",
    "es:santona": "spain",
    "ca:franca": "france",
    "francia-espana": "spain",
    "fr:belgie": "belgium",
    "francia": "france",
    "cote-d-ivoire": "ivory-coast",
    "fr:francia": "france",
    "fr:belgica": "belgium",
    "en": "unknown",
    "alemania": "germany",
    "south-korea": "south-korea",
    "fr:frankrijk": "france",  # Added mapping for "fr:frankrijk"
    # Add more mappings as needed
}

# Map nutriscore grades to numerical values
grade_mapping = {'a': 5, 'b': 4, 'c': 3, 'd': 2, 'e': 1}

# Split the countries_tags column by comma to get a list of countries for each product
df['countries_list'] = df['countries_tags'].str.split(',')

# Explode the list of countries into separate rows
df = df.explode('countries_list')

# Remove the 'en:' prefix from each country
df['countries_list'] = df['countries_list'].str.replace('en:', '')

# Apply the mapping to unify the country names
df['countries_list'] = df['countries_list'].apply(lambda x: country_mapping.get(x, x))

# Remove rows with specified values
df = df[~df['countries_list'].isin(['french-polynesia', 'guadeloupe', 'martinique', 'reunion', 'new-caledonia', 'world', 'unknown'])]

# Filter the columns to select only nutrient columns
nutrient_columns = [
    'sugars_value', 'energy-kcal_value', 'fat_value', 'saturated-fat_value',
    'carbohydrates_value', 'fiber_value', 'proteins_value', 'salt_value',
    'sodium_value', 'energy_value', 'off:nutriscore_score', 'off:nutriscore_grade'
]

# Define numerical and categorical nutrient columns
numerical_nutrient_columns = [col for col in nutrient_columns if col != 'off:nutriscore_grade']

# Calculate the average for numerical columns, dropping missing values
country_nutrient_avg_numerical = df.groupby('countries_list')[numerical_nutrient_columns].mean().reset_index().dropna()

# Calculate the mode for the categorical 'nutriscore_grade' column
country_nutrient_mode_categorical = df.groupby('countries_list')['off:nutriscore_grade'].agg(lambda x: x.mode()[0]).reset_index()

# Map nutriscore grades to numerical values
country_nutrient_mode_categorical['nutriscore_grade_numeric'] = country_nutrient_mode_categorical['off:nutriscore_grade'].map(grade_mapping)

# Merge the numerical and categorical dataframes
country_nutrient_avg = pd.merge(country_nutrient_avg_numerical, country_nutrient_mode_categorical, on='countries_list')

# Group the data by country and aggregate the distribution of each nutrient
country_nutrient_distribution = df.groupby('countries_list')[nutrient_columns].agg(list).reset_index()


####################################################################################################

####################################################################################################
# CREATE DASH APP STRUCTURE
####################################################################################################
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
        ]),
        dbc.Tab(label='NutriScore and Nutrient Distribution', children=[
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Graph(id='nutriscore-map'),
                        html.Label('Select Variable:', style={'text-align': 'center'}),
                        dcc.Dropdown(
                            id='nutrient-dropdown',
                            options=[
                                {'label': 'Sugars', 'value': 'sugars_value'},
                                {'label': 'Energy (kcal)', 'value': 'energy-kcal_value'},
                                {'label': 'Fat', 'value': 'fat_value'},
                                {'label': 'Saturated Fat', 'value': 'saturated-fat_value'},
                                {'label': 'Carbohydrates', 'value': 'carbohydrates_value'},
                                {'label': 'Fiber', 'value': 'fiber_value'},
                                {'label': 'Proteins', 'value': 'proteins_value'},
                                {'label': 'Salt', 'value': 'salt_value'},
                                {'label': 'Sodium', 'value': 'sodium_value'},
                                {'label': 'Energy (kJ)', 'value': 'energy_value'},
                                {'label': 'NutriScore Score', 'value': 'off:nutriscore_score'},
                                {'label': 'NutriScore Grade', 'value': 'off:nutriscore_grade'}
                            ],
                            value='sugars_value'
                        )
                    ])
                ], width=8),
                dbc.Col([
                    dcc.Graph(id='nutrient-histogram')
                ], width=4)
            ])
        ])
    ])
])


####################################################################################################
# CALLBACK FUNCTION FOR THE FIRST TAB
####################################################################################################


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
    fig.update_xaxes(title_text="", row=1, col=2)  # Leave x-axis label empty for the box plot
    fig.update_yaxes(title_text="Fat Content (g)", row=1, col=2)  # Update y-axis label for the box plot
    fig.update_yaxes(title_text=f"Average Sugar Content ({sugars_unit})", row=1, col=1)
    fig.update_xaxes(title_text=f"Sugar Content ({sugars_unit})", row=2, col=1)
    fig.update_yaxes(title_text=f"Fat Content ({fat_unit})", row=2, col=1)




    return fig

####################################################################################################
# CALLBACK FUNCTIONS FOR THE SECOND TAB
####################################################################################################

# Callback to update the choropleth map based on selected nutrient
@app.callback(
    Output('nutriscore-map', 'figure'),
    Input('nutrient-dropdown', 'value')
)
def update_nutriscore_map(selected_nutrient):
    
    # Define data for the choropleth map
    if selected_nutrient == 'off:nutriscore_grade':
        avg_nutrient_by_country = country_nutrient_avg[['countries_list', 'off:nutriscore_grade']]
        z = country_nutrient_avg['nutriscore_grade_numeric']  # Use the numerical values for coloring
        colorbar_title = 'Nutriscore Grade'
        hover_text = ['Mode Nutriscore Grade: {}'.format(grade) for grade in country_nutrient_avg['off:nutriscore_grade']]
    else:
        avg_nutrient_by_country = country_nutrient_avg[['countries_list', selected_nutrient]]
        z = avg_nutrient_by_country[selected_nutrient]
        colorbar_title = selected_nutrient.replace('_value', '').replace('-', ' ').title()
        hover_text = [f'{selected_nutrient.replace("_value", "").replace("-", " ").title()}: {value:.2f}' for value in avg_nutrient_by_country[selected_nutrient]]

    nutriscore_map_trace = go.Choropleth(
        locations=avg_nutrient_by_country['countries_list'],
        z=z,
        locationmode='country names',
        colorscale='Viridis' if selected_nutrient != 'off:nutriscore_grade' else 'Viridis',
        colorbar_title=colorbar_title,
        hoverinfo='text',
        text=hover_text
    )
    
    # Set colorbar labels to Nutriscore grades
    if selected_nutrient == 'off:nutriscore_grade':
        nutriscore_map_trace.colorbar.tickvals = [1, 2, 3, 4, 5]  # Numerical values corresponding to grades
        nutriscore_map_trace.colorbar.ticktext = [str(grade) for grade in ['E', 'D', 'C', 'B', 'A']]  # Corresponding grade labels


    # Define layout for the choropleth map
    nutriscore_map_layout = go.Layout(
        title='Average {} by Country'.format(selected_nutrient.replace('_value', '').replace('-', ' ').replace('_', ' ').replace('off:', '').title()),
        geo=dict(showframe=False, 
                 projection={'type': 'mercator'}, 
                 showland=True, 
                 showcoastlines=True,
                 lataxis=dict(range=[-60, 90])
        ),
        height=600,  # Set map height
        width=800   # Set map width
    )

    return {'data': [nutriscore_map_trace], 'layout': nutriscore_map_layout}


# Callback to update the histogram based on selected nutrient
@app.callback(
    Output('nutrient-histogram', 'figure'),
    [Input('nutrient-dropdown', 'value'),
     Input('nutriscore-map', 'clickData')]
)
def update_histogram(selected_nutrient, clickData):
    # Filter data based on selected country from map
    if clickData:
        country = clickData['points'][0]['location']
        filtered_df = df[df['countries_list'] == country]
    else:
        filtered_df = df

    nutrient_values = filtered_df[selected_nutrient].dropna()

    # Create histogram trace
    histogram_trace = go.Histogram(
        x=nutrient_values,
        nbinsx=50,
        marker_color='blue',
        opacity=0.75
    )

    # Create layout for histogram
    histogram_layout = go.Layout(
        title='Distribution of {}'.format(selected_nutrient.replace('_value', '').replace('-', ' ').replace('_', ' ').replace('off:', '').title()),
        xaxis_title=selected_nutrient.replace('_value', '').replace('-', ' ').replace('_', ' ').replace('off:', '').title(),
        yaxis_title='Count',
        bargap=0.2,
        height=400,  # Set histogram height
        width=800   # Set histogram width        
    )

    return {'data': [histogram_trace], 'layout': histogram_layout}



####################################################################################################
# RUN THE DASH APP
####################################################################################################
if __name__ == '__main__':
    app.run_server(debug=True)


