import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
from plotly.subplots import make_subplots
from scipy.spatial.distance import euclidean
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
# DATA PREPROCESSING FOR THE THIRD TAB
####################################################################################################

# Define the columns to keep from the original DataFrame
columns_to_keep = ['code', 'product_name_es', 'energy-kcal_value', 'fat_value', 'saturated-fat_value', 'carbohydrates_value', 'sugars_value', 'fiber_value', 'proteins_value', 'salt_value', 'sodium_value', 'energy_value', 'off:nutriscore_score', 'off:ecoscore_score', 'off:nutriscore_grade', 'countries_list']

# Aggregate the data for each product, concatenating the countries list
df_3_tab = df.groupby('code')[columns_to_keep].agg({
    'product_name_es': 'first',
    'energy-kcal_value': 'first',
    'fat_value': 'first',
    'saturated-fat_value': 'first',
    'carbohydrates_value': 'first',
    'sugars_value': 'first',
    'fiber_value': 'first',
    'proteins_value': 'first',
    'salt_value': 'first',
    'sodium_value': 'first',
    'energy_value': 'first',
    'off:nutriscore_score': 'first',
    'off:ecoscore_score': 'first',
    'off:nutriscore_grade': 'first',
    'countries_list': lambda x: ', '.join(x)
}).reset_index()

# Load the clustering data CSV file
clustering_data = pd.read_csv('cereal_clustering_data.csv')

# Merge the clustering data with the df_3_tab based on the product code column.
df_3_tab = df_3_tab.merge(clustering_data[['code', 'cluster']], on='code', how='left')


# Now, df_3_tab contains one row per product with aggregated information

# Define the columns for similarity, these are the same columns that were used for clustering
columns_for_clustering = [
    'energy-kcal_value', 'fat_value', 'saturated-fat_value', 'carbohydrates_value',
    'sugars_value', 'fiber_value', 'proteins_value', 'salt_value', 'sodium_value',
    'energy_value', 'off:nutriscore_score', 'off:ecoscore_score']

# Define the Euclidean distance function for calculating similarity
def calculate_similarity_scores(dframe, product_code, dimensions):
    # Filter DataFrame to include only rows without missing values in the selected dimensions
    filtered_dframe = dframe.dropna(subset=dimensions)
    
    # Calculate similarity scores based on selected dimensions
    product_values = dframe[dframe['code'] == product_code][dimensions].values[0]
    filtered_dframe['similarity_score'] = filtered_dframe.apply(lambda row: euclidean(row[dimensions], product_values), axis=1)
    
    return filtered_dframe

####################################################################################################
# CREATE DASH APP STRUCTURE
####################################################################################################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define layout with multiple tabs
app.layout = dbc.Container([
    html.Img(src='/assets/image.jpg', className='img'),
    dbc.Tabs([
        dbc.Tab(label='Dashboard and Data Description', children=[
            html.Div([
                html.Div([
                    html.H1('Dashboard and Data Description'),
                    html.P('Welcome to our Nutritional Analysis Dashboard! This interactive tool is designed to provide insights into the nutritional content of cereal products and encourage data-driven food decisions!'),
                    html.Hr(),
                    html.H2('Tab 0: Dashboard and Data Description'),
                    html.P('In the current tab, you\'ll find an overview of the dashboard\'s functionality and the underlying dataset. The data used in this dashboard is sourced from the OpenFoodFacts website, a collaborative project that collects and shares nutritional information about food products worldwide. Specifically, the dataset includes products from the "Cereals" category that are available at least in Spain. However, many of these products are also sold in other countries, which increases the diversity of the dataset.'),
                    html.H2('Tab 1: Brand Nutritional Analysis'),
                    html.P('This tab presents a dashboard to analyze the sugar, fat, and NutriScore grade values by brand. The bar chart displays the average sugar content per brand, sorted from the brands with the highest average to the lowest. With the slider in the lower part of the tab one can select specific rank (e.g., 20th) and the bar chart will display the brands ranked 20th through 29th. By clicking on a specific bar, you can explore the fat content of products from that brand on a box plot to the right of the bar chart. At the same time all the products of the selected brand are plotted according to their sugar vs. fat content on a scatter plot in the lower part of the dashboard. The scatter plot uses the NutriScore color scale to represent the healthiness of the products. As you might guess, most of the time the products high in fat and sugar, are also low in the nutriscore grade. However, is it always the case? Explore the dashboard to find out!'),
                    html.H2('Tab 2: NutriScore and Nutrient Geographical Distribution'),
                    html.P('Here, you can dive deeper into the distribution of NutriScore and various nutrients across different countries. The choropleth map displays the average value of the nutrient or NutriScore grade selected through the dropdown for each country. The histogram on the right side of the tab shows the distribution of the selected nutrient across all products and all countries. By clicking on a specific country on the choropleth map, you can explore the distribution of the selected nutrient for that country on the histogram. This feature allows you to compare the nutritional quality of cereals across different countries in more detail.'),
                    html.H2('Tab 3: Clustering and Suggestions'),
                    html.P('This tab provides insights into product clustering based on the nutritional attributes. You can select dimensions for the X and Y axes and visualize the clustering patterns using a scatter plot. Additionally, you can search for specific products using the name search bar to narrow down the search and finally selecting the needed product from a dropdown. The dashboard will display the top-3 products that are similar to the selected one in terms of nutritional values and hava the same or even higher NutriScore grade. This feature allows you to discover alternative products with equal or higher health ratings compared to your current product of choice.')
                ], className='content')
            ], className='container')
        ]),
        dbc.Tab(label='Brand Nutritional Analysis', children=[
            html.Div([
                html.H1('Brand Nutritional Analysis'),
                dcc.Graph(id='nutritional-analysis'),
                html.Label('Select the rank of the brand by sugar content, this brand and the 9 ranked directly after it will be displayed on the bar chart', style={'font-weight': 'bold', 'margin-top': '10px', 'margin-bottom': '5px'}),
                dcc.Slider(
                    id='brand-slider',
                    min=0,
                    max=len(avg_sugar_by_brand) - 10,
                    step=1,
                    value=0,
                    marks={i: str(i) for i in range(0, len(avg_sugar_by_brand), 50)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.P('Interesting Insight', style={'font-weight': 'bold', 'margin-top': '10px', 'margin-bottom': '5px'}),
                html.P('Try selecting Carrefour, ranking number 260 according to the average sugar content. You can see that this brand has quite a few products that are high in either sugar or fat and are still graded as A or B (which is super healthy). That undermines the common understanding that high sugar and fat content always leads to a product being unhealthy.'),
                html.Div(style={'margin-bottom': '100px'})
            ])
        ]),
        dbc.Tab(label='NutriScore and Nutrient Geographical Distribution', children=[
            html.H1('NutriScore and Nutrient Geographical Distribution'),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Graph(id='nutriscore-map'),
                        html.Label('Select Variable:', style={'font-weight': 'bold', 'margin-top': '10px', 'margin-bottom': '5px'}),
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
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='nutrient-histogram')
                ], width=4)
            ]),
            html.P('Interesting Insight', style={'font-weight': 'bold', 'margin-top': '10px', 'margin-bottom': '5px'}),
            html.P('Interestingly, the US is not among the top countries in average sugar or fat, contrary to the stereotypes. And, underlining the complexity of working with real-life data, India (with only one product) has both the best NutriScore average and the worst mode NutriScore grade.'),
            html.P('Also, quite unexpectedly, Algeria and Saudi Arabia have products that are by far the highest in proteins, fiber and energy, while being low in sugar and carbohydrates.'),
            html.P('Finally, as expected, most of the countries for which more data is available, are consuming the cereals of mostly A NutriScore grade'),
            html.Div(style={'margin-bottom': '100px'})   
        ]),
        dcc.Tab(label='Clustering and Suggestions', children=[
        html.Div([
            html.H1('Clustering and Suggestions'),
            html.H2('Bivariate Cluster Visualization'),
            # Scatter plot for cluster visualization
            dcc.Graph(id='cluster-scatter-plot'),
            
            html.Div([
                html.Label('Select X Axis Dimension:', style={'font-weight': 'bold', 'margin-top': '10px', 'margin-bottom': '5px'}),
                dcc.Dropdown(
                    id='x-axis-dimension',
                    options=[
                        {'label': 'Energy (kcal)', 'value': 'energy-kcal_value'},
                        {'label': 'Fat', 'value': 'fat_value'},
                        {'label': 'Saturated Fat', 'value': 'saturated-fat_value'},
                        {'label': 'Carbohydrates', 'value': 'carbohydrates_value'},
                        {'label': 'Sugars', 'value': 'sugars_value'},
                        {'label': 'Proteins', 'value': 'proteins_value'},
                        {'label': 'Salt', 'value': 'salt_value'},
                        {'label': 'Sodium', 'value': 'sodium_value'},
                        {'label': 'Energy (kj)', 'value': 'energy_value'},
                        {'label': 'Nutriscore Score', 'value': 'off:nutriscore_score'},
                        {'label': 'Ecoscore Score', 'value': 'off:ecoscore_score'},
                        {'label': 'NutriScore Grade', 'value': 'off:nutriscore_grade'}
                    ],
                    value='energy-kcal_value'  # Default value
                ),
                html.Label('Select Y Axis Dimension:', style={'font-weight': 'bold', 'margin-top': '10px', 'margin-bottom': '5px'}),
                dcc.Dropdown(
                    id='y-axis-dimension',
                    options=[
                        {'label': 'Energy (kcal)', 'value': 'energy-kcal_value'},
                        {'label': 'Fat', 'value': 'fat_value'},
                        {'label': 'Saturated Fat', 'value': 'saturated-fat_value'},
                        {'label': 'Carbohydrates', 'value': 'carbohydrates_value'},
                        {'label': 'Sugars', 'value': 'sugars_value'},
                        {'label': 'Proteins', 'value': 'proteins_value'},
                        {'label': 'Salt', 'value': 'salt_value'},
                        {'label': 'Sodium', 'value': 'sodium_value'},
                        {'label': 'Energy (kJ)', 'value': 'energy_value'},
                        {'label': 'Nutriscore Score', 'value': 'off:nutriscore_score'},
                        {'label': 'Ecoscore Score', 'value': 'off:ecoscore_score'},
                        {'label': 'NutriScore Grade', 'value': 'off:nutriscore_grade'}
                    ],
                    value='fat_value'  # Default value
                ),
            ]),
            
            html.Div(id='hover-info'),  # Placeholder for hover information
            
            html.P('Interesting Insight', style={'font-weight': 'bold', 'margin-top': '10px', 'margin-bottom': '5px'}),
            html.P('The clusters mostly vary according to the energy content and main nutrients like fat or carbohyrdates. Overall, there is a cluster that is low in energy and fat, the one that is higher in energy, but still low in fat, and the other just grow the amount of energy proportionally to the amount of fat. Logically, the cluster that is high and energy and fat, does not need much sugar (as kcal come from both sugar and fat), and to contrast, the cluster with relatively low fat but high energy has to get it from somewhere, and therefore is pretty high in sugars. Overall, as expected, cereal products are quite similar in terms of salt and sodium content.'),
            

            html.Hr(),  # Add a horizontal line

            html.Div([
                html.H2("Find Similar Products"),
                dcc.Input(id='search-input', type='text', placeholder='Enter product name...'),
                html.Div(style={'margin-bottom': '20px'}),
                dcc.Dropdown(id='product-dropdown', placeholder='Select a product...'),
                html.Div(style={'margin-bottom': '20px'}),
                html.Div(id='product-details')
            ]),
            
            # Placeholder for displaying similar products
            html.Div(id='similar-products')
            ])
            ])
    ])
], fluid=True)


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
# CALLBACK FUNCTIONS FOR THE THIRD TAB
####################################################################################################

# Define callback to update scatter plot based on selected dimensions
@app.callback(
    Output('cluster-scatter-plot', 'figure'),
    [Input('x-axis-dimension', 'value'), Input('y-axis-dimension', 'value')]
)
def update_scatter_plot(x_axis_dimension, y_axis_dimension):
    # Use the selected dimensions to create the scatter plot
    scatter_plot_trace = go.Scatter(
        x=df_3_tab[x_axis_dimension],
        y=df_3_tab[y_axis_dimension],
        mode='markers',
        marker=dict(color=df_3_tab['cluster']),
        text=df_3_tab['product_name_es'],
        hoverinfo='text+x+y',
        name='Cluster',
    )

    layout = go.Layout(
        xaxis=dict(title=x_axis_dimension),
        yaxis=dict(title=y_axis_dimension),
        template='plotly'
    )

    fig = go.Figure(data=[scatter_plot_trace], layout=layout)
    return fig

# Define callback to update the dropdown options based on the search input
@app.callback(
    Output('product-dropdown', 'options'),
    [Input('search-input', 'value')]
)
def update_product_dropdown(search_input):
    if search_input is None:
        return []
    elif search_input == '':
        return [{'label': 'Start typing to search...', 'value': ''}]
    else:
        # Fill missing values with an empty string
        df_3_tab['product_name_es'] = df_3_tab['product_name_es'].fillna('')
        
        # Filter the DataFrame based on the search input
        filtered_df = df_3_tab[df_3_tab['product_name_es'].str.contains(search_input, case=False)]
        
        # Generate options for the dropdown
        options = [{'label': row['product_name_es'], 'value': row['code']} for index, row in filtered_df.iterrows()]
        return options


# Define callback to update the product details based on the selected product code
@app.callback(
    Output('product-details', 'children'),
    [Input('product-dropdown', 'value')]
)
def update_product_details(product_code):
    if product_code:
        # Look up product details in the DataFrame based on the product code
        product_details = df_3_tab[df_3_tab['code'] == product_code].iloc[0]

        # Extract relevant information from the product details
        product_name = product_details['product_name_es']
        nutritional_info = {
            'Energy (kcal)': product_details['energy-kcal_value'],
            'Fat': product_details['fat_value'],
            'Saturated Fat': product_details['saturated-fat_value'],
            'Carbohydrates': product_details['carbohydrates_value'],
            'Sugars': product_details['sugars_value'],
            'Fiber': product_details['fiber_value'],
            'Proteins': product_details['proteins_value']
        }

        countries_available = product_details['countries_list']

        # Split the string by comma and capitalize the first letter of each word
        countries_available = [country.strip().capitalize() for country in countries_available.split(',')]

        nutriscore = product_details['off:nutriscore_grade'].upper()

        # Construct HTML elements to display the product details
        product_details_html = html.Div([
            html.H3('Details of Selected Product'),
            html.Div([
                html.P(f'{product_name}', style={'font-weight': 'bold'}),
                html.Ul([
                    html.Li(f'{key}: {value}') for key, value in nutritional_info.items()
                ]),
                html.P(f'Countries Available: {", ".join(countries_available)}'),
                html.P(f'Nutriscore Grade: {nutriscore}')
            ], className = 'content')
        ])
    else:
        # If no product is selected, display a message
        product_details_html = html.Div([
            html.H3('Details of Selected Product'),
            html.P('No product is currently selected. Please select a product from the dropdown.'),
            html.Div(style={'margin-bottom': '100px'})
        ])

    return product_details_html


@app.callback(
    Output('similar-products', 'children'),
    [Input('product-dropdown', 'value')]
)
def update_similar_products(product_code):
    if product_code:
        # Calculate similarity scores based on clustering dimensions
        similarity_scores_df = calculate_similarity_scores(df_3_tab, product_code, columns_for_clustering)
        
        # Filter products with better or equal nutriscore grade and non-empty Spanish names
        similar_products = similarity_scores_df[
            (similarity_scores_df['code'] != product_code) & 
            (similarity_scores_df['off:nutriscore_grade'] <= similarity_scores_df[similarity_scores_df['code'] == product_code]['off:nutriscore_grade'].values[0]) &
            (~similarity_scores_df['product_name_es'].isnull())  # Only include products with non-empty Spanish names
        ]
        
        # Sort similar products by similarity score
        similar_products = similar_products.sort_values(by='similarity_score').head(3)
        
        # Create HTML elements to display similar products
        similar_products_html = html.Div([
            html.H3('Similar Products'),
            html.Div([
                html.Div([
                            html.P(f'{row["product_name_es"]}', style={'font-weight': 'bold'}),
                            html.Ul([
                                    html.Li(f'Energy (kcal): {row["energy-kcal_value"]}'),
                                    html.Li(f'Fat: {row["fat_value"]}'),
                                    html.Li(f'Saturated Fat: {row["saturated-fat_value"]}'),
                                    html.Li(f'Carbohydrates: {row["carbohydrates_value"]}'),
                                    html.Li(f'Sugars: {row["sugars_value"]}'),
                                    html.Li(f'Fiber: {row["fiber_value"]}'),
                                    html.Li(f'Proteins: {row["proteins_value"]}')
                            ]),
                            html.P(f'Countries Available: {", ".join([country.strip().capitalize() for country in row["countries_list"].split(",")])}'),
                            html.P(f'Nutriscore Grade: {row["off:nutriscore_grade"].upper()}')
                        ], className = 'content', style={'display': 'inline-block', 'margin-right': '10px'})
            for index, row in similar_products.iterrows()
            ])
        ])
    else:
        similar_products_html = html.Div()
    
    return similar_products_html

####################################################################################################
# RUN THE DASH APP
####################################################################################################
if __name__ == '__main__':
    app.run_server(debug=True)


