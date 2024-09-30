# dash_nutrition_dashboard

This repository contains all the files needed to run the Dash application about the Cereals food category.

**Project Description**
Nutritional Analysis Dashboard is a prototype of an interactive tool, which is designed to provide insights into the nutritional content of cereal products and encourage data-driven food decisions.

Tab 0: Dashboard and Data Description

In the current tab, you'll find an overview of the dashboard's functionality and the underlying dataset. The data used in this dashboard is sourced from the OpenFoodFacts website, a collaborative project that collects and shares nutritional information about food products worldwide. Specifically, the dataset includes products from the "Cereals" category that are available at least in Spain. However, many of these products are also sold in other countries, which increases the diversity of the dataset.

Tab 1: Brand Nutritional Analysis

This tab presents a dashboard to analyze the sugar, fat, and NutriScore grade values by brand. The bar chart displays the average sugar content per brand, sorted from the brands with the highest average to the lowest. With the slider in the lower part of the tab one can select specific rank (e.g., 20th) and the bar chart will display the brands ranked 20th through 29th. By clicking on a specific bar, you can explore the fat content of products from that brand on a box plot to the right of the bar chart. At the same time all the products of the selected brand are plotted according to their sugar vs. fat content on a scatter plot in the lower part of the dashboard. The scatter plot uses the NutriScore color scale to represent the healthiness of the products. As you might guess, most of the time the products high in fat and sugar, are also low in the NutriScore grade. However, is it always the case? Explore the dashboard to find out!

Tab 2: NutriScore and Nutrient Geographical Distribution

Here, you can dive deeper into the distribution of NutriScore and various nutrients across different countries. The choropleth map displays the average value of the nutrient or NutriScore grade selected through the dropdown for each country. The histogram on the right side of the tab shows the distribution of the selected nutrient across all products and all countries. By clicking on a specific country on the choropleth map, you can explore the distribution of the selected nutrient for that country on the histogram. This feature allows you to compare the nutritional quality of cereals across different countries in more detail.

Tab 3: Clustering and Suggestions

This tab provides insights into product clustering based on the nutritional attributes. You can select dimensions for the X and Y axes and visualize the clustering patterns using a scatter plot. Additionally, you can search for specific products using the name search bar to narrow down the search and finally selecting the needed product from a dropdown. The dashboard will display the top-3 products that are similar to the selected one in terms of nutritional values and have the same or even higher NutriScore grade. This feature allows you to discover alternative products with equal or higher health ratings compared to your current product of choice.

**File Tree**
Folder "assets":
  * image.jpg - background image
  * styles.css - css styles for the web page

Default folder:
  * cereal_clustering_data.csv - data obtained from running k-means clustering on the dataset
  * cereal_spain.csv - initial raw csv obtained from the OpenFoodFacts
  * cereal_spain_cleaned_v0.csv - cleaned data
  * clustering_elbow_plot.png - elbow plot used to decide on the number of clusters
  * clustering_v0.py - Python script used to cluster products
  * initial_analysis_v0.py - Python script used to analyze the data structure and perform some basic cleaning and preprocessing
  * visualizations_dash_v0.py - first iteration of the dashboard
  * visualizations_dash_v1.py - second iteration of the dashboard
  * visualizations_dash_v2.py - third iteration of the dashboard
  * visualizations_dash_v3.py - fourth and final iteration of the dashboard
