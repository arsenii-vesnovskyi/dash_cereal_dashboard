import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('cereal_spain_cleaned_v0.csv')

# Select relevant columns for clustering
columns_for_clustering = [
    'energy-kcal_value', 'fat_value', 'saturated-fat_value', 'carbohydrates_value',
    'sugars_value', 'fiber_value', 'proteins_value', 'salt_value', 'sodium_value',
    'energy_value', 'off:nutriscore_score', 'off:ecoscore_score', 'off:nova_groups'
]

# After checking the percentage of the missing values in each columns,
# We eliminated the columns with more than 50% of missing values: fiber_value and nova_groups

columns_for_clustering = [
    'energy-kcal_value', 'fat_value', 'saturated-fat_value', 'carbohydrates_value',
    'sugars_value', 'proteins_value', 'salt_value', 'sodium_value',
    'energy_value', 'off:nutriscore_score', 'off:ecoscore_score'
]

# Create a dataframe with the selected columns
clustering_data = df[columns_for_clustering]

# Handle the rest of missing values by filling them with the mean of the column
# Note: this simplified approach is only used to facilitate the clustering process
# In a real-world scenario, we would need to handle missing values more carefully

clustering_data.fillna(clustering_data.mean(), inplace=True)

# Normalize the numerical columns
scaler = StandardScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data)

#Apply the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(clustering_data_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Based on the visual inspection of the elbow plot, we can choose the optimal number of clusters.
optimal_clusters = 5

# Apply K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=77)
clustering_data['cluster'] = kmeans.fit_predict(clustering_data_scaled)

# Ensure the indices are aligned
assert clustering_data.index.equals(df.index), "Indices are not aligned. The 'code' column may not correspond correctly."

# Add the 'code' column to the clustering data
clustering_data['code'] = df['code']

# Save the processed data to a new CSV file
clustering_data.to_csv('cereal_clustering_data.csv', index=False)
