import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# Read the latitude and longitude data from the excel file
data = pd.read_excel('C:/Users/uqjwil54/OneDrive - The University of Queensland/DBBD/tools/counts.xlsx', sheet_name='Data')

# Extract latitude, longitude, and root columns
coordinates = data[['latitude', 'longitude']]
root_data = data['root']

# Standardize the data
scaler = StandardScaler()
scaled_coordinates = scaler.fit_transform(coordinates)

# Cluster data
def cluster(data, epsilon, min_samples, col_name):
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_coordinates)
    clusters = clusters + abs(clusters.min())

    # Add cluster labels to the DataFrame
    data[col_name] = clusters

    # Calculate the number of points in each cluster
    cluster_sizes = data[col_name].value_counts()

    # Calculate the cluster colour based on the most common contributor
    cluster_colours = data.groupby(col_name)['root'].unique()
    cluster_colours = [name[0] if len(name) == 1 else 'multiple' for name in cluster_colours]

    # Calculate average latitude and longitude for each cluster
    cluster_centers = data.groupby(col_name)[['latitude', 'longitude']].mean()

    return cluster_centers, cluster_sizes, cluster_colours

# Create clusters
centers, sizes, colours = cluster(data, 0.2, 5, 'coarse_cluster')

# Create a figure and gridspec layout
fig, ax = plt.subplots()
for spine in ax.spines.values(): spine.set_visible(False)

# Global figure
m = Basemap(projection='mill', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c', ax=ax)
m.drawcoastlines(color='grey')
m.fillcontinents(color='white', lake_color='white')

# Add clusters
a = 0.4
b = 3
for cluster_label, (avg_lat, avg_lon) in centers.iterrows():
    x, y = m(avg_lon, avg_lat)
    size = (sizes[cluster_label] ** a) * b
    m.scatter(x, y, s = size, color ='black', zorder=5)
    label = f"{sizes[cluster_label]:,.0f}"
    ax.text(x + 1000000, y - 400000, label, fontsize=8, weight='bold')

# Plot for size legend
# sizes = [10, 100, 1000, 10000]
# legend_labels = [str(i) for i in sizes]
# sizes = [(i**a)*b for i in sizes]

# size_handles = []
# for i, size in enumerate(sizes):
#     handle = plt.scatter([], [], s=size, marker='o', color='black', edgecolor='black', linewidth=1, label=legend_labels[i])
#     size_handles.append(handle)

# # Show size legend separately
# size_legend = plt.legend(loc='lower left', bbox_to_anchor=(0.0, 0.2), frameon=False, title='Instances', fontsize="7")

# Add the size legend to the axis
# plt.gca().add_artist(size_legend)

# Plot figure
plt.savefig('locations.png')
