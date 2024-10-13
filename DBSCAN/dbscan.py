import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class DBSCANClustering:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit(self, X):
        """Fit the DBSCAN model to the dataset X."""
        # Initialize all points as unvisited
        n_samples = len(X)
        self.labels = -1 * np.ones(n_samples)  # Initialize all points as noise (-1)

        cluster_label = 0

        for point_idx in range(n_samples):
            if self.labels[point_idx] != -1:
                # Skip if the point has already been labeled (visited)
                continue

            # Find neighbors of the current point
            neighbors = self._region_query(X, point_idx)

            if len(neighbors) < self.min_samples:
                # Mark as noise if not enough neighbors
                self.labels[point_idx] = -1  # Noise
            else:
                # Expand the cluster
                self._expand_cluster(X, point_idx, neighbors, cluster_label)
                cluster_label += 1

        return self.labels

    def _region_query(self, X, point_idx):
        """Find the neighbors of the point at point_idx using eps."""
        neighbors = []
        for idx, point in enumerate(X):
            if np.linalg.norm(X[point_idx] - point) <= self.eps:
                neighbors.append(idx)
        return neighbors

    def _expand_cluster(self, X, point_idx, neighbors, cluster_label):
        """Expand the cluster recursively."""
        self.labels[point_idx] = cluster_label  # Label the initial point
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if self.labels[neighbor_idx] == -1:
                # If point was labeled as noise, make it part of the cluster
                self.labels[neighbor_idx] = cluster_label

            elif self.labels[neighbor_idx] == -1 * np.ones(1):
                # If it's not visited, label it as part of the cluster
                self.labels[neighbor_idx] = cluster_label

                # Find neighbors of the current neighbor
                neighbor_neighbors = self._region_query(X, neighbor_idx)

                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors += neighbor_neighbors  # Add the new neighbors

            i += 1


# Load and scale the data (assuming the dataset is in 'disney_movies.csv')
file_path = 'disney_movies.csv'
df = pd.read_csv(file_path)
features = df[['Release Year', 'IMDb Rating']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Instantiate and fit DBSCAN
eps = 0.5  # The maximum distance between two points to be considered in the same neighborhood
min_samples = 5  # The minimum number of points required to form a dense region (cluster)
dbscan = DBSCANClustering(eps=eps, min_samples=min_samples)
labels = dbscan.fit(X_scaled)

# Assign labels to DataFrame
df['Cluster'] = labels

# Visualization
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap='viridis',
    label='Movies'
)
plt.title('DBSCAN Clustering of Disney Movies')
plt.xlabel('Release Year (standardized)')
plt.ylabel('IMDb Rating (standardized)')
plt.legend()
plt.colorbar(scatter, label='Cluster')
plt.show()

# Analyzing clusters
for cluster_num in np.unique(labels):
    if cluster_num == -1:
        print("\nNoise points:")
    else:
        print(f"\nCluster {cluster_num}:")
    cluster_movies = df[df['Cluster'] == cluster_num]
    print(cluster_movies[['Movie Name', 'Release Year', 'IMDb Rating']])