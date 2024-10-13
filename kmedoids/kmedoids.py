import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class KMedoidsClustering:
    def __init__(self, k=5):
        self.k = k
        self.medoids = None

    @staticmethod
    def euclidean_distance(data_point, medoids):
        """Calculate the Euclidean distance between a data point and medoids."""
        return np.sqrt(np.sum((medoids - data_point) ** 2, axis=1))

    def fit(self, X, max_iter=200):
        """Fit the KMedoids model to the dataset X."""
        # randomly initialize medoids from the data points
        self.medoids = X[np.random.choice(len(X), self.k, replace=False)]

        for iteration in range(max_iter):
            # assign each data point to the nearest medoid
            labels = []
            for data_point in X:
                distances = self.euclidean_distance(data_point, self.medoids)
                cluster_num = np.argmin(distances)
                labels.append(cluster_num)
            labels = np.array(labels)

            new_medoids = []
            for i in range(self.k):
                points = X[labels == i]
                if len(points) == 0:
                    # If a cluster has no points, keep the current medoid
                    new_medoids.append(self.medoids[i])
                else:
                    # Calculate the total distance between each point in the cluster and all others
                    distances = np.sum([np.linalg.norm(points - point, axis=1) for point in points], axis=0)
                    new_medoid_idx = np.argmin(distances)  # The point with the minimum total distance
                    new_medoids.append(points[new_medoid_idx])
            new_medoids = np.array(new_medoids)

            # Check for convergence
            if np.allclose(self.medoids, new_medoids, atol=1e-4):
                print(f"Converged after {iteration + 1} iterations.")
                break
            self.medoids = new_medoids

        return labels


# Load and scale the data (assuming the dataset is in 'disney_movies.csv')
file_path = 'disney_movies.csv'
df = pd.read_csv(file_path)
features = df[['Release Year', 'IMDb Rating']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Instantiate and fit K-Medoids
k = 3  # Number of clusters
kmedoids = KMedoidsClustering(k=k)
labels = kmedoids.fit(X_scaled)

# Assign labels to DataFrame
df['Cluster'] = labels

# Visualization with smaller medoids
plt.figure(figsize=(10, 6))

# Scatter plot for the data points with the cluster labels
scatter = plt.scatter(
    X_scaled[:, 0],         # Release Year (standardized)
    X_scaled[:, 1],         # IMDb Rating (standardized)
    c=labels,               # Cluster assignment
    cmap='viridis',         # Colormap
    label='Movies',
    alpha=0.8,              # Slight transparency to highlight overlaps
    edgecolor='k',          # Add border for visibility
    zorder=1                # Ensure data points are below the medoids
)

# Scatter plot for the medoids
plt.scatter(
    kmedoids.medoids[:, 0],  # Medoid X-coordinate (Release Year)
    kmedoids.medoids[:, 1],  # Medoid Y-coordinate (IMDb Rating)
    c='red',                 # Medoid color
    marker='*',              # Star marker for medoids
    s=150,                   # Smaller size for medoids (adjusted from 300)
    edgecolor='white',       # White border around the medoids
    label='Medoids',
    zorder=2                 # Stars on top of data points
)

# Set title and labels
plt.title('K-Medoids Clustering of Disney Movies')
plt.xlabel('Release Year (standardized)')
plt.ylabel('IMDb Rating (standardized)')

plt.legend()
plt.colorbar(scatter, ticks=range(k), label='Cluster')

plt.show()
