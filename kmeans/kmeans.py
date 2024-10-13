import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

file_path = 'disney_movies.csv'
df = pd.read_csv(file_path)
features = df[['Release Year', 'IMDb Rating']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)


class KMeansClustering:
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidean_distance(data_point, centroids):
        """Calculate the Euclidean distance between a data point and centroids."""
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))

    def fit(self, X, max_iter=200):
        """Fit the KMeans model to the dataset X."""
        # randomly initialize centroids within the range of the dataset
        self.centroids = np.random.uniform(np.amin(X, axis=0),
                                           np.amax(X, axis=0),
                                           size=(self.k, X.shape[1]))

        for iteration in range(max_iter):
            # assign each data point to the nearest centroid
            labels = []
            for data_point in X:
                distances = self.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)
                labels.append(cluster_num)
            labels = np.array(labels)

            # Calculate new centroids
            new_centroids = []
            for i in range(self.k):
                points = X[labels == i]
                if len(points) == 0:
                    # if a cluster gets no points, reinitialize its centroid randomly
                    new_centroids.append(self.centroids[i])
                else:
                    new_centroids.append(np.mean(points, axis=0))
            new_centroids = np.array(new_centroids)

            # check for convergence (if centroids do not change)
            if np.allclose(self.centroids, new_centroids, atol=1e-4):
                print(f"Converged after {iteration + 1} iterations.")
                break
            self.centroids = new_centroids

        return labels


# instantiate and fit KMeans
k = 3  # Number of clusters
kmeans = KMeansClustering(k=k)
labels = kmeans.fit(X_scaled)

# assign labels to DataFrame
df['Cluster'] = labels

# visualization
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap='viridis',
    label='Movies'
)
plt.scatter(
    kmeans.centroids[:, 0],
    kmeans.centroids[:, 1],
    c='red',
    marker='*',
    s=200,
    label='Centroids'
)
plt.title('K-Means Clustering of Disney Movies')
plt.xlabel('Release Year (standardized)')
plt.ylabel('IMDb Rating (standardized)')
plt.legend()
plt.colorbar(scatter, ticks=range(k), label='Cluster')
plt.show()

# analyzing clusters
for cluster_num in range(k):
    cluster_movies = df[df['Cluster'] == cluster_num]
    print(f"\nCluster {cluster_num + 1}:")
    print(cluster_movies[['Movie Name', 'Release Year', 'IMDb Rating']])
