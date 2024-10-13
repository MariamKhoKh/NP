# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


# This implementation clusters random data
class KMeansClustering:

    def __init__(self, k=3):
        """
        Initialize the KMeansClustering class with a number of clusters 'k'.
        Set centroids to None initially.
        """
        self.k = k
        self.centroids = None  # Centroids to be initialized during fitting

    @staticmethod
    def euclidian_distance(data_point, centroids):
        """
        Calculate the Euclidean distance between a data point and all centroids.
        Returns a list of distances.
        """
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

    def fit(self, X, max_iter=200):
        """
        Fit the KMeans model on the given dataset X.
        Randomly initialize centroids and update them iteratively.
        """
        # Randomly initialize the centroids within the range of the dataset
        self.centroids = np.random.uniform(np.amin(X, axis=0),
                                           np.amax(X, axis=0),
                                           size=(self.k, X.shape[1]))
        # Iterate to update centroids
        for _ in range(max_iter):
            y = []  # List to hold cluster assignments for each data point

            # Assign each data point to the nearest centroid
            for data_point in X:
                distance = KMeansClustering.euclidian_distance(data_point, self.centroids)
                cluster_num = np.argmin(distance)  # Get index of nearest centroid
                y.append(cluster_num)

            y = np.array(y)  # Convert list to array for further processing

            cluster_indexes = []  # To store indices of points in each cluster

            # Group data points by clusters
            for i in range(self.k):
                cluster_indexes.append(np.argwhere(y == i))

            cluster_centers = []  # List to store updated centroids

            # Recompute the centroids by calculating the mean of points in each cluster
            for i, indices in enumerate(cluster_indexes):
                if len(indices) == 0:  # If no points are assigned to this cluster
                    cluster_centers.append(self.centroids[i])  # Keep old centroid
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])  # Update centroid

            # If centroids don't change much, break out of the loop (convergence)
            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(cluster_centers)  # Update centroids

        return y  # Return final cluster assignments


# Generate random data points for clustering
data = make_blobs(n_samples=100, n_features=2, centers=3)
random_points = data[0]

# Initialize and fit the KMeans model
kmeans = KMeansClustering(k=3)
labels = kmeans.fit(random_points)

# Print the true labels and predicted labels
print(data[1])  # True labels
print(labels)   # Predicted labels by KMeans

# Calculate Adjusted Rand Index (ARI) to measure clustering performance
ari = adjusted_rand_score(data[1], labels)
print(ari)

plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)),
            marker="*", s=200)  # Mark centroids with a star

plt.show()
