import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


class KMeansClustering:
    def __init__(self, k=3, ord_number=2):
        self.k = k
        self.centroids = None
        self.ord_number = ord_number

    def distance(self, data_point, centroids):
        return np.linalg.norm(centroids - data_point, ord=self.ord_number, axis=1)

    def fit(self, X, max_iter=200):
        self.centroids = np.random.uniform(np.amin(X, axis=0),
                                           np.amax(X, axis=0),
                                           size=(self.k, X.shape[1]))

        for iteration in range(max_iter):
            labels = []
            for data_point in X:
                distances = self.distance(data_point, self.centroids)
                labels.append(np.argmin(distances))
            labels = np.array(labels)

            # Calculate new centroids
            new_centroids = []
            for i in range(self.k):
                points = X[labels == i]
                if len(points) == 0:
                    new_centroids.append(self.centroids[i])
                else:
                    new_centroids.append(np.mean(points, axis=0))
            new_centroids = np.array(new_centroids)

            if np.allclose(self.centroids, new_centroids, atol=1e-4):
                print(f"Converged after {iteration + 1} iterations.")
                break

            self.centroids = new_centroids

        return labels



# Generate matrix data for testing
matrix_data = np.array([
    [10.2, 3.5, 5.8, 8.7],
    [6.1, 7.3, 9.2, 4.4],
    [5.7, 6.8, 3.2, 9.5],
    [2.9, 8.4, 7.1, 5.6],
    [4.3, 3.9, 6.7, 2.8]
])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(matrix_data)

# Initialize and run KMeansClustering
k = 2  # Adjust the number of clusters if needed
kmeans = KMeansClustering(k=k, ord_number=2)
kmeans_labels = kmeans.fit(X_scaled)

# Assign labels to DataFrame (Optional)
df_test = pd.DataFrame(matrix_data, columns=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'])
df_test['Cluster'] = kmeans_labels

# Visualize clusters (for 2D data, just pick first two features for visualization)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis', label='Samples')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='*', s=200, label='Centroids')
plt.title('K-Means Clustering of Test Matrix Data')
plt.xlabel('Feature 1 (standardized)')
plt.ylabel('Feature 2 (standardized)')
plt.legend()
plt.show()


print(df_test)
