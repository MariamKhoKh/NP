import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


class KMedoidsClustering:
    def __init__(self, k=5, p=2):
        self.k = k
        self.p = p
        self.medoids = None

    def distance(self, data_point, medoids):
        return np.linalg.norm(medoids - data_point, ord=self.p, axis=1)

    def fit(self, X, max_iter=200):
        """Fit the KMedoids model to the dataset X.
           Randomly initialize medoids from the data points
        """
        self.medoids = X[np.random.choice(len(X), self.k, replace=False)]

        for iteration in range(max_iter):
            labels = []
            for data_point in X:
                distances = self.distance(data_point, self.medoids)
                cluster_num = np.argmin(distances)
                labels.append(cluster_num)
            labels = np.array(labels)

            new_medoids = []
            for i in range(self.k):
                points = X[labels == i]
                if len(points) == 0:
                    new_medoids.append(self.medoids[i])
                else:
                    distances = np.sum([np.linalg.norm(points - point, ord=self.p, axis=1) for point in points], axis=0)
                    new_medoid_idx = np.argmin(distances)
                    new_medoids.append(points[new_medoid_idx])
            new_medoids = np.array(new_medoids)

            if np.allclose(self.medoids, new_medoids, atol=1e-4):
                print(f"Converged after {iteration + 1} iterations.")
                break

            self.medoids = new_medoids

        return labels


# Load data
file_path = 'disney_movies.csv'
df = pd.read_csv(file_path)
features = df[['Release Year', 'IMDb Rating']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

k = 3
kmedoids = KMedoidsClustering(k=k, p=2)
kmedoids_labels = kmedoids.fit(X_scaled)


df['Cluster'] = kmedoids_labels


plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=kmedoids_labels,
    cmap='viridis',
    label='Movies',
    alpha=0.8,
    edgecolor='k',
    zorder=1
)

plt.scatter(
    kmedoids.medoids[:, 0],
    kmedoids.medoids[:, 1],
    c='red',
    marker='*',
    s=150,
    edgecolor='white',
    label='Medoids',
    zorder=2
)

plt.title('K-Medoids Clustering of Disney Movies')
plt.xlabel('Release Year (standardized)')
plt.ylabel('IMDb Rating (standardized)')
plt.legend()
plt.colorbar(scatter, ticks=range(k), label='Cluster')
plt.show()

for cluster_num in range(k):
    cluster_movies = df[df['Cluster'] == cluster_num]
    print(f"\nCluster {cluster_num + 1}:")
    print(cluster_movies[['Movie Name', 'Release Year', 'IMDb Rating']])


def evaluate_clustering(X, labels):
    """Evaluate clustering using Inertia and Silhouette Score."""
    # Inertia (Sum of squared distances to the closest medoid)
    inertia = np.sum([np.linalg.norm(X[i] - X[labels == labels[i]].mean(axis=0)) ** 2 for i in range(len(X))])

    # Silhouette score
    silhouette = silhouette_score(X, labels)

    return inertia, silhouette

kmedoids_inertia, kmedoids_silhouette = evaluate_clustering(X_scaled, kmedoids_labels)
print(f"K-Medoids Inertia: {kmedoids_inertia}, Silhouette: {kmedoids_silhouette}")
