# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
#
# data = [
#     {"Movie Name": "Snow White and the Seven Dwarfs", "Release Year": 1937, "IMDb Rating": 7.6},
#     {"Movie Name": "Pinocchio", "Release Year": 1940, "IMDb Rating": 7.5},
#     {"Movie Name": "Fantasia", "Release Year": 1940, "IMDb Rating": 7.7},
#     {"Movie Name": "Dumbo", "Release Year": 1941, "IMDb Rating": 7.2},
#     {"Movie Name": "Bambi", "Release Year": 1942, "IMDb Rating": 7.3},
#     {"Movie Name": "Victory Through Air Power", "Release Year": 1943, "IMDb Rating": 6.5},
#     {"Movie Name": "The Three Caballeros", "Release Year": 1944, "IMDb Rating": 6.3},
#     {"Movie Name": "Make Mine Music", "Release Year": 1946, "IMDb Rating": 6.1},
#     {"Movie Name": "Song of the South", "Release Year": 1946, "IMDb Rating": 6.9},
#     {"Movie Name": "Fun and Fancy Free", "Release Year": 1947, "IMDb Rating": 6.4},
#     {"Movie Name": "Melody Time", "Release Year": 1948, "IMDb Rating": 6.1},
#     {"Movie Name": "The Adventures of Ichabod and Mr. Toad", "Release Year": 1949, "IMDb Rating": 6.8},
#     {"Movie Name": "Cinderella", "Release Year": 1950, "IMDb Rating": 7.3},
#     {"Movie Name": "Alice in Wonderland", "Release Year": 1951, "IMDb Rating": 7.3},
#     {"Movie Name": "Peter Pan", "Release Year": 1953, "IMDb Rating": 7.3},
#     {"Movie Name": "Lady and the Tramp", "Release Year": 1955, "IMDb Rating": 7.3},
#     {"Movie Name": "Sleeping Beauty", "Release Year": 1959, "IMDb Rating": 7.2},
#     {"Movie Name": "One Hundred and One Dalmatians", "Release Year": 1961, "IMDb Rating": 7.3},
#     {"Movie Name": "The Sword in the Stone", "Release Year": 1963, "IMDb Rating": 7.1},
#     {"Movie Name": "The Jungle Book", "Release Year": 1967, "IMDb Rating": 7.6},
#     {"Movie Name": "The Aristocats", "Release Year": 1970, "IMDb Rating": 7.1},
#     {"Movie Name": "Robin Hood", "Release Year": 1973, "IMDb Rating": 7.5},
#     {"Movie Name": "The Many Adventures of Winnie the Pooh", "Release Year": 1977, "IMDb Rating": 7.5},
#     {"Movie Name": "The Rescuers", "Release Year": 1977, "IMDb Rating": 6.9},
#     {"Movie Name": "The Fox and the Hound", "Release Year": 1981, "IMDb Rating": 7.2},
#     {"Movie Name": "The Black Cauldron", "Release Year": 1985, "IMDb Rating": 6.3},
#     {"Movie Name": "The Great Mouse Detective", "Release Year": 1986, "IMDb Rating": 7.1},
#     {"Movie Name": "Who Framed Roger Rabbit", "Release Year": 1988, "IMDb Rating": 7.7},
#     {"Movie Name": "Oliver & Company", "Release Year": 1988, "IMDb Rating": 6.6},
#     {"Movie Name": "The Little Mermaid", "Release Year": 1989, "IMDb Rating": 7.6},
#     {"Movie Name": "DuckTales the Movie: Treasure of the Lost Lamp", "Release Year": 1990, "IMDb Rating": 6.8},
#     {"Movie Name": "The Rescuers Down Under", "Release Year": 1990, "IMDb Rating": 6.8},
#     {"Movie Name": "Beauty and the Beast", "Release Year": 1991, "IMDb Rating": 8.0},
#     {"Movie Name": "Aladdin", "Release Year": 1992, "IMDb Rating": 8.0},
#     {"Movie Name": "The Nightmare Before Christmas", "Release Year": 1993, "IMDb Rating": 7.9},
#     {"Movie Name": "The Lion King", "Release Year": 1994, "IMDb Rating": 8.5},
#     {"Movie Name": "A Goofy Movie", "Release Year": 1995, "IMDb Rating": 6.9},
#     {"Movie Name": "Pocahontas", "Release Year": 1995, "IMDb Rating": 6.7},
#     {"Movie Name": "Toy Story", "Release Year": 1995, "IMDb Rating": 8.3},
#     {"Movie Name": "James and the Giant Peach", "Release Year": 1996, "IMDb Rating": 6.7}
#     # src: https://www.imdb.com/list/ls026785255/
#     ]
#
#
# df = pd.DataFrame(data)
# features = df[['Release Year', 'IMDb Rating']]
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(features)
#
#
# class KMeansClustering:
#     def __init__(self, k=3):
#         self.k = k
#         self.centroids = None
#
#     @staticmethod
#     def euclidean_distance(data_point, centroids):
#         """Calculate the Euclidean distance between a data point and centroids"""
#         return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))
#
#     def fit(self, X, max_iter=200):
#         """Fit the KMeans model to the dataset X
#         and Randomly initialize centroids within the range of the dataset"""
#         self.centroids = np.random.uniform(np.amin(X, axis=0),
#                                            np.amax(X, axis=0),
#                                            size=(self.k, X.shape[1]))
#
#         for iteration in range(max_iter):
#             # assign each data point to the nearest centroid
#             labels = []
#             for data_point in X:
#                 distances = self.euclidean_distance(data_point, self.centroids)
#                 cluster_num = np.argmin(distances)
#                 labels.append(cluster_num)
#             labels = np.array(labels)
#
#             # calculate new centroids
#             new_centroids = []
#             for i in range(self.k):
#                 points = X[labels == i]
#                 if len(points) == 0:
#                     # If a cluster gets no points, reinitialize its centroid randomly
#                     new_centroids.append(self.centroids[i])
#                 else:
#                     new_centroids.append(np.mean(points, axis=0))
#             new_centroids = np.array(new_centroids)
#
#             # check for convergence (if centroids don't change)
#             if np.allclose(self.centroids, new_centroids, atol=1e-4):
#                 print(f"Converged after {iteration + 1} iterations.")
#                 break
#             self.centroids = new_centroids
#
#         return labels
#
#
# # instantiate and fit k-means
# k = 3  # number of clusters
# kmeans = KMeansClustering(k=k)
# labels = kmeans.fit(X_scaled)
#
# # assign labels to DataFrame
# df['Cluster'] = labels
#
# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(
#     X_scaled[:, 0],
#     X_scaled[:, 1],
#     c=labels,
#     cmap='viridis',
#     label='Movies'
# )
# plt.scatter(
#     kmeans.centroids[:, 0],
#     kmeans.centroids[:, 1],
#     c='red',
#     marker='*',
#     s=200,
#     label='Centroids'
# )
# plt.title('K-Means Clustering of Disney Movies')
# plt.xlabel('Release Year (standardized)')
# plt.ylabel('IMDb Rating (standardized)')
# plt.legend()
# plt.colorbar(scatter, ticks=range(k), label='Cluster')
# plt.show()
#
# # analyzing clusters
# for cluster_num in range(k):
#     cluster_movies = df[df['Cluster'] == cluster_num]
#     print(f"\nCluster {cluster_num + 1}:")
#     print(cluster_movies[['Movie Name', 'Release Year', 'IMDb Rating']])
