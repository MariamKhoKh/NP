### Clustering Analysis of Disney Movies

#### Objective
To analyze the clustering behavior of Disney movies based on their release year and IMDb rating.

#### Methodology
- **Algorithms Used**: K-Means and K-Medoids
- **Distance Metrics**: Euclidean distance and Manhattan distance
- **Range of Clusters Tested**: 2 to 10

#### Results
- **Optimal Number of Clusters**: 
  - K-Means optimal \( k \) = 3 (elbow method observed)
  - K-Medoids optimal \( k \) = 3

- **Evaluation Metrics**:
  - **K-Means**:
    - Inertia: 30.1641
    - Silhouette Score: 0.4150
  - **K-Medoids**:
    - Inertia: 52.3685
    - Silhouette Score: 0.1657

#### Visualizations
- Scatter plots of clusters showing distinct separation.
- Elbow curve for inertia vs. number of clusters.
- Pair plot for exploratory data analysis.

#### Conclusions
K-Means provided better-defined clusters compared to K-Medoids based on silhouette score and inertia. The insights suggest a potential for movie recommendation based on clustered genres and ratings.

