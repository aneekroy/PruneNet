from typing import List
import numpy as np
from sklearn.cluster import KMeans


def kmeans_cluster_pool(embeddings: np.ndarray, k: int, random_state: int = 42) -> List[int]:
    kmeans = KMeans(n_clusters=k, random_state=random_state).fit(embeddings)
    centers = kmeans.cluster_centers_
    indices = []
    for c in centers:
        dists = ((embeddings - c)**2).sum(axis=1)
        indices.append(int(dists.argmin()))
    return indices
