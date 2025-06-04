from typing import List
import numpy as np

from ..utils.embeddings import embed_texts
from ..utils.clustering import kmeans_cluster_pool


class DiversitySampler:
    def select(self, pool_dataset, k: int) -> List[int]:
        texts = pool_dataset.get_all_texts()
        emb = embed_texts(texts)
        return kmeans_cluster_pool(emb, k)
