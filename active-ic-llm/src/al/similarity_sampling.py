from typing import List
import numpy as np

from ..utils.embeddings import embed_texts


class SimilaritySampler:
    def select_for_one_test(self, pool_dataset, test_text: str, k: int) -> List[int]:
        pool_emb = embed_texts(pool_dataset.get_all_texts())
        test_emb = embed_texts([test_text])
        sims = pool_emb @ test_emb.T
        indices = np.argsort(-sims.squeeze())[:k]
        return indices.tolist()
