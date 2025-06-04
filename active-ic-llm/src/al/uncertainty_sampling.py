from typing import List

from ..utils.perplexity import compute_perplexities
from ..config import cfg


class UncertaintySampler:
    def select(self, pool_dataset, k: int) -> List[int]:
        texts = pool_dataset.get_all_texts()
        perps = compute_perplexities(texts, cfg.model_name, cfg.device)
        indices = list(range(len(texts)))
        indices.sort(key=lambda i: perps[i], reverse=True)
        return indices[:k]
