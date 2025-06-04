import random
from typing import List


class RandomSampler:
    def __init__(self, seed: int = 42):
        random.seed(seed)

    def select(self, pool_dataset, k: int) -> List[int]:
        indices = list(range(len(pool_dataset)))
        random.shuffle(indices)
        return indices[:k]
