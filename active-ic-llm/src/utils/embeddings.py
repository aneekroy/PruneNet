from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

_sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: List[str]) -> np.ndarray:
    embeddings = _sbert_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings
