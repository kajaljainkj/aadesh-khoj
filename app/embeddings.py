"""Local multilingual embeddings (free, no API, no rate limits).

E5 models expect 'query: ' / 'passage: ' prefixes; we add them automatically.
"""
from functools import lru_cache

import numpy as np

from app.config import settings


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embed_model)


def _is_e5() -> bool:
    return "e5" in settings.embed_model.lower()


def embed_passages(texts: list[str], batch_size: int = 32) -> np.ndarray:
    if _is_e5():
        texts = [f"passage: {t}" for t in texts]
    return _model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 64
    )


def embed_query(text: str) -> np.ndarray:
    if _is_e5():
        text = f"query: {text}"
    return _model().encode([text], normalize_embeddings=True)[0]
