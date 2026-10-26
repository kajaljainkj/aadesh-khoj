"""Second-stage reranking with a cross-encoder.

The e5 retriever is a *bi-encoder*: it embeds the question and each chunk separately, which is
fast enough to search everything but only approximately ranks them. A *cross-encoder* reads the
question and a chunk together and scores how well that chunk answers the question: much more
accurate, but too slow to run on every chunk. So we run it only on the top N candidates.

Default model: BAAI/bge-reranker-v2-m3 (multilingual, handles Hindi, runs locally on CPU).
"""
from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.rerank_model, max_length=settings.rerank_max_length)


def passage_text(hit: dict) -> str:
    """What the reranker reads for one hit: the order's subject (context) + the chunk."""
    subject = hit.get("subject") or hit.get("title") or ""
    return f"{subject}\n{hit['content']}" if subject else hit["content"]


def score(query: str, passages: list[str]) -> list[float]:
    if not passages:
        return []
    return [float(s) for s in _model().predict([(query, p) for p in passages], batch_size=16)]


def rerank(query: str | list[str], hits: list[dict], k: int) -> list[dict]:
    """Return the top-k hits re-ordered by cross-encoder score (score stored in 'rerank_score').

    `query` can be several phrasings (the user's question + the formal-Hindi rewrite); each hit
    gets its best score across them. A romanized Hinglish question ("IAS officers ka DA") barely
    matches OCR'd Hindi text, while the rewrite ("महंगाई भत्ता वृद्धि") does.

    If even the best hit scores below RERANK_MIN_SCORE, the cross-encoder isn't confident about
    anything, and its order is mostly noise. Then the first-stage (RRF) order is kept.
    """
    if not hits:
        return []
    queries = [query] if isinstance(query, str) else list(dict.fromkeys(q for q in query if q))
    passages = [passage_text(h) for h in hits]
    per_query = [score(q, passages) for q in queries]
    scores = [max(col) for col in zip(*per_query)]
    if max(scores) < settings.rerank_min_score:
        return [{**h, "rerank_score": round(s, 4)} for h, s in zip(hits, scores)][:k]
    ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:k]
    return [{**h, "score": round(s, 4), "rerank_score": round(s, 4)} for h, s in ranked]
