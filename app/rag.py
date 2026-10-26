"""Retrieval + grounded answer generation.

Query strategies:
  direct   embed / keyword-search the user's question as typed
  rewrite  LLM first turns the question (Hindi, Hinglish or English) into a formal
           Hindi search query, the vocabulary government orders actually use
           ("chhutti" -> "अवकाश", "DA" -> "महंगाई भत्ता")
Search modes:
  vector   pgvector cosine similarity only
  hybrid   vector + keyword (Postgres full-text), fused with Reciprocal Rank Fusion.
           Keyword search matters here: order numbers, dates and exact terms.
Reranking (optional, RERANK_ENABLED or rerank=True):
  the top RERANK_CANDIDATES hits are re-scored by a cross-encoder (app/rerank.py) and the
  best k are kept. Slower, but puts the right order at rank 1 more often.
"""
import logging
import re
import time
from dataclasses import dataclass, field

from app import llm
from app.config import settings
from app.lang import LANGUAGE_NAMES, detect_language

log = logging.getLogger(__name__)

STRATEGIES = ("direct", "rewrite")
MODES = ("vector", "hybrid")

REWRITE_SYSTEM = (
    "You turn a user's question about Indian government orders into a short search query in "
    "formal Hindi (Devanagari script), using the official words such orders use, e.g. "
    "महंगाई भत्ता, स्थानांतरण, अवकाश, पेंशन, छात्रवृत्ति, आयु सीमा, दैनिक भत्ता, विद्यालय, कर्मचारी. "
    "The question may be in Hindi, Hinglish (Hindi in Roman script) or English. "
    "Keep any numbers, years, acronyms and order numbers from the question. "
    "Do NOT answer the question. Never add names, numbers, themes or any other fact that is not "
    "in the question. Output ONLY 3 to 12 keywords, no full sentence, nothing else.\n"
    "Example: 'What is the theme of Yoga Day 2026?' -> योग दिवस 2026 थीम"
)

_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")
_QUOTES = re.compile(r"[\"“”‘’]")
_NUMBER = re.compile(r"\d+")


def _rewrite_is_safe(question: str, rewrite: str) -> bool:
    """Reject rewrites where the LLM answered the question instead of rephrasing it.

    Seen with gpt-oss on Groq: 'What is the theme of Yoga Day 2026?' was "rewritten" to a full
    sentence containing an invented theme. Signals: English words or numbers that weren't in the
    question (the rewrite should only rephrase), quoted text, or a much longer output.
    """
    if not rewrite:
        return False
    q_words = {w.lower() for w in _LATIN_WORD.findall(question)}
    new_latin = [w for w in _LATIN_WORD.findall(rewrite) if w.lower() not in q_words]
    new_numbers = set(_NUMBER.findall(rewrite)) - set(_NUMBER.findall(question))
    too_long = len(rewrite.split()) > max(16, 2 * len(question.split()) + 4)
    return not new_latin and not new_numbers and not _QUOTES.search(rewrite) and not too_long

ANSWER_SYSTEM = """You are Aadesh Khoj, an assistant that answers questions about government orders.

Rules:
1. Answer ONLY from the numbered context passages. Never use outside knowledge or guess.
2. Cite passages inline like [1], [2]. When you state a rule, also name the order number and date
   from that passage's header if available.
3. If the passages don't contain the answer, say so plainly.
   Users often use English words or abbreviations for official Hindi terms (DA = महंगाई भत्ता,
   IAS/IPS/IFS officers = अखिल भारतीय सेवा के अधिकारी, transfer = स्थानांतरण, leave = अवकाश).
   Treat these as the same thing. The passages are OCR'd, so expect small spelling errors.
4. Remind the user to check the latest official order if the answer depends on dates or rates,
   because orders are often amended or superseded.
5. Reply in {language}. Keep official Hindi terms where helpful.
6. Be concise: 2-6 sentences or a short list."""


@dataclass
class RagResult:
    answer: str
    language: str
    strategy: str
    mode: str
    search_query: str
    sources: list[dict] = field(default_factory=list)
    llm_provider: str | None = None  # which LLM actually answered (after any fallback)
    timings: dict = field(default_factory=dict)  # seconds per stage: rewrite, search, rerank, answer


def rewrite_query(question: str) -> str:
    out = llm.chat(REWRITE_SYSTEM, question, temperature=0.0).strip()
    out = out.splitlines()[0].strip() if out else ""
    if not _rewrite_is_safe(question, out):
        log.warning("Discarding unsafe query rewrite %r; searching with the original question", out)
        return question
    return out


def retrieve(question: str, strategy: str = "rewrite", mode: str = "hybrid", k: int | None = None,
             collection: str | None = None, rerank: bool | None = None,
             timings: dict | None = None) -> tuple[str, list[dict]]:
    from app import db
    from app.embeddings import embed_query

    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    k = k or settings.top_k
    collection = collection or settings.collection
    rerank = settings.rerank_enabled if rerank is None else rerank
    # with reranking, first-stage retrieval keeps more candidates for the cross-encoder to judge
    first_k = max(k, settings.rerank_candidates) if rerank else k

    timings = {} if timings is None else timings
    t = time.perf_counter()
    query = rewrite_query(question) if strategy == "rewrite" else question
    if strategy == "rewrite":
        timings["rewrite"] = time.perf_counter() - t
    t = time.perf_counter()
    pool = max(20, 4 * first_k)
    vec_hits = db.vector_search(embed_query(query), pool, collection)
    if query != question:
        # also search with the user's own words, so one bad rewrite can't hide the right order
        vec_hits = db.rrf_fuse(vec_hits, db.vector_search(embed_query(question), pool, collection), limit=pool)
    if mode == "vector":
        hits = [{**h, "score": round(1 / (60 + r), 4)} for r, h in enumerate(vec_hits[:first_k], 1)]
    else:
        kw_hits = db.keyword_search(query, pool, collection)
        if strategy == "rewrite" and query != question:
            kw_hits = db.rrf_fuse(kw_hits, db.keyword_search(question, pool, collection), limit=pool)
        hits = db.rrf_fuse(vec_hits, kw_hits, limit=first_k)
    timings["search"] = time.perf_counter() - t

    if rerank:
        from app.rerank import rerank as cross_encoder_rerank

        # judge against the user's own question plus the formal-Hindi rewrite, in ONE pass:
        # a Hinglish question alone barely matches Hindi order text
        t = time.perf_counter()
        rr_query = question if query == question else f"{question}\n{query}"
        hits = cross_encoder_rerank(rr_query, hits, k)
        timings["rerank"] = time.perf_counter() - t
    return query, hits[:k]


def _header(h: dict) -> str:
    bits = [h.get("subject") or h.get("title") or h["source"]]
    if h.get("order_no"):
        bits.append(f"क्रमांक {h['order_no']}")
    if h.get("order_date"):
        bits.append(f"दिनांक {h['order_date'].strftime('%d/%m/%Y')}")
    return " | ".join(bits)


def build_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{i}] ({_header(h)})\n{h['content']}" for i, h in enumerate(hits, 1))


NO_DOCS = {
    "hi": "अभी कोई दस्तावेज़ इंडेक्स नहीं है। पहले ingest चलाएँ।",
    "hinglish": "Abhi koi document index nahi hai. Pehle ingest chalaiye.",
    "en": "No documents are indexed yet. Run ingestion first.",
}


def answer(question: str, strategy: str = "rewrite", mode: str = "hybrid", k: int | None = None,
           collection: str | None = None, rerank: bool | None = None) -> RagResult:
    language = detect_language(question)
    timings: dict = {}
    query, hits = retrieve(question, strategy, mode, k, collection, rerank, timings)
    if not hits:
        return RagResult(NO_DOCS[language], language, strategy, mode, query, [], timings=timings)

    hint = f"\n(Search terms in official Hindi: {query})" if query != question else ""
    user_msg = f"Context:\n{build_context(hits)}\n\nQuestion: {question}{hint}"
    t = time.perf_counter()
    text = llm.chat(ANSWER_SYSTEM.format(language=LANGUAGE_NAMES[language]), user_msg)
    timings["answer"] = time.perf_counter() - t

    sources = [
        {
            "n": i,
            "source": h["source"],
            "subject": h.get("subject") or h.get("title"),
            "order_no": h.get("order_no"),
            "order_date": h["order_date"].isoformat() if h.get("order_date") else None,
            "page": h.get("page"),
            "score": h.get("score"),
            "snippet": h["content"][:240] + ("…" if len(h["content"]) > 240 else ""),
        }
        for i, h in enumerate(hits, 1)
    ]
    timings = {name: round(sec, 2) for name, sec in timings.items()}
    return RagResult(text, language, strategy, mode, query, sources, llm.last_provider(), timings)
