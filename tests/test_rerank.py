"""Reranker tests. The real cross-encoder (~2 GB) is replaced by a fake scorer."""
from pathlib import Path

import pytest

from app import rerank as rr
from tests.conftest import requires_db

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def fake_scorer(monkeypatch):
    """Score = how many query words appear in the passage (a stand-in for a cross-encoder)."""
    calls = []

    def score(query, passages):
        calls.append((query, len(passages)))
        words = set(query.split())
        return [float(sum(w in p for w in words)) for p in passages]

    monkeypatch.setattr(rr, "score", score)
    return calls


def test_rerank_reorders_and_truncates(fake_scorer):
    hits = [
        {"id": 1, "content": "गर्मी की छुट्टियाँ", "subject": "अवकाश"},
        {"id": 2, "content": "महंगाई भत्ता 46 प्रतिशत", "subject": "महंगाई भत्ता वृद्धि"},
        {"id": 3, "content": "पेंशन", "subject": None},
    ]
    out = rr.rerank("महंगाई भत्ता", hits, k=2)
    assert [h["id"] for h in out] == [2, 1]
    assert out[0]["rerank_score"] > out[1]["rerank_score"]
    assert "महंगाई भत्ता वृद्धि" in rr.passage_text(hits[1])   # subject is given to the reranker


def test_rerank_empty(fake_scorer):
    assert rr.rerank("x", [], 5) == []


@requires_db
def test_retrieve_with_rerank_uses_more_candidates(fake_embeddings, fake_scorer):
    from app import rag
    from app.ingest import ingest_folder

    ingest_folder(ROOT / "data" / "samples", "pytest_rr", ocr=False, quiet=True)
    _, hits = rag.retrieve("महंगाई भत्ता", "direct", "hybrid", k=3, collection="pytest_rr", rerank=True)
    assert len(hits) == 3
    assert all("rerank_score" in h for h in hits)
    question, n_candidates = fake_scorer[-1]
    assert question == "महंगाई भत्ता"
    assert n_candidates > 3                     # the cross-encoder saw more than the final k
    assert hits[0]["source"] == "order_01_mahngai_bhatta.pdf"


@requires_db
def test_retrieve_without_rerank_skips_cross_encoder(fake_embeddings, fake_scorer):
    from app import rag
    from app.ingest import ingest_folder

    ingest_folder(ROOT / "data" / "samples", "pytest_rr", ocr=False, quiet=True)
    _, hits = rag.retrieve("महंगाई भत्ता", "direct", "hybrid", k=3, collection="pytest_rr", rerank=False)
    assert len(hits) == 3 and not fake_scorer


def test_rerank_uses_best_score_across_queries(fake_scorer):
    hits = [{"id": 1, "content": "अवकाश सूची", "subject": None},
            {"id": 2, "content": "महंगाई भत्ता 60 प्रतिशत", "subject": None}]
    out = rr.rerank(["IAS officers ka DA", "महंगाई भत्ता"], hits, k=2)
    assert [h["id"] for h in out] == [2, 1]          # only the Hindi rewrite matches
    assert [q for q, _ in fake_scorer] == ["IAS officers ka DA", "महंगाई भत्ता"]


def test_rerank_keeps_first_stage_order_when_unsure(fake_scorer):
    hits = [{"id": 1, "content": "क", "subject": None}, {"id": 2, "content": "ख", "subject": None},
            {"id": 3, "content": "ग", "subject": None}]
    out = rr.rerank("xyz", hits, k=2)                 # every score is 0 -> below RERANK_MIN_SCORE
    assert [h["id"] for h in out] == [1, 2]
    assert all(h["rerank_score"] == 0 for h in out)
