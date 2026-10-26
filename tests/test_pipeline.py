"""Integration tests against a real Postgres + pgvector (skipped if not running).

Embeddings and the LLM are faked, so no model download or API is needed.
"""
from pathlib import Path

import pytest

from app import llm
from tests.conftest import requires_db

ROOT = Path(__file__).resolve().parent.parent
COLL = "pytest"


@pytest.fixture
def fake_llm(monkeypatch):
    calls = []

    def chat(system, user, temperature=0.2, provider=None, max_retries=5):
        calls.append(system)
        if "search query" in system:
            return "महंगाई भत्ता वृद्धि"
        return "महंगाई भत्ता 46 प्रतिशत किया गया है [1]।"

    monkeypatch.setattr(llm, "chat", chat)
    return calls


@requires_db
def test_ingest_and_answer(fake_embeddings, fake_llm):
    from app import db, rag
    from app.ingest import ingest_folder

    summary = ingest_folder(ROOT / "data" / "samples", COLL, ocr=False, quiet=True)
    assert summary["methods"].get("krutidev-font") == 5
    docs = {d["source"]: d for d in db.list_documents(COLL)}
    assert docs["order_01_mahngai_bhatta.pdf"]["order_no"] == "एफ 4-12/2024/नियम/चार"

    # keyword search finds the Kruti Dev document only because it was converted
    hits = db.keyword_search("महंगाई भत्ते", 5, COLL)
    assert hits and hits[0]["source"] == "order_01_mahngai_bhatta.pdf"

    r = rag.answer("DA kitna badha hai?", "rewrite", "hybrid", 5, COLL)
    assert r.language == "hinglish"
    assert r.search_query == "महंगाई भत्ता वृद्धि"
    assert r.sources[0]["source"] == "order_01_mahngai_bhatta.pdf"
    assert r.sources[0]["order_no"] == "एफ 4-12/2024/नियम/चार"
    assert len(fake_llm) == 2


@requires_db
def test_raw_ingest_cannot_find_krutidev_doc(fake_embeddings):
    from app import db
    from app.ingest import ingest_folder

    ingest_folder(ROOT / "data" / "samples", COLL + "_raw", fix_legacy=False, ocr=False, quiet=True)
    sources = [h["source"] for h in db.keyword_search("महंगाई भत्ते", 5, COLL + "_raw")]
    assert "order_01_mahngai_bhatta.pdf" not in sources


def test_rrf_fuse():
    from app.db import rrf_fuse

    a = [{"id": 1}, {"id": 2}, {"id": 3}]
    b = [{"id": 3}, {"id": 1}]
    fused = rrf_fuse(a, b, limit=3)
    assert [x["id"] for x in fused][:2] == [1, 3]


def test_keyword_query_drops_stopwords():
    from app.db import keyword_query

    assert keyword_query("DA kitna badha hai?") == "da | badha"
    assert keyword_query("क्या है") is None


@requires_db
def test_api(fake_embeddings, fake_llm):
    from fastapi.testclient import TestClient

    from app.main import app

    c = TestClient(app)
    res = c.post("/ask", json={"question": "महंगाई भत्ता कितना है?", "collection": COLL})
    assert res.status_code == 200, res.text
    assert res.json()["sources"][0]["order_date"] == "2024-03-15"
    assert any(d["source"].startswith("order_") for d in c.get("/documents", params={"collection": COLL}).json())


def test_ollama_client_needs_no_key():
    llm._client.cache_clear()
    c = llm._client("ollama")
    assert "11434" in str(c.base_url)


# ---------------- LLM fallback ----------------

class _FakeCompletions:
    def __init__(self, behaviour):
        self.behaviour = behaviour

    def create(self, **kw):
        b = self.behaviour
        if isinstance(b, Exception):
            raise b
        msg = type("M", (), {"content": b})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()


def _fake_client(behaviour):
    return type("Cl", (), {"chat": type("Ch", (), {"completions": _FakeCompletions(behaviour)})()})()


def _status_error(code):
    import httpx
    from openai import APIStatusError, RateLimitError

    req = httpx.Request("POST", "https://x")
    resp = httpx.Response(code, request=req)
    cls = RateLimitError if code == 429 else APIStatusError
    return cls("err", response=resp, body=None)


@pytest.fixture
def providers(monkeypatch):
    """Point 'groq' and 'ollama' at fake clients whose behaviour each test sets."""
    from app import config

    beh = {}
    monkeypatch.setattr(llm, "_client", lambda p: _fake_client(beh[p]))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    new = config.Settings(llm_provider="groq", llm_fallback="ollama", llm_min_interval=0)
    monkeypatch.setattr(llm, "settings", new)
    llm._cooldown_until.clear()
    return beh


def test_primary_used_when_ok(providers):
    providers.update(groq="from groq", ollama="from ollama")
    assert llm.chat("s", "u") == "from groq"
    assert llm.last_provider() == "groq"


def test_falls_back_on_rate_limit_then_cools_down(providers):
    providers.update(groq=_status_error(429), ollama="from ollama")
    assert llm.chat("s", "u") == "from ollama"
    assert llm.last_provider() == "ollama"
    assert llm._cooldown_until["groq"] > 0
    providers["groq"] = "groq is back"
    assert llm.chat("s", "u") == "from ollama"   # still cooling down, primary skipped


def test_falls_back_on_missing_key(providers, monkeypatch):
    def client(p):
        if p == "groq":
            raise llm.ProviderError("No API key set for provider 'groq'")
        return _fake_client("from ollama")

    monkeypatch.setattr(llm, "_client", client)
    assert llm.chat("s", "u") == "from ollama"


def test_falls_back_on_bad_key_and_server_error(providers):
    providers.update(groq=_status_error(401), ollama="ok")
    assert llm.chat("s", "u") == "ok"
    llm._cooldown_until.clear()
    providers.update(groq=_status_error(503))
    assert llm.chat("s", "u") == "ok"


def test_no_fallback_for_bad_request(providers):
    from openai import APIStatusError

    providers.update(groq=_status_error(400), ollama="ok")
    with pytest.raises(APIStatusError):
        llm.chat("s", "u")


def test_fallback_disabled_raises(providers, monkeypatch):
    from app import config

    monkeypatch.setattr(llm, "settings", config.Settings(llm_provider="groq", llm_fallback="", llm_min_interval=0))
    providers.update(groq=_status_error(429), ollama="ok")
    with pytest.raises(llm.ProviderError):
        llm.chat("s", "u")


def test_falls_back_when_hosted_model_retired(providers):
    providers.update(groq=_status_error(404), ollama="ok")
    assert llm.chat("s", "u") == "ok"
    assert llm._cooldown_until["groq"] > 0


# ---------------- query rewrite guard ----------------

@pytest.mark.parametrize("question,rewrite,ok", [
    ("What is the theme of International Yoga Day 2026?",
     "International Yoga Day 2026 का थीम “Yoga for a Healthy Nation” (स्वस्थ राष्ट्र के लिए योग) है।", False),
    ("What is the theme of International Yoga Day 2026?", "अंतर्राष्ट्रीय योग दिवस 2026 थीम", True),
    ("iGOT karmayogi pe kitne course karne hain?", "iGOT कर्मयोगी पाठ्यक्रम संख्या", True),
    ("DA kitna badha?", "महंगाई भत्ता 60 प्रतिशत किया गया है।", False),
    ("DA kitna badha?", "", False),
])
def test_rewrite_guard(question, rewrite, ok):
    from app.rag import _rewrite_is_safe

    assert _rewrite_is_safe(question, rewrite) is ok


def test_unsafe_rewrite_falls_back_to_question(monkeypatch):
    from app import rag

    monkeypatch.setattr(llm, "chat", lambda *a, **k: 'थीम "Yoga for a Healthy Nation" है।')
    q = "What is the theme of International Yoga Day 2026?"
    assert rag.rewrite_query(q) == q
