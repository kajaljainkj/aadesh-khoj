"""Shared fixtures.

`fake_embeddings` replaces the e5 model with a deterministic character-trigram hashing
embedder, so tests run without downloading a 1 GB model. It is lexical, not semantic,
which is fine for wiring tests (and is NOT what the real eval uses).
"""
import hashlib
import os
import sys
import types

import numpy as np
import pytest

from app.config import settings


def _hash_embed(text: str) -> np.ndarray:
    v = np.zeros(settings.embed_dim, dtype=np.float32)
    t = f"  {text.lower()}  "
    for i in range(len(t) - 2):
        h = int.from_bytes(hashlib.md5(t[i:i + 3].encode()).digest()[:4], "little")
        v[h % settings.embed_dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


@pytest.fixture
def fake_embeddings(monkeypatch):
    mod = types.SimpleNamespace(
        embed_passages=lambda texts, batch_size=32: np.stack([_hash_embed(t) for t in texts]),
        embed_query=_hash_embed,
    )
    monkeypatch.setitem(sys.modules, "app.embeddings", mod)
    return mod


def _db_available() -> bool:
    try:
        import psycopg

        with psycopg.connect(settings.database_url, connect_timeout=2) as c:
            c.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'").fetchone()
        return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(
    os.getenv("SKIP_DB_TESTS") == "1" or not _db_available(),
    reason="Postgres with pgvector not reachable (start it with: docker compose up -d)",
)
