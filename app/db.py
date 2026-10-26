"""Postgres storage: pgvector for semantic search, tsvector for keyword search.

A 'collection' column lets several indexes live side by side (the eval uses
'eval_raw' and 'eval_fixed' to compare ingestion with and without the legacy-font fix).
"""
import re
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from app.config import settings


@contextmanager
def get_conn():
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        yield conn


def init_schema() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id          BIGSERIAL PRIMARY KEY,
                collection  TEXT NOT NULL,
                source      TEXT NOT NULL,
                title       TEXT,
                order_no    TEXT,
                order_date  DATE,
                department  TEXT,
                subject     TEXT,
                pages       INT,
                methods     JSONB,          -- extraction method per page
                created_at  TIMESTAMPTZ DEFAULT now(),
                UNIQUE (collection, source)
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id           BIGSERIAL PRIMARY KEY,
                document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                collection   TEXT NOT NULL,
                page         INT,
                chunk_index  INT NOT NULL,
                content      TEXT NOT NULL,
                search_text  TEXT NOT NULL,     -- header (subject, order no) + content
                embedding    VECTOR({settings.embed_dim}) NOT NULL,
                tsv          TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', search_text)) STORED
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)")
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING gin (tsv)")
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_collection ON chunks (collection)")


def replace_document(collection: str, source: str, meta: dict, methods: list[str],
                     chunks: list[dict], embeddings) -> int:
    """Insert (or re-insert) one document and its chunks. Idempotent per (collection, source)."""
    with get_conn() as conn, conn.transaction():
        conn.execute("DELETE FROM documents WHERE collection = %s AND source = %s", (collection, source))
        doc_id = conn.execute(
            """
            INSERT INTO documents (collection, source, title, order_no, order_date, department, subject, pages, methods)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
            """,
            (collection, source, meta.get("title"), meta.get("order_no"), meta.get("order_date"),
             meta.get("department"), meta.get("subject"), len(methods), Jsonb(methods)),
        ).fetchone()[0]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (document_id, collection, page, chunk_index, content, search_text, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [(doc_id, collection, c["page"], i, c["content"], c["search_text"], e)
                 for i, (c, e) in enumerate(zip(chunks, embeddings))],
            )
    return len(chunks)


_SELECT = """
    SELECT c.id, d.source, d.title, d.order_no, d.order_date, d.department, d.subject,
           c.page, c.chunk_index, c.content
    FROM chunks c JOIN documents d ON d.id = c.document_id
"""
_COLS = ["id", "source", "title", "order_no", "order_date", "department", "subject", "page", "chunk_index", "content"]


def vector_search(query_embedding, k: int, collection: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            _SELECT + " WHERE c.collection = %s ORDER BY c.embedding <=> %s LIMIT %s",
            (collection, query_embedding, k),
        ).fetchall()
    return [dict(zip(_COLS, r)) for r in rows]


# Words too common to help keyword search (Hindi, Hinglish, English)
STOPWORDS = set("""
का के की को में से पर है हैं था थी थे हो और या एवं तथा भी तो ही कि जो यह वह इस उस इन उन
क्या कब कैसे क्यों कितना कितने कितनी कौन किस किसे किसको कहाँ कहां लिए द्वारा किया गया गई गए
होगा होगी होंगे होता होती होते करना करने करें कर सकते सकता सकती जाएगा जाएगी जाता जाती
ka ke ki ko mein me se par hai hain tha thi the ho aur ya bhi to toh hi kya kab kaise kyun kitna
kitne kitni kaun kis kahan liye kiya gaya hoga hogi karna kar sakte sakta milega milti milta wala wali
the a an of to in is are was were be for on at by with what when how why which who does do did
""".split())
_TOKEN = re.compile(r"[0-9A-Za-zऀ-ॿ]+")


def keyword_query(text: str) -> str | None:
    tokens = [t.lower() for t in _TOKEN.findall(text)]
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = list(dict.fromkeys(tokens))[:32]
    return " | ".join(tokens) if tokens else None


def keyword_search(text: str, k: int, collection: str) -> list[dict]:
    q = keyword_query(text)
    if not q:
        return []
    with get_conn() as conn:
        rows = conn.execute(
            _SELECT + """
            , to_tsquery('simple', %s) q
            WHERE c.collection = %s AND c.tsv @@ q
            ORDER BY ts_rank_cd(c.tsv, q) DESC
            LIMIT %s
            """,
            (q, collection, k),
        ).fetchall()
    return [dict(zip(_COLS, r)) for r in rows]


def rrf_fuse(*ranked_lists: list[dict], k: int = 60, limit: int = 5) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) over lists."""
    scores: dict[int, float] = {}
    items: dict[int, dict] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst, 1):
            scores[item["id"]] = scores.get(item["id"], 0.0) + 1.0 / (k + rank)
            items.setdefault(item["id"], item)
    best = sorted(scores, key=scores.get, reverse=True)[:limit]
    return [{**items[i], "score": round(scores[i], 4)} for i in best]


def stats(collection: str | None = None) -> dict:
    with get_conn() as conn:
        if collection:
            docs = conn.execute("SELECT count(*) FROM documents WHERE collection=%s", (collection,)).fetchone()[0]
            chunks = conn.execute("SELECT count(*) FROM chunks WHERE collection=%s", (collection,)).fetchone()[0]
        else:
            docs, chunks = (conn.execute("SELECT count(*) FROM documents").fetchone()[0],
                            conn.execute("SELECT count(*) FROM chunks").fetchone()[0])
    return {"documents": docs, "chunks": chunks}


def list_documents(collection: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT source, order_no, order_date, department, subject, pages, methods "
            "FROM documents WHERE collection = %s ORDER BY order_date DESC NULLS LAST, source",
            (collection,),
        ).fetchall()
    cols = ["source", "order_no", "order_date", "department", "subject", "pages", "methods"]
    return [dict(zip(cols, r)) for r in rows]
