"""FastAPI service.

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs
"""
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app import db, rag
from app.config import settings

app = FastAPI(title="Aadesh Khoj", description="Multilingual RAG over Hindi government orders, "
                                               "including legacy Kruti Dev and scanned PDFs")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, examples=["DA kitna badha hai 2024 mein?"])
    strategy: Literal["direct", "rewrite"] = "rewrite"
    mode: Literal["vector", "hybrid"] = "hybrid"
    top_k: int = Field(5, ge=1, le=20)
    collection: str = settings.collection
    rerank: bool | None = Field(None, description="Cross-encoder reranking; default from RERANK_ENABLED")


class Source(BaseModel):
    n: int
    source: str
    subject: str | None
    order_no: str | None
    order_date: str | None
    page: int | None
    score: float | None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    language: str
    strategy: str
    mode: str
    search_query: str
    sources: list[Source]
    llm_provider: str | None = None
    timings: dict[str, float] = {}


@app.get("/health")
def health():
    try:
        return {"status": "ok", **db.stats(settings.collection)}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        return asdict(rag.answer(req.question, req.strategy, req.mode, req.top_k, req.collection, req.rerank))
    except RuntimeError as e:  # e.g. Ollama not running, missing API key
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/documents")
def documents(collection: str = settings.collection):
    return db.list_documents(collection)


@app.post("/documents")
def upload(file: UploadFile = File(...), collection: str = settings.collection):
    """Upload a PDF; it is extracted (legacy fonts fixed, OCR if needed) and indexed."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    from app.ingest import ingest_folder

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / Path(file.filename).name
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        summary = ingest_folder(Path(tmp), collection, ocr=settings.ocr_enabled, quiet=True)
    return {"file": file.filename, **summary}
