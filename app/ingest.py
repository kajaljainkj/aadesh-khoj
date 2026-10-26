"""Ingest government orders (PDF / txt / html) into Postgres.

Usage:
    python -m app.ingest data/samples                    # full pipeline, collection 'main'
    python -m app.ingest my_pdfs/ --collection mp_gad    # your own PDFs
    python -m app.ingest data/samples --no-fix --no-ocr  # baseline, no legacy-font fix
"""
import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from app.chunking import chunk_text
from app.config import settings
from app.extract import PageText, extract_pdf
from app.legacy_fonts import fix_text
from app.metadata import extract_metadata
from app.textnorm import normalize_devanagari

SUPPORTED = {".pdf", ".txt", ".md", ".html", ".htm"}


def load_pages(path: Path, fix_legacy: bool, ocr: bool) -> list[PageText]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf(str(path), fix_legacy=fix_legacy, ocr=ocr, ocr_lang=settings.ocr_lang)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if ext in {".html", ".htm"}:
        from bs4 import BeautifulSoup

        raw = BeautifulSoup(raw, "html.parser").get_text("\n")
    if not fix_legacy:
        return [PageText(1, raw, "raw")]
    text, converted = fix_text(raw)
    return [PageText(1, normalize_devanagari(text), "krutidev-text" if converted else "unicode")]


def header_for(meta: dict, fallback_title: str) -> str:
    """Short context header prepended to every chunk before embedding / keyword indexing."""
    parts = [meta.get("subject") or fallback_title]
    if meta.get("order_no"):
        parts.append(f"क्रमांक {meta['order_no']}")
    if meta.get("order_date"):
        parts.append(f"दिनांक {meta['order_date'].strftime('%d/%m/%Y')}")
    return " | ".join(parts)


def build_chunks(pages: list[PageText], header: str) -> list[dict]:
    out = []
    for p in pages:
        for c in chunk_text(p.text, settings.chunk_size, settings.chunk_overlap):
            out.append({"page": p.page, "content": c, "search_text": f"{header}\n{c}"})
    return out


def ingest_folder(folder: Path, collection: str, fix_legacy: bool = True, ocr: bool = True,
                  quiet: bool = False) -> dict:
    from app import db
    from app.embeddings import embed_passages

    db.init_schema()
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"No supported files in {folder}")
        sys.exit(1)

    methods_total: Counter = Counter()
    n_chunks = 0
    for path in files:
        pages = load_pages(path, fix_legacy, ocr)
        methods = [p.method for p in pages]
        methods_total.update(methods)
        full_text = "\n".join(p.text for p in pages)
        meta = extract_metadata(full_text) if fix_legacy else {}
        meta["title"] = meta.get("subject") or path.stem
        chunks = build_chunks(pages, header_for(meta, path.stem))
        if not chunks:
            if not quiet:
                print(f"  skip (no text): {path.name}  [{', '.join(methods)}]")
            continue
        vectors = embed_passages([c["search_text"] for c in chunks])
        n = db.replace_document(collection, path.name, meta, methods, chunks, vectors)
        n_chunks += n
        if not quiet:
            print(f"  {path.name}: {n} chunks  [{', '.join(sorted(set(methods)))}]  "
                  f"no={meta.get('order_no')!s:.30}  date={meta.get('order_date')}")
    summary = {"files": len(files), "chunks": n_chunks, "methods": dict(methods_total)}
    if not quiet:
        print(f"Done: {summary}  ->  collection '{collection}' now has {db.stats(collection)}")
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default="data/samples")
    ap.add_argument("--collection", default=settings.collection)
    ap.add_argument("--no-fix", action="store_true", help="don't convert legacy fonts (baseline)")
    ap.add_argument("--no-ocr", action="store_true", help="skip OCR for scanned pages")
    args = ap.parse_args()
    ingest_folder(Path(args.folder), args.collection, fix_legacy=not args.no_fix,
                  ocr=settings.ocr_enabled and not args.no_ocr)


if __name__ == "__main__":
    main()
