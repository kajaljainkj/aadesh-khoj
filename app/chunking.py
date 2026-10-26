"""Paragraph-aware chunking with character overlap. No heavy dependencies, easy to unit test."""
import re


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    # PDFs sometimes contain NUL/control bytes; Postgres TEXT rejects NUL
    text = _CONTROL.sub("", text)
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _split_long(paragraph: str, size: int) -> list[str]:
    """Split a paragraph longer than `size` on sentence boundaries (English and Hindi '।')."""
    sentences = re.split(r"(?<=[.!?।])\s+", paragraph)
    out, buf = [], ""
    for s in sentences:
        if len(s) > size:  # pathological sentence: hard split
            if buf:
                out.append(buf)
                buf = ""
            out.extend(s[i:i + size] for i in range(0, len(s), size))
            continue
        if len(buf) + len(s) + 1 > size and buf:
            out.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        out.append(buf)
    return out


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = clean_text(text)
    if not text:
        return []

    pieces: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        pieces.extend(_split_long(para, size) if len(para) > size else [para])

    chunks: list[str] = []
    buf = ""
    for p in pieces:
        if buf and len(buf) + len(p) + 2 > size:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            # start the tail at a word boundary
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1:]
            buf = f"{tail}\n\n{p}".strip() if tail else p
        else:
            buf = f"{buf}\n\n{p}".strip()
    if buf:
        chunks.append(buf)
    return chunks
