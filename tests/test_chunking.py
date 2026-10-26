import pytest

from app.chunking import chunk_text


def test_empty():
    assert chunk_text("   ") == []


def test_short_text_single_chunk():
    assert chunk_text("Hello world.\n\nSecond para.", size=500, overlap=50) == ["Hello world.\n\nSecond para."]


def test_chunks_respect_size_roughly():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(30))
    chunks = chunk_text(text, size=600, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 600 + 100 + 10 for c in chunks)


def test_overlap_carries_context():
    text = "\n\n".join(f"Para {i} " + "alpha " * 30 for i in range(10))
    chunks = chunk_text(text, size=400, overlap=80)
    # some text from the end of chunk 0 should appear at the start of chunk 1
    assert chunks[0][-40:].split()[-1] in chunks[1][:120]


def test_hindi_sentence_split():
    para = "यह पहला वाक्य है। " * 200
    chunks = chunk_text(para, size=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 560 for c in chunks)


def test_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("x", size=100, overlap=100)
