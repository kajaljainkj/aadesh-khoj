"""Extraction on the bundled sample PDFs (one per kind)."""
import difflib
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from app.extract import extract_pdf  # noqa: E402
from app.metadata import extract_metadata  # noqa: E402
from app.textnorm import garbled_ratio, normalize_devanagari  # noqa: E402
from sample_orders import ORDERS, order_text  # noqa: E402

SAMPLES = ROOT / "data" / "samples"
BY_KIND = {k: next(o for o in ORDERS if o["kind"] == k) for k in ("unicode", "krutidev", "scanned")}


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, " ".join(a.split()), " ".join(b.split())).ratio()


def _tesseract_hindi() -> bool:
    if not shutil.which("tesseract"):
        return False
    try:
        import pytesseract

        return "hin" in pytesseract.get_languages()
    except Exception:
        return False


def test_unicode_pdf():
    o = BY_KIND["unicode"]
    pages = extract_pdf(str(SAMPLES / f"{o['id']}.pdf"), ocr=False)
    assert pages[0].method == "unicode"
    assert similarity(pages[0].text, normalize_devanagari(order_text(o))) > 0.98


def test_krutidev_pdf_is_garbage_without_fix_and_clean_with_it():
    o = BY_KIND["krutidev"]
    path = str(SAMPLES / f"{o['id']}.pdf")
    truth = normalize_devanagari(order_text(o))
    raw = extract_pdf(path, fix_legacy=False)[0].text
    fixed = extract_pdf(path, ocr=False)[0]
    assert similarity(raw, truth) < 0.2
    assert fixed.method == "krutidev-font"
    assert similarity(fixed.text, truth) > 0.98
    assert o["date"] in fixed.text  # English-font span left untouched


@pytest.mark.skipif(not _tesseract_hindi(), reason="Tesseract with Hindi language data not installed")
def test_scanned_pdf_ocr():
    o = BY_KIND["scanned"]
    path = str(SAMPLES / f"{o['id']}.pdf")
    assert extract_pdf(path, ocr=False)[0].text.strip() == ""
    page = extract_pdf(path)[0]
    assert page.method == "ocr"
    assert similarity(page.text, normalize_devanagari(order_text(o))) > 0.95


def test_metadata():
    o = BY_KIND["krutidev"]
    text = extract_pdf(str(SAMPLES / f"{o['id']}.pdf"), ocr=False)[0].text
    m = extract_metadata(text)
    assert m["order_no"] == o["no"]
    assert m["order_date"].strftime("%d/%m/%Y") == o["date"]
    assert m["subject"].startswith(o["subject"][:20])
    assert "विभाग" in m["department"]


def test_metadata_devanagari_digits():
    m = extract_metadata("क्रमांक १२३/२०२४ दिनांक ०५/०२/२०२४\nविषय - परीक्षण")
    assert m["order_no"] == "123/2024"
    assert m["order_date"].isoformat() == "2024-02-05"


def test_normalize_and_garbled():
    assert normalize_devanagari("स्कूूल प्राारंभ") == "स्कूल प्रारंभ"
    assert garbled_ratio("राज्य शासन, आदेश 12/2024") == 0
    assert garbled_ratio("का͛नύ क Ǹदेश ΋ूल") > 0.05


def _scanned_with_text_layer(text: str) -> str:
    """Copy of a scanned sample with an invisible text layer added on top of the image."""
    import tempfile

    import pymupdf

    o = BY_KIND["scanned"]
    doc = pymupdf.open(str(SAMPLES / f"{o['id']}.pdf"))
    page = doc[0]
    y = 60
    for line in [text[i:i + 80] for i in range(0, len(text), 80)]:
        page.insert_text((40, y), line, fontsize=9, render_mode=3)  # 3 = invisible, like scanner OCR
        y += 12
    path = tempfile.mktemp(suffix=".pdf")
    doc.save(path)
    return path


def test_junk_english_ocr_layer_detected():
    """Real MP orders: the scanner OCR'd Hindi as Latin junk. That page must be re-OCR'd."""
    import pymupdf

    from app.extract import _is_junk_ocr_layer

    junk = "ar{rIeer ilrg;T TrIIrdI srngfr E$rrr dTrdrc *q6 1 t 1 cfr, etqrd, Edin srmfr t e-ffFd EsTra " * 4
    english = ("The state government has decided that all offices shall remain open on the day of the "
               "election and that the staff will be paid for the work done in the office. ") * 2
    page_junk = pymupdf.open(_scanned_with_text_layer(junk))[0]
    page_en = pymupdf.open(_scanned_with_text_layer(english))[0]
    assert _is_junk_ocr_layer(page_junk, page_junk.get_text())
    assert not _is_junk_ocr_layer(page_en, page_en.get_text())   # a genuine English text layer is kept


@pytest.mark.skipif(not _tesseract_hindi(), reason="Tesseract with Hindi language data not installed")
def test_junk_ocr_layer_is_replaced_by_hindi_ocr():
    o = BY_KIND["scanned"]
    junk = "ar{rIeer ilrg;T TrIIrdI srngfr E$rrr dTrdrc *q6 1 t 1 cfr, etqrd, Edin srmfr t e-ffFd EsTra " * 4
    page = extract_pdf(_scanned_with_text_layer(junk))[0]
    assert page.method == "ocr"
    assert similarity(page.text, normalize_devanagari(order_text(o))) > 0.9
