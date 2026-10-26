"""Turn a PDF into clean Unicode text, page by page.

For each page we pick the best method:
  unicode        text layer is fine, just normalize it
  krutidev-font  some spans use a Kruti Dev / DevLys font -> convert only those spans
  krutidev-text  no font info, but the text looks like Kruti Dev -> convert (dates protected)
  ocr            no usable text layer (scanned), text is garbled, or a scanned page carries a
                 junk OCR layer (see _is_junk_ocr_layer) -> Tesseract eng+hin
                 ('eng+hin' at 400 DPI read 23/23 numbers on the samples vs 12/23 for 'hin+eng':
                  the Hindi model misreads Latin digits such as 1 in dates and order numbers)
  raw            fix_legacy=False: text as-is (used by the eval as the baseline)
"""
import logging
import os
import re
from dataclasses import dataclass

import pymupdf

from app.legacy_fonts import KRUTIDEV_FONT_RE, fix_text, krutidev_to_unicode
from app.textnorm import garbled_ratio, normalize_devanagari

log = logging.getLogger(__name__)

MIN_TEXT_CHARS = 40          # fewer visible chars than this on a page -> treat as scanned
GARBLED_THRESHOLD = 0.05     # >5% unexpected characters -> OCR instead

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_EN_STOP = re.compile(r"(?i)\b(the|and|of|to|in|is|for|that|with|on|by|this|be|are|as|from|at|"
                      r"or|an|it|was|which|shall|will|has|have|not)\b")


def _image_coverage(page: "pymupdf.Page") -> float:
    area = page.rect.get_area() or 1.0
    return max((pymupdf.Rect(i["bbox"]).get_area() / area for i in page.get_image_info()), default=0.0)


def _is_junk_ocr_layer(page: "pymupdf.Page", text: str) -> bool:
    """A scanned Hindi page whose invisible text layer came from an English-only OCR.

    Seen on real MP government PDFs: the scanner OCR'd Devanagari as Latin, giving text like
    'ar{rIeer ilrg;T TrIIrdI'. It isn't Unicode Hindi, isn't Kruti Dev, and isn't English
    (almost no common English words), and the page is one full-page image.
    """
    words = text.split()
    if len(words) < 20:
        return False
    visible = len("".join(words))
    if len(_DEVANAGARI.findall(text)) > 0.1 * visible:
        return False
    english_ratio = len(_EN_STOP.findall(text)) / len(words)
    return english_ratio < 0.05 and _image_coverage(page) > 0.8


@dataclass
class PageText:
    page: int          # 1-based
    text: str
    method: str


def _page_text_font_aware(page: pymupdf.Page) -> tuple[str, bool]:
    """Rebuild page text from spans, converting runs that use legacy fonts."""
    used_legacy = False
    lines_out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            # merge consecutive spans with the same "is legacy" flag, so that the
            # Kruti Dev i-matra / reph reordering never breaks across a span boundary
            runs: list[list] = []
            for span in line["spans"]:
                legacy = bool(KRUTIDEV_FONT_RE.search(span["font"]))
                if runs and runs[-1][0] == legacy:
                    runs[-1][1] += span["text"]
                else:
                    runs.append([legacy, span["text"]])
            parts = []
            for legacy, text in runs:
                if legacy:
                    used_legacy = True
                    parts.append(krutidev_to_unicode(text))
                else:
                    parts.append(text)
            lines_out.append(" ".join(p.strip() for p in parts if p.strip()))
        lines_out.append("")  # blank line between blocks = paragraph break
    return "\n".join(lines_out).strip(), used_legacy


_ocr_warned = False


def ocr_page(page: pymupdf.Page, lang: str = "eng+hin", dpi: int = 400) -> str | None:
    """OCR a page with Tesseract. Returns None if Tesseract isn't installed."""
    global _ocr_warned
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None
    cmd = os.getenv("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    try:
        return pytesseract.image_to_string(img, lang=lang)
    except (pytesseract.TesseractNotFoundError, pytesseract.TesseractError) as e:
        if not _ocr_warned:
            log.warning("OCR unavailable (%s). Install Tesseract with the Hindi language pack, "
                        "or set TESSERACT_CMD. Scanned pages will be skipped.", e)
            _ocr_warned = True
        return None


def extract_pdf(path: str, fix_legacy: bool = True, ocr: bool = True, ocr_lang: str = "eng+hin") -> list[PageText]:
    pages: list[PageText] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, 1):
            if not fix_legacy:
                pages.append(PageText(i, page.get_text(), "raw"))
                continue

            text, used_font = _page_text_font_aware(page)
            method = "krutidev-font" if used_font else "unicode"
            if not used_font:
                text, converted = fix_text(text)
                if converted:
                    method = "krutidev-text"

            visible = len("".join(text.split()))
            junk_layer = method == "unicode" and _is_junk_ocr_layer(page, text)
            if ocr and (junk_layer or visible < MIN_TEXT_CHARS or garbled_ratio(text) > GARBLED_THRESHOLD):
                ocr_text = ocr_page(page, ocr_lang)
                if ocr_text and (junk_layer or len("".join(ocr_text.split())) >= visible):
                    text, method = ocr_text, "ocr"

            pages.append(PageText(i, normalize_devanagari(text), method))
    return pages
