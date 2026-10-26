"""Build the sample PDFs in data/samples/ from scripts/sample_orders.py.

You do NOT need to run this: the generated PDFs are already in the repo.
It's here so the test data is reproducible and transparent.

    python scripts/make_samples.py --devanagari-font Hind-Regular.ttf --latin-font DejaVuSans.ttf

How each kind is simulated:
  unicode  - Hindi text printed to PDF by headless Chromium (normal modern PDF)
  krutidev - text converted to Kruti Dev codepoints and drawn with a Latin font that is
             renamed 'KrutiDev010', so the PDF looks to extractors exactly like a real
             legacy document. Dates/times/numbers go in a separate 'Times' span, which
             is how real legacy documents are usually typed.
  scanned  - the unicode page rendered to an image (no text layer at all)
"""
import argparse
import io
import re
import sys
from pathlib import Path

import pymupdf as fitz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.legacy_fonts import unicode_to_krutidev  # noqa: E402
from sample_orders import ORDERS, order_text  # noqa: E402

OUT = ROOT / "data" / "samples"
PAGE = fitz.paper_rect("a4")
MARGIN = 56
ASCII_TOKEN = re.compile(r"^[0-9A-Za-z/:.,\-()]+$")


def renamed_font(latin_ttf: Path, new_name: str = "KrutiDev010") -> bytes:
    from fontTools.ttLib import TTFont

    f = TTFont(str(latin_ttf))
    for rec in f["name"].names:
        if rec.nameID in (1, 3, 4, 6, 16):
            rec.string = new_name
    buf = io.BytesIO()
    f.save(buf)
    return buf.getvalue()


def unicode_pdf(text: str, dev_font: Path) -> fitz.Document:
    """Print HTML to PDF with headless Chromium (proper Devanagari shaping)."""
    import html as htmllib
    import tempfile

    from playwright.sync_api import sync_playwright

    body = "".join(f"<p>{htmllib.escape(line)}</p>" for line in text.split("\n"))
    page_html = (
        "<html><head><meta charset='utf-8'><style>"
        f"@font-face{{font-family:dev;src:url('{dev_font.resolve().as_uri()}');}}"
        "body{font-family:dev;font-size:12pt;line-height:1.6;margin:0}"
        "</style></head><body>" + body + "</body></html>"
    )
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "o.html", Path(tmp) / "o.pdf"
        src.write_text(page_html, encoding="utf-8")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            pg = browser.new_page()
            pg.goto(src.as_uri())
            pg.wait_for_timeout(300)
            pg.pdf(path=str(out), format="A4", margin={k: "20mm" for k in ("top", "bottom", "left", "right")})
            browser.close()
        return fitz.open(stream=out.read_bytes(), filetype="pdf")


def krutidev_pdf(text: str, kd_font_bytes: bytes) -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    kd = fitz.Font(fontbuffer=kd_font_bytes)
    page.insert_font(fontname="KD", fontbuffer=kd_font_bytes)
    latin = fitz.Font("tiro")
    size, lead = 12, 18
    x0, x1 = MARGIN, PAGE.width - MARGIN
    y = MARGIN + size

    for line in text.split("\n"):
        x = x0
        for token in line.split(" "):
            if not token:
                x += kd.text_length(" ", size)
                continue
            if ASCII_TOKEN.match(token):
                font, name, glyphs = latin, "tiro", token
            else:
                font, name, glyphs = kd, "KD", unicode_to_krutidev(token)
            w = font.text_length(glyphs, size)
            if x + w > x1 and x > x0:
                x, y = x0, y + lead
            page.insert_text((x, y), glyphs, fontname=name, fontsize=size)
            x += w + kd.text_length(" ", size)
        y += lead
    return doc


def scanned_pdf(text: str, dev_font: Path) -> fitz.Document:
    src = unicode_pdf(text, dev_font)
    pix = src[0].get_pixmap(dpi=200, colorspace=fitz.csGRAY)
    doc = fitz.open()
    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    page.insert_image(page.rect, stream=pix.tobytes("png"))
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devanagari-font", required=True, type=Path, help="e.g. Hind-Regular.ttf (OFL)")
    ap.add_argument("--latin-font", required=True, type=Path, help="e.g. DejaVuSans.ttf")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    kd_bytes = renamed_font(args.latin_font)
    for o in ORDERS:
        text = order_text(o)
        if o["kind"] == "unicode":
            doc = unicode_pdf(text, args.devanagari_font)
        elif o["kind"] == "krutidev":
            doc = krutidev_pdf(text, kd_bytes)
        else:
            doc = scanned_pdf(text, args.devanagari_font)
        doc.set_metadata({"title": o["subject"], "subject": "Sample (fictional) order - test data"})
        path = OUT / f"{o['id']}.pdf"
        doc.subset_fonts()
        doc.save(path, garbage=4, deflate=True)
        print(f"{o['kind']:9} {path.name}")


if __name__ == "__main__":
    main()
