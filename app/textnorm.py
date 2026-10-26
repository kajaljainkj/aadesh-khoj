"""Clean up Devanagari text that comes out of PDFs."""
import re
import unicodedata

_DEPENDENT_VOWELS = "ािीुूृॄेैोौंँ़"
# Same matra twice in a row is never valid Hindi; PDF ToUnicode maps often produce it
_DOUBLE_MATRA = re.compile(f"([{_DEPENDENT_VOWELS}])\\1+")
_INVISIBLE = re.compile("[​‌‍­﻿]")
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize_devanagari(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE.sub("", text)
    text = _DOUBLE_MATRA.sub(r"\1", text)
    text = text.replace("।।", "॥")  # ।। -> ॥
    return text


# Characters we expect in Hindi/English government documents
_EXPECTED = re.compile(r"[ऀ-ॿ -~\s–—‘’“”₹ ]")


def garbled_ratio(text: str) -> float:
    """Share of visible characters outside Devanagari / ASCII / common punctuation.

    Broken ToUnicode maps in PDFs produce Greek, Latin-extended or private-use characters
    in the middle of Hindi words (e.g. 'का͛नύ क'). High ratio -> OCR the page instead.
    """
    visible = [c for c in text if not c.isspace()]
    if not visible:
        return 0.0
    bad = sum(1 for c in visible if not _EXPECTED.match(c))
    return bad / len(visible)
