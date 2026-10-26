"""Detect and convert legacy (non-Unicode) Hindi font text.

Many Indian government PDFs were typed in Kruti Dev 010. The PDF stores Latin
codepoints that only *look* like Hindi with that font, so text extraction gives
garbage like 'jkT; ljdkj' instead of 'राज्य सरकार'. This module fixes that.

Algorithm ported from the MIT-licensed hindi-font-converter (see _krutidev_tables).
"""
import re

from app._krutidev_tables import KD_TO_UNICODE, UNICODE_TO_KD

# Font names (as reported by PyMuPDF) that use Kruti Dev-compatible encoding.
# DevLys and Kruti Dev share the same keyboard layout.
KRUTIDEV_FONT_RE = re.compile(r"kruti|krutidev|devlys|k010|kd0", re.I)

_MATRAS_AND_VOWELS = set("अआइईउऊएऐओऔािीुूृेैोौं:ँॅ")


def _replace_all(text: str, old: str, new: str) -> str:
    # The JS original loops until no occurrence remains (replacements can create new matches).
    if not old or old in new:
        return text.replace(old, new)
    while old in text:
        text = text.replace(old, new)
    return text


def krutidev_to_unicode(text: str) -> str:
    if not text:
        return text
    s = text
    for old, new in KD_TO_UNICODE:
        s = _replace_all(s, old, new)

    # Glyphs that combine several marks
    s = s.replace("±", "Zं").replace("Æ", "र्f")

    # 'f' is the short-i matra typed BEFORE its consonant; move it after.
    i = s.find("f")
    while i != -1:
        if i + 1 < len(s):
            s = s[:i] + s[i + 1] + "ि" + s[i + 2:]
        else:
            s = s[:i] + s[i + 1:]  # dangling f at end: drop it
        i = s.find("f", i + 1)

    s = s.replace("Ç", "fa").replace("É", "र्fa")
    i = s.find("fa")
    while i != -1:
        if i + 2 < len(s):
            s = s[:i] + s[i + 2] + "िं" + s[i + 3:]
        else:
            s = s[:i] + s[i + 2:]
        i = s.find("fa", i + 2)

    s = s.replace("Ê", "ीZ")

    # i-matra landed after a halant: 'ि्X' -> '्Xि' (conjunct + i)
    i = s.find("ि्")
    while i != -1:
        if i + 2 < len(s):
            s = s[:i] + "्" + s[i + 2] + "ि" + s[i + 3:]
        i = s.find("ि्", i + 2)

    # 'Z' is the reph (half-r) typed AFTER the syllable; move 'र्' before it.
    pos = s.find("Z")
    while pos > 0:
        start = pos - 1
        while start > 0 and s[start] in _MATRAS_AND_VOWELS:
            start -= 1
        s = s[:start] + "र्" + s[start:pos] + s[pos + 1:]
        pos = s.find("Z")
    if pos == 0:  # stray Z at the very start
        s = s[1:]
    return s


_MATRAS_U2K = "ािीुूृेैोौं:ँॅ"


def unicode_to_krutidev(text: str) -> str:
    """Reverse conversion. Used to build realistic legacy-encoded test data."""
    if not text:
        return text
    s = text
    # i-matra: move 'ि' before its consonant (and before any half-consonants joined to it)
    i = s.find("ि")
    while i != -1:
        if i > 0:
            left = s[i - 1]
            s = s[:i - 1] + "f" + left + s[i + 1:]
            p = i - 1  # index of 'f'
            while p >= 2 and s[p - 1] == "्":
                s = s[:p - 2] + "f" + s[p - 2:p] + s[p + 1:]
                p -= 2
        i = s.find("ि", i + 1)

    # reph: 'र्' + syllable -> syllable + 'Z'
    s += "  "
    i = s.find("र्")
    while i != -1:
        z = i + 2
        while z + 1 < len(s) and s[z + 1] in _MATRAS_U2K:
            z += 1
        syll = s[i + 2:z + 1]
        s = s[:i] + syll + "Z" + s[z + 1:]
        i = s.find("र्")
    s = s[:-2]

    # In Kruti Dev '/' renders as 'ध', so a literal slash is typed as '@'
    s = s.replace("/", "@")
    for old, new in UNICODE_TO_KD:
        s = _replace_all(s, old, new)
    return s


# --- detection -------------------------------------------------------------

# Very frequent Kruti Dev "words" for common Hindi words (के, का, की, में, है, और, से, को, किया, दिनांक, आदेश, सरकार).
_KD_MARKERS = re.compile(
    r"(?<![A-Za-z])(ds|dk|dh|esa|gS|vkSj|ls|dks|fd;k|fnukad|vkns'k|ljdkj|;g|gSa|fd|ij|Hkh|rFkk|fy,|}kjk)(?![A-Za-z])"
)
_WORD = re.compile(r"\S+")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_ENGLISH_STOP = re.compile(r"(?i)\b(the|and|of|to|in|is|for|that|with|on|by|this|be|are|as)\b")


def krutidev_score(text: str) -> float:
    """Fraction of words that look like Kruti Dev encoded Hindi (0..1)."""
    words = _WORD.findall(text)
    if not words:
        return 0.0
    markers = len(_KD_MARKERS.findall(text))
    english = len(_ENGLISH_STOP.findall(text))
    # Kruti Dev text is full of these symbols mid-word
    symbol_words = sum(1 for w in words if re.search(r"[A-Za-z][;'\[\]{}~\\|`%^&*+=@#][A-Za-z]|[A-Za-z]k[A-Za-z]S|Z\b", w))
    return max(0.0, (2 * markers + symbol_words - 2 * english)) / len(words)


def looks_like_krutidev(text: str, threshold: float = 0.15) -> bool:
    if len(_DEVANAGARI.findall(text)) > 0.3 * max(1, len(text.replace(" ", ""))):
        return False  # already Unicode Hindi
    return krutidev_score(text) >= threshold


# Dates / times / numbers like 15/03/2024, 10:00, 4-12 are usually typed in an
# English font even inside Kruti Dev documents. Converting them would turn
# '/' into 'ध' and ':' into 'रू', so we protect them in whole-text mode.
_PROTECT = re.compile(r"(?<!\S)\d+(?:[/:.\-]\d+)+(?!\S)")


def krutidev_to_unicode_protected(text: str) -> str:
    out, last = [], 0
    for m in _PROTECT.finditer(text):
        out.append(krutidev_to_unicode(text[last:m.start()]))
        out.append(m.group())
        last = m.end()
    out.append(krutidev_to_unicode(text[last:]))
    return "".join(out)


def fix_text(text: str) -> tuple[str, bool]:
    """Heuristic whole-text fix, used when font names aren't available.

    Returns (text, converted?). Prefer font-aware conversion in extract.py when possible.
    """
    if looks_like_krutidev(text):
        return krutidev_to_unicode_protected(text), True
    return text, False
