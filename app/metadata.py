"""Pull order number, date, department and subject out of a government order's text."""
import re
from datetime import date

from app.textnorm import DEVANAGARI_DIGITS

_NO = re.compile(
    r"(?:क्रमांक|क्र\.|आदेश\s*क्र\.?|No\.?|Memo\s*No\.?)\s*[:\-]?\s*"
    r"([^\s].{1,50}?)(?=\s{2,}|\s+दिनांक|\s+dated|\s*,|\n|$)",
    re.I,
)
_DATE = re.compile(r"(?:दिनांक|dated|date|दि\.)\s*[:\-]?\s*(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})", re.I)
_ANY_DATE = re.compile(r"\b(\d{1,2})[./\-](\d{1,2})[./\-](\d{4})\b")
_SUBJECT = re.compile(r"(?:विषय|Subject)\s*[:\-]\s*(.+)", re.I)
_DEPT = re.compile(r"[^\n]*(?:विभाग|Department|निदेशालय|कार्यालय)[^\n]*", re.I)


def _to_date(d: str, m: str, y: str) -> date | None:
    year = int(y) + (2000 if len(y) == 2 else 0)
    try:
        return date(year, int(m), int(d))
    except ValueError:
        return None


def extract_metadata(text: str) -> dict:
    t = text.translate(DEVANAGARI_DIGITS)
    head = t[:1500]
    meta: dict = {"order_no": None, "order_date": None, "department": None, "subject": None}

    if m := _NO.search(head):
        meta["order_no"] = m.group(1).strip(" ,;:-")
    if m := _DATE.search(head) or _ANY_DATE.search(head):
        meta["order_date"] = _to_date(*m.groups())
    if m := _SUBJECT.search(head):
        meta["subject"] = m.group(1).strip()[:300]
    if m := _DEPT.search(head):
        meta["department"] = m.group(0).strip()[:200]
    return meta
