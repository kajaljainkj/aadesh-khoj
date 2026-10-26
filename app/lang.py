"""Lightweight language detection: 'hi' (Devanagari Hindi), 'hinglish' (Roman-script Hindi), 'en'.

Generic detectors (langdetect, fastText) usually label Hinglish as English,
which is exactly the case this project cares about, so we use a small heuristic.
"""
import re

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
WORD = re.compile(r"[A-Za-z']+")

# Common Roman-script Hindi words. Kept to words that are rare as English words.
HINGLISH_WORDS = {
    "hai", "hain", "ho", "hota", "hoti", "hote", "hua", "hui", "tha", "thi",
    "ka", "ki", "ke", "ko", "se", "mein", "par", "pe", "aur", "ya", "bhi", "toh",
    "kya", "kaise", "kaisa", "kaisi", "kyun", "kyon", "kab", "kahan", "kaun", "kaunsa", "kaunsi", "konsa", "konsi","kitna", "kitne", "kitni",
    "nahi", "nahin", "na", "mat", "mujhe", "mera", "meri", "mere", "hum", "humein", "aap", "apna", "apni",
    "yeh", "ye", "woh", "wo", "isko", "usko", "iska", "uska", "kuch", "sab", "bahut", "zyada", "kam",
    "karna", "karein", "karte", "karta", "karti", "kar", "kare", "sakte", "sakta", "sakti", "chahiye",
    "raha", "rahi", "rahe", "wala", "wali", "liye", "lie", "agar", "lekin", "phir", "abhi", "jab",
    "ilaaj", "ilaj", "bimari", "bimaari", "dawai", "dawa", "dard", "bukhar", "khana", "peena",
    "sharir", "khoon", "machhar", "failta", "failata", "phailta", "faila", "lakshan", "upay", "tika","shakkar", "yojana", "milta", "milti", "milega", "batao", "bataiye", "samjhao",
}


def detect_language(text: str) -> str:
    """Return 'hi', 'hinglish' or 'en'."""
    if not text or not text.strip():
        return "en"

    deva = len(DEVANAGARI.findall(text))
    latin = len(LATIN.findall(text))
    if deva and deva >= 0.3 * (deva + latin):
        return "hi"

    words = [w.lower() for w in WORD.findall(text)]
    if not words:
        return "en"
    hits = sum(1 for w in words if w in HINGLISH_WORDS)
    if hits >= 2 or (hits >= 1 and len(words) <= 4) or hits / len(words) >= 0.25:
        return "hinglish"
    return "en"


LANGUAGE_NAMES = {
    "hi": "Hindi (Devanagari script)",
    "hinglish": "Hinglish (Hindi written in Roman/English script, the way the user wrote it)",
    "en": "English",
}
