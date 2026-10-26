import pytest

from app.legacy_fonts import (fix_text, krutidev_to_unicode, krutidev_to_unicode_protected,
                              looks_like_krutidev, unicode_to_krutidev)

# Known Kruti Dev 010 encodings of common government-order words
KNOWN = [
    ("jkT; ljdkj", "राज्य सरकार"),
    ("Hkkjr", "भारत"),
    ("vkns'k", "आदेश"),
    ("fnukad", "दिनांक"),
    ("deZpkfj;ksa", "कर्मचारियों"),
    ("fo|ky;", "विद्यालय"),
    ("iwoZ", "पूर्व"),
    ("dk;kZy;", "कार्यालय"),
    ("vf/kdkjh", "अधिकारी"),
    ("f'k{kd", "शिक्षक"),
]


@pytest.mark.parametrize("kd,uni", KNOWN)
def test_known_words(kd, uni):
    assert krutidev_to_unicode(kd) == uni


@pytest.mark.parametrize("word", [
    "महंगाई भत्ता", "प्रधानमंत्री", "स्थानांतरण", "निर्देशित", "मध्यप्रदेश शासन",
    "सामान्य प्रशासन विभाग", "क्रमांक", "धर्म", "स्त्री", "श्रमिक", "जिलाधिकारी",
    "छात्रवृत्ति", "ई-केवाईसी", "पेंशनरों", "ग्रीष्मकालीन अवकाश",
])
def test_round_trip(word):
    assert krutidev_to_unicode(unicode_to_krutidev(word)) == word


def test_detection():
    assert looks_like_krutidev("jkT; 'kklu }kjk ;g fu.kZ; fy;k x;k gS fd deZpkfj;ksa dks")
    assert not looks_like_krutidev("The state government has decided to increase the allowance.")
    assert not looks_like_krutidev("राज्य शासन द्वारा यह निर्णय लिया गया है")
    assert not looks_like_krutidev("")


def test_dates_protected_in_text_mode():
    kd = "fnukad 15/03/2024 ls dk;kZy; 10:00 cts"
    out = krutidev_to_unicode_protected(kd)
    assert "15/03/2024" in out and "10:00" in out
    assert out.startswith("दिनांक")


def test_fix_text_leaves_unicode_alone():
    t = "यह आदेश तत्काल प्रभाव से लागू होगा।"
    assert fix_text(t) == (t, False)
