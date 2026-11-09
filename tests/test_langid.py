"""Tests for LegalMT.langid on synthetic strings (no organizer text)."""
from __future__ import annotations

import pytest

from LegalMT.langid import VI_THRESHOLD, detect_lang, direction_of, other_lang
from LegalMT.quality import column_language

EN = "The applicant shall submit the form within ten working days after the notice is received."
VI = "Người nộp đơn phải nộp tờ khai trong thời hạn mười ngày làm việc kể từ ngày nhận thông báo."


def test_detects_both_languages():
    assert detect_lang(EN) == "en"
    assert detect_lang(VI) == "vi"
    import unicodedata

    assert detect_lang(unicodedata.normalize("NFD", VI)) == "vi"


def test_threshold_and_empty():
    assert VI_THRESHOLD == 0.2
    assert detect_lang("") == "en"
    assert detect_lang("12/2031 - 45 67") == "en"
    assert detect_lang("alpha beta gamma delta được") == "en"
    assert detect_lang("alpha beta gamma được được") == "vi"


def test_agrees_with_quality_column_language():
    for text in (EN, VI, EN + " " + VI, "Điều 5. Scope of application and general terms"):
        assert detect_lang(text) == column_language(text)[0]


def test_other_lang_and_direction():
    assert other_lang("en") == "vi" and other_lang("vi") == "en"
    assert direction_of("en") == "en2vi" and direction_of("vi") == "vi2en"
    with pytest.raises(ValueError):
        other_lang("fr")
