"""Character-level normalisation shared by alignment, inference and scoring."""
from __future__ import annotations

import re
import unicodedata

_INVISIBLE = re.compile("[​‌‍­‎‏﻿]")
_PRIVATE_USE = re.compile("[-]")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE.sub("", text)
    return _PRIVATE_USE.sub("", text)


VI_SPECIFIC_CHARS = frozenset(
    "ăâđêôơư"
    "áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    "ĂÂĐÊÔƠƯ"
    "ÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
)

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def vietnamese_word_share(text: str, max_words: int | None = 20000) -> float:
    words = _WORD.findall(normalize_text(text))
    if max_words is not None:
        words = words[:max_words]
    if not words:
        return 0.0
    hits = sum(1 for w in words if any(ch in VI_SPECIFIC_CHARS for ch in w))
    return hits / len(words)
