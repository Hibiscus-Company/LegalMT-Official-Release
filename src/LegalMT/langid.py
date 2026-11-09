"""Source-language detection for inference: English or Vietnamese, nothing else."""
from __future__ import annotations

from .textnorm import vietnamese_word_share

VI_THRESHOLD = 0.2
LANGS = ("en", "vi")


def detect_lang(text: str, threshold: float = VI_THRESHOLD) -> str:
    return "vi" if vietnamese_word_share(text) > threshold else "en"


def other_lang(lang: str) -> str:
    if lang not in LANGS:
        raise ValueError(f"language must be one of {LANGS}, got {lang!r}")
    return "vi" if lang == "en" else "en"


def direction_of(src_lang: str) -> str:
    return f"{src_lang}2{other_lang(src_lang)}"
