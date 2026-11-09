"""Typed loading of the raw LegalMT data. No analysis here."""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass

import pandas as pd

from .paths import split_path

csv.field_size_limit(min(sys.maxsize, 100_000_000))

EXPECTED_COLUMNS = [
    "sample_id",
    "category",
    "english_path",
    "vietnamese_path",
    "english_markdown",
    "vietnamese_markdown",
]


@dataclass(frozen=True)
class LegalMTSplit:
    name: str
    df: pd.DataFrame

    def __len__(self) -> int:
        return len(self.df)


def load_split(split: str) -> LegalMTSplit:
    path = split_path(split)
    if not path.is_file():
        raise FileNotFoundError(f"No data file for split '{split}' at {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Split '{split}' at {path} is missing columns: {missing}")
    return LegalMTSplit(name=split, df=df)


def load_train() -> LegalMTSplit:
    return load_split("train")


def load_public_test() -> LegalMTSplit:
    return load_split("public_test")
