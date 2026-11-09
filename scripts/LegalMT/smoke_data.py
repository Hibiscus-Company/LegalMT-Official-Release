#!/usr/bin/env python3
"""Smoke test for LegalMT's raw data: counts and structural invariants only, no EDA."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from LegalMT.data import load_public_test, load_train
from LegalMT.metrics import self_test
from LegalMT.paths import load_config

EXPECTED_ROWS: dict[str, int] = load_config("dataset")["expected_rows"]


def check_split(name: str, split) -> list[str]:
    failures: list[str] = []
    df = split.df
    n = len(df)

    expected = EXPECTED_ROWS[name]
    print(f"[{name}] rows: {n} (expected {expected})")
    if n != expected:
        failures.append(
            f"[{name}] row count mismatch: got {n}, expected {expected} (per Description.pdf)"
        )

    n_unique_ids = df["sample_id"].nunique()
    print(f"[{name}] unique sample_id: {n_unique_ids}")
    if n_unique_ids != n:
        failures.append(
            f"[{name}] sample_id is not unique: {n_unique_ids} unique out of {n} rows"
        )

    empty_en = int((df["english_markdown"].str.strip() == "").sum())
    empty_vi = int((df["vietnamese_markdown"].str.strip() == "").sum())
    print(f"[{name}] empty english_markdown: {empty_en}")
    print(f"[{name}] empty vietnamese_markdown: {empty_vi}")
    if empty_en:
        failures.append(f"[{name}] {empty_en} rows have empty english_markdown")
    if empty_vi:
        failures.append(f"[{name}] {empty_vi} rows have empty vietnamese_markdown")

    cat_counts = df["category"].value_counts().to_dict()
    print(f"[{name}] category counts: {cat_counts}")

    return failures


def main() -> int:
    failures: list[str] = []

    print("=== LegalMT smoke test (counts only) ===")

    print("\n-- loading train --")
    train = load_train()
    failures += check_split("train", train)

    print("\n-- loading public_test --")
    public_test = load_public_test()
    failures += check_split("public_test", public_test)

    print("\n-- metric self-test (SacreBLEU/chrF, document-level scoring) --")
    sample = train.df.iloc[0]["vietnamese_markdown"]
    result = self_test(sample)
    print(f"self-test BLEU: {result.bleu:.4f}  chrF: {result.chrf:.4f}  (expect both ~100.0)")
    if result.bleu < 99.9 or result.chrf < 99.9:
        failures.append(
            f"metric self-test failed: reference-vs-itself scored BLEU={result.bleu:.4f}, "
            f"chrF={result.chrf:.4f} (expected both ~100.0)"
        )

    print()
    if failures:
        print(f"FAIL: {len(failures)} invariant(s) violated:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("PASS: all invariants held.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
