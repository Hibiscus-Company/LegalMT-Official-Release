#!/usr/bin/env python3
"""Validate the LegalMT glossary against the train_fit documents (document-level co-occurrence).

    uv run python scripts/LegalMT/check_glossary.py
    uv run python scripts/LegalMT/check_glossary.py --rewrite
    uv run python scripts/LegalMT/check_glossary.py --limit 100
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from LegalMT import terminology as T
from LegalMT.paths import artifacts
from LegalMT.splits import docs_for

TABLE_COLUMNS = ["en", "vi", "note", "n_en_docs", "n_vi_docs", "n_both_docs", "cooc_en2vi", "cooc_vi2en",
                 "en_occurrences", "vi_occurrences", "keep", "reason"]


def judge(row: dict, min_docs: int, min_cooc: float) -> tuple[bool, str]:
    reasons = []
    if row["n_en_docs"] < min_docs:
        reasons.append(f"n_en_docs {row['n_en_docs']} < {min_docs}")
    if row["n_vi_docs"] < min_docs:
        reasons.append(f"n_vi_docs {row['n_vi_docs']} < {min_docs}")
    if row["cooc_en2vi"] < min_cooc:
        reasons.append(f"cooc_en2vi {row['cooc_en2vi']:.2f} < {min_cooc}")
    if row["cooc_vi2en"] < min_cooc:
        reasons.append(f"cooc_vi2en {row['cooc_vi2en']:.2f} < {min_cooc}")
    return (not reasons), "; ".join(reasons)


def rewrite_glossary(path: Path, entries: list[T.GlossaryEntry], keep: list[bool]) -> None:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    head = []
    for ln in lines:
        head.append(ln)
        if ln.strip() and not ln.startswith("#"):
            break
    if not head or head[-1].split("\t")[:2] != ["en", "vi"]:
        raise ValueError(f"{path}: could not find the 'en<TAB>vi...' header row after the comment block")
    kept = [T.format_entry(T.GlossaryEntry(**{**e.__dict__, "dropped": False})) for e, k in zip(entries, keep) if k]
    dropped = [T.format_entry(T.GlossaryEntry(**{**e.__dict__, "dropped": True})) for e, k in zip(entries, keep) if not k]
    body = head + kept
    if dropped:
        body += ["# Rejected by scripts/LegalMT/check_glossary.py (reasons in <ARTIFACTS>/glossary/check_v0.tsv):"] + dropped
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glossary", type=Path, default=T.default_glossary_path())
    ap.add_argument("--split-version", default="v0")
    ap.add_argument("--out", type=Path, default=None, help="default <ARTIFACTS>/glossary/check_v0.tsv")
    ap.add_argument("--min-docs", type=int, default=5)
    ap.add_argument("--min-cooc", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=None, help="only the first N train_fit docs (quick look)")
    ap.add_argument("--rewrite", action="store_true", help="rewrite --glossary: rejected entries become #DROPPED lines")
    args = ap.parse_args(argv)

    entries = T.load_glossary(args.glossary, include_dropped=True)
    df = docs_for("train_fit", args.split_version)
    if args.limit:
        df = df.head(args.limit)
    print(f"glossary {args.glossary.name}: {len(entries)} entries ({sum(e.dropped for e in entries)} previously #DROPPED); "
          f"train_fit docs: {len(df)}")
    rows = T.cooccurrence_stats(entries, df["english_markdown"].tolist(), df["vietnamese_markdown"].tolist())
    keep = []
    for r in rows:
        ok, why = judge(r, args.min_docs, args.min_cooc)
        r["keep"], r["reason"] = ok, why
        keep.append(ok)

    out = args.out if args.out is not None else artifacts("glossary", f"check_{args.split_version}.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(TABLE_COLUMNS)
        for r in rows:
            w.writerow([f"{r[c]:.4f}" if isinstance(r[c], float) else r[c] for c in TABLE_COLUMNS])
    n_keep = sum(keep)
    print(f"kept {n_keep} / {len(rows)} (min docs {args.min_docs} per side, min co-occurrence {args.min_cooc} both ways)")
    by_note = Counter(r["note"] for r in rows if r["keep"])
    print("kept by note:", dict(sorted(by_note.items())))
    fail = Counter()
    for r in rows:
        if not r["keep"]:
            for part in r["reason"].split("; "):
                fail[part.split(" ")[0]] += 1
    print("rejection reasons (an entry can have several):", dict(fail))
    kept_rows = [r for r in rows if r["keep"]]
    if kept_rows:
        c1 = sorted(r["cooc_en2vi"] for r in kept_rows)
        c2 = sorted(r["cooc_vi2en"] for r in kept_rows)
        print(f"kept entries: median cooc_en2vi {c1[len(c1) // 2]:.2f}, median cooc_vi2en {c2[len(c2) // 2]:.2f}")
    print(f"wrote {out}")
    if args.rewrite:
        rewrite_glossary(args.glossary, entries, keep)
        print(f"rewrote {args.glossary}: {n_keep} kept, {len(rows) - n_keep} #DROPPED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
