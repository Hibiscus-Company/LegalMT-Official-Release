#!/usr/bin/env python3
"""Render results_index.jsonl (written by evaluate.py) as a markdown table.

    uv run python scripts/LegalMT/make_results_table.py
    uv run python scripts/LegalMT/make_results_table.py --split dev
    uv run python scripts/LegalMT/make_results_table.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from LegalMT import runs as R

HEADER = ["run", "split", "direction", "BLEU", "chrF", "chrF++", "term recall", "consistency", "sec/doc", "peak MiB",
          "flags"]


def _num(x: Any, nd: int) -> str:
    return "n/a" if x is None else f"{float(x):.{nd}f}"


def row_flags(rec: dict, m: dict) -> str:
    flags = []
    status = rec.get("run_status")
    if status is not None and status != "done":
        flags.append(f"status {status}")
    for key, label in (("n_missing", "missing"), ("n_empty", "empty"), ("n_wrong_language", "wrong-lang"),
                       ("n_direction_field_mismatch", "dir-field")):
        if m.get(key):
            flags.append(f"{label} {m[key]}/{m.get('n_docs', '?')}")
    return "; ".join(flags)


def latest_per_run(records: list[dict]) -> list[dict]:
    order: list[tuple[str, str]] = []
    last: dict[tuple[str, str], dict] = {}
    for rec in records:
        key = (rec.get("run_name", ""), rec.get("split", ""))
        if key not in last:
            order.append(key)
        last[key] = rec
    return [last[k] for k in order]


def context_flags(records: list[dict]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {i: [] for i in range(len(records))}
    glossaries = {r.get("glossary_sha256") for r in records if r.get("glossary_sha256")}
    if len(glossaries) > 1:
        for i, r in enumerate(records):
            g = r.get("glossary_sha256")
            out[i].append(f"glossary {g[:8]}" if g else "glossary unknown")
    versions: dict[str, set] = {}
    for r in records:
        versions.setdefault(r.get("split", ""), set()).add(r.get("split_version"))
    for i, r in enumerate(records):
        if len(versions.get(r.get("split", ""), ())) > 1:
            out[i].append(f"split version {r.get('split_version')}")
    return out


def render(records: list[dict]) -> str:
    lines = ["| " + " | ".join(HEADER) + " |", "|" + "|".join(["---"] * 3 + ["---:"] * 7 + ["---"]) + "|"]
    ctx = context_flags(records)
    for i, rec in enumerate(records):
        metrics = rec.get("metrics", {})
        run_cost = metrics.get("run") or {}
        peak = run_cost.get("peak_mib_nvidia_smi")
        if peak is None:
            peak = run_cost.get("peak_mib_torch_allocated")
        for d in ("en2vi", "vi2en"):
            m = metrics.get(d)
            if not m:
                continue
            lines.append("| " + " | ".join([
                rec.get("run_name", ""), rec.get("split", ""), d,
                _num(m.get("bleu"), 2), _num(m.get("chrf"), 2), _num(m.get("chrfpp"), 2),
                _num(m.get("term_recall"), 3), _num(m.get("term_consistency"), 3),
                _num(m.get("sec_per_doc"), 2), _num(peak, 0), "; ".join(x for x in [row_flags(rec, m), *ctx[i]] if x),
            ]) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, default=None, help="default <ARTIFACTS>/results_index.jsonl")
    ap.add_argument("--split", default=None)
    ap.add_argument("--tag", default=None, help="keep runs whose tag contains this string")
    ap.add_argument("--all", action="store_true", help="keep every line (re-evaluations included)")
    args = ap.parse_args(argv)

    path = args.index or R.results_index_path()
    if not path.is_file():
        print(f"no results index at {path}")
        return 1
    records, truncated = R.read_jsonl_tolerant(path)
    if truncated:
        print(f"WARNING: {path} ends in a half-written line (skipped)")
    if args.split:
        records = [r for r in records if r.get("split") == args.split]
    if args.tag:
        records = [r for r in records if args.tag in (r.get("tag") or "")]
    if not args.all:
        records = latest_per_run(records)
    print(render(records))
    print(f"\n{len(records)} evaluation(s) from {path}. Metrics are local approximations: whole documents as SacreBLEU segments; "
          f"term metrics on the seed glossary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
