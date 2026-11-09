#!/usr/bin/env python3
"""Sample aligned links for a manual spot check.

    uv run python scripts/LegalMT/inspect_alignment.py --n 50
    uv run python scripts/LegalMT/inspect_alignment.py --n 50 --only-gc-nm --context
    uv run python scripts/LegalMT/inspect_alignment.py --dropped --n 30
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from LegalMT.paths import artifacts, load_config

ZONES = ("header", "signature")


def is_gc_nm(p: dict) -> bool:
    return p["method"] == "gc" and p["link"] not in ("1-1", "1-0", "0-1") and p.get("anchor") not in ZONES


def read_pairs(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def clip(text: str, max_chars: int) -> str:
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + f" [... {len(text) - max_chars} more chars]"
    return text


def format_link(k: int, n: int, p: dict, max_chars: int, neighbours: tuple[dict | None, dict | None] | None) -> str:
    head = (f"========== [{k}/{n}] {p['pair_id']}  link={p['link']} method={p['method']} anchor={p.get('anchor')} "
            f"len_ratio={p['len_ratio']} num_f1={p['num_f1']} cost={p['cost']}"
            + (f" DROPPED={p['drop_reason']}" if p.get("drop_reason") else ""))
    lines = [head, f"EN units {p['en_units']} uids {p.get('en_uids')} | VI units {p['vi_units']} uids {p.get('vi_uids')}"]
    if neighbours and neighbours[0]:
        prev = neighbours[0]
        lines += [f"  (prev #{prev['order']}) EN: {clip(prev['en'], 200)}", f"  (prev #{prev['order']}) VI: {clip(prev['vi'], 200)}"]
    lines += [f"EN: {clip(p['en'], max_chars)}", f"VI: {clip(p['vi'], max_chars)}"]
    if neighbours and neighbours[1]:
        nxt = neighbours[1]
        lines += [f"  (next #{nxt['order']}) EN: {clip(nxt['en'], 200)}", f"  (next #{nxt['order']}) VI: {clip(nxt['vi'], 200)}"]
    lines += ["verdict: [ ] ok   [ ] partial   [ ] wrong   note:", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=("train", "public_test"))
    ap.add_argument("--version", default=None, help="aligned version (default: configs/LegalMT/align.yaml version)")
    ap.add_argument("--n", type=int, default=50, help="number of links to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-gc-nm", action="store_true", help="only Gale-Church n-m links from unequal-count chunks")
    ap.add_argument("--dropped", action="store_true", help="sample from <split>.dropped.jsonl (build with --keep-dropped)")
    ap.add_argument("--context", action="store_true", help="also show the previous and next link of the same doc")
    ap.add_argument("--max-chars", type=int, default=1500, help="truncate each side beyond this many chars (0 = never)")
    ap.add_argument("--name", default=None, help="output file stem (default derived from the options)")
    args = ap.parse_args()

    version = args.version or load_config("align").get("version", "v0")
    base = artifacts("aligned", version)
    src = base / (f"{args.split}.dropped.jsonl" if args.dropped else f"{args.split}.jsonl")
    if not src.is_file():
        print(f"missing {src}: run scripts/LegalMT/build_aligned.py first" + (" with --keep-dropped" if args.dropped else ""))
        return 1
    pairs = read_pairs(src)
    pool = [p for p in pairs if is_gc_nm(p)] if args.only_gc_nm else pairs
    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n, len(pool)))
    sample.sort(key=lambda p: (p["sample_id"], p["order"]))

    by_doc: dict[str, list[dict]] = {}
    if args.context:
        for p in pairs:
            by_doc.setdefault(p["sample_id"], []).append(p)

    def neighbours(p: dict) -> tuple[dict | None, dict | None]:
        doc = by_doc.get(p["sample_id"], [])
        pos = next((k for k, q in enumerate(doc) if q["order"] == p["order"]), None)
        if pos is None:
            return None, None
        return (doc[pos - 1] if pos > 0 else None, doc[pos + 1] if pos + 1 < len(doc) else None)

    filt = "gc-nm-unequal" if args.only_gc_nm else "all"
    name = args.name or f"{args.split}_{'dropped_' if args.dropped else ''}{filt}_n{len(sample)}_seed{args.seed}"
    out_dir = base / "inspect"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.txt"
    header = [
        f"# Alignment spot check: split={args.split} version={version} source={src.name} filter={filt} "
        f"n={len(sample)} seed={args.seed} population={len(pool)}",
        "# Each link is one line per side (n-m links join their paragraphs with one space).",
        "# Mark each verdict: ok (translation of each other), partial (overlap but extra/missing text), wrong.",
        "",
    ]
    body = [format_link(k + 1, len(sample), p, args.max_chars, neighbours(p) if args.context else None)
            for k, p in enumerate(sample)]
    out.write_text("\n".join(header + body), encoding="utf-8")
    links = {}
    for p in sample:
        links[p["link"]] = links.get(p["link"], 0) + 1
    print(f"pool {len(pool)} links, sampled {len(sample)} from {len({p['sample_id'] for p in sample})} docs, link types {links}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
