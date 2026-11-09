#!/usr/bin/env python3
"""Secondary views of a dev-split run, so decisions do not hinge on quirks of the 40 dev documents.

    uv run python scripts/LegalMT/dev_views.py RUN
    uv run python scripts/LegalMT/dev_views.py RUN_A RUN_B
    uv run python scripts/LegalMT/dev_views.py RUN --mean novel
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import sacrebleu

from LegalMT import metrics as M
from LegalMT import runs as R
from LegalMT import splits as S

OUTLIERS = ("So_huu_tri_tue/46",)
REF_COL = {"en2vi": "vietnamese_markdown", "vi2en": "english_markdown"}
DOTS = re.compile(r"\.{4,}|(?:\. ){3,}\.")


def load(run_arg: str) -> tuple[str, dict[str, dict[str, str]], str]:
    run = R.load_run(run_arg)
    hyps: dict[str, dict[str, str]] = {}
    for h in run.read_jsonl("hyps.jsonl"):
        hyps.setdefault(h["direction"], {})[h["sample_id"]] = h["hyp"]
    return run.name, hyps, run.manifest.get("split_version") or "v0"


def views(ids: list[str], info: dict) -> dict[str, list[str]]:
    return {
        "all": ids,
        "no_outlier": [s for s in ids if s not in OUTLIERS],
        "novel": [s for s in ids if not info[s].get("near_relative")],
        "no_dots": ids,
    }


def score(hyps: dict[str, str], refs: dict[str, str], ids: list[str], collapse: bool) -> tuple[float, float]:
    f = (lambda t: DOTS.sub("...", t)) if collapse else (lambda t: t)
    h = [M.prepare_for_bleu(f(hyps[s])) for s in ids]
    r = [[M.prepare_for_bleu(f(refs[s])) for s in ids]]
    return sacrebleu.corpus_bleu(h, r).score, sacrebleu.corpus_chrf(h, r).score


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="one run (views) or two runs (B - A per view)")
    ap.add_argument("--mean", choices=["all", "no_outlier", "novel", "no_dots"], default=None,
                    help="print only the mean BLEU over both directions in this view (one run)")
    args = ap.parse_args(argv)
    if len(args.runs) > 2 or (args.mean and len(args.runs) != 1):
        ap.error("give one or two runs (exactly one with --mean)")
    loaded = [load(r) for r in args.runs]
    version = loaded[0][2]
    info = S.load_split_spec(version).doc_info
    df = S.docs_for("dev", version)
    if args.mean:
        _, hyps, _ = loaded[0]
        vals = []
        for d in ("en2vi", "vi2en"):
            if not hyps.get(d):
                print("none")
                return 1
            refs = dict(zip(df.sample_id, df[REF_COL[d]]))
            ids = views(sorted(hyps[d]), info)[args.mean]
            vals.append(score(hyps[d], refs, ids, args.mean == "no_dots")[0])
        print(f"{sum(vals) / len(vals):.3f}")
        return 0
    print(f"runs: {', '.join(n for n, _, _ in loaded)} (dev {version})")
    print("| view | dir | docs | " + (" | ".join(["BLEU", "chrF"]) if len(loaded) == 1 else "BLEU A | BLEU B | B - A | chrF B - A") + " |")
    for d in ("en2vi", "vi2en"):
        refs = dict(zip(df.sample_id, df[REF_COL[d]]))
        common = sorted(set.intersection(*[set(h.get(d, {})) for _, h, _ in loaded]))
        if not common:
            continue
        for v, ids in views(common, info).items():
            sc = [score(h[d], refs, ids, v == "no_dots") for _, h, _ in loaded]
            if len(sc) == 1:
                print(f"| {v} | {d} | {len(ids)} | {sc[0][0]:.2f} | {sc[0][1]:.2f} |")
            else:
                print(f"| {v} | {d} | {len(ids)} | {sc[0][0]:.2f} | {sc[1][0]:.2f} | {sc[1][0] - sc[0][0]:+.2f} | "
                      f"{sc[1][1] - sc[0][1]:+.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
