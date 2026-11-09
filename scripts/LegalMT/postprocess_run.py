#!/usr/bin/env python3
"""Re-render a finished translation run with source-following output clean-up (CPU only).

    uv run python scripts/LegalMT/postprocess_run.py RUN --rules bold,dashes --evaluate
    uv run python scripts/LegalMT/postprocess_run.py RUN --rules all --tag g1b_clean_all --evaluate
    uv run python scripts/LegalMT/postprocess_run.py RUN --rules none
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from LegalMT import postprocess as P
from LegalMT import runs as R

COLUMNS = {"en": "english_markdown", "vi": "vietnamese_markdown"}
COPIED_MANIFEST_FIELDS = ("split_version", "directions", "n_docs", "subset", "docs_per_call", "translator",
                          "timing", "memory", "throughput", "translate_stats")


def parse_rules(spec: str) -> dict[str, bool]:
    if spec == "all":
        names = list(P.RULES)
    elif spec == "none":
        names = []
    else:
        names = [s.strip() for s in spec.split(",") if s.strip()]
    return P.resolve_rules({n: True for n in names})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="source run directory or run name (status done)")
    ap.add_argument("--rules", default="all", help=f"comma list of {list(P.RULES)}, 'all' or 'none'")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--evaluate", action="store_true", help="score the new run with evaluate.py")
    ap.add_argument("--no-index", action="store_true", help="with --evaluate: do not append to the results index")
    args = ap.parse_args(argv)

    rules = parse_rules(args.rules)
    enabled = [r for r in P.RULES if rules[r]]
    src_run = R.load_run(args.run)
    man = src_run.manifest
    if man.get("status") != "done":
        ap.error(f"{src_run.name} has status {man.get('status')!r}; re-render only finished runs")
    split, version = man.get("split"), man.get("split_version") or "v0"
    cfg = man.get("config") or {}
    if cfg.get("postprocess") and any(cfg["postprocess"].values()):
        ap.error(f"{src_run.name} already has postprocess {cfg['postprocess']}: re-render its source run instead")

    from LegalMT.splits import docs_for

    df = docs_for(split, version)
    rows = {r.sample_id: r for r in df.itertuples(index=False)}
    hyps = src_run.read_jsonl("hyps.jsonl")
    units_by_doc: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    unit_recs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for u in src_run.read_jsonl("units.jsonl"):
        key = (u["sample_id"], u["direction"])
        units_by_doc[key][u["uid"]] = u["hyp"]
        unit_recs[key].append(u)

    t0 = time.monotonic()
    new_hyps, new_units, mismatch = [], [], []
    fired_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for h in hyps:
        key = (h["sample_id"], h["direction"])
        src_text = getattr(rows[h["sample_id"]], COLUMNS[h["src_lang"]])
        ident, _ = P.rerender(src_text, h["src_lang"], units_by_doc[key], cfg)
        if ident != h["hyp"]:
            mismatch.append(f"{h['direction']}:{h['sample_id']}")
            continue
        doc, fired = P.rerender(src_text, h["src_lang"], units_by_doc[key], cfg, rules)
        for names in fired.values():
            for n in names:
                fired_count[h["direction"]][n] += 1
        rec = dict(h)
        rec["hyp"] = doc
        rec["postprocess_fired_units"] = len(fired)
        new_hyps.append(rec)
        for u in unit_recs[key]:
            nu = dict(u)
            if u["uid"] in fired:
                nu["hyp"], _ = P.clean_unit(u["src"], u["hyp"], rules)
                nu["postprocess"] = fired[u["uid"]]
            new_units.append(nu)
    pp_seconds = round(time.monotonic() - t0, 3)
    if mismatch:
        print(f"ABORT: re-rendering without rules does not reproduce {len(mismatch)} of {len(hyps)} documents "
              f"(first: {mismatch[:5]}); the layout/render code or config differs from the run's")
        return 1
    print(f"identity check: {len(hyps)} of {len(hyps)} documents re-render exactly; rules {enabled or 'none'}")
    for d in sorted(fired_count):
        print(f"  {d}: units changed per rule {dict(fired_count[d])}")
    if not enabled:
        return 0

    new_cfg = copy.deepcopy(cfg)
    new_cfg["postprocess"] = rules
    tag = args.tag or f"{man.get('tag')}_pp_{'-'.join(enabled) if len(enabled) < len(P.RULES) else 'all'}"
    extra = {k: man[k] for k in COPIED_MANIFEST_FIELDS if k in man}
    extra.update({"derived_from": src_run.name, "postprocess": rules, "postprocess_seconds": pp_seconds,
                  "postprocess_fired": {d: dict(v) for d, v in fired_count.items()},
                  "cost_note": "timing/memory/throughput copied from derived_from; the clean-up adds postprocess_seconds"})
    new = R.new_run(tag, new_cfg, man.get("config_path"), split, extra=extra)
    new.write_jsonl("hyps.jsonl", new_hyps)
    new.write_jsonl("units.jsonl", new_units)
    new.finish("done")
    print(f"wrote {new.dir}")
    if args.evaluate:
        import importlib.util

        spec = importlib.util.spec_from_file_location("legalmt_evaluate", SCRIPT_DIR / "evaluate.py")
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)
        return ev.main([str(new.dir)] + (["--no-index"] if args.no_index else []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
