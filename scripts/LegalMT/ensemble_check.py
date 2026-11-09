#!/usr/bin/env python3
"""Estimate the gain of selecting among finished dev runs (doc_mbr, unit_mbr, doc_oracle). CPU only.

    .venv/bin/python scripts/LegalMT/ensemble_check.py RUN_1 RUN_2 RUN_3 [RUN_4 ...] --tag ens4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sacrebleu.metrics import BLEU, CHRF

from LegalMT import metrics as M
from LegalMT import paths
from LegalMT import postprocess as P
from LegalMT import runs as R

COLUMNS = {"en": "english_markdown", "vi": "vietnamese_markdown"}
TGT = {"en2vi": "vi", "vi2en": "en"}
MIN_UNIT_CANDIDATES = 3


def _evaluate_module():
    spec = importlib.util.spec_from_file_location("legalmt_evaluate", SCRIPT_DIR / "evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


UTILITIES = ("chrf", "mix", "bleu")
_BLEU = BLEU(effective_order=True)


def _consensus(cands: list[str], chrf: CHRF, utility: str = "chrf") -> int:
    def util(h: str, r: str) -> float:
        if utility == "chrf":
            return chrf.sentence_score(h, [r]).score
        b = _BLEU.sentence_score(h, [r]).score
        return b if utility == "bleu" else (chrf.sentence_score(h, [r]).score + b) / 2

    best, best_i = -1.0, 0
    for i, c in enumerate(cands):
        s = sum(util(c, o) for j, o in enumerate(cands) if j != i) / (len(cands) - 1)
        if s > best + 1e-9:
            best, best_i = s, i
    return best_i


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="member runs (status done, same split); the first is the primary")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--utility", choices=UTILITIES, default="chrf", help="MBR agreement measure (default chrf)")
    args = ap.parse_args(argv)
    if len(args.runs) < 3:
        ap.error("give at least 3 members (with 2, agreement cannot pick a side)")

    members = [R.load_run(r) for r in args.runs]
    for m in members:
        if m.manifest.get("status") != "done":
            ap.error(f"{m.name} has status {m.manifest.get('status')!r}")
    split = members[0].manifest.get("split")
    version = members[0].manifest.get("split_version") or "v0"
    if any(m.manifest.get("split") != split or (m.manifest.get("split_version") or "v0") != version for m in members):
        ap.error("members were made on different splits")
    cfg0 = members[0].manifest.get("config") or {}
    names = [m.name for m in members]

    from LegalMT.splits import docs_for

    rows = {r.sample_id: r for r in docs_for(split, version).itertuples(index=False)}
    hyps = [{(h["sample_id"], h["direction"]): h for h in m.read_jsonl("hyps.jsonl")} for m in members]
    keys = list(hyps[0])
    if any(set(h) != set(keys) for h in hyps):
        ap.error("members do not cover the same documents and directions")
    units = []
    for m in members:
        by_doc: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for u in m.read_jsonl("units.jsonl"):
            by_doc[(u["sample_id"], u["direction"])].append(u)
        units.append(by_doc)

    chrf = CHRF()
    doc_pick: dict[tuple[str, str], int] = {}
    unit_docs: dict[tuple[str, str], str] = {}
    unit_stats = {d: Counter() for d in TGT}
    mismatch = []
    for key in keys:
        sid, direction = key
        src_lang = hyps[0][key]["src_lang"]
        src_text = getattr(rows[sid], COLUMNS[src_lang])
        prepared = [M.prepare_for_bleu(h[key]["hyp"]) for h in hyps]
        doc_pick[key] = _consensus(prepared, chrf, args.utility)
        own = {u["uid"]: u["hyp"] for u in units[0][key]}
        ident, _ = P.rerender(src_text, src_lang, own, cfg0)
        if ident != hyps[0][key]["hyp"]:
            mismatch.append(f"{direction}:{sid}")
            continue
        by_src = []
        for by_doc in units[1:]:
            count = Counter(u["src"] for u in by_doc[key])
            by_src.append({u["src"]: u["hyp"] for u in by_doc[key] if count[u["src"]] == 1})
        count0 = Counter(u["src"] for u in units[0][key])
        chosen = {}
        st = unit_stats[direction]
        for u in units[0][key]:
            cands = [u["hyp"]]
            if count0[u["src"]] == 1:
                cands += [m[u["src"]] for m in by_src if u["src"] in m]
            st["units"] += 1
            if len(cands) < MIN_UNIT_CANDIDATES:
                st["kept_too_few_candidates"] += 1
                chosen[u["uid"]] = u["hyp"]
                continue
            st[f"candidates_{len(cands)}"] += 1
            if len(set(cands)) == 1:
                st["all_identical"] += 1
                chosen[u["uid"]] = u["hyp"]
                continue
            i = _consensus(cands, chrf, args.utility)
            st["changed" if cands[i] != u["hyp"] else "kept_primary_won"] += 1
            chosen[u["uid"]] = cands[i]
        unit_docs[key], _ = P.rerender(src_text, src_lang, chosen, cfg0)
    if mismatch:
        print(f"ABORT: the primary's units do not re-render {len(mismatch)} documents exactly (first: {mismatch[:5]})")
        return 1

    oracle = {}
    for direction in TGT:
        dk = [k for k in keys if k[1] == direction]
        refs = [getattr(rows[k[0]], COLUMNS[TGT[direction]]) for k in dk]
        per_member = [M.per_document_bleu([h[k]["hyp"] for k in dk], refs) for h in hyps]
        best = [max(range(len(members)), key=lambda i: (per_member[i][j], -i)) for j in range(len(dk))]
        o_hyps = [hyps[best[j]][k]["hyp"] for j, k in enumerate(dk)]
        oracle[direction] = {"bleu": M.corpus_scores(o_hyps, refs)["bleu"],
                             "picks": dict(Counter(names[b] for b in best))}

    ev = _evaluate_module()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    derived = {}
    secs = {d: [sum(h[k].get("seconds") or 0 for k in keys if k[1] == d) for h in hyps] for d in TGT}
    for strategy in ("doc_mbr", "unit_mbr"):
        recs = []
        for key in keys:
            if strategy == "doc_mbr":
                rec = dict(hyps[doc_pick[key]][key])
                rec["ensemble_member"] = names[doc_pick[key]]
            else:
                rec = dict(hyps[0][key])
                rec["hyp"] = unit_docs[key]
            rec["seconds"] = sum(h[key].get("seconds") or 0 for h in hyps)
            recs.append(rec)
        extra = {"split_version": version, "derived_from": names, "ensemble_strategy": strategy, "mbr_utility": args.utility,
                 "cost_note": "an ensemble decodes every member: seconds per document are the SUM of the members'; "
                              "peak memory is the base model plus every member's adapter if served as LoRA adapters"}
        run = R.new_run(f"{args.tag}_{strategy}", cfg0, members[0].manifest.get("config_path"), split, extra=extra)
        run.write_jsonl("hyps.jsonl", recs)
        run.finish("done")
        ev.main([str(run.dir), "--no-index"])
        out = run.path("compare_vs_primary.json")
        ev.main(["--compare", str(members[0].dir), str(run.dir), "--out", str(out)])
        derived[strategy] = run.name

    picks = {d: dict(Counter(names[doc_pick[k]] for k in keys if k[1] == d)) for d in TGT}
    report = {"created": datetime.now().isoformat(timespec="seconds"), "members": names, "split": split, "mbr_utility": args.utility,
              "split_version": version, "derived_runs": derived, "doc_mbr_picks": picks,
              "unit_mbr_stats": {d: dict(c) for d, c in unit_stats.items()}, "doc_oracle": oracle,
              "member_seconds_total": secs}
    out = paths.artifacts("analysis", f"ensemble_check_{stamp}_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("\n## ensemble check summary")
    print(f"members: {names}")
    for d in TGT:
        print(f"{d}: doc_mbr picks {picks[d]}; unit_mbr {dict(unit_stats[d])}")
        print(f"{d}: doc oracle BLEU {oracle[d]['bleu']:.2f}, picks {oracle[d]['picks']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
