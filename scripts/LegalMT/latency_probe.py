#!/usr/bin/env python3
"""Per-document serving latency: translate dev documents one at a time through the server's Translator.

    scripts/LegalMT/gpu_run.sh scripts/LegalMT/latency_probe.py --config hymt2_vllm_greedy \\
        --set model.local_path=<merged model> [--set backend.enforce_eager=false ...] \\
        --ref-run REF_RUN --per-dir 8 --tag NAME
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import sacrebleu

from LegalMT import metrics as M
from LegalMT import paths as P
from LegalMT import runs as R
from LegalMT import splits as S
from LegalMT.translate import Translator, load_infer_config, parse_set_overrides

SRC_COL = {"en2vi": "english_markdown", "vi2en": "vietnamese_markdown"}
REF_COL = {"en2vi": "vietnamese_markdown", "vi2en": "english_markdown"}
OUT_DIR = P.artifacts("latency")
WARMUP = {"en": "Article 1. Scope\n\nThis regulation applies to public parks.\n",
          "vi": "Điều 1. Phạm vi\n\nQuy định này áp dụng cho công viên.\n"}


def pick(ids: list[str], lengths: dict[str, int], k: int) -> list[str]:
    order = sorted(ids, key=lambda s: (lengths[s], s))
    if k <= 0 or k >= len(order):
        return order
    idx = sorted({round(i * (len(order) - 1) / (k - 1)) for i in range(k)}) if k > 1 else [0]
    return [order[i] for i in idx]


def corpus(hyps: dict[str, str], refs: dict[str, str], ids: list[str]) -> dict[str, float]:
    h = [M.prepare_for_bleu(hyps[s]) for s in ids]
    r = [[M.prepare_for_bleu(refs[s]) for s in ids]]
    return {"bleu": round(sacrebleu.corpus_bleu(h, r).score, 2), "chrf": round(sacrebleu.corpus_chrf(h, r).score, 2)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--ref-run", required=True)
    ap.add_argument("--per-dir", type=int, default=8)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args(argv)

    run = R.load_run(args.ref_run)
    version = run.manifest.get("split_version") or "v0"
    df = S.docs_for("dev", version)
    ref: dict[str, dict[str, str]] = {}
    for h in run.read_jsonl("hyps.jsonl"):
        ref.setdefault(h["direction"], {})[h["sample_id"]] = h["hyp"]
    src = {d: dict(zip(df.sample_id, df[c])) for d, c in SRC_COL.items()}
    gold = {d: dict(zip(df.sample_id, df[c])) for d, c in REF_COL.items()}

    cfg, _ = load_infer_config(args.config, parse_set_overrides(args.set))
    report: dict = {"tag": args.tag, "config": args.config, "set": args.set, "ref_run": run.name,
                    "started": datetime.now().isoformat(timespec="seconds"), "docs": []}
    got: dict[str, dict[str, str]] = {"en2vi": {}, "vi2en": {}}
    rc = 0
    try:
        t0 = time.monotonic()
        tr = Translator(cfg)
        for lang, text in WARMUP.items():
            tr.translate_text(text, lang, doc_id=f"warmup_{lang}")
        report["load_and_warmup_s"] = round(time.monotonic() - t0, 1)
        report["engine_kwargs"] = getattr(tr.backend, "engine_kwargs", None)
        for d in ("en2vi", "vi2en"):
            ids = [s for s in ref.get(d, {}) if s in src[d]]
            for sid in pick(ids, {s: len(src[d][s]) for s in ids}, args.per_dir):
                t1 = time.monotonic()
                res = tr.translate_text(src[d][sid], d[:2], doc_id=sid)
                sec = time.monotonic() - t1
                got[d][sid] = res.text
                report["docs"].append({
                    "direction": d, "sample_id": sid, "src_chars": len(src[d][sid]), "seconds": round(sec, 2),
                    "bleu_vs_ref_run": round(sacrebleu.corpus_bleu([res.text], [[ref[d][sid]]]).score, 2),
                    "stats": {k: res.stats.get(k) for k in ("chunks", "fallback_units", "truncated_requests",
                                                           "failed_chunks") if k in res.stats}})
                print(f"{d} {sid}: {sec:.1f} s, BLEU vs ref run {report['docs'][-1]['bleu_vs_ref_run']}")
        report["by_direction"] = {}
        for d in ("en2vi", "vi2en"):
            ids = sorted(got[d])
            if ids:
                report["by_direction"][d] = {
                    "docs": len(ids), "seconds": round(sum(r["seconds"] for r in report["docs"] if r["direction"] == d), 1),
                    "probe_vs_gold": corpus(got[d], gold[d], ids), "ref_run_vs_gold": corpus(ref[d], gold[d], ids)}
                print(d, json.dumps(report["by_direction"][d]))
    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}"[:2000]
        report["traceback_tail"] = traceback.format_exc()[-3000:]
        print(report["error"][:500], file=sys.stderr)
        rc = 3
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{args.tag}.json"
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    with path.with_suffix(".hyps.jsonl").open("w", encoding="utf-8") as fh:
        for d in ("en2vi", "vi2en"):
            for sid, hyp in sorted(got[d].items()):
                fh.write(json.dumps({"direction": d, "sample_id": sid, "hyp": hyp}, ensure_ascii=False) + "\n")
    print(f"report: {path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
