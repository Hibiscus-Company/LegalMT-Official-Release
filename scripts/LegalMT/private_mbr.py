#!/usr/bin/env python3
"""Translate an input with unit-level MBR: one greedy pass plus N sampled passes, selected per unit by chrF.

    scripts/LegalMT/gpu_run.sh scripts/LegalMT/private_mbr.py --input FILE --model MERGED_DIR --out OUT_DIR \
        [--seeds 1,2,3,4] [--extra-model NAME=MERGED_DIR ...] [--extra-seeds 1,2] [--score-public]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from LegalMT import runs as R

SAMPLING = ["generation.do_sample=true", "generation.temperature=0.7", "generation.top_p=0.6",
            "generation.top_k=20", "generation.repetition_penalty=1.0"]
MIN_UNIT_CANDIDATES = 3


def _module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def run_candidate(name: str, sets: list[str], docs: list[dict[str, Any]], a, out: Path, model: str) -> dict[str, Any]:
    from LegalMT.translate import Translator, load_infer_config, parse_set_overrides

    cdir = out / f"cand_{name}"
    cdir.mkdir(parents=True, exist_ok=True)
    mpath = cdir / "model.json"
    if mpath.is_file():
        was = json.loads(mpath.read_text(encoding="utf-8"))["model"]
        if Path(was).resolve() != Path(model).resolve():
            raise SystemExit(f"candidate {name}: {cdir} was made by {was}, not {model}; use another --out or name")
    else:
        mpath.write_text(json.dumps({"model": model}) + "\n", encoding="utf-8")
    tpath, upath = cdir / "translations.jsonl", cdir / "units.jsonl"
    done = {(t["sample_id"], t["direction"]): t for t in read_jsonl(tpath)}
    by_uid: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for u in read_jsonl(upath):
        by_uid.setdefault((u["sample_id"], u["direction"]), {})[u["uid"]] = u
    units = {k: list(v.values()) for k, v in by_uid.items()}
    done = {k: v for k, v in done.items() if k in units}
    cfg, _ = load_infer_config(a.config, parse_set_overrides(list(a.set) + sets + [f"model.local_path={model}"]))
    todo = [d for d in docs if (d["sample_id"], d["direction"]) not in done]
    seconds = 0.0
    if todo:
        tr = Translator(cfg)
        try:
            for src_lang in ("en", "vi"):
                group_all = [d for d in todo if d["src_lang"] == src_lang]
                for k in range(0, len(group_all), max(1, a.docs_per_call)):
                    group = group_all[k:k + a.docs_per_call]
                    t0 = time.monotonic()
                    results = tr.translate_documents([(d["sample_id"], d["text"], d["src_lang"]) for d in group])
                    seconds += time.monotonic() - t0
                    with upath.open("a", encoding="utf-8") as fu, tpath.open("a", encoding="utf-8") as ft:
                        for d, res in zip(group, results):
                            key = (d["sample_id"], d["direction"])
                            recs = [{"sample_id": key[0], "direction": key[1], "uid": u["uid"], "src": u["src"], "hyp": u["hyp"]}
                                    for u in res.units]
                            for r in recs:
                                fu.write(json.dumps(r, ensure_ascii=False) + "\n")
                            t = {"sample_id": key[0], "direction": key[1], "translation": res.text}
                            ft.write(json.dumps(t, ensure_ascii=False) + "\n")
                            done[key], units[key] = t, recs
                    print(f"[{name}] {src_lang}: {min(k + a.docs_per_call, len(group_all))}/{len(group_all)} documents, "
                          f"{seconds:.0f} s", flush=True)
        finally:
            tr.close()
    return {"cfg": cfg, "docs": {k: {"text": v["translation"], "units": units[k]} for k, v in done.items()},
            "seconds": round(seconds, 1)}


def unit_mbr(docs: list[dict[str, Any]], cands: dict[str, dict[str, Any]], cfg0: dict[str, Any]):
    from sacrebleu.metrics import CHRF

    from LegalMT import postprocess as P

    consensus = _module("legalmt_ensemble_check", "ensemble_check.py")._consensus
    chrf = CHRF()
    names = list(cands)
    out, stats, mismatch = {}, Counter(), []
    for d in docs:
        key = (d["sample_id"], d["direction"])
        g = cands[names[0]]["docs"][key]
        own = {u["uid"]: u["hyp"] for u in g["units"]}
        ident, _ = P.rerender(d["text"], d["src_lang"], own, cfg0)
        if ident != g["text"]:
            mismatch.append(key[0])
            out[key] = g["text"]
            continue
        others = []
        for n in names[1:]:
            us = cands[n]["docs"][key]["units"]
            cnt = Counter(u["src"] for u in us)
            others.append({u["src"]: u["hyp"] for u in us if cnt[u["src"]] == 1})
        cnt0 = Counter(u["src"] for u in g["units"])
        chosen = {}
        for u in g["units"]:
            c = [u["hyp"]] + ([m[u["src"]] for m in others if u["src"] in m] if cnt0[u["src"]] == 1 else [])
            stats["units"] += 1
            if len(c) < MIN_UNIT_CANDIDATES or len(set(c)) == 1:
                stats["kept_greedy"] += 1
                chosen[u["uid"]] = u["hyp"]
                continue
            i = consensus(c, chrf)
            stats["changed" if c[i] != u["hyp"] else "greedy_won"] += 1
            chosen[u["uid"]] = c[i]
        out[key], _ = P.rerender(d["text"], d["src_lang"], chosen, cfg0)
    return out, dict(stats), mismatch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True, help="merged model directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="hymt2_final_graphs")
    ap.add_argument("--set", action="append", default=[], metavar="KEY.PATH=VALUE", help="applied to every candidate")
    ap.add_argument("--seeds", default="1,2,3,4", help="sampled candidates, one seed each")
    ap.add_argument("--extra-model", action="append", default=[], metavar="NAME=MERGED_DIR",
                    help="one greedy pass of another merged model joins the pools (repeatable)")
    ap.add_argument("--extra-seeds", default="", help="sampled passes of every --extra-model, for pool_balanced (each also in --seeds)")
    ap.add_argument("--docs-per-call", type=int, default=10)
    ap.add_argument("--score-public", action="store_true", help="rehearsal: score greedy and MBR against the public references")
    a = ap.parse_args(argv)
    pr = _module("legalmt_private_round", "private_round.py")
    docs, report = pr.read_input(Path(a.input))
    print("input: " + json.dumps({k: v for k, v in report.items() if k != "langid_disagrees"}, ensure_ascii=False, default=str))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    extras = dict(x.split("=", 1) for x in a.extra_model)
    if len(extras) != len(a.extra_model) or not all(k and v for k, v in extras.items()):
        ap.error("--extra-model takes distinct NAME=MERGED_DIR")
    for k, v in extras.items():
        if not (Path(v) / "config.json").is_file():
            ap.error(f"--extra-model {k}: no merged model at {v}")
    if len(seeds) + 1 + len(extras) < MIN_UNIT_CANDIDATES:
        ap.error(f"give at least {MIN_UNIT_CANDIDATES - 1} seeds or other models")
    xseeds = [int(s) for s in a.extra_seeds.split(",") if s.strip()]
    if xseeds and (not extras or not set(xseeds) <= set(seeds)):
        ap.error("--extra-seeds needs --extra-model, and each of its seeds in --seeds")
    from LegalMT.profiling import GpuMemorySampler, memory_report
    sampler = GpuMemorySampler(interval=0.2).start()
    cands: dict[str, dict[str, Any]] = {}
    try:
        cands["greedy"] = run_candidate("greedy", [], docs, a, out, a.model)
        g0 = {k: v["text"] for k, v in cands["greedy"]["docs"].items()}
        if all((d["sample_id"], d["direction"]) in g0 for d in docs):
            early = pr.write_submission(docs, g0, out / "cand_greedy", datetime.now().strftime("%Y%m%d-%H%M%S"))
            print(f"greedy zip, written before the other candidates: {early['zip']} ({early['rows']} rows, "
                  f"sha256 {early['zip_sha256']})", flush=True)
        for s in seeds:
            cands[f"s{s}"] = run_candidate(f"s{s}", SAMPLING + [f"generation.seed={s}"], docs, a, out, a.model)
        for k, v in extras.items():
            cands[f"m_{k}"] = run_candidate(f"m_{k}", [], docs, a, out, v)
        for k, v in extras.items():
            for s in xseeds:
                cands[f"m_{k}_s{s}"] = run_candidate(f"m_{k}_s{s}", SAMPLING + [f"generation.seed={s}"], docs, a, out, v)
    finally:
        sampler.stop()
    for n, c in cands.items():
        miss = [(d["sample_id"], d["direction"]) for d in docs if (d["sample_id"], d["direction"]) not in c["docs"]]
        if miss:
            raise SystemExit(f"candidate {n}: {len(miss)} documents untranslated, e.g. {miss[:3]}")
    xnames = {f"m_{k}_s{s}" for k in extras for s in xseeds}
    t0 = time.monotonic()
    mbr, stats, mismatch = unit_mbr(docs, {n: c for n, c in cands.items() if n not in xnames}, cands["greedy"]["cfg"])
    mbr_s = round(time.monotonic() - t0, 1)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    greedy = {k: v["text"] for k, v in cands["greedy"]["docs"].items()}
    sub_g = pr.write_submission(docs, greedy, out / "cand_greedy", stamp)
    sub_m = pr.write_submission(docs, mbr, out, stamp)
    pools: dict[str, Any] = {}
    texts = {"greedy": greedy, "mbr": mbr}
    if extras:
        subsets = {"samples": ["greedy"] + [f"s{s}" for s in seeds], "models": ["greedy"] + [f"m_{k}" for k in extras]}
        if xseeds:
            subsets["balanced"] = (["greedy"] + [f"s{s}" for s in xseeds]
                                   + [n for k in extras for n in [f"m_{k}"] + [f"m_{k}_s{s}" for s in xseeds]])
        for pn, members in subsets.items():
            if len(members) < MIN_UNIT_CANDIDATES:
                continue
            ptext, pstats, pmis = unit_mbr(docs, {n: cands[n] for n in members}, cands["greedy"]["cfg"])
            (out / f"pool_{pn}").mkdir(exist_ok=True)
            pools[pn] = {"members": members, "mbr_unit_stats": pstats, "greedy_rerender_mismatch_docs": pmis,
                         "submission": pr.write_submission(docs, ptext, out / f"pool_{pn}", stamp)}
            texts[pn] = ptext
    weights = Path(a.model) / "model.safetensors"
    manifest = {"created": R.now_iso(), "input": report, "model": a.model,
                "model_sha256": pr.sha256_file(weights) if weights.is_file() else None,
                "extra_models": {k: {"model": v, "model_sha256": pr.sha256_file(Path(v) / "model.safetensors")
                                     if (Path(v) / "model.safetensors").is_file() else None} for k, v in extras.items()},
                "config": a.config, "set": a.set, "seeds": seeds, "extra_seeds": xseeds, "sampling": SAMPLING,
                "mbr_members": [n for n in cands if n not in xnames],
                "candidate_seconds": {n: c["seconds"] for n, c in cands.items()}, "mbr_seconds": mbr_s,
                "mbr_unit_stats": stats, "greedy_rerender_mismatch_docs": mismatch,
                "code_fingerprint": R.code_fingerprint(), "memory": memory_report(sampler),
                "submission_mbr": sub_m, "submission_greedy": sub_g, "pools": pools}
    if a.score_public:
        manifest["public_score"] = {n: pr.score_public(docs, x) for n, x in texts.items()}
        print("public score (rehearsal): " + json.dumps(manifest["public_score"]))
    (out / f"manifest_mbr_{stamp}.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str) + "\n",
                                                    encoding="utf-8")
    print(f"MBR units: {stats}; greedy re-render mismatches: {len(mismatch)}")
    print(f"wrote MBR {sub_m['zip']} ({sub_m['rows']} rows, sha256 {sub_m['zip_sha256']}) and greedy {sub_g['zip']} "
          f"({sub_g['rows']} rows); not in the target language: MBR {len(sub_m['translation_not_in_target_language'])}, "
          f"greedy {len(sub_g['translation_not_in_target_language'])}")
    for pn, pv in pools.items():
        s = pv["submission"]
        print(f"pool {pn} ({'+'.join(pv['members'])}): {s['zip']} ({s['rows']} rows, sha256 {s['zip_sha256']}); "
              f"not in the target language {len(s['translation_not_in_target_language'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
