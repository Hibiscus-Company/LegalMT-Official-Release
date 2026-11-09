#!/usr/bin/env python3
"""Translate LegalMT documents with a configured Translator (configs/LegalMT/infer/<name>.yaml).

    uv run python scripts/LegalMT/translate.py --config echo_smoke --split dev_mini --limit 2 --evaluate
    scripts/LegalMT/gpu_run.sh scripts/LegalMT/translate.py --config hymt15_vllm_greedy --split dev
    ... --directions en2vi --ids Quyen_dan_su/12,Bat_dong_san/3 --tag my_test
    ... --set generation.do_sample=true --set backend.batch_size=4
    ... --resume <run dir or name>

    ... --config hymt15_hf_greedy --input-file in.txt --output-file out.txt [--src-lang en]
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from LegalMT import runs as R
from LegalMT.langid import direction_of
from LegalMT.profiling import GpuMemorySampler, Stopwatch, memory_report
from LegalMT.translate import Translator, load_infer_config, parse_set_overrides

COLUMNS = {"en": "english_markdown", "vi": "vietnamese_markdown"}
DIRECTIONS = {"en2vi": ("en", "vi"), "vi2en": ("vi", "en")}
SPLITS = ("dev", "dev_mini", "public_test", "train_fit")
COUNTERS = ("n_units", "n_chunks", "n_pieces", "n_requests", "failed_chunks", "truncated_chunks", "fallback_units",
            "source_copied_units", "truncated_requests", "in_tokens", "out_tokens", "src_tokens", "render_missing")

log = logging.getLogger("LegalMT.translate")


def load_evaluate_module():
    spec = importlib.util.spec_from_file_location("legalmt_evaluate", SCRIPT_DIR / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_directions(s: str) -> list[str]:
    out = [d.strip() for d in s.split(",") if d.strip()]
    bad = [d for d in out if d not in DIRECTIONS]
    if bad or not out:
        raise SystemExit(f"--directions must be a comma list of {sorted(DIRECTIONS)}, got {s!r}")
    return list(dict.fromkeys(out))


def select_docs(split: str, version: str, ids: list[str] | None, limit: int | None):
    from LegalMT.splits import docs_for

    df = docs_for(split, version)
    n_split = len(df)
    if ids:
        known = set(df["sample_id"])
        unknown = [i for i in ids if i not in known]
        if unknown:
            raise SystemExit(f"{len(unknown)} --ids are not in {split}: {unknown[:5]}")
        pos = {s: k for k, s in enumerate(df["sample_id"])}
        df = df.iloc[[pos[i] for i in ids]].reset_index(drop=True)
    if limit is not None:
        df = df.head(limit).reset_index(drop=True)
    return df, n_split


def records_for(res, sample_id: str, category: str, direction: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    src, tgt = DIRECTIONS[direction]
    st = res.stats
    hyp = {
        "sample_id": sample_id, "category": category, "direction": direction, "src_lang": src, "tgt_lang": tgt,
        "hyp": res.text, "seconds": st["seconds"], "n_units": st["n_units"], "n_chunks": st["n_chunks"],
        "in_tokens": st["in_tokens"], "out_tokens": st["out_tokens"], "fallback_units": st["fallback_units"],
        "source_copied_units": st["source_copied_units"], "truncated_requests": st["truncated_requests"],
        "failed_chunks": st["failed_chunks"], "truncated_chunks": st["truncated_chunks"], "n_pieces": st["n_pieces"],
        "detected_src_lang": st["detected_src_lang"],
        "max_request_tokens": st["max_request_tokens"],
        "layout_family": st["layout_family"],
    }
    units = [{"sample_id": sample_id, "direction": direction, **u} for u in res.units]
    return hyp, units


MAXIMA = ("max_chunk_tokens", "max_prompt_tokens", "max_request_tokens")


def _new_totals() -> dict[str, Any]:
    return {"n_docs": 0, "src_words": 0, **{k: 0 for k in COUNTERS}, **{k: 0 for k in MAXIMA},
            "chunk_cuts": {}, "src_lang_mismatch": []}


def _accumulate(tot: dict[str, Any], res, sample_id: str, src_words: int) -> None:
    tot["n_docs"] += 1
    tot["src_words"] += src_words
    for k in COUNTERS:
        tot[k] = tot.get(k, 0) + int(res.stats.get(k, 0))
    for k in MAXIMA:
        tot[k] = max(tot[k], int(res.stats.get(k, 0)))
    for k, v in res.stats["chunk_cuts"].items():
        tot["chunk_cuts"][k] = tot["chunk_cuts"].get(k, 0) + v
    if res.stats["detected_src_lang"] != res.src_lang:
        tot["src_lang_mismatch"].append(sample_id)


def throughput(words: int, out_tokens: int, seconds: float | None) -> dict[str, Any]:
    ok = seconds is not None and seconds > 0
    return {
        "src_words": words, "out_tokens": out_tokens,
        "words_per_second": round(words / seconds, 3) if ok else None,
        "out_tokens_per_second": round(out_tokens / seconds, 3) if ok else None,
    }


TIMING_KEYS = ("load_seconds", "translate_seconds", "total_seconds")


def previous_sessions(m: dict[str, Any], done: set[tuple[str, str]], directions: list[str]) -> dict[str, Any]:
    prev_stats = m.get("translate_stats") or {}
    n_done = {d: sum(1 for _, dd in done if dd == d) for d in directions}
    timing = m.get("timing") or {}
    consistent = (not timing.get("incomplete")
                  and all(isinstance(timing.get(k), (int, float)) for k in TIMING_KEYS)
                  and all((prev_stats.get(d) or {}).get("n_docs", 0) == n_done[d] for d in directions))
    totals = {d: _new_totals() for d in directions}
    if consistent:
        for d in directions:
            totals[d].update(copy.deepcopy(prev_stats.get(d) or {}))
    return {
        "consistent": consistent,
        "totals": totals,
        "timing": {k: timing[k] for k in TIMING_KEYS} if consistent else None,
        "memory": m.get("memory") or {},
        "sessions": list(m["sessions"]) if m.get("sessions") else
                    ([{"timing": {k: timing.get(k) for k in TIMING_KEYS}, "memory": m.get("memory") or {}}]
                     if timing else []),
    }


def merge_timing(prev: dict[str, Any] | None, cur: dict[str, float], consistent: bool) -> dict[str, Any]:
    if not consistent:
        return {**{k: None for k in TIMING_KEYS}, "incomplete": True}
    if not prev:
        return dict(cur)
    return {k: round(float(prev[k]) + cur[k], 3) for k in TIMING_KEYS}


def merge_memory(prev: dict[str, Any], cur: dict[str, Any]) -> dict[str, Any]:
    out = dict(cur)
    for k, v in prev.items():
        if k.startswith("peak_") and isinstance(v, (int, float)):
            out[k] = v if out.get(k) is None else max(out[k], v)
    return out


def run_split(args: argparse.Namespace) -> int:
    prior: dict[str, Any] | None = None
    if args.resume:
        run = R.load_run(args.resume)
        m = run.manifest
        fp_now = R.code_fingerprint()
        if m.get("code_fingerprint") and fp_now != m["code_fingerprint"] and not args.allow_code_change:
            raise SystemExit(f"src/ or configs/ changed since {run.name} started (code_fingerprint "
                             f"{m['code_fingerprint'][:12]} -> {fp_now[:12]}): start a new run (rerun from zero), "
                             f"or pass --allow-code-change if the change cannot affect translations")
        cfg, cfg_path = m["config"], m.get("config_path")
        split, version, directions = m["split"], m.get("split_version", "v0"), m["directions"]
        subset = m.get("subset")
        df, n_split = select_docs(split, version, subset["ids"] if subset else None, None)
        hyps_path = run.path("hyps.jsonl")
        prev, truncated = R.read_jsonl_tolerant(hyps_path) if hyps_path.is_file() else ([], False)
        done = {(h["sample_id"], h["direction"]) for h in prev}
        run.write_jsonl("hyps.jsonl", prev)
        if run.path("units.jsonl").is_file():
            units_prev, _ = R.read_jsonl_tolerant(run.path("units.jsonl"))
            run.write_jsonl("units.jsonl", [u for u in units_prev if (u["sample_id"], u["direction"]) in done])
        prior = previous_sessions(m, done, directions)
        entry = {"at": R.now_iso(), "n_done": len(done), "truncated_tail": truncated,
                 "earlier_sessions_consistent": prior["consistent"]}
        if fp_now != m.get("code_fingerprint"):
            entry["code_fingerprint"] = fp_now
        run.update_manifest(status="running", resumed=(m.get("resumed") or []) + [entry])
        run.t0 = time.monotonic()
        print(f"resuming {run.dir}: {len(done)} (document, direction) pairs already done")
        if not prior["consistent"]:
            print("WARNING: the earlier sessions' counters do not match the committed documents (killed run?): "
                  "run-level timing and throughput will be recorded as unknown (None)")
    else:
        cfg, cfg_path = load_infer_config(args.config, parse_set_overrides(args.set or []))
        split, version = args.split, args.split_version
        directions = parse_directions(args.directions)
        ids = [i.strip() for i in args.ids.split(",") if i.strip()] if args.ids else None
        df, n_split = select_docs(split, version, ids, args.limit)
        if len(df) == 0:
            raise SystemExit("no documents selected")
        subset = None
        if len(df) != n_split:
            subset = {"n_docs": len(df), "of": n_split, "ids": df["sample_id"].tolist(),
                      "limit": args.limit, "ids_arg": bool(ids)}
        tag = args.tag or f"{cfg['config_name']}_{split}" + (f"_n{len(df)}" if subset else "")
        run = R.new_run(tag, cfg, cfg_path, split, extra={
            "split_version": version, "directions": directions, "n_docs": len(df), "subset": subset,
            "docs_per_call": args.docs_per_call})
        done = set()
        print(f"run dir {run.dir}")
    print(f"{len(df)} documents ({split}, split {version}) x {directions}"
          + (f" (subset of {n_split})" if subset else ""))

    sampler = GpuMemorySampler(interval=args.mem_interval).start()
    sw = Stopwatch()
    totals = prior["totals"] if prior else {d: _new_totals() for d in directions}
    tr = None

    def session_fields() -> dict[str, Any]:
        cur = {"load_seconds": round(sw.seconds.get("load", 0.0), 3),
               "translate_seconds": round(sw.seconds.get("translate", 0.0), 3),
               "total_seconds": round(sw.total(), 3)}
        mem = memory_report(sampler)
        timing = merge_timing(prior["timing"] if prior else None, cur, prior["consistent"] if prior else True)
        words = sum(t["src_words"] for t in totals.values())
        out_tok = sum(t["out_tokens"] for t in totals.values())
        fields = {"timing": timing, "memory": merge_memory(prior["memory"], mem) if prior else mem,
                  "throughput": throughput(words, out_tok, timing["translate_seconds"]), "translate_stats": totals}
        if prior:
            fields["sessions"] = prior["sessions"] + [{"timing": cur, "memory": mem}]
            fields["translate_stats_partial"] = not prior["consistent"]
        return fields

    try:
        with sw.lap("load"):
            tr = Translator(cfg)
        print(f"loaded {tr.info().get('config_name')} ({tr.backend_name}) in {tr.load_seconds:.1f} s")
        for direction in directions:
            src_lang, _ = DIRECTIONS[direction]
            todo = [row for row in df.itertuples(index=False) if (row.sample_id, direction) not in done]
            step = args.docs_per_call if args.docs_per_call > 0 else max(1, len(todo))
            for k in range(0, len(todo), step):
                group = todo[k:k + step]
                docs = [(row.sample_id, getattr(row, COLUMNS[src_lang]), src_lang) for row in group]
                with sw.lap("translate"):
                    results = tr.translate_documents(docs)
                for row, res in zip(group, results):
                    if res.direction != direction:
                        raise RuntimeError(f"{row.sample_id}: translator produced {res.direction}, expected {direction}")
                    hyp, units = records_for(res, row.sample_id, row.category, direction)
                    run.append_jsonl("units.jsonl", units)
                    run.append_jsonl("hyps.jsonl", hyp)
                    _accumulate(totals[direction], res, row.sample_id, len(getattr(row, COLUMNS[src_lang]).split()))
                t = totals[direction]
                print(f"[{direction}] {min(k + step, len(todo))}/{len(todo)} docs  chunks {t['n_chunks']}  "
                      f"fallback units {t['fallback_units']}  truncated {t['truncated_requests']}  "
                      f"translate {sw.seconds.get('translate', 0.0):.1f} s", flush=True)
        sampler.stop()
        run.finish("done", translator=tr.info(), **session_fields())
    except BaseException as exc:
        sampler.stop()
        run.finish("failed", error=f"{type(exc).__name__}: {exc}", **session_fields())
        raise
    finally:
        if tr is not None:
            tr.close()

    m = R.load_run(run.dir).manifest
    print(f"done: load {m['timing']['load_seconds']} s, translate {m['timing']['translate_seconds']} s, "
          f"total {m['timing']['total_seconds']} s; {m['throughput']['words_per_second']} words/s, "
          f"{m['throughput']['out_tokens_per_second']} out tok/s; peak MiB nvidia-smi "
          f"{m['memory']['peak_mib_nvidia_smi']}, torch {m['memory']['peak_mib_torch_allocated']}")
    for d, t in totals.items():
        if t["src_lang_mismatch"]:
            print(f"WARNING: {d}: langid disagrees with the source language for {t['src_lang_mismatch'][:5]}")
    if args.evaluate:
        return evaluate(run.dir, split, version, subset, args)
    return 0


def evaluate(run_dir: Path, split: str, version: str, subset: dict | None, args: argparse.Namespace) -> int:
    ev = load_evaluate_module()
    run = R.load_run(run_dir)
    ref_df = ev.references(split, version)
    if subset:
        ref_df = ref_df[ref_df["sample_id"].isin(set(subset["ids"]))].reset_index(drop=True)
    gpath = ev.T.default_glossary_path()
    glossary = ev.T.load_glossary(gpath)
    doc_info = ev.S.load_split_spec(version).doc_info if split in ("dev", "dev_mini") else None
    results = ev.evaluate_run(run, ref_df, glossary, split=split, split_version=version, glossary_path=str(gpath),
                              n_bootstrap=args.n_bootstrap, seed=12345, doc_info=doc_info,
                              references_fingerprint=ev.split_fingerprint(split, version))
    if subset:
        results["subset"] = {"n_docs": subset["n_docs"], "of": subset["of"],
                             "note": "scored on the translated documents only; not appended to the results index"}
    run.write_json("results.json", results)
    print(ev.markdown_summary(results))
    print(f"\nwrote {run.path('results.json')}")
    if not results["directions"]:
        return 1
    if subset:
        print("subset run: results index NOT appended")
    elif not args.no_index:
        print(f"appended {ev.R.append_results_index(ev.index_record(results))}")
    return 0


def run_file(args: argparse.Namespace) -> int:
    if not args.output_file:
        raise SystemExit("--input-file needs --output-file")
    cfg, _ = load_infer_config(args.config, parse_set_overrides(args.set or []))
    text = Path(args.input_file).read_text(encoding="utf-8")
    sampler = GpuMemorySampler(interval=args.mem_interval).start()
    sw = Stopwatch()
    tr = None
    try:
        with sw.lap("load"):
            tr = Translator(cfg)
        with sw.lap("translate"):
            res = tr.translate_text(text, args.src_lang, doc_id=Path(args.input_file).name)
    finally:
        sampler.stop()
        if tr is not None:
            tr.close()
    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(res.text, encoding="utf-8")
    words = len(text.split())
    tsec = sw.seconds["translate"]
    mem = memory_report(sampler)
    thr = throughput(words, res.stats["out_tokens"], tsec)
    print(f"{direction_of(res.src_lang)} ({'given' if args.src_lang else 'langid'}): {words} source words, "
          f"{res.stats['n_units']} units, {res.stats['n_chunks']} chunks, {res.stats['fallback_units']} fallback units")
    print(f"load {sw.seconds['load']:.2f} s, translate {tsec:.2f} s, total {sw.total():.2f} s; "
          f"{thr['words_per_second']} words/s, {thr['out_tokens_per_second']} out tok/s; "
          f"peak MiB nvidia-smi {mem['peak_mib_nvidia_smi']}, torch {mem['peak_mib_torch_allocated']}")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config name under configs/LegalMT/infer/ or a path to a yaml file")
    ap.add_argument("--set", action="append", metavar="KEY.PATH=VALUE", help="override a config value (repeatable)")
    ap.add_argument("--split", choices=SPLITS)
    ap.add_argument("--split-version", default="v0")
    ap.add_argument("--directions", default="en2vi,vi2en")
    ap.add_argument("--limit", type=int, default=None, help="first N documents of the split (after --ids)")
    ap.add_argument("--ids", default=None, help="comma-separated sample_ids")
    ap.add_argument("--tag", default=None, help="run tag (default <config>_<split>[_n<docs>])")
    ap.add_argument("--docs-per-call", type=int, default=0,
                    help="documents per translate call; output is appended after each call (0 = all)")
    ap.add_argument("--resume", default=None, help="continue a run directory (config, split and documents from its manifest)")
    ap.add_argument("--allow-code-change", action="store_true",
                    help="with --resume: continue even though src/ or configs/ changed since the run started")
    ap.add_argument("--evaluate", action="store_true", help="score the run with evaluate.py's logic when done")
    ap.add_argument("--no-index", action="store_true", help="with --evaluate: do not append to results_index.jsonl")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--input-file", default=None, help="single-file mode: translate this text file (no run dir)")
    ap.add_argument("--output-file", default=None)
    ap.add_argument("--src-lang", choices=("en", "vi"), default=None, help="single-file mode: source language (default langid)")
    ap.add_argument("--mem-interval", type=float, default=0.2, help="nvidia-smi poll interval in seconds")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.input_file:
        if not args.config:
            ap.error("--input-file needs --config")
        return run_file(args)
    if args.resume:
        if args.config or args.split or args.set:
            ap.error("--resume takes config, split and documents from the run's manifest")
        return run_split(args)
    if not args.config or not args.split:
        ap.error("give --config and --split (or --input-file, or --resume)")
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be >= 1")
    return run_split(args)


if __name__ == "__main__":
    raise SystemExit(main())
