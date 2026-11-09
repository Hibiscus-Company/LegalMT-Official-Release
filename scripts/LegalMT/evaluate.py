#!/usr/bin/env python3
"""Evaluate a LegalMT translation run directory, or compare two runs with a paired document bootstrap.

    uv run python scripts/LegalMT/evaluate.py RUN_DIR
    uv run python scripts/LegalMT/evaluate.py RUN_DIR --no-index
    uv run python scripts/LegalMT/evaluate.py --compare RUN_A RUN_B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from LegalMT import metrics as M
from LegalMT import quality as Q
from LegalMT import runs as R
from LegalMT import splits as S
from LegalMT import terminology as T

COLUMNS = {"en": "english_markdown", "vi": "vietnamese_markdown"}
DIRECTIONS = ("en2vi", "vi2en")


def load_hyps(run: R.Run) -> tuple[dict[str, dict[str, dict]], dict[str, Any]]:
    path = run.path("hyps.jsonl")
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    records, truncated = R.read_jsonl_tolerant(path)
    out: dict[str, dict[str, dict]] = {}
    dups: dict[str, int] = {}
    field_mismatch: dict[str, list[str]] = {}
    for rec in records:
        d = rec.get("direction")
        if d not in DIRECTIONS:
            raise ValueError(f"hyps.jsonl record for {rec.get('sample_id')} has direction {d!r}")
        src_lang, tgt_lang = T.DIRECTIONS[d]
        if rec.get("src_lang", src_lang) != src_lang or rec.get("tgt_lang", tgt_lang) != tgt_lang:
            field_mismatch.setdefault(d, []).append(rec["sample_id"])
        bucket = out.setdefault(d, {})
        if rec["sample_id"] in bucket:
            dups[d] = dups.get(d, 0) + 1
        bucket[rec["sample_id"]] = rec
    info = {"duplicates": dups, "direction_field_mismatch": field_mismatch, "truncated_last_line": truncated}
    return out, info


def hyp_language_problems(ids: list[str], hyp_map: dict[str, dict], tgt_lang: str) -> tuple[list[str], list[str]]:
    empty, wrong = [], []
    for s in ids:
        if s not in hyp_map:
            continue
        hyp = hyp_map[s].get("hyp") or ""
        if not hyp.strip():
            empty.append(s)
        elif Q.column_language(hyp)[0] != tgt_lang:
            wrong.append(s)
    return empty, wrong


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def evaluate_direction(
    ref_df, hyp_map: dict[str, dict], direction: str, glossary: list[T.GlossaryEntry], *,
    n_bootstrap: int = 1000, seed: int = 12345, doc_info: dict[str, dict] | None = None,
) -> dict[str, Any]:
    src_lang, tgt_lang = T.DIRECTIONS[direction]
    ids = ref_df["sample_id"].tolist()
    cats = ref_df["category"].tolist()
    srcs = ref_df[COLUMNS[src_lang]].tolist()
    refs = ref_df[COLUMNS[tgt_lang]].tolist()
    hyps = [(hyp_map.get(s) or {}).get("hyp") or "" for s in ids]
    missing = [s for s in ids if s not in hyp_map]
    extra = sorted(set(hyp_map) - set(ids), key=S.natural_key)
    empty, wrong_lang = hyp_language_problems(ids, hyp_map, tgt_lang)

    ds = M.doc_statistics(hyps, refs)
    scores = ds.scores()
    ci = M.bootstrap_ci(hyps, refs, n=n_bootstrap, seed=seed, stats=ds) if n_bootstrap > 1 else {}
    per_doc_bleu = ds.per_doc("bleu")
    res: dict[str, Any] = {
        "n_docs": len(ids),
        "n_hyps": len(ids) - len(missing),
        "n_missing": len(missing),
        "missing_hyps": missing,
        "n_empty": len(empty),
        "empty_hyps": empty,
        "n_wrong_language": len(wrong_lang),
        "wrong_language_hyps": wrong_lang,
        "extra_hyps_ignored": extra,
        "bleu": scores["bleu"],
        "chrf": scores["chrf"],
        "chrfpp": scores["chrfpp"],
        "doc_macro_bleu": float(np.mean(per_doc_bleu)),
    }
    for m, v in ci.items():
        res[f"{m}_ci95"] = v
    res["bootstrap"] = {"n": n_bootstrap, "seed": seed, "unit": "document"}
    res["per_category"] = M.per_category_scores(hyps, refs, cats, stats=ds)
    if doc_info:
        groups = {"near_relative": [], "novel": []}
        for i, s in enumerate(ids):
            if s in doc_info:
                groups["near_relative" if doc_info[s].get("near_relative") else "novel"].append(i)
        res["by_near_relative"] = {
            g: ({"n_docs": len(rows), **ds.scores(np.asarray(rows))} if rows else {"n_docs": 0})
            for g, rows in groups.items()
        }
    res["per_doc_bleu"] = {s: round(b, 4) for s, b in zip(ids, per_doc_bleu)}
    res["terms"] = T.term_metrics(glossary, srcs, refs, hyps, direction).as_dict()
    secs = [float(hyp_map[s]["seconds"]) for s in ids if s in hyp_map and hyp_map[s].get("seconds") is not None]
    res["seconds_per_doc_mean"] = _mean(secs)
    res["seconds_total"] = float(np.sum(secs)) if secs else None
    for k in ("out_tokens", "in_tokens", "fallback_units", "n_chunks", "n_units"):
        vals = [hyp_map[s].get(k) for s in ids if s in hyp_map and hyp_map[s].get(k) is not None]
        if vals:
            res[f"{k}_total"] = int(np.sum(vals))
    return res


def run_cost(manifest: dict) -> dict[str, Any]:
    timing, memory, thr = manifest.get("timing") or {}, manifest.get("memory") or {}, manifest.get("throughput") or {}
    return {
        "total_seconds": timing.get("total_seconds"),
        "load_seconds": timing.get("load_seconds"),
        "translate_seconds": timing.get("translate_seconds"),
        "peak_mib_nvidia_smi": memory.get("peak_mib_nvidia_smi"),
        "peak_mib_torch_allocated": memory.get("peak_mib_torch_allocated"),
        "words_per_second": thr.get("words_per_second"),
        "out_tokens_per_second": thr.get("out_tokens_per_second"),
    }


def split_fingerprint(split: str, version: str) -> dict[str, Any]:
    if split == "public_test":
        return {"public_test_csv_sha256": S.file_sha256(S.paths.split_path("public_test"))}
    p = S.spec_path(version)
    return {"split_spec_sha256": S.file_sha256(p)} if p.is_file() else {}


def evaluate_run(
    run: R.Run, ref_df, glossary: list[T.GlossaryEntry], *, split: str, split_version: str = "v0",
    glossary_path: str = "", n_bootstrap: int = 1000, seed: int = 12345, doc_info: dict | None = None,
    references_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hyps, info = load_hyps(run)
    declared = [d for d in (run.manifest.get("directions") or []) if d in DIRECTIONS]
    directions = [d for d in DIRECTIONS if d in hyps or d in declared]
    out: dict[str, Any] = {
        "run_name": run.name,
        "tag": run.manifest.get("tag"),
        "created": run.manifest.get("created"),
        "run_status": run.manifest.get("status"),
        "split": split,
        "split_version": split_version,
        "references": references_fingerprint or {},
        "evaluated": R.now_iso(),
        "metric_note": M.PROVISIONAL_NOTE,
        "signatures": M.signatures(),
        "preprocess": M.PREPROCESS_NOTE,
        "glossary": {"path": glossary_path, "n_entries": len(glossary),
                     "sha256": S.file_sha256(Path(glossary_path)) if glossary_path and Path(glossary_path).is_file() else None},
        "eval_code_fingerprint": R.code_fingerprint(),
        "duplicate_hyp_records": info["duplicates"],
        "direction_field_mismatch": info["direction_field_mismatch"],
        "hyps_truncated_last_line": info["truncated_last_line"],
        "directions": {},
        "cost": run_cost(run.manifest),
    }
    for d in directions:
        out["directions"][d] = evaluate_direction(
            ref_df, hyps.get(d, {}), d, glossary, n_bootstrap=n_bootstrap, seed=seed, doc_info=doc_info)
    return out


def index_record(results: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for d, r in results["directions"].items():
        metrics[d] = {
            "bleu": r["bleu"], "chrf": r["chrf"], "chrfpp": r["chrfpp"],
            "bleu_ci95": r.get("bleu_ci95"), "doc_macro_bleu": r["doc_macro_bleu"],
            "term_recall": r["terms"]["term_recall"], "term_consistency": r["terms"]["term_consistency"],
            "sec_per_doc": r["seconds_per_doc_mean"], "n_docs": r["n_docs"], "n_missing": r["n_missing"],
            "n_empty": r.get("n_empty", 0), "n_wrong_language": r.get("n_wrong_language", 0),
            "n_direction_field_mismatch": len(results.get("direction_field_mismatch", {}).get(d, [])),
        }
    metrics["run"] = results["cost"]
    return {
        "run_name": results["run_name"], "created": results["created"], "split": results["split"],
        "tag": results["tag"], "metrics": metrics,
        "split_version": results["split_version"], "evaluated": results["evaluated"],
        "run_status": results.get("run_status"),
        "glossary_sha256": (results.get("glossary") or {}).get("sha256"),
        "eval_code_fingerprint": results.get("eval_code_fingerprint"),
    }


def result_warnings(results: dict[str, Any]) -> list[str]:
    out = []
    status = results.get("run_status")
    if status != "done":
        out.append(f"run status is {status!r}, not 'done': scores cover only what the run wrote so far")
    if results.get("hyps_truncated_last_line"):
        out.append("hyps.jsonl ended in a half-written line (dropped)")
    if not results["directions"]:
        out.append("no hypotheses for any direction (empty hyps.jsonl and no 'directions' in the manifest)")
    for d, n in (results.get("duplicate_hyp_records") or {}).items():
        out.append(f"{d}: {n} duplicate hypothesis records (the last one per document was scored)")
    for d, ids in (results.get("direction_field_mismatch") or {}).items():
        out.append(f"{d}: {len(ids)} records whose src_lang/tgt_lang contradict the direction label, e.g. {ids[:3]}")
    for d, r in results["directions"].items():
        if r["n_missing"]:
            out.append(f"{d}: {r['n_missing']} of {r['n_docs']} docs have no hypothesis (scored as empty)")
        if r.get("n_empty"):
            out.append(f"{d}: {r['n_empty']} hypotheses are blank, e.g. {r['empty_hyps'][:3]}")
        if r.get("n_wrong_language"):
            out.append(f"{d}: {r['n_wrong_language']} hypotheses are not in the target language, "
                       f"e.g. {r['wrong_language_hyps'][:3]}")
    return out


def _fmt(x: Any, nd: int = 2) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def markdown_summary(results: dict[str, Any]) -> str:
    lines = [f"### {results['run_name']} on {results['split']} (split {results['split_version']})", "",
             f"{results['metric_note']}", "",
             "| direction | docs | missing | BLEU [95% CI] | chrF | chrF++ | doc-macro BLEU | term recall | term consistency (ref) | sec/doc |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for d, r in results["directions"].items():
        ci = r.get("bleu_ci95")
        ci_s = f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
        t = r["terms"]
        lines.append(
            f"| {d} | {r['n_docs']} | {r['n_missing']} | {r['bleu']:.2f}{ci_s} | {r['chrf']:.2f} | {r['chrfpp']:.2f} | "
            f"{r['doc_macro_bleu']:.2f} | {_fmt(t['term_recall'], 3)} ({t['term_recall_hits']}/{t['term_recall_eligible']}) | "
            f"{_fmt(t['term_consistency'], 3)} ({_fmt(t['ref_term_consistency'], 3)}) | {_fmt(r['seconds_per_doc_mean'])} |")
    for d, r in results["directions"].items():
        lines += ["", f"{d} per category:", "", "| category | docs | BLEU | chrF | doc-macro BLEU |", "|---|---:|---:|---:|---:|"]
        for c, s in r["per_category"].items():
            lines.append(f"| {c} | {s['n_docs']} | {s['bleu']:.2f} | {s['chrf']:.2f} | {s['doc_macro_bleu']:.2f} |")
        if r.get("by_near_relative"):
            parts = [f"{g}: {v['n_docs']} docs, BLEU {_fmt(v.get('bleu'))}" for g, v in r["by_near_relative"].items()]
            lines += ["", f"{d} by near_relative tag: " + "; ".join(parts)]
        if r["missing_hyps"]:
            lines += ["", f"{d}: {r['n_missing']} docs had no hypothesis (scored as empty): {r['missing_hyps'][:10]}"]
    c = results["cost"]
    lines += ["", f"cost: total {_fmt(c['total_seconds'])} s, peak MiB (nvidia-smi) {_fmt(c['peak_mib_nvidia_smi'], 0)}, "
                  f"peak MiB (torch) {_fmt(c['peak_mib_torch_allocated'], 0)}, words/s {_fmt(c['words_per_second'])}"]
    warn = result_warnings(results)
    if warn:
        lines += [""] + [f"WARNING: {w}" for w in warn]
    return "\n".join(lines)


def compare_runs(
    run_a: R.Run, run_b: R.Run, ref_df, *, n_bootstrap: int = 1000, seed: int = 12345,
) -> dict[str, Any]:
    hyps_a, _ = load_hyps(run_a)
    hyps_b, _ = load_hyps(run_b)
    out: dict[str, Any] = {"run_a": run_a.name, "run_b": run_b.name, "directions": {}, "metric_note": M.PROVISIONAL_NOTE,
                           "run_status_a": run_a.manifest.get("status"), "run_status_b": run_b.manifest.get("status"),
                           "skipped_directions": []}
    ids = ref_df["sample_id"].tolist()
    for d in DIRECTIONS:
        if d not in hyps_a or d not in hyps_b:
            if d in hyps_a or d in hyps_b:
                out["skipped_directions"].append(d)
            continue
        _, tgt = T.DIRECTIONS[d]
        common = [s for s in ids if s in hyps_a[d] and s in hyps_b[d]]
        if not common:
            out["skipped_directions"].append(d)
            continue
        ref_col = dict(zip(ids, ref_df[COLUMNS[tgt]].tolist()))
        refs = [ref_col[s] for s in common]
        ha = [(hyps_a[d].get(s) or {}).get("hyp") or "" for s in common]
        hb = [(hyps_b[d].get(s) or {}).get("hyp") or "" for s in common]
        res = M.paired_bootstrap(ha, hb, refs, n=n_bootstrap, seed=seed)
        res["missing_a"] = [s for s in ids if s not in hyps_a[d]]
        res["missing_b"] = [s for s in ids if s not in hyps_b[d]]
        res["n_common"] = len(common)
        out["directions"][d] = res
    return out


def markdown_compare(cmp: dict[str, Any]) -> str:
    lines = [f"### paired document bootstrap: B = {cmp['run_b']} vs A = {cmp['run_a']}", "", cmp["metric_note"], "",
             "| direction | metric | A | B | B - A | 95% CI of B - A | B better in | p (sacrebleu paired bs) |",
             "|---|---|---:|---:|---:|---|---:|---:|"]
    for d, r in cmp["directions"].items():
        for m in M.METRICS:
            x = r[m]
            lines.append(f"| {d} | {m} | {x['a']:.2f} | {x['b']:.2f} | {x['delta']:+.2f} | "
                         f"[{x['delta_ci95'][0]:+.2f}, {x['delta_ci95'][1]:+.2f}] | {x['b_better_share']:.1%} | {x['p_value']:.4f} |")
        if r["missing_a"] or r["missing_b"]:
            lines.append(f"| {d} | missing hyps (excluded from the pairing) | {len(r['missing_a'])} | "
                         f"{len(r['missing_b'])} | | | | |")
    lines.append("")
    lines.append(f"({next(iter(cmp['directions'].values()))['n_docs'] if cmp['directions'] else 0} documents, "
                 f"resampled with replacement; unit = document)")
    for side in ("a", "b"):
        status = cmp.get(f"run_status_{side}")
        if status is not None and status != "done":
            lines.append(f"WARNING: run {side.upper()} has status {status!r}, not 'done'")
    if cmp.get("skipped_directions"):
        lines.append(f"WARNING: directions present in only one run, not compared: {cmp['skipped_directions']}")
    return "\n".join(lines)


def references(split: str, version: str):
    return S.docs_for(split, version)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="run directory or run name")
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="paired bootstrap B vs A")
    ap.add_argument("--split", default=None, help="override the manifest split (train_fit/dev/dev_mini/public_test)")
    ap.add_argument("--split-version", default=None,
                    help="dev-split version (default: the manifest's split_version if it has one, else v0)")
    ap.add_argument("--glossary", type=Path, default=None)
    ap.add_argument("--n-bootstrap", type=int, default=1000, help="document resamples (<= 1 skips the CI; --compare needs >= 1)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no-index", action="store_true", help="do not append to results_index.jsonl")
    ap.add_argument("--out", type=Path, default=None, help="--compare: also write the comparison JSON here")
    args = ap.parse_args(argv)

    if args.compare:
        if args.n_bootstrap < 1:
            ap.error("--compare needs --n-bootstrap >= 1")
        ra, rb = R.load_run(args.compare[0]), R.load_run(args.compare[1])
        split = args.split or ra.manifest.get("split")
        if rb.manifest.get("split") != split:
            print(f"WARNING: runs were made on different splits ({ra.manifest.get('split')} vs {rb.manifest.get('split')}); "
                  f"comparing on {split}")
        version = args.split_version or ra.manifest.get("split_version") or "v0"
        cmp = compare_runs(ra, rb, references(split, version), n_bootstrap=args.n_bootstrap, seed=args.seed)
        cmp["split"] = split
        cmp["split_version"] = version
        print(markdown_compare(cmp))
        if args.out:
            args.out.write_text(json.dumps(cmp, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(f"\nwrote {args.out}")
        return 0

    if not args.run:
        ap.error("give RUN_DIR or --compare RUN_A RUN_B")
    run = R.load_run(args.run)
    split = args.split or run.manifest.get("split")
    if not split:
        ap.error("the manifest has no split; pass --split")
    version = args.split_version or run.manifest.get("split_version") or "v0"
    ref_df = references(split, version)
    gpath = args.glossary or T.default_glossary_path()
    glossary = T.load_glossary(gpath)
    doc_info = None
    if split in ("dev", "dev_mini"):
        doc_info = S.load_split_spec(version).doc_info
    results = evaluate_run(run, ref_df, glossary, split=split, split_version=version,
                           glossary_path=str(gpath), n_bootstrap=args.n_bootstrap, seed=args.seed, doc_info=doc_info,
                           references_fingerprint=split_fingerprint(split, version))
    run.write_json("results.json", results)
    print(markdown_summary(results))
    print(f"\nwrote {run.path('results.json')}")
    if not results["directions"]:
        print("nothing was scored: not appending to the results index")
        return 1
    if not args.no_index:
        p = R.append_results_index(index_record(results))
        print(f"appended {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
