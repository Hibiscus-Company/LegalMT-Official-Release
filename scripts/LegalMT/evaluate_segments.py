#!/usr/bin/env python3
"""Segment-level evaluation of a translation run: segment BLEU / chrF / chrF++ per direction, optionally COMET.

    uv run python scripts/LegalMT/evaluate_segments.py RUN_DIR
    uv run python scripts/LegalMT/evaluate_segments.py RUN_DIR --limit 3
    uv run python scripts/LegalMT/evaluate_segments.py RUN_DIR --comet --comet-gpus 0
    scripts/LegalMT/gpu_run.sh scripts/LegalMT/evaluate_segments.py RUN_DIR --comet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from LegalMT import metrics as M
from LegalMT import runs as R
from LegalMT import segments as SEG

COMET_PROJECT = REPO_ROOT / "tools" / "comet"
COMET_SCRIPT = SCRIPT_DIR / "comet_score.py"
ENV_SH = SCRIPT_DIR / "env.sh"
COMET_INPUT = "comet_input.jsonl"
COMET_OUTPUT = "comet_scores.json"


def comet_pins() -> dict[str, Any]:
    return tomllib.loads((COMET_PROJECT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["legalmt-comet"]


def default_comet_env() -> Path:
    p = Path(os.environ.get("LEGALMT_COMET_ENV") or comet_pins()["env_dir"])
    return p if p.is_absolute() else REPO_ROOT / p


def _with_env_sh(base: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    proc = subprocess.run(["bash", "-c", 'source "$1" >/dev/null 2>&1 && env -0', "bash", str(ENV_SH)],
                          env=env, capture_output=True, check=True)
    for item in proc.stdout.decode("utf-8", "replace").split("\0"):
        if "=" in item:
            k, v = item.split("=", 1)
            env[k] = v
    return env


def comet_subprocess_env(comet_env: Path, base: dict[str, str] | None = None) -> dict[str, str]:
    env = _with_env_sh(dict(os.environ if base is None else base))
    env["UV_PROJECT_ENVIRONMENT"] = str(comet_env)
    return env


def gpu_env_dir(base: dict[str, str] | None = None) -> Path | None:
    env = {k: v for k, v in (os.environ if base is None else base).items() if k != "UV_PROJECT_ENVIRONMENT"}
    gpu = _with_env_sh(env).get("UV_PROJECT_ENVIRONMENT")
    return Path(gpu) if gpu else None


def check_comet_env(comet_env: Path, base: dict[str, str] | None = None) -> None:
    gpu = gpu_env_dir(base)
    if gpu is not None and gpu.resolve() == comet_env.resolve():
        raise SystemExit(f"refusing COMET env {comet_env}: it is the LegalMT GPU env")
    has_comet = any(comet_env.glob("lib/python3*/site-packages/comet/__init__.py"))
    if not (comet_env / "bin" / "python").exists() or not has_comet:
        raise SystemExit(f"no COMET env at {comet_env} (no python with the comet package): "
                         "build it with scripts/LegalMT/setup_comet_env.sh")


def comet_command(uv: str, input_path: Path, output_path: Path, *, gpus: int, batch_size: int,
                  precision: str) -> list[str]:
    return [uv, "run", "--project", str(COMET_PROJECT), "--locked", "python", str(COMET_SCRIPT),
            "--input", str(input_path), "--output", str(output_path), "--gpus", str(gpus),
            "--batch-size", str(batch_size), "--precision", precision]


def run_comet(args: argparse.Namespace, input_path: Path, output_path: Path) -> dict[str, Any]:
    comet_env = Path(args.comet_env) if args.comet_env else default_comet_env()
    check_comet_env(comet_env)
    uv = args.uv or shutil.which("uv")
    if not uv:
        raise SystemExit("uv not found on PATH (pass --uv)")
    cmd = comet_command(uv, input_path, output_path, gpus=args.comet_gpus, batch_size=args.comet_batch_size,
                        precision=args.comet_precision)
    if args.comet_gpus > 0:
        print("note: COMET on the GPU; run this script under scripts/LegalMT/gpu_run.sh")
    print(f"COMET env {comet_env}: {' '.join(cmd[1:4])} ... --gpus {args.comet_gpus} --batch-size {args.comet_batch_size}",
          flush=True)
    try:
        subprocess.run(cmd, env=comet_subprocess_env(comet_env), check=True, cwd=str(REPO_ROOT))
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"COMET scoring failed (exit {e.returncode}, see its messages above); "
                         f"{SEG.SEGMENTS_FILE} and {SEG.RESULTS_FILE} were not updated") from None
    return json.loads(output_path.read_text(encoding="utf-8"))


PIN_FIELDS = ("model", "model_revision", "encoder", "encoder_revision")


def comet_meta(out: dict[str, Any], *, reused: bool) -> dict[str, Any]:
    keys = (*PIN_FIELDS, "input_sha256", "n_segments", "gpus", "batch_size", "precision", "device",
            "seconds_load", "seconds_predict", "segments_per_second", "peak_mib_torch_allocated", "versions")
    trunc = {k: v for k, v in (out.get("truncation") or {}).items() if k != "ids"}
    return {**{k: out.get(k) for k in keys}, "system_score_pooled": out.get("system_score"),
            "system_score_note": "mean over all scored segments of all directions; per direction: directions[d].comet.mean",
            "truncation": trunc, "reused": reused}


def comet_reuse_problem(out: dict[str, Any], in_sha: str, pins: dict[str, Any]) -> str | None:
    if out.get("input_sha256") != in_sha:
        return "scored different segments"
    diff = [k for k in PIN_FIELDS if out.get(k) != pins.get(k)]
    if diff:
        return f"scored with other pins ({', '.join(diff)}) than tools/comet/pyproject.toml"
    n_ids, n_scores = len(out.get("ids") or []), len(out.get("scores") or [])
    if n_ids != n_scores or n_ids != out.get("n_segments"):
        return f"inconsistent file ({n_ids} ids, {n_scores} scores, n_segments {out.get('n_segments')})"
    if not all(isinstance(s, (int, float)) and math.isfinite(s) for s in out.get("scores") or []):
        return "holds NaN, infinite or non-numeric scores"
    return None


def load_refs_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".csv":
        import pandas as pd

        return pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records")
    return R.read_jsonl(path)


def _fmt(x: Any, nd: int = 2) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def result_warnings(res: dict[str, Any]) -> list[str]:
    out = []
    if res.get("run_status") != "done":
        out.append(f"run status is {res.get('run_status')!r}, not 'done'")
    if res.get("subset"):
        out.append(f"partial evaluation: {res['subset']['n_docs']} documents selected with --limit/--ids")
    if res.get("directions_requested"):
        out.append(f"partial evaluation: directions restricted to {res['directions_requested']} with --directions")
    if not res["align"]["filters_applied"]:
        out.append("alignment filters were OFF (--no-filter): segment sets are not comparable with filtered runs")
    if (res.get("comet") or {}).get("stale"):
        out.append(res["comet"].get("note") or f"{COMET_OUTPUT} not merged (rerun with --comet)")
    if not res["directions"]:
        out.append("no direction to evaluate: hyps.jsonl is empty and the manifest declares no directions")
    elif not any(r.get("n_segments") for r in res["directions"].values()):
        out.append("no segments at all: nothing was scored")
    for flag in ("hyps_truncated_last_line", "units_truncated_last_line"):
        if res.get(flag):
            out.append(f"{flag.replace('_', ' ')} (dropped)")
    for d, n in (res.get("duplicate_hyp_records") or {}).items():
        if n:
            out.append(f"{d}: {n} duplicate hypothesis records (the last one per document was used)")
    for d, r in res["directions"].items():
        if r["missing_hyps"]:
            out.append(f"{d}: {len(r['missing_hyps'])} documents have no hypothesis, e.g. {r['missing_hyps'][:3]}")
        skipped = r["coverage"]["n_docs_skipped"]
        if skipped:
            out.append(f"{d}: documents skipped {skipped} (unit_mismatch: units.jsonl does not match the re-parsed "
                       f"source; no_units: hyps line without unit records), e.g. {r['skipped_docs'][:3]}")
        if r["coverage"].get("mt_missing_units"):
            out.append(f"{d}: {r['coverage']['mt_missing_units']} source units inside segments had no hypothesis")
        if r.get("hyps_not_in_refs"):
            out.append(f"{d}: {len(r['hyps_not_in_refs'])} hypothesis documents are not in the references (ignored)")
        c = r.get("comet")
        if c is not None and c.get("n_scored") != c.get("n_segments"):
            out.append(f"{d}: COMET scored {c.get('n_scored')} of {c.get('n_segments')} segments")
        if c is not None and c.get("n_truncated"):
            out.append(f"{d}: {c['n_truncated']} segments exceed COMET's 510-token window in src, mt or ref "
                       f"(scored on their beginning only)")
    return out


def markdown_summary(res: dict[str, Any]) -> str:
    lines = [f"### segments: {res['run_name']} on {res['split']} (split {res['split_version']})", "", SEG.NOTE, "",
             "| direction | docs | segments | src coverage | ref coverage | BLEU [95% CI] | chrF | chrF++ | mean seg chrF | COMET [95% CI] |",
             "|---|---:|---:|---:|---:|---|---:|---:|---:|---|"]
    for d, r in res["directions"].items():
        cov = r["coverage"]
        ci = r.get("bleu_ci95")
        ci_s = f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
        c = r.get("comet") or {}
        cci = c.get("ci95")
        comet_s = "n/a" if c.get("mean") is None else f"{c['mean']:.4f}" + (f" [{cci[0]:.4f}, {cci[1]:.4f}]" if cci else "")
        lines.append(
            f"| {d} | {cov['n_docs_scored']} | {r.get('n_segments', 0)} | {_fmt(cov['src_coverage'], 3)} | "
            f"{_fmt(cov['ref_coverage'], 3)} | {_fmt(r.get('bleu'))}{ci_s} | {_fmt(r.get('chrf'))} | "
            f"{_fmt(r.get('chrfpp'))} | {_fmt(r.get('segment_mean_chrf'))} | {comet_s} |")
    for d, r in res["directions"].items():
        cov = r["coverage"]
        lines += ["", f"{d}: links {cov['links']}; kept {cov['kept_links']}; source chars dropped {cov['src_chars_dropped']}"]
        if r.get("per_kind"):
            parts = [f"{k}: {v['n_segments']} seg, BLEU {v['bleu']:.2f}, chrF {v['chrf']:.2f}" for k, v in r["per_kind"].items()]
            lines.append(f"{d} per kind: " + "; ".join(parts))
    warn = result_warnings(res)
    if warn:
        lines += [""] + [f"WARNING: {w}" for w in warn]
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> int:
    run = R.load_run(args.run)
    split = args.split or run.manifest.get("split")
    if not split and not args.refs_file:
        raise SystemExit("the manifest has no split; pass --split or --refs-file")
    version = args.split_version or run.manifest.get("split_version") or "v0"
    if args.refs_file:
        refs = load_refs_file(args.refs_file)
    else:
        from LegalMT.splits import docs_for

        refs = docs_for(split, version)
    directions = [d.strip() for d in args.directions.split(",") if d.strip()] if args.directions else None
    ids = [i.strip() for i in args.ids.split(",") if i.strip()] if args.ids else None
    if args.limit is not None:
        hyp_ids = {h["sample_id"] for h in R.read_jsonl_tolerant(run.path("hyps.jsonl"))[0]}
        pool = ids or list(SEG.reference_table(refs))
        ids = [s for s in pool if s in hyp_ids][: args.limit]

    seg = SEG.build_segments(run.dir, split, refs=refs, split_version=version, directions=directions,
                             align_cfg=args.align_config, filters=not args.no_filter, ids=ids)
    info = seg.info
    triples = seg.triples
    res: dict[str, Any] = {
        "run_name": run.name, "tag": run.manifest.get("tag"), "created": run.manifest.get("created"),
        "run_status": run.manifest.get("status"), "split": split, "split_version": version,
        "references": str(args.refs_file) if args.refs_file else f"splits.docs_for({split!r}, {version!r})",
        "evaluated": R.now_iso(), "note": SEG.NOTE, "signatures": M.signatures(), "preprocess": M.PREPROCESS_NOTE,
        "eval_code_fingerprint": R.code_fingerprint(),
        "align": {"config": info["align_config"], **info["align"], "layout_cfg": info["layout_cfg"]},
        "hyps_truncated_last_line": info.get("hyps_truncated_last_line"),
        "units_truncated_last_line": info.get("units_truncated_last_line"),
        "duplicate_hyp_records": info["duplicate_hyp_records"],
        "subset": {"n_docs": len(ids), "ids": ids} if ids is not None else None,
        "directions_requested": directions,
        "directions": {},
    }

    per_seg: dict[str, dict[str, float]] = {}
    for d in info["directions"]:
        ts = [t for t in triples if t["direction"] == d]
        docs_d = [x for x in seg.docs if x["direction"] == d]
        summary, seg_scores = SEG.score_triples(ts, n_bootstrap=args.n_bootstrap, seed=args.seed)
        for t, s in zip(ts, seg_scores):
            per_seg[t["seg_id"]] = s
        res["directions"][d] = {
            **summary,
            "coverage": SEG.coverage(docs_d),
            "missing_hyps": info["missing_hyps"].get(d, []),
            "hyps_not_in_refs": info["hyps_not_in_refs"].get(d, []),
            "skipped_docs": [x["sample_id"] for x in docs_d if x.get("status") != "ok"],
        }

    comet_scores: dict[str, float] | None = None
    comet_truncated: list[str] = []
    comet_blob = SEG.dumps_jsonl(SEG.comet_input_records(triples)).encode("utf-8")
    in_sha = hashlib.sha256(comet_blob).hexdigest()
    out_path = run.path(COMET_OUTPUT)
    if args.comet and triples:
        in_path = run.path(COMET_INPUT)
        tmp = in_path.with_name(in_path.name + ".tmp")
        tmp.write_bytes(comet_blob)
        os.replace(tmp, in_path)
        out = run_comet(args, in_path, out_path)
        if out.get("input_sha256") != in_sha:
            raise SystemExit(f"{out_path}: input sha256 does not match the segments just written")
        res["comet"] = comet_meta(out, reused=False)
        comet_scores = dict(zip(out["ids"], out["scores"]))
        comet_truncated = (out.get("truncation") or {}).get("ids") or []
    elif out_path.is_file():
        out = json.loads(out_path.read_text(encoding="utf-8"))
        problem = comet_reuse_problem(out, in_sha, comet_pins())
        if problem is None:
            res["comet"] = comet_meta(out, reused=True)
            comet_scores = dict(zip(out["ids"], out["scores"]))
            comet_truncated = (out.get("truncation") or {}).get("ids") or []
        else:
            res["comet"] = {"stale": True, "note": f"{COMET_OUTPUT} {problem}; not merged (rerun with --comet)"}
    if comet_scores is not None:
        for d, c in SEG.comet_by_direction(triples, comet_scores, n_bootstrap=args.n_bootstrap, seed=args.seed,
                                           truncated=comet_truncated).items():
            res["directions"][d]["comet"] = c

    res["docs"] = seg.docs
    run.write_jsonl(SEG.SEGMENTS_FILE, (
        {**t, **per_seg.get(t["seg_id"], {}), **({"comet": comet_scores.get(t["seg_id"])} if comet_scores else {})}
        for t in triples))
    run.write_json(SEG.RESULTS_FILE, res)
    print(markdown_summary(res))
    print(f"\nwrote {run.path(SEG.SEGMENTS_FILE)} ({len(triples)} segments) and {run.path(SEG.RESULTS_FILE)}")
    return 0 if triples else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run directory or run name")
    ap.add_argument("--split", default=None, help="override the manifest split (train_fit/dev/dev_mini/public_test)")
    ap.add_argument("--split-version", default=None, help="dev-split version (default: the manifest's, else v0)")
    ap.add_argument("--refs-file", type=Path, default=None,
                    help="references from this JSONL/CSV (sample_id, category, english_markdown, vietnamese_markdown)")
    ap.add_argument("--directions", default=None, help="comma list (default: every direction in hyps.jsonl)")
    ap.add_argument("--ids", default=None, help="comma-separated sample_ids (partial evaluation)")
    ap.add_argument("--limit", type=int, default=None, help="first N documents with hypotheses (partial evaluation)")
    ap.add_argument("--align-config", default="align", help="alignment config name under configs/LegalMT/")
    ap.add_argument("--no-filter", action="store_true", help="keep links the training filters would drop")
    ap.add_argument("--n-bootstrap", type=int, default=1000, help="document resamples for the CIs (<= 1 skips them)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--comet", action="store_true", help="score the segments with COMET in the separate env")
    ap.add_argument("--comet-gpus", type=int, default=1, help="0 = CPU (slow); GPU runs belong under gpu_run.sh")
    ap.add_argument("--comet-batch-size", type=int, default=16)
    ap.add_argument("--comet-precision", choices=("auto", "fp16", "fp32"), default="auto")
    ap.add_argument("--comet-env", default=None, help="COMET env dir (default $LEGALMT_COMET_ENV or tools/comet env_dir)")
    ap.add_argument("--uv", default=None, help="uv executable (default: from PATH)")
    args = ap.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be >= 1")
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
