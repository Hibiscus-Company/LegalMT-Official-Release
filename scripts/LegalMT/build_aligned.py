#!/usr/bin/env python3
"""Build the aligned EN/VI paragraph pairs used for SFT data (train) and segment-level analysis (public test).

    uv run python scripts/LegalMT/build_aligned.py
    uv run python scripts/LegalMT/build_aligned.py --limit 20
    uv run python scripts/LegalMT/build_aligned.py --fit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from LegalMT.align import (
    align_documents,
    fit_length_model,
    fit_numeric_table,
    fit_numeric_weight,
    load_align_config,
    numeric_outcome,
    numeric_tokens,
)
from LegalMT.data import load_split
from LegalMT.paths import CONFIG_DIR, artifacts, load_config

SPLITS = ("train", "public_test")


def excluded_ids(dq: dict) -> dict[str, list[str]]:
    broken = sorted({sid for ids in (dq.get("broken_pairs") or {}).values() for sid in (ids or [])})
    dup = sorted({pair[1] for pair in (dq.get("identical_pairs") or [])})
    return {"broken": broken, "duplicate_dropped": dup}


def code_fingerprint(root: Path) -> str:
    files = sorted((root / "src").rglob("*.py")) if (root / "src").is_dir() else []
    if (root / "configs").is_dir():
        files += sorted(p for p in (root / "configs").rglob("*") if p.is_file())
    files = sorted({p for p in files if "__pycache__" not in p.parts}, key=lambda p: p.relative_to(root).as_posix())
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(root).as_posix().encode("utf-8") + b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def env_info() -> dict:
    from importlib import metadata

    out = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "vllm", "peft"):
        try:
            out[mod] = metadata.version(mod)
        except metadata.PackageNotFoundError:
            pass
    return out


def write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows) -> int:
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def default_out_dir(version: str, limit: int | None) -> Path:
    return artifacts("aligned", version if not limit else f"{version}-limit{limit}")


def _align_task(task: tuple[str, str, str, str, dict]) -> tuple[str, list[dict], dict]:
    sid, cat, en, vi, cfg = task
    pairs, stats = align_documents(en, vi, sample_id=sid, category=cat, cfg=cfg)
    return sid, [p.to_dict() for p in pairs], stats


def run_split(df, cfg: dict, workers: int) -> list[tuple[str, list[dict], dict]]:
    tasks = [(r.sample_id, r.category, r.english_markdown, r.vietnamese_markdown, cfg) for r in df.itertuples()]
    order = {t[0]: k for k, t in enumerate(tasks)}
    tasks.sort(key=lambda t: -(len(t[2]) + len(t[3])))
    if workers <= 1:
        results = [_align_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_align_task, tasks, chunksize=1))
    results.sort(key=lambda r: order[r[0]])
    return results


def _share(num: float, den: float) -> float:
    return round(num / den, 4) if den else 0.0


def aggregate(results: list[tuple[str, list[dict], dict]]) -> dict:
    tot: dict = {"docs": len(results), "pairs": 0}
    sums = {k: 0 for k in ("chars_en", "chars_vi", "nonws_en", "nonws_vi", "kept_nonws_en", "kept_nonws_vi",
                            "gc_en_chars", "gc_nm_en_chars", "chunks", "chunks_equal", "anchors_matched",
                            "units_en", "units_vi")}
    ctr: dict[str, dict] = {k: {} for k in ("links", "kept_links", "kept_en_chars", "kept_vi_chars",
                                             "kept_en_chars_by_link", "dropped_en_chars", "dropped_vi_chars",
                                             "nm_unequal", "gc11_unequal", "header_mode", "signature_mode",
                                             "table_chunks")}
    zero = {"links": 0, "with_numbers": 0, "f1_ge_0.8": 0, "f1_eq_0": 0,
            "both_with_numbers": 0, "both_f1_ge_0.8": 0, "both_f1_eq_0": 0}
    post = {"nm": dict(zero), "gc11": dict(zero)}
    secs = []
    for _sid, pairs, st in results:
        tot["pairs"] += len([p for p in pairs if not p.get("drop_reason")])
        for k in sums:
            sums[k] += st[k]
        for k in ("links", "kept_links", "kept_en_chars", "kept_vi_chars", "kept_en_chars_by_link",
                  "dropped_en_chars", "dropped_vi_chars", "nm_unequal", "gc11_unequal"):
            for kk, v in st[k].items():
                ctr[k][kk] = ctr[k].get(kk, 0) + v
        for k in ("header_mode", "signature_mode"):
            ctr[k][st[k]] = ctr[k].get(st[k], 0) + 1
        for kk, v in st["table_chunks"].items():
            ctr["table_chunks"][kk] = ctr["table_chunks"].get(kk, 0) + v
        secs.append(st["seconds"])
        for p in pairs:
            if p.get("drop_reason") or p["method"] != "gc" or p["anchor"] in ("header", "signature"):
                continue
            b = post["gc11"] if p["link"] == "1-1" else post["nm"]
            b["links"] += 1
            f1 = p["num_f1"]
            if f1 is not None:
                b["with_numbers"] += 1
                b["f1_ge_0.8"] += f1 >= 0.8
                b["f1_eq_0"] += f1 == 0.0
                if numeric_tokens(p["en"]) and numeric_tokens(p["vi"]):
                    b["both_with_numbers"] += 1
                    b["both_f1_ge_0.8"] += f1 >= 0.8
                    b["both_f1_eq_0"] += f1 == 0.0
    kept_en = sum(ctr["kept_en_chars"].values())
    kept_vi = sum(ctr["kept_vi_chars"].values())

    def proxy(b: dict) -> dict:
        n, nb = b.get("with_numbers", 0), b.get("both_with_numbers", 0)
        return {**b, "share_f1_ge_0.8": _share(b.get("f1_ge_0.8", 0), n), "share_f1_eq_0": _share(b.get("f1_eq_0", 0), n),
                "share_both_f1_ge_0.8": _share(b.get("both_f1_ge_0.8", 0), nb),
                "share_both_f1_eq_0": _share(b.get("both_f1_eq_0", 0), nb)}

    tot.update({
        "chars": {k: sums[k] for k in ("chars_en", "chars_vi", "nonws_en", "nonws_vi")},
        "kept_en_char_share": _share(kept_en, sums["chars_en"]),
        "kept_vi_char_share": _share(kept_vi, sums["chars_vi"]),
        "kept_en_nonws_share": _share(sums["kept_nonws_en"], sums["nonws_en"]),
        "kept_vi_nonws_share": _share(sums["kept_nonws_vi"], sums["nonws_vi"]),
        "kept_en_char_share_by_method": {k: _share(v, sums["chars_en"]) for k, v in sorted(ctr["kept_en_chars"].items())},
        "dropped_en_char_share_by_reason": {k: _share(v, sums["chars_en"]) for k, v in sorted(ctr["dropped_en_chars"].items())},
        "dropped_vi_char_share_by_reason": {k: _share(v, sums["chars_vi"]) for k, v in sorted(ctr["dropped_vi_chars"].items())},
        "links_all": dict(sorted(ctr["links"].items())),
        "links_kept": dict(sorted(ctr["kept_links"].items())),
        "kept_link_en_char_share": {k: _share(v, kept_en) for k, v in sorted(ctr["kept_en_chars_by_link"].items())},
        "chunks": sums["chunks"], "chunks_equal": sums["chunks_equal"], "anchors_matched": sums["anchors_matched"],
        "units_en": sums["units_en"], "units_vi": sums["units_vi"],
        "header_mode": ctr["header_mode"], "signature_mode": ctr["signature_mode"],
        "table_chunks": ctr["table_chunks"],
        "gc_unequal_chunks": {
            "gc_en_chars_share_of_all_en": _share(sums["gc_en_chars"], sums["chars_en"]),
            "nm_share_of_gc_en_chars": _share(sums["gc_nm_en_chars"], sums["gc_en_chars"]),
            "numeric_proxy_nm_before_filters": proxy(ctr["nm_unequal"]),
            "numeric_proxy_11_before_filters": proxy(ctr["gc11_unequal"]),
            "numeric_proxy_nm_kept": proxy(post["nm"]),
            "numeric_proxy_11_kept": proxy(post["gc11"]),
        },
        "align_seconds_sum": round(sum(secs), 2),
        "align_seconds_max": round(max(secs), 2) if secs else 0.0,
    })
    return tot


def fit(results: list[tuple[str, list[dict], dict]]) -> dict:
    lens, f1_either, f1_both, out_aligned, out_shifted = [], [], [], [], []
    for _sid, pairs, _st in results:
        zips = [p for p in pairs if p["method"] == "zip" and p["anchor"] not in ("header", "signature")
                and p["en"] and p["vi"] and not any("c" in u for u in p["en_uids"] + p["vi_uids"])]
        for k, p in enumerate(zips):
            lens.append((len(p["en"]), len(p["vi"])))
            ne, nv = numeric_tokens(p["en"]), numeric_tokens(p["vi"])
            out_aligned.append(numeric_outcome(ne, nv))
            if p["num_f1"] is not None:
                f1_either.append(p["num_f1"])
                if ne and nv:
                    f1_both.append(p["num_f1"])
            if k + 1 < len(zips) and zips[k + 1]["anchor"] == p["anchor"] and zips[k + 1]["order"] == p["order"] + 1:
                out_shifted.append(numeric_outcome(ne, numeric_tokens(zips[k + 1]["vi"])))
    return {"length_model": fit_length_model(lens), "numeric_either": fit_numeric_weight(f1_either),
            "numeric_both": fit_numeric_weight(f1_both), "numeric_table": fit_numeric_table(out_aligned, out_shifted)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="align", help="config name under configs/LegalMT (default: align)")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--limit", type=int, default=None, help="align only the first N docs of each split")
    ap.add_argument("--workers", type=int, default=None, help="process pool size (default: config build.workers)")
    ap.add_argument("--version", default=None, help="output version dir (default: config version)")
    ap.add_argument("--out-dir", default=None, help="override the output dir (default <ARTIFACTS>/aligned/<version>, or <version>-limit<N> with --limit)")
    ap.add_argument("--keep-dropped", action="store_true", help="also write <split>.dropped.jsonl")
    ap.add_argument("--fit", action="store_true", help="fit c, s2 and num_weight on train zip links; write fit.json only")
    args = ap.parse_args()

    raw_cfg = load_config(args.config)
    cfg = load_align_config(args.config)
    version = args.version or raw_cfg.get("version", "v0")
    workers = args.workers or int((raw_cfg.get("build") or {}).get("workers", 16))
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(version, args.limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    dq_name = (raw_cfg.get("build") or {}).get("data_quality", "data_quality_v0")
    excl = excluded_ids(load_config(dq_name))
    excl_set = set(excl["broken"]) | set(excl["duplicate_dropped"])

    prev_stats = {} if args.fit else read_json(out_dir / "stats.json")
    prev_manifest = {} if args.fit else read_json(out_dir / "manifest.json")

    t_start = time.time()
    manifest = {
        "version": version, "created": datetime.now().isoformat(timespec="seconds"), "argv": sys.argv,
        "host": socket.gethostname(), "config": cfg, "config_path": str(CONFIG_DIR / f"{args.config}.yaml"),
        "splits": args.splits, "limit": args.limit, "workers": workers, "code_fingerprint": code_fingerprint(REPO_ROOT),
        "env": env_info(), "status": "running", "excluded_train": excl, "data_quality_config": dq_name,
        "notes": ("train.jsonl: training pool (dev docs included; sft_data applies the dev split). "
                  "public_test.jsonl is for analysis and segment scoring only, NEVER for training."),
    }
    if not args.fit:
        write_manifest(out_dir / "manifest.json", manifest)

    if args.fit:
        df = load_split("train").df
        df = df[~df.sample_id.isin(excl_set)]
        if args.limit:
            df = df.head(args.limit)
        fit_cfg = {**cfg, "keep_dropped": True}
        t0 = time.time()
        results = run_split(df, fit_cfg, workers)
        res = fit(results)
        res.update({"docs": len(df), "created": datetime.now().isoformat(timespec="seconds"),
                    "config_before_fit": {"gc": cfg["gc"]}, "seconds": round(time.time() - t0, 1),
                    "code_fingerprint": manifest["code_fingerprint"]})
        (out_dir / "fit.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
        print(f"wrote {out_dir / 'fit.json'}")
        return 0

    all_stats: dict = {"version": version, "created": manifest["created"], "splits": {}, "docs": {}}
    provenance: dict = {}
    for split in SPLITS:
        if split not in args.splits and split in (prev_stats.get("splits") or {}):
            all_stats["splits"][split] = prev_stats["splits"][split]
            all_stats["docs"][split] = (prev_stats.get("docs") or {}).get(split, [])
            provenance[split] = (prev_manifest.get("split_provenance") or {}).get(split) or {
                k: prev_manifest.get(k) for k in ("created", "argv", "code_fingerprint", "limit")}
    manifest["split_provenance"] = provenance
    timing: dict = {}
    try:
        for split in args.splits:
            df = load_split(split).df
            n_all = len(df)
            if split == "train":
                df = df[~df.sample_id.isin(excl_set)]
            if args.limit:
                df = df.head(args.limit)
            split_cfg = {**cfg, "keep_dropped": bool(args.keep_dropped)}
            t0 = time.time()
            results = run_split(df, split_cfg, workers)
            wall = time.time() - t0
            write_jsonl(out_dir / f"{split}.jsonl",
                        (p for _sid, pairs, _st in results for p in pairs if not p.get("drop_reason")))
            dropped_path = out_dir / f"{split}.dropped.jsonl"
            if args.keep_dropped:
                write_jsonl(dropped_path, (p for _sid, pairs, _st in results for p in pairs if p.get("drop_reason")))
            elif dropped_path.exists():
                dropped_path.unlink()
            agg = aggregate(results)
            agg.update({"docs_in_split": n_all, "docs_excluded": n_all - len(df) if split == "train" and not args.limit else None,
                        "wall_seconds": round(wall, 1)})
            all_stats["splits"][split] = agg
            all_stats["docs"][split] = [st for _sid, _pairs, st in results]
            timing[split] = round(wall, 1)
            provenance[split] = {"created": datetime.now().isoformat(timespec="seconds"), "argv": sys.argv,
                                 "code_fingerprint": manifest["code_fingerprint"], "limit": args.limit,
                                 "docs": agg["docs"], "pairs": agg["pairs"], "keep_dropped": bool(args.keep_dropped)}
            print(f"[{split}] docs {agg['docs']} pairs {agg['pairs']} kept EN chars {agg['kept_en_char_share']} "
                  f"VI chars {agg['kept_vi_char_share']} (non-ws EN {agg['kept_en_nonws_share']} VI {agg['kept_vi_nonws_share']}) "
                  f"wall {wall:.1f}s")
            print(f"[{split}] kept EN share by method {agg['kept_en_char_share_by_method']}; dropped EN share {agg['dropped_en_char_share_by_reason']}")
            print(f"[{split}] kept link EN-char shares {agg['kept_link_en_char_share']}")
            print(f"[{split}] links kept {agg['links_kept']}")
            print(f"[{split}] gc (unequal chunks) {json.dumps(agg['gc_unequal_chunks'])}")
        manifest["status"] = "done"
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        manifest["timing"] = {"per_split_seconds": timing, "total_seconds": round(time.time() - t_start, 1)}
        write_manifest(out_dir / "manifest.json", manifest)
        (out_dir / "stats.json").write_text(json.dumps(all_stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
