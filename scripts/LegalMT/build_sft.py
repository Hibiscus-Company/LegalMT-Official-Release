#!/usr/bin/env python
"""Build LegalMT SFT data: aligned links -> prompt/completion examples.

    source scripts/LegalMT/env.sh
    CUDA_VISIBLE_DEVICES="" uv run --group gpu python scripts/LegalMT/build_sft.py --config configs/LegalMT/sft/sft_v0.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml

from LegalMT import paths, runs
from LegalMT import sft_data as S
from LegalMT.data import load_split
from LegalMT.splits import file_sha256, load_split_spec
from LegalMT.translate.core import load_infer_config, resolve_model_path

log = logging.getLogger("build_sft")
SFT_CONFIG_DIR = paths.CONFIG_DIR / "sft"
KNOWN_KEYS = {"name", "description", "aligned_version", "split_version", "infer_config", "infer_overrides",
              "directions", "filters", "packing", "context", "seed", "max_train_tokens", "extra", "final"}


def load_sft_config(name_or_path: str | Path) -> tuple[dict, Path]:
    p = Path(name_or_path)
    if not p.is_file():
        p = SFT_CONFIG_DIR / (p.name if p.suffix == ".yaml" else f"{p.name}.yaml")
    if not p.is_file():
        raise FileNotFoundError(f"no SFT config at {name_or_path} or {p}")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    unknown = sorted(set(cfg) - KNOWN_KEYS)
    if unknown:
        raise ValueError(f"unknown SFT config keys {unknown} (known: {sorted(KNOWN_KEYS)})")
    for key in ("name", "aligned_version", "split_version", "infer_config", "seed"):
        if cfg.get(key) is None:
            raise ValueError(f"SFT config needs '{key}'")
    S.validate_extra(cfg.get("extra"))
    S.validate_final(cfg.get("final"))
    return cfg, p.resolve()


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def stale_alignment(kind_stats: dict) -> str | None:
    bad = {k: v for k, v in kind_stats.items()
           if k in ("missing_uid", "no_uids", "text_mismatch_en", "text_mismatch_vi") and v}
    if not bad:
        return None
    return (f"the aligned data does not match the current layout parser ({bad}): it is stale or was built with "
            f"another normalisation; rebuild it with scripts/LegalMT/build_aligned.py")


def load_tokenizer(infer_cfg: dict, mode: str):
    if mode == "whitespace":
        if infer_cfg["model"]["family"] != "hymt":
            raise ValueError("--tokenizer whitespace only works for the hymt family (hand-rendered template)")
        return None, None
    model_dir = resolve_model_path(infer_cfg["model"])
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir), model_dir


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="SFT config yaml (or a name under configs/LegalMT/sft/)")
    ap.add_argument("--out-name", default=None, help="output folder name under <ARTIFACTS>/sft/ (default: config name)")
    ap.add_argument("--limit-docs", type=int, default=None,
                    help="use only the first N documents of each role (quick runs; the folder name gets '-limitN')")
    ap.add_argument("--tokenizer", choices=("model", "whitespace"), default="model",
                    help="model: the inference config's tokenizer (default); whitespace: CPU dry run, not for training")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output folder")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t0 = time.monotonic()
    cfg, cfg_path = load_sft_config(args.config)
    for _e in cfg.get("extra") or []:
        if not Path(_e["path"]).is_absolute():
            _e["path"] = str(paths.ARTIFACTS_ROOT / _e["path"])
    infer_cfg, infer_path = load_infer_config(cfg["infer_config"], cfg.get("infer_overrides") or None)
    family = infer_cfg["model"]["family"]

    name = args.out_name or cfg["name"]
    if args.limit_docs:
        name += f"-limit{args.limit_docs}"
    if args.tokenizer == "whitespace":
        name += "-wstok"
    out_dir = paths.artifacts("sft", runs.sanitize_tag(name))
    if (out_dir / "manifest.json").is_file() and not args.force:
        raise SystemExit(f"{out_dir} already holds a build; pass --force to overwrite (a training run may depend on it)")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").unlink(missing_ok=True)

    aligned_dir = paths.artifacts("aligned", str(cfg["aligned_version"]))
    aligned_path = aligned_dir / "train.jsonl"
    if not aligned_path.is_file():
        raise SystemExit(f"no aligned data at {aligned_path}; build it with scripts/LegalMT/build_aligned.py")
    aligned_manifest = json.loads((aligned_dir / "manifest.json").read_text(encoding="utf-8"))
    aligned_layout = (aligned_manifest.get("config") or {}).get("layout") or {}
    if aligned_layout != (infer_cfg.get("layout") or {}):
        raise SystemExit(f"layout config differs between the aligned data ({aligned_layout}) and the inference config "
                         f"({infer_cfg.get('layout')}): training units would not match inference units")

    extras = S.validate_extra(cfg.get("extra"))
    for e in extras:
        if not Path(e["path"]).is_file():
            raise SystemExit(f"extra source {e['name']}: no file at {e['path']}")

    spec = load_split_spec(str(cfg["split_version"]))
    final = S.validate_final(cfg.get("final"))
    public_raw = load_split("public_test").df if final and final["public_test"] else None
    public_ids = [] if public_raw is None else public_raw["sample_id"].tolist()
    if final is None:
        roles = S.split_roles(spec.train_fit_ids, spec.dev_ids, limit=args.limit_docs)
    else:
        roles = S.final_roles(spec.train_fit_ids, spec.dev_ids, public_ids, limit=args.limit_docs)
        log.warning("FINAL build: dev%s documents go into train; the dev loss is a fit monitor only",
                    " and public-test" if public_ids else "")
    wanted = set(roles["train"]) | set(roles["dev"])
    log.info("reading %s", aligned_path)
    docs = S.read_aligned(aligned_path, keep_ids=wanted)
    public_aligned = aligned_dir / "public_test.jsonl"
    if public_ids:
        if not public_aligned.is_file():
            raise SystemExit(f"no aligned public-test data at {public_aligned}; build it with scripts/LegalMT/build_aligned.py")
        log.info("reading %s", public_aligned)
        docs.update(S.read_aligned(public_aligned, keep_ids=wanted))
    missing = {r: [s for s in ids if s not in docs] for r, ids in roles.items()}
    for r, m in missing.items():
        if m:
            log.warning("%d %s documents have no aligned links, e.g. %s", len(m), r, m[:3])

    raw = load_split("train").df
    if public_raw is not None:
        import pandas as pd

        raw = pd.concat([raw, public_raw], ignore_index=True)
    raw = raw[raw["sample_id"].isin(docs)]
    kind_stats: dict[str, int] = {}
    t_parse = time.monotonic()
    for row in raw.itertuples(index=False):
        st = S.attach_kinds(docs[row.sample_id], row.english_markdown, row.vietnamese_markdown, infer_cfg.get("layout"))
        for k, v in st.items():
            kind_stats[k] = kind_stats.get(k, 0) + v
    log.info("parsed %d documents in %.1fs: %s", len(raw), time.monotonic() - t_parse, kind_stats)
    problem = stale_alignment(kind_stats)
    if problem:
        raise SystemExit(problem)

    tokenizer, model_dir = load_tokenizer(infer_cfg, args.tokenizer)
    settings = S.BuildSettings(
        family=family, chunking=infer_cfg["chunking"], prompt=infer_cfg["prompt"],
        directions=tuple(cfg.get("directions") or S.DIRECTIONS), filters=cfg.get("filters") or {},
        packing=cfg.get("packing") or {}, context_p=float((cfg.get("context") or {}).get("p", 0.0)),
        seed=int(cfg["seed"]))
    counter = S.TokenCounter(tokenizer)
    stats: dict = {"name": name, "roles": {}}
    extra_stats: dict[str, dict] = {}
    split_ids = set(spec.train_fit_ids) | set(spec.dev_ids) | set(public_ids)
    for role, ids in roles.items():
        role_docs = {s: docs[s] for s in ids if s in docs}
        t_role = time.monotonic()
        exs, counts = S.build_examples(role_docs, settings, tokenizer, counter)
        extra_exs: list[dict] = []
        if role == "train":
            for e in extras:
                t_extra = time.monotonic()
                log.info("extra %s: reading %s", e["name"], e["path"])
                pairs = S.read_pairs(e["path"])
                got, st = S.build_extra_examples(pairs, settings, name=e["name"], seed=e["seed"],
                                                 max_tokens=e["max_tokens"], max_lines=e["max_lines"],
                                                 block_pairs=e["block_pairs"], tokenizer=tokenizer, counter=counter)
                del pairs
                extra_exs.extend(got)
                extra_stats[e["name"]] = {**st, "seconds": round(time.monotonic() - t_extra, 1)}
                log.info("extra %s: %d examples from %d of %d pairs, %d tokens (%s)", e["name"], st["examples"],
                         st["pairs_taken"], st["pairs_in_file"], st["total_tokens"], st["stopped_by"])
        ids_all = [ex["id"] for ex in exs] + [ex["id"] for ex in extra_exs]
        if len(set(ids_all)) != len(ids_all):
            raise AssertionError(f"{role}: duplicate example ids (the shuffle sorts by id)")
        exs = S.shuffle_examples(exs + extra_exs, int(cfg["seed"]))
        n_before = len(exs)
        if role == "train" and cfg.get("max_train_tokens"):
            exs = S.cap_tokens(exs, int(cfg["max_train_tokens"]))
        leak = sorted({ex["sample_id"] for ex in exs if "source" not in ex} - set(ids))
        if leak:
            raise AssertionError(f"{role}: examples from documents outside the role: {leak[:3]}")
        bad = sorted({ex["sample_id"] for ex in exs if "source" in ex
                      and (role != "train" or ex["sample_id"] in split_ids
                           or not ex["sample_id"].startswith(S.EXTRA_PREFIX))})
        if bad:
            raise AssertionError(f"{role}: extra examples outside train or named like split documents: {bad[:3]}")
        S.write_jsonl(out_dir / f"{role}.jsonl", exs)
        summary = S.summarize(exs)
        summary["build"] = counts
        summary["examples_before_cap"] = n_before
        summary["seconds"] = round(time.monotonic() - t_role, 1)
        stats["roles"][role] = summary
        log.info("%s: %d examples (%d docs), %d prompt + %d completion tokens", role, summary["examples"],
                 summary["docs"], summary["prompt_tokens"], summary["completion_tokens"])
    stats["parse"] = kind_stats
    if extras:
        stats["extra"] = extra_stats
    write_json(out_dir / "stats.json", stats)

    manifest = {
        "name": name,
        "created": runs.now_iso(),
        "argv": sys.argv if argv is None else ["build_sft.py", *argv],
        "host": socket.gethostname(),
        "config": cfg,
        "config_path": str(cfg_path),
        "infer_config": {"name": infer_cfg.get("config_name"), "path": str(infer_path), "resolved": infer_cfg},
        "family": family,
        "tokenizer": {"mode": args.tokenizer, "model_dir": model_dir,
                      "repo_id": infer_cfg["model"].get("repo_id"), "revision": infer_cfg["model"].get("revision"),
                      **(S.tokenizer_fingerprint(model_dir) if model_dir else {"sha256": None, "files": []})},
        "trainable": args.tokenizer == "model",
        "inputs": {"aligned": str(aligned_path), "aligned_sha256": file_sha256(aligned_path),
                   "aligned_code_fingerprint": aligned_manifest.get("code_fingerprint"),
                   "split_version": spec.version, "train_csv_sha256": spec.data.get("train_sha256")},
        "roles": {r: {"documents": len(ids), "documents_without_links": missing[r]} for r, ids in roles.items()},
        "leak_check": ({"train_dev_overlap": 0, "train_docs_from": "train_fit_ids", "dev_docs_from": "dev_ids"}
                       if final is None else
                       {"final": True, "dev_in_train": True, "public_test_in_train": bool(public_ids),
                        "train_docs_from": "train_fit_ids + dev_ids" + (" + public_test" if public_ids else ""),
                        "dev_docs_from": "dev_ids (also in train: a fit monitor, not a held-out loss)"}),
        "limit_docs": args.limit_docs,
        "code_fingerprint": runs.code_fingerprint(),
        "env": runs.env_versions(),
        "seconds": round(time.monotonic() - t0, 1),
        "status": "done",
    }
    if final is not None:
        manifest["final"] = final
        if public_ids:
            manifest["inputs"]["aligned_public_test"] = str(public_aligned)
            manifest["inputs"]["aligned_public_test_sha256"] = file_sha256(public_aligned)
    if extras:
        train_src = stats["roles"]["train"].get("per_source", {})
        manifest["extra"] = [{
            **e, "sha256": file_sha256(Path(e["path"])),
            **{k: extra_stats[e["name"]][k] for k in ("pairs_in_file", "pairs_taken", "blocks_taken", "stopped_by",
                                                      "max_units_per_chunk", "links_dropped")},
            "examples": train_src.get(e["name"], {}).get("examples", 0),
            "total_tokens": train_src.get(e["name"], {}).get("total_tokens", 0),
            "per_direction": {d: {"examples": b["examples"], "pairs_used": b["lines"], "total_tokens": b["total_tokens"],
                                  "prompt_tokens": b["prompt_tokens"], "completion_tokens": b["completion_tokens"]}
                              for d, b in train_src.get(e["name"], {}).get("per_direction", {}).items()},
        } for e in extras]
        manifest["leak_check"]["extra_sources_in"] = "train only"
    write_json(out_dir / "manifest.json", manifest)

    print(f"SFT data written to {out_dir}")
    if final is not None:
        print("FINAL build: dev" + (" and public-test" if public_ids else "") + " documents are in train; "
              "the dev loss is a fit monitor, and no score of a model trained on this build can select anything")
    for role, s in stats["roles"].items():
        print(f"[{role}] examples {s['examples']} from {s['docs']} docs; prompt tokens {s['prompt_tokens']:,}, "
              f"completion (loss) tokens {s['completion_tokens']:,}, total {s['total_tokens']:,}")
        for d, b in s["per_direction"].items():
            print(f"  {d}: {b['examples']} examples, {b['lines']} lines, total tokens {b['total_tokens']:,} "
                  f"(P50 {b['total_tokens_pct']['p50']}, P99 {b['total_tokens_pct']['p99']}, max {b['total_tokens_pct']['max']}), "
                  f"src tokens P50 {b['src_tokens_pct']['p50']} max {b['src_tokens_pct']['max']}, "
                  f"packing: {s['build']['per_direction'].get(d, {})}")
        print(f"  link filter drops: {s['build']['links_dropped']}")
        for src, b in s.get("per_source", {}).items():
            print(f"  source {src}: {b['examples']} examples, {b['lines']} lines, total tokens {b['total_tokens']:,}; "
                  + ", ".join(f"{d} {x['examples']} ex / {x['lines']} lines / {x['total_tokens']:,} tok"
                              for d, x in b["per_direction"].items()))
    for n, st in extra_stats.items():
        print(f"[extra {n}] {st['pairs_taken']:,} of {st['pairs_in_file']:,} pairs in {st['blocks_taken']} blocks "
              f"({st['stopped_by']}); dropped {st['links_dropped']}; packing {st['per_direction']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
