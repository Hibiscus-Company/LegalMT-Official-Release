#!/usr/bin/env python3
"""Score aligned MT segments with COMET (pinned Unbabel/wmt22-comet-da) inside the separate COMET env.

    UV_PROJECT_ENVIRONMENT=.venv-comet \\
        uv run --project tools/comet python scripts/LegalMT/comet_score.py \\
        --input RUN/segments.jsonl --output RUN/comet_scores.json --gpus 1 --batch-size 32
    ... --gpus 0
    ... --download
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
COMET_PROJECT = REPO_ROOT / "tools" / "comet"
PATHS_CONFIG = REPO_ROOT / "configs" / "LegalMT" / "paths.yaml"
PIN_KEYS = ("model", "model_revision", "encoder_hparam", "encoder", "encoder_revision", "encoder_files")
VERSIONED = ("unbabel-comet", "torch", "transformers", "tokenizers", "numpy", "pytorch-lightning",
             "huggingface-hub", "sentencepiece")


def load_pins(pyproject: Path = COMET_PROJECT / "pyproject.toml") -> dict[str, Any]:
    table = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("tool", {}).get("legalmt-comet", {})
    missing = [k for k in PIN_KEYS if not table.get(k)]
    if missing:
        raise SystemExit(f"{pyproject}: [tool.legalmt-comet] is missing {missing}")
    return table


def ensure_hf_home() -> str:
    if not os.environ.get("HF_HOME"):
        import yaml

        cfg = yaml.safe_load(PATHS_CONFIG.read_text(encoding="utf-8"))
        os.environ["HF_HOME"] = os.path.expanduser(str(cfg["hf_cache_default"]))
    return os.environ["HF_HOME"]


def read_segments(path: Path, id_key: str, data: bytes | None = None) -> tuple[list[dict[str, str]], list[Any]]:
    samples: list[dict[str, str]] = []
    ids: list[Any] = []
    text = (path.read_bytes() if data is None else data).decode("utf-8")
    for n, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        bad = [k for k in ("src", "mt", "ref") if not isinstance(rec.get(k), str)]
        if bad:
            raise SystemExit(f"{path}:{n}: fields {bad} missing or not strings")
        samples.append({"src": rec["src"], "mt": rec["mt"], "ref": rec["ref"]})
        ids.append(rec.get(id_key, len(ids)))
    return samples, ids


def resolve_snapshots(pins: dict[str, Any], *, download: bool) -> tuple[Path, Path]:
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(pins["model"], revision=pins["model_revision"], local_files_only=not download)
    enc_dir = snapshot_download(pins["encoder"], revision=pins["encoder_revision"],
                                allow_patterns=list(pins["encoder_files"]), local_files_only=not download)
    return Path(model_dir), Path(enc_dir)


def load_model(model_dir: Path, enc_dir: Path, pins: dict[str, Any]):
    import torch
    import yaml
    from comet.models import str2model

    hparams = yaml.safe_load((model_dir / "hparams.yaml").read_text(encoding="utf-8"))
    if hparams.get("pretrained_model") != pins["encoder_hparam"]:
        raise SystemExit(f"checkpoint encoder {hparams.get('pretrained_model')!r} is not the pinned "
                         f"{pins['encoder_hparam']!r}: update [tool.legalmt-comet] on purpose")
    kwargs: dict[str, Any] = {
        "checkpoint_path": model_dir / "checkpoints" / "model.ckpt",
        "load_pretrained_weights": False,
        "map_location": torch.device("cpu"),
        "strict": False,
        "local_files_only": True,
        "pretrained_model": str(enc_dir),
    }
    if hparams.get("layer_transformation") == "sparsemax_patch":
        kwargs["layer_transformation"] = "softmax"
    model = str2model[hparams["class_identifier"]].load_from_checkpoint(**kwargs)
    check_weights_loaded(model, kwargs["checkpoint_path"])
    return model


def check_weights_loaded(model, checkpoint_path: Path) -> None:
    import torch

    state = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)["state_dict"]
    missing = sorted(set(model.state_dict()) - set(state))
    if missing:
        raise SystemExit(f"{checkpoint_path}: {len(missing)} model weights are not in the checkpoint "
                         f"(they would stay random), e.g. {missing[:5]}")


def truncation_report(model, samples: list[dict[str, str]], ids: list[Any]) -> dict[str, Any]:
    tok = model.encoder.tokenizer
    limit = int(model.encoder.max_positions) - 2
    over: dict[int, list[str]] = {}
    by_field: dict[str, int] = {}
    for f in ("src", "mt", "ref"):
        lens = [len(x) for x in tok([s[f] for s in samples], add_special_tokens=True, verbose=False)["input_ids"]]
        by_field[f] = sum(n > limit for n in lens)
        for i, n in enumerate(lens):
            if n > limit:
                over.setdefault(i, []).append(f)
    return {"max_tokens": limit, "n_segments": len(over), "by_field": by_field, "ids": [ids[i] for i in sorted(over)]}


def nonfinite_problem(scores: list[float], ids: list[Any], precision: str) -> str | None:
    bad = [ids[i] for i, s in enumerate(scores) if not math.isfinite(s)]
    if not bad:
        return None
    hint = " (rerun with --precision fp32)" if precision == "fp16" else ""
    return f"{len(bad)} of {len(scores)} COMET scores are NaN or infinite, e.g. ids {bad[:3]}{hint}"


def versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": sys.version.split()[0]}
    for name in VERSIONED:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def download(pins: dict[str, Any]) -> int:
    hf_home = ensure_hf_home()
    t0 = time.monotonic()
    model_dir, enc_dir = resolve_snapshots(pins, download=True)
    ckpt = model_dir / "checkpoints" / "model.ckpt"
    print(f"HF_HOME {hf_home}")
    print(f"model   {pins['model']}@{pins['model_revision']}: {model_dir} (checkpoint {ckpt.stat().st_size:,} bytes)")
    print(f"encoder {pins['encoder']}@{pins['encoder_revision']}: {enc_dir} "
          f"({', '.join(sorted(p.name for p in enc_dir.iterdir()))})")
    print(f"fetched in {time.monotonic() - t0:.1f} s")
    return 0


def score(args: argparse.Namespace, pins: dict[str, Any]) -> int:
    ensure_hf_home()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    data = args.input.read_bytes()
    input_sha256 = hashlib.sha256(data).hexdigest()
    samples, ids = read_segments(args.input, args.id_key, data)
    if not samples:
        raise SystemExit(f"{args.input}: no segments")
    if len(set(map(str, ids))) != len(ids):
        raise SystemExit(f"{args.input}: duplicate {args.id_key!r} values")

    import torch

    if args.gpus > 0 and not torch.cuda.is_available():
        raise SystemExit(f"--gpus {args.gpus} but CUDA is not available (CUDA_VISIBLE_DEVICES="
                         f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}); pass --gpus 0 to score on CPU")
    precision = args.precision if args.precision != "auto" else ("fp16" if args.gpus > 0 else "fp32")
    if args.gpus > 0:
        torch.set_float32_matmul_precision("high")
    from pytorch_lightning import seed_everything

    seed_everything(args.seed, verbose=False)
    t0 = time.monotonic()
    model_dir, enc_dir = resolve_snapshots(pins, download=False)
    model = load_model(model_dir, enc_dir, pins)
    model.eval()
    if precision == "fp16":
        model.half()
    t_load = time.monotonic() - t0
    truncation = truncation_report(model, samples, ids)
    t1 = time.monotonic()
    out = model.predict(samples, batch_size=args.batch_size, gpus=args.gpus, progress_bar=args.progress,
                        accelerator="cpu" if args.gpus == 0 else "auto", num_workers=args.num_workers,
                        length_batching=True)
    t_pred = time.monotonic() - t1
    scores = [float(s) for s in out.scores]
    if len(scores) != len(samples):
        raise RuntimeError(f"COMET returned {len(scores)} scores for {len(samples)} segments")
    problem = nonfinite_problem(scores, ids, precision)
    if problem:
        raise SystemExit(f"{problem}; nothing written")
    peak = torch.cuda.max_memory_allocated() / 2**20 if args.gpus > 0 else None
    result = {
        "model": pins["model"], "model_revision": pins["model_revision"],
        "encoder": pins["encoder"], "encoder_revision": pins["encoder_revision"],
        "input": str(args.input), "input_sha256": input_sha256,
        "id_key": args.id_key,
        "n_segments": len(scores),
        "system_score": sum(scores) / len(scores),
        "gpus": args.gpus, "batch_size": args.batch_size, "precision": precision, "seed": args.seed,
        "device": torch.cuda.get_device_name(0) if args.gpus > 0 else "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seconds_load": round(t_load, 3), "seconds_predict": round(t_pred, 3),
        "segments_per_second": round(len(scores) / t_pred, 3) if t_pred > 0 else None,
        "peak_mib_torch_allocated": None if peak is None else round(peak, 1),
        "truncation": truncation,
        "versions": versions(),
        "ids": ids,
        "scores": scores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, args.output)
    print(f"COMET {pins['model']}@{pins['model_revision'][:10]} on {len(scores)} segments ({precision}, "
          f"{result['device']}): system {result['system_score']:.4f}; load {t_load:.1f} s, predict {t_pred:.1f} s; "
          f"{truncation['n_segments']} segments exceed {truncation['max_tokens']} tokens in some field "
          f"(src/mt/ref {truncation['by_field']['src']}/{truncation['by_field']['mt']}/{truncation['by_field']['ref']})")
    print(f"wrote {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, help="JSONL of {src, mt, ref, ...} segments")
    ap.add_argument("--output", type=Path, help="output JSON (per-segment and system scores)")
    ap.add_argument("--gpus", type=int, default=1, help="0 = CPU; the default 1 fails loudly without CUDA")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--precision", choices=("auto", "fp16", "fp32"), default="auto",
                    help="auto: fp16 on GPU (comet-score CLI parity), fp32 on CPU")
    ap.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (COMET default: 2 x gpus)")
    ap.add_argument("--id-key", default="seg_id", help="input field echoed back as the segment id")
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--progress", action="store_true", help="show COMET's progress bar")
    ap.add_argument("--pins", type=Path, default=COMET_PROJECT / "pyproject.toml",
                    help="pyproject.toml holding [tool.legalmt-comet]")
    ap.add_argument("--download", action="store_true",
                    help="fetch the pinned checkpoint and encoder files into HF_HOME, then exit")
    args = ap.parse_args(argv)
    pins = load_pins(args.pins)
    if args.download:
        return download(pins)
    if not args.input or not args.output:
        ap.error("--input and --output are required (or --download)")
    if args.gpus < 0 or args.batch_size < 1:
        ap.error("--gpus must be >= 0 and --batch-size >= 1")
    if args.gpus > 1:
        ap.error("--gpus > 1 is not supported by this wrapper")
    return score(args, pins)


if __name__ == "__main__":
    raise SystemExit(main())
