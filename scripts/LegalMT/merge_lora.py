#!/usr/bin/env python
"""Merge a LoRA adapter into its base model and save a plain model directory (bf16 by default).

    source scripts/LegalMT/env.sh
    CUDA_VISIBLE_DEVICES="" uv run --group gpu python scripts/LegalMT/merge_lora.py --adapter <run_name>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from LegalMT import paths, runs
from LegalMT.translate.core import resolve_model_path

log = logging.getLogger("merge_lora")
DTYPES = ("bfloat16", "float16", "float32")


def resolve_adapter(spec: str) -> Path:
    p = Path(spec)
    for cand in (p, p / "adapter", paths.artifacts("checkpoints", spec, "adapter")):
        if (cand / "adapter_config.json").is_file():
            return cand.resolve()
    raise FileNotFoundError(f"no adapter_config.json at {p}, {p / 'adapter'} or under checkpoints/{spec}/adapter")


def default_name(adapter: Path) -> str:
    if adapter.name == "adapter":
        return adapter.parent.name
    if adapter.name.startswith("checkpoint-"):
        return f"{adapter.parent.name}-{adapter.name}"
    return adapter.name


def resolve_base(adapter: Path, override: str | None) -> tuple[str, dict[str, Any]]:
    if override:
        return str(Path(override).resolve()), {"source": "--base"}
    for tc in (adapter.parent / "train_config.json", adapter.parent.parent / "train_config.json"):
        if tc.is_file():
            model = json.loads(tc.read_text(encoding="utf-8"))["config"]["model"]
            return resolve_model_path(model), {"source": str(tc), **{k: model.get(k) for k in ("repo_id", "revision", "local_path")}}
    base = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8")).get("base_model_name_or_path")
    if not base or not Path(base).is_dir():
        raise FileNotFoundError(f"cannot find the base model of {adapter}; pass --base")
    return base, {"source": "adapter_config.json base_model_name_or_path"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def merge(adapter_dir: Path, out_dir: Path, *, base: str | None = None, merge_dtype: str = "float32",
          save_dtype: str = "bfloat16", device: str = "cpu") -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if merge_dtype not in DTYPES or save_dtype not in DTYPES:
        raise ValueError(f"dtypes must be in {DTYPES}")
    t0 = time.monotonic()
    base_dir, base_info = resolve_base(adapter_dir, base)
    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=getattr(torch, merge_dtype)).to(device)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    merged = model.merge_and_unload()
    leftover = [n for n, _ in merged.named_parameters() if "lora_" in n]
    if leftover:
        raise RuntimeError(f"merge left LoRA parameters behind, e.g. {leftover[:3]}")
    merged = merged.to(getattr(torch, save_dtype))
    merged.config.use_cache = True
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tok_src = adapter_dir if (adapter_dir / "tokenizer_config.json").is_file() else Path(base_dir)
    AutoTokenizer.from_pretrained(str(tok_src)).save_pretrained(str(out_dir))
    gen = Path(base_dir) / "generation_config.json"
    if gen.is_file():
        shutil.copyfile(gen, out_dir / "generation_config.json")
    weights = next((p for p in (adapter_dir / "adapter_model.safetensors", adapter_dir / "adapter_model.bin") if p.is_file()), None)
    manifest = {
        "created": runs.now_iso(),
        "adapter_dir": str(adapter_dir),
        "adapter_sha256": sha256_file(weights) if weights else None,
        "adapter_config": json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8")),
        "base_dir": base_dir,
        "base": base_info,
        "merge_dtype": merge_dtype,
        "save_dtype": save_dtype,
        "device": device,
        "parameters": int(sum(p.numel() for p in merged.parameters())),
        "tokenizer_from": str(tok_src),
        "code_fingerprint": runs.code_fingerprint(),
        "env": runs.env_versions(),
        "seconds": round(time.monotonic() - t0, 1),
    }
    (out_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="training run name, checkpoint folder or peft adapter directory")
    ap.add_argument("--name", default=None, help="output name under <ARTIFACTS>/merged/ (default: the run name)")
    ap.add_argument("--base", default=None, help="base model directory (default: from the run's train_config.json)")
    ap.add_argument("--merge-dtype", choices=DTYPES, default="float32")
    ap.add_argument("--save-dtype", choices=DTYPES, default="bfloat16")
    ap.add_argument("--device", default="cpu", help="cpu (default, leaves the GPU free) or cuda")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output folder")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    adapter = resolve_adapter(args.adapter)
    name = args.name or default_name(adapter)
    out = paths.artifacts("merged", runs.sanitize_tag(name))
    if (out / "merge_manifest.json").is_file() and not args.force:
        raise SystemExit(f"{out} already holds a merged model; pass --force to overwrite")
    man = merge(adapter, out, base=args.base, merge_dtype=args.merge_dtype, save_dtype=args.save_dtype, device=args.device)
    print(f"merged {adapter} into {man['base_dir']} -> {out} ({man['parameters']:,} parameters, {args.save_dtype}, "
          f"{man['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
