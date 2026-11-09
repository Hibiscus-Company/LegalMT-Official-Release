#!/usr/bin/env python
"""Weight-space average of several LoRA fine-tunes of the same base, saved as one merged model directory.

    source scripts/LegalMT/env.sh
    CUDA_VISIBLE_DEVICES="" uv run --group gpu python scripts/LegalMT/soup_merge.py \
        --adapter RUN_OR_DIR --adapter RUN_OR_DIR [...] [--weights 1,1] --name NAME
"""
from __future__ import annotations

import argparse
import importlib.util
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

log = logging.getLogger("soup_merge")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def _merge_lora_module():
    spec = importlib.util.spec_from_file_location("legalmt_merge_lora", SCRIPT_DIR / "merge_lora.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def soup(adapter_dirs: list[Path], weights: list[float], out_dir: Path, *, save_dtype: str = "bfloat16") -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ml = _merge_lora_module()
    t0 = time.monotonic()
    if len(adapter_dirs) != len(weights) or not adapter_dirs or min(weights) <= 0:
        raise ValueError("give one positive weight per adapter")
    total = sum(weights)
    weights = [w / total for w in weights]
    bases = [ml.resolve_base(a, None) for a in adapter_dirs]
    base_dir, base_info = bases[0]
    if any(b[0] != base_dir for b in bases):
        raise ValueError(f"adapters have different bases: {sorted({b[0] for b in bases})}")
    for name in TOKENIZER_FILES:
        blobs = {(a / name).read_bytes() if (a / name).is_file() else None for a in adapter_dirs}
        if len(blobs) != 1:
            raise ValueError(f"adapters carry different {name} files")

    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
    base_params = dict(model.named_parameters())
    acc: dict[str, Any] = {}
    for a, w in zip(adapter_dirs, weights):
        m = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
        m = PeftModel.from_pretrained(m, str(a)).merge_and_unload()
        leftover = [n for n, _ in m.named_parameters() if "lora_" in n]
        if leftover:
            raise RuntimeError(f"merge left LoRA parameters behind in {a}, e.g. {leftover[:3]}")
        changed = 0
        with torch.no_grad():
            for n, p in m.named_parameters():
                d = p.detach() - base_params[n].detach()
                if torch.count_nonzero(d):
                    changed += 1
                    acc[n] = acc[n].add_(d, alpha=w) if n in acc else d.mul_(w)
        log.info("adapter %s: weight %.4f, %d changed tensors", a, w, changed)
        del m
    with torch.no_grad():
        for n, d in acc.items():
            base_params[n].add_(d)
    model = model.to(getattr(torch, save_dtype))
    model.config.use_cache = True
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter_dirs[0])).save_pretrained(str(out_dir))
    gen = Path(base_dir) / "generation_config.json"
    if gen.is_file():
        shutil.copyfile(gen, out_dir / "generation_config.json")
    members = []
    for a, w in zip(adapter_dirs, weights):
        wf = next((p for p in (a / "adapter_model.safetensors", a / "adapter_model.bin") if p.is_file()), None)
        members.append({"adapter_dir": str(a), "weight": w, "adapter_sha256": ml.sha256_file(wf) if wf else None})
    manifest = {
        "created": runs.now_iso(),
        "kind": "soup",
        "members": members,
        "adapter_config": json.loads((adapter_dirs[0] / "adapter_config.json").read_text(encoding="utf-8")),
        "base_dir": base_dir,
        "base": base_info,
        "merge_dtype": "float32",
        "save_dtype": save_dtype,
        "changed_tensors": len(acc),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "tokenizer_from": str(adapter_dirs[0]),
        "code_fingerprint": runs.code_fingerprint(),
        "env": runs.env_versions(),
        "seconds": round(time.monotonic() - t0, 1),
    }
    (out_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", action="append", required=True, help="run name, checkpoint folder or peft adapter dir (repeat)")
    ap.add_argument("--weights", default=None, help="comma list, one per adapter (default: uniform)")
    ap.add_argument("--name", required=True, help="output name under <ARTIFACTS>/merged/")
    ap.add_argument("--save-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output folder")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ml = _merge_lora_module()
    adapters = [ml.resolve_adapter(a) for a in args.adapter]
    weights = [float(x) for x in args.weights.split(",")] if args.weights else [1.0] * len(adapters)
    out = paths.artifacts("merged", runs.sanitize_tag(args.name))
    if (out / "merge_manifest.json").is_file() and not args.force:
        raise SystemExit(f"{out} already holds a merged model; pass --force to overwrite")
    man = soup(adapters, weights, out, save_dtype=args.save_dtype)
    print(f"soup of {len(adapters)} adapters -> {out} ({man['changed_tensors']} changed tensors, "
          f"{man['parameters']:,} parameters, {args.save_dtype}, {man['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
