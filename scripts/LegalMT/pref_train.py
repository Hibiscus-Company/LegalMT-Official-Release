#!/usr/bin/env python3
"""Preference training, step 2 of 2: DPO with a small NLL term, as a new LoRA adapter on a merged model.

    scripts/LegalMT/gpu_run.sh scripts/LegalMT/pref_train.py --base MERGED_DIR --pairs PAIRS_DIR --sft sft_v1 \
        --out OUT_DIR [--beta 0.1] [--nll 0.1] [--lr 2e-5] [--r 64] [--pairs-per-step 16]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from LegalMT import paths
from LegalMT import runs as R

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def completion_logp(hf: Any, ids: Any, n_prompt: int, chunk: int, grad: bool) -> Any:
    import torch
    from torch.utils.checkpoint import checkpoint

    h = hf.model(input_ids=ids).last_hidden_state[0]
    hs, tgt = h[n_prompt - 1:-1], ids[0, n_prompt:]
    head = hf.lm_head

    def piece(x: Any, t: Any) -> Any:
        return torch.log_softmax(head(x).float(), dim=-1).gather(-1, t[:, None]).sum()

    total = hs.new_zeros((), dtype=torch.float32)
    for k in range(0, hs.shape[0], chunk):
        x, t = hs[k:k + chunk], tgt[k:k + chunk]
        total = total + (checkpoint(piece, x, t, use_reentrant=False) if grad else piece(x, t))
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="merged model directory (policy start and reference)")
    ap.add_argument("--pairs", required=True, help="pref_sample.py output folder (pairs.jsonl)")
    ap.add_argument("--sft", required=True, help="the SFT build the pairs' prompts come from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--nll", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=None, help="default 2 x r")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--pairs-per-step", type=int, default=16)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--logit-chunk", type=int, default=1024)
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = all (a smoke test sets a few)")
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--save-every", type=int, default=50, help="optimizer steps between adapter saves")
    a = ap.parse_args(argv)
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from LegalMT.prompts import encode_batch

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base)
    prompts = {e["id"]: e for e in read_jsonl(paths.artifacts("sft", a.sft, "train.jsonl"))}
    pairs = read_jsonl(Path(a.pairs) / "pairs.jsonl")
    random.Random(a.seed).shuffle(pairs)
    rows, too_long = [], 0
    ids_all = encode_batch(tok, [prompts[p["id"]]["prompt"] for p in pairs])
    for p, pid in zip(pairs, ids_all):
        if len(pid) != prompts[p["id"]]["prompt_tokens"]:
            raise SystemExit(f"{p['id']}: prompt tokenizes to {len(pid)} ids, the build says {prompts[p['id']]['prompt_tokens']}")
        if len(pid) + max(len(p["chosen_ids"]), len(p["rejected_ids"])) > a.max_len or not p["rejected_ids"]:
            too_long += 1
            continue
        rows.append((pid, p["chosen_ids"], p["rejected_ids"]))
    if a.max_pairs:
        rows = rows[:a.max_pairs]
    steps = max(1, len(rows) // a.pairs_per_step)
    rows = rows[:steps * a.pairs_per_step]
    print(f"{len(rows)} pairs ({too_long} over {a.max_len} tokens or without a rejected side dropped), {steps} steps", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=a.r, lora_alpha=a.alpha or 2.0 * a.r, lora_dropout=a.dropout, target_modules=TARGETS,
                     bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.train()
    hf = model.get_base_model()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters {sum(p.numel() for p in params):,}", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)
    warm = max(1, int(a.warmup * steps))

    def lr_at(s: int) -> float:
        if s < warm:
            return a.lr * (s + 1) / warm
        q = (s - warm) / max(1, steps - warm)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * q)))

    mpath = out / "metrics.jsonl"
    t0, peak = time.monotonic(), 0
    acc: dict[str, float] = {}
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        acc = {"loss": 0.0, "dpo": 0.0, "nll": 0.0, "margin": 0.0, "win": 0.0, "rw_c": 0.0, "rw_r": 0.0}
        for pid, c, r in rows[s * a.pairs_per_step:(s + 1) * a.pairs_per_step]:
            n = len(pid)
            ic = torch.tensor([pid + c], device="cuda")
            ir = torch.tensor([pid + r], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad(), model.disable_adapter():
                    ref_c = completion_logp(hf, ic, n, a.logit_chunk, grad=False)
                    ref_r = completion_logp(hf, ir, n, a.logit_chunk, grad=False)
                pi_c = completion_logp(hf, ic, n, a.logit_chunk, grad=True)
                pi_r = completion_logp(hf, ir, n, a.logit_chunk, grad=True)
            margin = a.beta * ((pi_c - ref_c) - (pi_r - ref_r))
            dpo = -F.logsigmoid(margin)
            nll = -pi_c / len(c)
            loss = dpo + a.nll * nll
            (loss / a.pairs_per_step).backward()
            for k, v in (("loss", loss), ("dpo", dpo), ("nll", nll), ("margin", margin),
                         ("rw_c", a.beta * (pi_c - ref_c)), ("rw_r", a.beta * (pi_r - ref_r))):
                acc[k] += float(v.detach()) / a.pairs_per_step
            acc["win"] += float(margin.detach() > 0) / a.pairs_per_step
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        peak = max(peak, torch.cuda.max_memory_reserved() // 2**20)
        rec = {"step": s + 1, "lr": lr_at(s), "grad_norm": round(gn, 4), "seconds": round(time.monotonic() - t0, 1),
               "peak_mib_reserved": int(peak), **{k: round(v, 5) for k, v in acc.items()}}
        with mpath.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        if (s + 1) % 5 == 0 or s == 0:
            print(json.dumps(rec), flush=True)
        if (s + 1) % a.save_every == 0 and s + 1 < steps:
            model.save_pretrained(str(out / f"checkpoint-{s + 1}"))
    adir = out / "adapter"
    model.save_pretrained(str(adir))
    for f in TOKENIZER_FILES:
        if (Path(a.base) / f).is_file():
            shutil.copyfile(Path(a.base) / f, adir / f)
    man = {"created": R.now_iso(), "base": a.base, "pairs": a.pairs, "sft": a.sft, "n_pairs": len(rows),
           "dropped": too_long, "steps": steps, "args": vars(a), "seconds": round(time.monotonic() - t0, 1),
           "peak_mib_reserved": int(peak), "last": acc, "code_fingerprint": R.code_fingerprint()}
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    print(f"saved {adir}; {json.dumps(man['last'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
