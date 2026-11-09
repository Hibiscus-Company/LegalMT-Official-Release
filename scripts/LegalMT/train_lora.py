#!/usr/bin/env python
"""LoRA SFT of a LegalMT base model on the prompt/completion data of build_sft.py.

    scripts/LegalMT/gpu_run.sh scripts/LegalMT/train_lora.py --config configs/LegalMT/train/smoke.yaml
    scripts/LegalMT/gpu_run.sh scripts/LegalMT/train_lora.py --resume <run_name>
    ... --config <cfg> --plan-only
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml

from LegalMT import paths, runs
from LegalMT.profiling import GpuMemorySampler, memory_report, torch_peak_mib
from LegalMT.prompts import HY_IDS, completion_suffix, encode_batch
from LegalMT.sft_data import DIRECTIONS, tokenizer_fingerprint
from LegalMT.translate.core import deep_merge, parse_set_overrides, registry_revision, resolve_model_path

log = logging.getLogger("train_lora")
TRAIN_CONFIG_DIR = paths.CONFIG_DIR / "train"
PLAN_FILE = "plan.json"
TRAIN_CONFIG_FILE = "train_config.json"
METRICS_FILE = "metrics.jsonl"
ADAPTER_DIR = "adapter"

SCHEMA: dict[str, set[str]] = {
    "model": {"family", "repo_id", "revision", "local_path", "dtype", "attn_implementation"},
    "data": {"sft", "dir", "directions", "max_seq_len", "token_budget", "order_seed", "reshuffle_each_epoch",
             "eval_max_examples", "allow_tokenizer_mismatch"},
    "lora": {"r", "alpha", "dropout", "target_modules", "use_rslora", "bias", "init_from"},
    "training": {"seed", "max_steps", "per_device_train_batch_size", "per_device_eval_batch_size",
                 "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type", "min_lr_rate", "warmup_ratio",
                 "weight_decay", "max_grad_norm", "optim", "bf16", "gradient_checkpointing", "loss_type",
                 "logging_steps", "eval_steps", "eval_on_start", "save_steps", "save_total_limit", "device",
                 "loraplus_lr_ratio", "plain_layers"},
}
TOP_KEYS = {"run_tag", "description", "model", "data", "lora", "training", "config_name"}


def config_path(name_or_path: str | Path) -> Path:
    p = Path(name_or_path)
    if p.is_file():
        return p.resolve()
    cand = TRAIN_CONFIG_DIR / (p.name if p.suffix == ".yaml" else f"{p.name}.yaml")
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"no training config at {p} or {cand}")


def load_train_config(name_or_path: str | Path, overrides: dict[str, Any] | None = None) -> tuple[dict[str, Any], Path]:
    path = config_path(name_or_path)
    seen: list[Path] = []

    def load(p: Path) -> dict[str, Any]:
        if p in seen:
            raise ValueError(f"config extends cycle: {[str(s) for s in seen + [p]]}")
        seen.append(p)
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        parent = raw.pop("extends", None)
        return deep_merge(load((p.parent / parent).resolve()), raw) if parent else raw

    cfg = deep_merge(load(path), overrides or {})
    cfg["config_name"] = cfg.get("config_name") or path.stem
    return resolve_train_config(cfg), path


def resolve_train_config(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    unknown = sorted(set(cfg) - TOP_KEYS)
    for block, keys in SCHEMA.items():
        cfg.setdefault(block, {})
        unknown += sorted(f"{block}.{k}" for k in set(cfg[block]) - keys)
    if unknown:
        raise ValueError(f"unknown training config keys {unknown} (a misspelled key would be ignored silently)")
    m, d, t, lo = cfg["model"], cfg["data"], cfg["training"], cfg["lora"]
    if m.get("family") not in ("hymt", "qwen"):
        raise ValueError(f"model.family must be hymt or qwen, got {m.get('family')!r}")
    if not m.get("local_path") and not (m.get("repo_id") and m.get("revision")):
        raise ValueError("model needs repo_id AND a pinned revision (or local_path): the base model has no default")
    for key in ("seed",):
        if not isinstance(t.get(key), int):
            raise ValueError(f"training.{key} must be listed explicitly as an int (seeds have no default)")
    if not isinstance(d.get("order_seed"), int):
        raise ValueError("data.order_seed must be listed explicitly as an int (seeds have no default)")
    if not (d.get("sft") or d.get("dir")):
        raise ValueError("data.sft (a build name under <ARTIFACTS>/sft/) or data.dir is required")
    if not d.get("max_seq_len"):
        raise ValueError("data.max_seq_len is required")
    if not (d.get("token_budget") or t.get("max_steps")):
        raise ValueError("set data.token_budget (tokens) or training.max_steps")
    for key in ("r", "alpha", "dropout", "target_modules"):
        if lo.get(key) is None:
            raise ValueError(f"lora.{key} is required")
    for key in ("per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type",
                "optim", "bf16", "gradient_checkpointing"):
        if t.get(key) is None:
            raise ValueError(f"training.{key} is required")
    if not 0.0 <= float(t.get("warmup_ratio") or 0.0) < 1.0:
        raise ValueError(f"training.warmup_ratio must be in [0, 1), got {t.get('warmup_ratio')!r}")
    pl = t.get("plain_layers")
    if pl is not None and (not isinstance(pl, int) or isinstance(pl, bool) or pl < 0):
        raise ValueError(f"training.plain_layers must be an int >= 0, got {pl!r}")
    if pl and not t["gradient_checkpointing"]:
        raise ValueError("training.plain_layers needs gradient_checkpointing: true (without it every layer is plain)")
    if t["lr_scheduler_type"] == "cosine_with_min_lr" and t.get("min_lr_rate") is None:
        raise ValueError("training.min_lr_rate is required with lr_scheduler_type cosine_with_min_lr")
    init = lo.get("init_from")
    if init is not None:
        if not isinstance(init, str) or not init:
            raise ValueError(f"lora.init_from must be an adapter directory path, got {init!r}")
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (Path(init) / name).is_file():
                raise ValueError(f"lora.init_from: no {name} in {init}")
    dirs = d.get("directions") or list(DIRECTIONS)
    if not set(dirs) <= set(DIRECTIONS):
        raise ValueError(f"data.directions must be a subset of {DIRECTIONS}")
    d["directions"] = list(dirs)
    d.setdefault("reshuffle_each_epoch", True)
    d.setdefault("eval_max_examples", 0)
    lo.setdefault("use_rslora", False)
    lo.setdefault("bias", "none")
    m.setdefault("dtype", "bfloat16")
    m.setdefault("attn_implementation", "sdpa")
    t.setdefault("loss_type", "chunked_nll")
    t.setdefault("per_device_eval_batch_size", 1)
    t.setdefault("logging_steps", 10)
    t.setdefault("eval_on_start", False)
    t.setdefault("save_total_limit", 2)
    return cfg


def sft_dir(cfg: dict[str, Any]) -> Path:
    d = cfg["data"]
    return Path(d["dir"]) if d.get("dir") else paths.artifacts("sft", str(d["sft"]))


def read_examples(path: Path, directions: Sequence[str]) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ex = json.loads(line)
                if ex["direction"] in directions:
                    out.append(ex)
    return out


def check_sft_manifest(sdir: Path, model_dir: str, family: str, allow_mismatch: bool) -> dict[str, Any]:
    man = json.loads((sdir / "manifest.json").read_text(encoding="utf-8"))
    if not man.get("trainable", False):
        raise ValueError(f"{sdir} was built with a whitespace tokenizer (dry run); rebuild with the model tokenizer")
    if man.get("family") != family:
        raise ValueError(f"SFT data family {man.get('family')!r} differs from model family {family!r}")
    want = (man.get("tokenizer") or {}).get("sha256")
    have = tokenizer_fingerprint(model_dir)["sha256"]
    if want != have and not allow_mismatch:
        raise ValueError(f"tokenizer files of {model_dir} differ from the ones {sdir} was built with "
                         f"({have[:12]} vs {str(want)[:12]}): prompts and token counts would not match")
    return man


def check_split_roles(train_ids: set[str], dev_ids: set[str], split_train_fit: set[str], split_dev: set[str]) -> None:
    leak = sorted(train_ids & split_dev)
    outside = sorted(train_ids - split_train_fit)
    dev_outside = sorted(dev_ids - split_dev)
    if leak or outside or dev_outside:
        raise ValueError(f"the SFT build does not match the current split: {len(leak)} train documents are dev "
                         f"documents (e.g. {leak[:3]}), {len(outside)} are not train_fit (e.g. {outside[:3]}), "
                         f"{len(dev_outside)} dev-file documents are not dev (e.g. {dev_outside[:3]}); rebuild the "
                         f"SFT data with scripts/LegalMT/build_sft.py --force")


def check_final_roles(train_ids: set[str], dev_ids: set[str], split_train_fit: set[str], split_dev: set[str],
                      public_ids: set[str]) -> None:
    outside = sorted(train_ids - split_train_fit - split_dev - public_ids)
    dev_outside = sorted(dev_ids - split_dev)
    if outside or dev_outside:
        raise ValueError(f"the FINAL SFT build does not match the current split: {len(outside)} train documents "
                         f"are neither split nor public-test documents (e.g. {outside[:3]}), {len(dev_outside)} "
                         f"dev-file documents are not dev (e.g. {dev_outside[:3]}); rebuild it with build_sft.py --force")


def sft_input_checks(man: dict[str, Any], train_ex: Sequence[dict[str, Any]],
                     dev_ex: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from LegalMT.splits import file_sha256, load_split_spec

    inputs = man.get("inputs") or {}
    out: dict[str, Any] = {}
    extra_train = Counter(ex["source"] for ex in train_ex if ex.get("source"))
    if any(ex.get("source") for ex in dev_ex):
        raise ValueError("the SFT dev file holds extra-source examples: the dev loss must stay in-domain; rebuild it")
    unlisted = sorted(set(extra_train) - {e.get("name") for e in man.get("extra") or []})
    if unlisted:
        raise ValueError(f"extra sources {unlisted} are not listed in the SFT manifest's extra section")
    version = inputs.get("split_version")
    if version is None:
        log.warning("SFT manifest has no inputs.split_version: split leak check skipped (not a build_sft output?)")
        out["split_check"] = "skipped: no inputs.split_version in the SFT manifest"
    else:
        spec = load_split_spec(str(version))
        final = man.get("final")
        public: set[str] = set()
        if final and final.get("public_test"):
            from LegalMT.data import load_split

            public = set(load_split("public_test").df["sample_id"])
        named = sorted({ex["sample_id"] for ex in train_ex if ex.get("source")}
                       & (set(spec.train_fit_ids) | set(spec.dev_ids) | public))
        if named:
            raise ValueError(f"{len(named)} extra-source examples carry a split document id (e.g. {named[:3]})")
        in_domain = {ex["sample_id"] for ex in train_ex if not ex.get("source")}
        dev_docs = {ex["sample_id"] for ex in dev_ex}
        if final:
            check_final_roles(in_domain, dev_docs, set(spec.train_fit_ids), set(spec.dev_ids), public)
            out["split_check"] = (f"FINAL build on split {version}: dev{' and public-test' if public else ''} "
                                  f"documents are in train; the dev loss is a fit monitor, not a held-out loss")
            log.warning(out["split_check"])
        else:
            check_split_roles(in_domain, dev_docs, set(spec.train_fit_ids), set(spec.dev_ids))
            out["split_check"] = f"ok: split {version}"
    aligned = Path(inputs["aligned"]) if inputs.get("aligned") else None
    if aligned is None or not aligned.is_file() or not inputs.get("aligned_sha256"):
        out["aligned_current"] = None
    else:
        out["aligned_current"] = file_sha256(aligned) == inputs["aligned_sha256"]
        if not out["aligned_current"]:
            log.warning("%s changed after the SFT build: the examples predate the current aligned data; "
                        "consider rebuilding the SFT data before a long run", aligned)
    if extra_train:
        out["extra_train_examples"] = dict(sorted(extra_train.items()))
    return out


def tokenize_examples(tokenizer: Any, family: str, examples: Sequence[dict[str, Any]],
                      batch_size: int = 256) -> list[dict[str, Any]]:
    suffix_id = tokenizer.convert_tokens_to_ids(completion_suffix(tokenizer, family))
    rows = []
    for k in range(0, len(examples), batch_size):
        part = examples[k:k + batch_size]
        p_ids = encode_batch(tokenizer, [ex["prompt"] for ex in part])
        c_ids = encode_batch(tokenizer, [ex["completion"] for ex in part])
        for ex, p, c in zip(part, p_ids, c_ids):
            if len(p) != ex["prompt_tokens"] or len(c) != ex["completion_tokens"]:
                raise ValueError(f"{ex['id']}: token counts ({len(p)}, {len(c)}) differ from the build "
                                 f"({ex['prompt_tokens']}, {ex['completion_tokens']}): different tokenizer?")
            if c[-1] != suffix_id:
                raise ValueError(f"{ex['id']}: completion does not end with the eos id {suffix_id}")
            ids = p + c
            if family == "hymt" and (p[0] != HY_IDS["bos"] or ids.count(HY_IDS["bos"]) != 1 or HY_IDS["no8"] in ids):
                raise ValueError(f"{ex['id']}: HY example must start with one BOS and never contain no_8")
            rows.append({"input_ids": ids, "labels": [-100] * len(p) + c, "n_prompt": len(p), "id": ex["id"]})
    return rows


@dataclass
class Plan:
    order: list[int]
    per_step: int
    step_tokens: list[int]
    max_steps: int
    epochs: float
    n_examples: int
    token_budget: int | None

    @property
    def total_tokens(self) -> int:
        return sum(self.step_tokens)

    def tokens_at(self, step: int) -> int:
        return sum(self.step_tokens[:step])

    def fingerprint(self) -> str:
        h = hashlib.sha256(json.dumps([self.order, self.per_step], separators=(",", ":")).encode())
        return h.hexdigest()

    def summary(self) -> dict[str, Any]:
        return {"max_steps": self.max_steps, "per_step": self.per_step, "n_examples": self.n_examples,
                "rows": len(self.order), "epochs": round(self.epochs, 4), "total_tokens": self.total_tokens,
                "token_budget": self.token_budget,
                "step_tokens_mean": round(self.total_tokens / max(1, self.max_steps), 1),
                "step_tokens_max": max(self.step_tokens, default=0), "fingerprint": self.fingerprint()}


def build_plan(lengths: Sequence[int], per_step: int, *, token_budget: int | None, max_steps: int | None,
               seed: int, reshuffle: bool = True) -> Plan:
    n = len(lengths)
    if n == 0 or per_step < 1:
        raise ValueError("no training examples (or per_step < 1)")
    if max_steps is None and token_budget is None:
        max_steps = math.ceil(n / per_step)
    epoch_orders: dict[int, list[int]] = {}

    def epoch_order(e: int) -> list[int]:
        key = e if reshuffle else 0
        if key not in epoch_orders:
            idx = list(range(n))
            random.Random(f"{seed}|{key}").shuffle(idx)
            epoch_orders[key] = idx
        return epoch_orders[key]

    order: list[int] = []
    step_tokens: list[int] = []
    pos = 0
    total = 0
    while True:
        if max_steps is not None and len(step_tokens) >= max_steps:
            break
        if token_budget is not None and total >= token_budget:
            break
        rows = []
        for _ in range(per_step):
            e, k = divmod(pos, n)
            rows.append(epoch_order(e)[k])
            pos += 1
        order.extend(rows)
        step_tokens.append(sum(lengths[i] for i in rows))
        total += step_tokens[-1]
    return Plan(order=order, per_step=per_step, step_tokens=step_tokens, max_steps=len(step_tokens),
                epochs=pos / n, n_examples=n, token_budget=token_budget)


def data_fingerprint(sdir: Path) -> str:
    h = hashlib.sha256()
    for name in ("train.jsonl", "dev.jsonl"):
        p = sdir / name
        if not p.is_file():
            continue
        h.update(name.encode() + b"\0")
        with p.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
        h.update(b"\0")
    return h.hexdigest()


def script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def check_resume_plan(old: dict[str, Any], new: dict[str, Any], *, allow_code_change: bool = False) -> None:
    for key, what in (("fingerprint", "the data plan (order or steps)"), ("data_sha256", "the SFT data files"),
                      ("sft_tokenizer_sha256", "the SFT build's tokenizer")):
        if key in old and old[key] != new.get(key):
            raise ValueError(f"resume: {what} differs from the run's start ({key}); rerun from zero")
    if "script_sha256" in old and old["script_sha256"] != new.get("script_sha256") and not allow_code_change:
        raise ValueError("resume: scripts/LegalMT/train_lora.py changed since the run started; rerun from zero or "
                         "pass --allow-code-change knowingly")


def eval_subset(examples: Sequence[dict[str, Any]], k: int, directions: Sequence[str]) -> list[dict[str, Any]]:
    if not k:
        return []
    per = {d: [ex for ex in examples if ex["direction"] == d] for d in directions}
    quota = {d: k // len(directions) + (1 if i < k % len(directions) else 0) for i, d in enumerate(directions)}
    return [ex for d in directions for ex in per[d][: quota[d]]]


def _trainer_class() -> type:
    from trl import SFTTrainer

    class LegalMTSFTTrainer(SFTTrainer):
        def setup_legalmt(self, plan: Plan, sampler: GpuMemorySampler | None, metrics_path: Path,
                          session: int = 1) -> None:
            self._lm_plan = plan
            self._lm_session = session
            self._lm_cum = [0]
            for n in plan.step_tokens:
                self._lm_cum.append(self._lm_cum[-1] + n)
            self._lm_sampler = sampler
            self._lm_metrics = metrics_path
            self._lm_last: tuple[float, int, float] | None = None
            self._lm_eval_seconds = 0.0

        def train(self, *args: Any, **kwargs: Any) -> Any:
            self._lm_last = None
            return super().train(*args, **kwargs)

        def create_optimizer(self, *args: Any, **kwargs: Any) -> Any:
            opt = super().create_optimizer(*args, **kwargs)
            ratio = getattr(self, "_lm_loraplus_ratio", None)
            if ratio is None or float(ratio) == 1.0 or getattr(self, "_lm_loraplus_done", False):
                return opt
            names = {id(p): n for n, p in self.model.named_parameters()}
            groups = []
            for g in opt.param_groups:
                base = {k: v for k, v in g.items() if k != "params"}
                rest = [p for p in g["params"] if "lora_B" not in names.get(id(p), "")]
                b = [p for p in g["params"] if "lora_B" in names.get(id(p), "")]
                if rest:
                    groups.append({**base, "params": rest})
                if b:
                    groups.append({**base, "params": b, "lr": base["lr"] * float(ratio)})
            opt.param_groups.clear()
            for g in groups:
                opt.add_param_group(g)
            self._lm_loraplus_done = True
            return opt

        def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
            step = self.state.global_step
            now = time.monotonic()
            tokens = self._lm_cum[min(step, len(self._lm_cum) - 1)]
            logs["tokens_seen"] = tokens
            if "eval_runtime" in logs:
                self._lm_eval_seconds += float(logs["eval_runtime"])
            if self._lm_last is None:
                self._lm_last = (now, tokens, self._lm_eval_seconds)
            elif "loss" in logs:
                t0, tok0, ev0 = self._lm_last
                dt = (now - t0) - (self._lm_eval_seconds - ev0)
                if dt > 0 and tokens > tok0:
                    logs["tokens_per_second"] = round((tokens - tok0) / dt, 1)
                self._lm_last = (now, tokens, self._lm_eval_seconds)
            peak = torch_peak_mib()
            if peak is not None:
                logs["peak_mem_mib_torch"] = round(peak, 1)
            if self._lm_sampler is not None and self._lm_sampler.peak_mib is not None:
                logs["peak_mem_mib_nvidia_smi"] = round(self._lm_sampler.peak_mib, 1)
            super().log(logs, start_time)
            rec = {"step": step, "session": self._lm_session, "time": runs.now_iso(), **logs}
            with self._lm_metrics.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=float) + "\n")

    return LegalMTSFTTrainer


def _torch_dtype(name: str) -> Any:
    import torch

    table = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16,
             "float32": torch.float32, "fp32": torch.float32}
    if name not in table:
        raise ValueError(f"model.dtype must be one of {sorted(table)}, got {name!r}")
    return table[name]


@dataclass
class Prepared:
    cfg: dict[str, Any]
    model_dir: str
    tokenizer: Any
    train_rows: list[dict[str, Any]]
    eval_rows: list[dict[str, Any]]
    plan: Plan
    sft_manifest: dict[str, Any]
    dropped_too_long: dict[str, int]
    input_checks: dict[str, Any]


def prepare_data(cfg: dict[str, Any]) -> Prepared:
    m, d, t = cfg["model"], cfg["data"], cfg["training"]
    model_dir = resolve_model_path(m)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    sdir = sft_dir(cfg)
    man = check_sft_manifest(sdir, model_dir, m["family"], bool(d.get("allow_tokenizer_mismatch")))
    limit = int(d["max_seq_len"])
    dropped: dict[str, int] = {}

    def fits(exs: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
        keep = [ex for ex in exs if ex["prompt_tokens"] + ex["completion_tokens"] <= limit]
        dropped[role] = len(exs) - len(keep)
        return keep

    train_ex = fits(read_examples(sdir / "train.jsonl", d["directions"]), "train")
    dev_ex = fits(read_examples(sdir / "dev.jsonl", d["directions"]), "dev") if (sdir / "dev.jsonl").is_file() else []
    checks = sft_input_checks(man, train_ex, dev_ex)
    per_step = int(t["per_device_train_batch_size"]) * int(t["gradient_accumulation_steps"])
    lengths = [ex["prompt_tokens"] + ex["completion_tokens"] for ex in train_ex]
    plan = build_plan(lengths, per_step, token_budget=d.get("token_budget"), max_steps=t.get("max_steps"),
                      seed=int(d["order_seed"]), reshuffle=bool(d["reshuffle_each_epoch"]))
    used = sorted(set(plan.order))
    rows = tokenize_examples(tokenizer, m["family"], [train_ex[i] for i in used])
    by_index = dict(zip(used, rows))
    train_rows = [by_index[i] for i in plan.order]
    eval_rows = tokenize_examples(tokenizer, m["family"],
                                  eval_subset(dev_ex, int(d.get("eval_max_examples") or 0), d["directions"]))
    return Prepared(cfg=cfg, model_dir=model_dir, tokenizer=tokenizer, train_rows=train_rows, eval_rows=eval_rows,
                    plan=plan, sft_manifest=man, dropped_too_long=dropped, input_checks=checks)


def build_trainer(prep: Prepared, output_dir: Path, sampler: GpuMemorySampler | None = None, *,
                  resuming: bool = False, session: int = 1) -> Any:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig

    from transformers import set_seed

    cfg = prep.cfg
    m, d, lo, t = cfg["model"], cfg["data"], cfg["lora"], cfg["training"]
    device = t.get("device")
    use_cpu = device == "cpu" or (device is None and not torch.cuda.is_available())
    set_seed(int(t["seed"]))
    model = AutoModelForCausalLM.from_pretrained(prep.model_dir, dtype=_torch_dtype(m["dtype"]),
                                                 attn_implementation=m.get("attn_implementation"))
    model.config.use_cache = False
    peft_config = LoraConfig(r=int(lo["r"]), lora_alpha=float(lo["alpha"]), lora_dropout=float(lo["dropout"]),
                             target_modules=list(lo["target_modules"]), use_rslora=bool(lo["use_rslora"]),
                             bias=str(lo["bias"]), task_type="CAUSAL_LM")
    sched_kwargs = {"min_lr_rate": float(t["min_lr_rate"])} if t["lr_scheduler_type"] == "cosine_with_min_lr" else None
    eval_on = bool(prep.eval_rows) and bool(t.get("eval_steps"))
    args = SFTConfig(
        output_dir=str(output_dir),
        max_length=int(d["max_seq_len"]),
        dataset_kwargs={"skip_prepare_dataset": True},
        packing=False,
        padding_free=False,
        loss_type=str(t["loss_type"]),
        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(t["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=str(t["lr_scheduler_type"]),
        lr_scheduler_kwargs=sched_kwargs,
        warmup_steps=float(t.get("warmup_ratio") or 0.0),
        weight_decay=float(t.get("weight_decay") or 0.0),
        max_grad_norm=float(t["max_grad_norm"]) if t.get("max_grad_norm") is not None else 1.0,
        optim=str(t["optim"]),
        bf16=bool(t["bf16"]),
        fp16=False,
        gradient_checkpointing=bool(t["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False} if t["gradient_checkpointing"] else None,
        max_steps=prep.plan.max_steps,
        train_sampling_strategy="sequential",
        logging_steps=int(t["logging_steps"]),
        logging_first_step=True,
        eval_strategy="steps" if eval_on else "no",
        eval_steps=int(t["eval_steps"]) if eval_on else None,
        eval_on_start=bool(t.get("eval_on_start")) and eval_on and not resuming,
        prediction_loss_only=True,
        save_strategy="steps" if t.get("save_steps") else "no",
        save_steps=int(t["save_steps"]) if t.get("save_steps") else 500,
        save_total_limit=int(t["save_total_limit"]),
        save_only_model=False,
        seed=int(t["seed"]),
        data_seed=int(d["order_seed"]),
        report_to="none",
        dataloader_num_workers=0,
        use_cpu=use_cpu,
        remove_unused_columns=True,
    )

    def dataset(rows: list[dict[str, Any]]) -> Any:
        return Dataset.from_dict({"input_ids": [r["input_ids"] for r in rows], "labels": [r["labels"] for r in rows]})

    trainer_cls = _trainer_class()
    trainer = trainer_cls(model=model, args=args, train_dataset=dataset(prep.train_rows),
                          eval_dataset=dataset(prep.eval_rows) if eval_on else None,
                          processing_class=prep.tokenizer, peft_config=peft_config)
    trainer.setup_legalmt(prep.plan, sampler, output_dir / METRICS_FILE, session)
    if lo.get("init_from"):
        loaded = load_adapter_weights(trainer.model, Path(lo["init_from"]), lo)
        log.info("init_from: %d LoRA tensors loaded from %s", loaded["tensors"], loaded["path"])
    trainer._lm_loraplus_ratio = t.get("loraplus_lr_ratio")
    if t.get("plain_layers"):
        trainer.add_callback(_plain_layers_callback(int(t["plain_layers"])))
    return trainer


def adapter_init_info(adapter_dir: Path) -> dict[str, Any]:
    from LegalMT.splits import file_sha256

    return {"path": str(adapter_dir), "adapter_sha256": file_sha256(adapter_dir / "adapter_model.safetensors")}


def load_adapter_weights(model: Any, adapter_dir: Path, lora: dict[str, Any]) -> dict[str, Any]:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    acfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    want = {"r": int(lora["r"]), "lora_alpha": float(lora["alpha"]), "use_rslora": bool(lora.get("use_rslora")),
            "target_modules": sorted(lora["target_modules"])}
    got = {"r": int(acfg["r"]), "lora_alpha": float(acfg["lora_alpha"]), "use_rslora": bool(acfg.get("use_rslora")),
           "target_modules": sorted(acfg["target_modules"])}
    if want != got:
        raise ValueError(f"lora.init_from: the adapter's LoRA shape {got} differs from this run's {want}")
    state = load_file(str(adapter_dir / "adapter_model.safetensors"))
    res = set_peft_model_state_dict(model, state)
    unexpected = sorted(res.unexpected_keys)
    missing = sorted(k for k in res.missing_keys if "lora_" in k)
    if unexpected or missing:
        raise ValueError(f"lora.init_from: {len(unexpected)} unexpected tensors (e.g. {unexpected[:2]}), "
                         f"{len(missing)} LoRA tensors not loaded (e.g. {missing[:2]})")
    return {"path": str(adapter_dir), "tensors": len(state)}


def _plain_layers_callback(k: int) -> Any:
    from transformers import TrainerCallback
    from transformers.modeling_layers import GradientCheckpointingLayer

    class PlainLayers(TrainerCallback):
        def on_train_begin(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
            layers = [mod for mod in model.modules() if isinstance(mod, GradientCheckpointingLayer)]
            for mod in layers[:k]:
                mod.gradient_checkpointing = False
            log.info("plain_layers: %d of %d decoder layers without gradient checkpointing",
                     sum(not mod.gradient_checkpointing for mod in layers), len(layers))

    return PlainLayers()


def trainable_params(model: Any) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable": int(train), "total": int(total)}


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def run(cfg: dict[str, Any], cfg_path: Path | None, *, resume: str | None = None, argv: list[str] | None = None,
        allow_code_change: bool = False, trainer_hook: Callable[[Any], None] | None = None) -> dict[str, Any]:
    from transformers.trainer_utils import get_last_checkpoint

    t_start = time.monotonic()
    fingerprint = runs.code_fingerprint()
    if resume:
        run_h = runs.load_run(resume)
        if run_h.manifest.get("config") != json.loads(json.dumps(cfg, default=str)):
            raise ValueError("resume: the resolved config differs from the run's manifest config")
        if run_h.manifest.get("code_fingerprint") != fingerprint and not allow_code_change:
            raise ValueError("resume: src/ or configs/ changed since the run started (code fingerprint differs); "
                             "rerun from zero or pass --allow-code-change knowingly")
        run_h.t0 = time.monotonic()
    else:
        m = cfg["model"]
        extra = {"task": "lora_sft", "model": {k: m.get(k) for k in ("family", "repo_id", "revision", "local_path")}}
        run_h = runs.new_run(cfg.get("run_tag") or cfg["config_name"], cfg, cfg_path, "train_fit", argv=argv, extra=extra)
    out_dir = paths.artifacts("checkpoints", run_h.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = cfg["model"]
    if m.get("repo_id") and m.get("revision"):
        reg = registry_revision(m["repo_id"])
        if reg and reg != m["revision"]:
            log.warning("%s: config revision %s differs from models.yaml %s", m["repo_id"], m["revision"], reg)

    sampler = GpuMemorySampler(interval=1.0).start()
    try:
        t0 = time.monotonic()
        prep = prepare_data(cfg)
        plan_info = {**prep.plan.summary(), "dropped_too_long": prep.dropped_too_long,
                     "eval_examples": len(prep.eval_rows), "sft_dir": str(sft_dir(cfg)),
                     "sft_name": prep.sft_manifest.get("name"), "sft_tokenizer_sha256": prep.sft_manifest["tokenizer"]["sha256"],
                     "data_sha256": data_fingerprint(sft_dir(cfg)), "script_sha256": script_sha256(),
                     "sft_input_checks": prep.input_checks}
        if cfg["lora"].get("init_from"):
            plan_info["init_from"] = adapter_init_info(Path(cfg["lora"]["init_from"]))
        script_now = plan_info["script_sha256"]
        plan_path = out_dir / PLAN_FILE
        if resume and plan_path.is_file():
            old = json.loads(plan_path.read_text(encoding="utf-8"))
            check_resume_plan(old, plan_info, allow_code_change=allow_code_change)
            plan_info = {**plan_info, **{k: old[k] for k in ("script_sha256",) if k in old}}
        _write_json(plan_path, plan_info)
        _write_json(out_dir / TRAIN_CONFIG_FILE, {"config": cfg, "config_path": str(cfg_path) if cfg_path else None,
                                                  "run_name": run_h.name, "model_dir": prep.model_dir})
        log.info("plan: %d steps x %d examples, %d tokens (%.3f epochs); dropped too long %s; eval %d examples",
                 prep.plan.max_steps, prep.plan.per_step, prep.plan.total_tokens, prep.plan.epochs,
                 prep.dropped_too_long, len(prep.eval_rows))
        session = int(run_h.manifest.get("sessions_started") or 0) + 1
        run_h.update_manifest(sessions_started=session)
        trainer = build_trainer(prep, out_dir, sampler, resuming=bool(resume), session=session)
        params = trainable_params(trainer.model)
        load_seconds = time.monotonic() - t0
        run_h.update_manifest(checkpoint_dir=str(out_dir), plan=plan_info, params=params)
        if trainer_hook is not None:
            trainer_hook(trainer)
        ckpt = None
        if resume:
            ckpt = get_last_checkpoint(str(out_dir))
            if ckpt is None:
                raise FileNotFoundError(f"resume: no checkpoint-* folder in {out_dir}")
            log.info("resuming from %s", ckpt)
        t1 = time.monotonic()
        result = trainer.train(resume_from_checkpoint=ckpt)
        train_seconds = time.monotonic() - t1
        adapter_dir = out_dir / ADAPTER_DIR
        trainer.save_model(str(adapter_dir))
        trainer.save_state()
        hist = trainer.state.log_history
        eval_losses = [h["eval_loss"] for h in hist if "eval_loss" in h]
        step = trainer.state.global_step
        complete = step >= prep.plan.max_steps
        tokens_session = prep.plan.tokens_at(step) - (prep.plan.tokens_at(int(Path(ckpt).name.split("-")[-1])) if ckpt else 0)
        sessions = list(run_h.manifest.get("sessions") or [])
        sessions.append({"session": session, "resumed_from": ckpt, "end_step": step,
                         "train_seconds": round(train_seconds, 1), "tokens": tokens_session,
                         "script_sha256": script_now, "finished": runs.now_iso()})
        summary = {
            "run_name": run_h.name,
            "checkpoint_dir": str(out_dir),
            "adapter_dir": str(adapter_dir),
            "global_step": step,
            "max_steps": prep.plan.max_steps,
            "complete": complete,
            "tokens_seen": prep.plan.tokens_at(step),
            "train_loss": result.training_loss,
            "eval_loss_first": eval_losses[0] if eval_losses else None,
            "eval_loss_last": eval_losses[-1] if eval_losses else None,
            "params": params,
        }
        sampler.stop()
        run_h.finish(
            "done" if complete else "failed",
            stopped_early=not complete,
            sessions=sessions,
            timing={"load_seconds": round(load_seconds, 1), "train_seconds": round(train_seconds, 1),
                    "total_seconds": round(time.monotonic() - t_start, 1)},
            memory=memory_report(sampler),
            throughput={"tokens_per_second": round(tokens_session / train_seconds, 1) if train_seconds > 0 else None,
                        "tokens_session": tokens_session},
            final=summary,
        )
        return summary
    except BaseException as exc:
        sampler.stop()
        run_h.finish("failed", error=f"{type(exc).__name__}: {exc}", memory=memory_report(sampler))
        raise


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="training config yaml (or a name under configs/LegalMT/train/)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted override, e.g. training.max_steps=5 (repeatable; recorded in the manifest)")
    ap.add_argument("--resume", metavar="RUN_NAME", help="continue this run from its last checkpoint (config from its manifest)")
    ap.add_argument("--allow-code-change", action="store_true", help="resume even if src/ or configs/ changed")
    ap.add_argument("--plan-only", action="store_true", help="print the data plan (steps, tokens, epochs) and exit")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    if not args.config and not args.resume:
        ap.error("--config or --resume is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.resume:
        man = runs.load_run(args.resume).manifest
        cfg, cfg_path = man["config"], Path(man["config_path"]) if man.get("config_path") else None
        if args.config or args.set:
            raise SystemExit("--resume takes the config from the run's manifest; do not pass --config/--set")
    else:
        cfg, cfg_path = load_train_config(args.config, parse_set_overrides(args.set))
    if args.plan_only:
        prep = prepare_data(cfg)
        print(json.dumps({**prep.plan.summary(), "dropped_too_long": prep.dropped_too_long,
                          "eval_examples": len(prep.eval_rows), "sft_input_checks": prep.input_checks}, indent=1))
        return 0
    summary = run(cfg, cfg_path, resume=args.resume, argv=sys.argv if argv is None else ["train_lora.py", *argv],
                  allow_code_change=args.allow_code_change)
    print(json.dumps(summary, indent=1, default=str))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
