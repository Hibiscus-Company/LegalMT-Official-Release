"""Generation backends for the LegalMT translator: "hf" (transformers generate), "vllm" (vllm.LLM)"""
from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Sequence

log = logging.getLogger("LegalMT.translate")

BACKENDS = ("hf", "vllm", "echo")


@dataclass
class GenRequest:
    prompt_ids: list[int]
    src_text: str
    max_new_tokens: int
    prompt_text: str = ""
    n_lines: int | None = None


def line_regex(n_lines: int) -> str:
    if n_lines < 1:
        raise ValueError(f"n_lines must be >= 1, got {n_lines}")
    line = r"[^\n]*\S[^\n]*"
    return line + (r"(?:\n" + line + r"){%d}" % (n_lines - 1) if n_lines > 1 else "")


@dataclass
class GenOutput:
    text: str
    n_in: int
    n_out: int
    finished: bool
    seconds: float = 0.0
    token_ids: list[int] | None = None


def _share_seconds(total: float, n_out: Sequence[int]) -> list[float]:
    s = sum(n_out)
    if not n_out:
        return []
    if s <= 0:
        return [total / len(n_out)] * len(n_out)
    return [total * n / s for n in n_out]


def decoding_params(gen: dict[str, Any]) -> dict[str, Any]:
    missing = [k for k in ("do_sample", "repetition_penalty") if k not in gen]
    if gen.get("do_sample"):
        missing += [k for k in ("temperature", "top_p", "top_k") if gen.get(k) is None]
    if missing:
        raise ValueError(f"generation config must set {missing} explicitly (the model's generation_config "
                         f"would otherwise fill them in)")
    out = {"do_sample": bool(gen["do_sample"]), "repetition_penalty": float(gen["repetition_penalty"]),
           "seed": int(gen.get("seed", 0))}
    if out["do_sample"]:
        out.update(temperature=float(gen["temperature"]), top_p=float(gen["top_p"]), top_k=int(gen["top_k"]))
    return out


class Backend:
    name = "base"

    def load(self) -> None:
        raise NotImplementedError

    def generate(self, requests: Sequence[GenRequest]) -> list[GenOutput]:
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        return {"backend": self.name}

    def close(self) -> None:
        return None


class EchoBackend(Backend):
    name = "echo"

    def __init__(self, cfg: dict[str, Any], tokenizer: Any = None) -> None:
        b = cfg.get("backend") or {}
        self.tokenizer = tokenizer
        self.prefix = str(b.get("prefix") or "")
        self.fail_mode = b.get("fail_mode")
        if self.fail_mode not in (None, "drop_last_line", "merge_lines"):
            raise ValueError(f"echo fail_mode must be null, drop_last_line or merge_lines, got {self.fail_mode!r}")

    def load(self) -> None:
        return None

    def generate(self, requests: Sequence[GenRequest]) -> list[GenOutput]:
        outs = []
        for r in requests:
            lines = r.src_text.split("\n")
            if len(lines) > 1 and self.fail_mode == "drop_last_line":
                lines = lines[:-1]
            elif len(lines) > 1 and self.fail_mode == "merge_lines":
                lines = [lines[0] + " " + lines[1]] + lines[2:]
            text = "\n".join(self.prefix + ln for ln in lines)
            n_out = self._count(text)
            if n_out > r.max_new_tokens:
                words = text.split(" ")
                lo, hi = 0, len(words)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if self._count(" ".join(words[:mid])) <= r.max_new_tokens:
                        lo = mid
                    else:
                        hi = mid - 1
                outs.append(GenOutput(text=" ".join(words[:lo]), n_in=len(r.prompt_ids), n_out=r.max_new_tokens,
                                      finished=False))
            else:
                outs.append(GenOutput(text=text, n_in=len(r.prompt_ids), n_out=n_out + 1, finished=True))
        return outs

    def _count(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text.split())
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "prefix": self.prefix, "fail_mode": self.fail_mode,
                "tokenizer": type(self.tokenizer).__name__ if self.tokenizer is not None else "whitespace"}


class HFBackend(Backend):
    name = "hf"

    def __init__(self, cfg: dict[str, Any], tokenizer: Any, model_path: str, stop_ids: list[int]) -> None:
        self.cfg = cfg
        self.b = cfg.get("backend") or {}
        self.gen = decoding_params(cfg.get("generation") or {})
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.stop_ids = stop_ids
        self.adapter = cfg.get("lora_adapter")
        self.model = None
        self.device = None
        self.dtype_name = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        self.device = self.b.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        want = self.b.get("dtype", "auto")
        if want in (None, "auto"):
            dtype = torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32
        else:
            dtype = getattr(torch, want)
        self.dtype_name = str(dtype).replace("torch.", "")
        kwargs: dict[str, Any] = {"dtype": dtype}
        if str(self.device).startswith("cuda"):
            kwargs["device_map"] = self.device
        if self.b.get("attn_implementation"):
            kwargs["attn_implementation"] = self.b["attn_implementation"]
        model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        if self.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(self.adapter))
            if self.b.get("lora_merge", True):
                model = model.merge_and_unload()
        model.to(self.device)
        model.eval()
        self.model = model
        pad = self.tokenizer.pad_token_id
        self.pad_id = int(pad if pad is not None else self.stop_ids[0])
        self.tokenizer.padding_side = "left"

    def _batches(self, order: list[int], requests: Sequence[GenRequest]) -> list[list[int]]:
        bs = int(self.b.get("batch_size", 8))
        max_tok = int(self.b.get("max_batch_tokens", 32768))
        batches: list[list[int]] = []
        cur: list[int] = []
        cur_len = cur_new = 0
        for i in order:
            r = requests[i]
            n_len, n_new = max(cur_len, len(r.prompt_ids)), max(cur_new, r.max_new_tokens)
            if cur and (len(cur) >= bs or (len(cur) + 1) * (n_len + n_new) > max_tok):
                batches.append(cur)
                cur, n_len, n_new = [], len(r.prompt_ids), r.max_new_tokens
            cur.append(i)
            cur_len, cur_new = n_len, n_new
        if cur:
            batches.append(cur)
        return batches

    def _gen_kwargs(self, max_new: int) -> dict[str, Any]:
        g = self.gen
        kw: dict[str, Any] = {
            "max_new_tokens": max_new,
            "do_sample": g["do_sample"],
            "repetition_penalty": g["repetition_penalty"],
            "eos_token_id": self.stop_ids,
            "pad_token_id": self.pad_id,
        }
        if g["do_sample"]:
            kw.update(temperature=g["temperature"], top_p=g["top_p"], top_k=g["top_k"])
        else:
            kw.update(temperature=None, top_p=None, top_k=None)
        return kw

    def generate(self, requests: Sequence[GenRequest]) -> list[GenOutput]:
        import torch

        if self.model is None:
            raise RuntimeError("HFBackend.generate before load()")
        out: list[GenOutput | None] = [None] * len(requests)
        order = sorted(range(len(requests)), key=lambda i: len(requests[i].prompt_ids), reverse=True)
        stop = set(self.stop_ids)
        for batch in self._batches(order, requests):
            t0 = time.monotonic()
            L = max(len(requests[i].prompt_ids) for i in batch)
            ids = torch.full((len(batch), L), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), L), dtype=torch.long)
            for row, i in enumerate(batch):
                p = requests[i].prompt_ids
                ids[row, L - len(p):] = torch.tensor(p, dtype=torch.long)
                mask[row, L - len(p):] = 1
            max_new = max(requests[i].max_new_tokens for i in batch)
            if self.gen["do_sample"]:
                torch.manual_seed(self.gen["seed"])
            with torch.inference_mode():
                gen = self.model.generate(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                          **self._gen_kwargs(max_new))
            new = gen[:, L:].tolist()
            rows = []
            for row, i in enumerate(batch):
                toks = new[row][: requests[i].max_new_tokens]
                cut = next((j for j, t in enumerate(toks) if t in stop), None)
                finished = cut is not None
                body = toks if cut is None else toks[:cut]
                n_out = len(body) + (1 if finished else 0)
                text = self.tokenizer.decode(body, skip_special_tokens=True)
                rows.append((i, text, n_out, finished, body))
            secs = _share_seconds(time.monotonic() - t0, [r[2] for r in rows])
            for (i, text, n_out, finished, body), s in zip(rows, secs):
                out[i] = GenOutput(text=text, n_in=len(requests[i].prompt_ids), n_out=n_out, finished=finished,
                                   seconds=s, token_ids=body)
        return [o for o in out if o is not None]

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "device": str(self.device), "dtype": self.dtype_name,
                "batch_size": int(self.b.get("batch_size", 8)), "max_batch_tokens": int(self.b.get("max_batch_tokens", 32768)),
                "lora_adapter": self.adapter, "decoding": self.gen, "stop_token_ids": self.stop_ids}

    def close(self) -> None:
        self.model = None
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def gpu_free_fraction() -> tuple[float, float] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        res = subprocess.run([exe, "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=False)
        free, total = (float(x) for x in res.stdout.splitlines()[0].split(","))
        return free, total
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class VLLMBackend(Backend):
    name = "vllm"
    ENGINE_KEYS = ("gpu_memory_utilization", "max_model_len", "enforce_eager", "max_num_seqs",
                   "max_num_batched_tokens", "kv_cache_memory_bytes", "quantization",
                   "enable_prefix_caching", "linear_backend",
                   "hf_overrides", "compilation_config")

    def __init__(self, cfg: dict[str, Any], tokenizer: Any, model_path: str, stop_ids: list[int]) -> None:
        self.cfg = cfg
        self.b = cfg.get("backend") or {}
        self.gen = decoding_params(cfg.get("generation") or {})
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.stop_ids = stop_ids
        self.adapter = cfg.get("lora_adapter")
        self.llm = None
        self.lora_request = None
        self.engine_kwargs: dict[str, Any] = {}

    def _preflight(self) -> None:
        util = self.b.get("gpu_memory_utilization")
        ft = gpu_free_fraction()
        if util is None or ft is None:
            return
        free, total = ft
        if free < float(util) * total:
            msg = (f"vLLM will refuse to start: free GPU memory {free:.0f} MiB < gpu_memory_utilization {util} x "
                   f"{total:.0f} MiB = {float(util) * total:.0f} MiB (another process holds the card?). Lower "
                   f"backend.gpu_memory_utilization below {free / total:.2f}; kv_cache_memory_bytes does not bypass "
                   f"this check.")
            log.warning(msg)
            print(f"WARNING: {msg}", file=sys.stderr)

    def load(self) -> None:
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        self._preflight()
        from vllm import LLM

        kw: dict[str, Any] = {
            "model": self.model_path,
            "tokenizer": self.model_path,
            "dtype": self.b.get("dtype", "bfloat16"),
            "seed": self.gen["seed"],
            "generation_config": self.b.get("generation_config", "vllm"),
            "trust_remote_code": False,
            "disable_log_stats": True,
        }
        for k in self.ENGINE_KEYS:
            if self.b.get(k) is not None:
                kw[k] = self.b[k]
        if self.adapter:
            kw["enable_lora"] = True
            kw["max_lora_rank"] = int(self.b.get("max_lora_rank", 64))
            kw["max_loras"] = 1
        self.engine_kwargs = dict(kw)
        self.llm = LLM(**kw)
        if self.adapter:
            from vllm.lora.request import LoRARequest

            self.lora_request = LoRARequest("legalmt_adapter", 1, str(self.adapter))
        self.max_model_len = int(self.llm.model_config.max_model_len)

    def _sampling(self, max_tokens: int, n_lines: int | None = None):
        from vllm import SamplingParams

        g = self.gen
        common = {"max_tokens": max_tokens, "repetition_penalty": g["repetition_penalty"],
                  "stop_token_ids": list(self.stop_ids), "skip_special_tokens": True, "seed": g["seed"]}
        if n_lines and n_lines > 1:
            from vllm.sampling_params import StructuredOutputsParams

            common["structured_outputs"] = StructuredOutputsParams(regex=line_regex(n_lines))
        if g["do_sample"]:
            return SamplingParams(temperature=g["temperature"], top_p=g["top_p"], top_k=g["top_k"], **common)
        return SamplingParams(temperature=0.0, **common)

    def generate(self, requests: Sequence[GenRequest]) -> list[GenOutput]:
        from vllm.inputs import TokensPrompt

        if self.llm is None:
            raise RuntimeError("VLLMBackend.generate before load()")
        out: list[GenOutput | None] = [None] * len(requests)
        live = []
        for i, r in enumerate(requests):
            room = self.max_model_len - len(r.prompt_ids)
            if room < 2:
                log.warning("prompt of %d tokens leaves no room under max_model_len %d: skipped",
                            len(r.prompt_ids), self.max_model_len)
                out[i] = GenOutput(text="", n_in=len(r.prompt_ids), n_out=0, finished=False)
            else:
                live.append((i, min(r.max_new_tokens, room)))
        if live:
            t0 = time.monotonic()
            prompts = [TokensPrompt(prompt_token_ids=list(requests[i].prompt_ids)) for i, _ in live]
            params = [self._sampling(m, requests[i].n_lines) for i, m in live]
            res = self.llm.generate(prompts, params, use_tqdm=bool(self.b.get("progress", False)),
                                    lora_request=self.lora_request)
            rows = []
            for (i, _), o in zip(live, res):
                c = o.outputs[0]
                ids = [int(t) for t in c.token_ids]
                rows.append((i, c.text, len(ids), c.finish_reason == "stop", [t for t in ids if t not in self.stop_ids]))
            secs = _share_seconds(time.monotonic() - t0, [r[2] for r in rows])
            for (i, text, n_out, fin, ids), s in zip(rows, secs):
                out[i] = GenOutput(text=text, n_in=len(requests[i].prompt_ids), n_out=n_out, finished=fin, seconds=s,
                                   token_ids=ids)
        return [o for o in out if o is not None]

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "engine_kwargs": self.engine_kwargs, "lora_adapter": self.adapter,
                "decoding": self.gen, "stop_token_ids": self.stop_ids,
                "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")}

    def close(self) -> None:
        llm, self.llm = self.llm, None
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown(timeout=60)
            except Exception as e:
                log.warning("vLLM engine shutdown failed: %s", e)
            del llm
            import gc

            gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


BACKEND_KEYS: dict[str, frozenset[str]] = {
    "echo": frozenset({"name", "prefix", "fail_mode", "tokenizer"}),
    "hf": frozenset({"name", "device", "dtype", "batch_size", "max_batch_tokens", "attn_implementation", "lora_merge"}),
    "vllm": frozenset({"name", "dtype", "generation_config", "max_lora_rank", "progress", *VLLMBackend.ENGINE_KEYS}),
}
GENERATION_KEYS = frozenset({"do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "seed",
                             "max_new_tokens_ratio", "max_new_tokens_margin", "max_new_tokens_cap"})


def make_backend(cfg: dict[str, Any], tokenizer: Any = None, model_path: str | None = None,
                 stop_ids: list[int] | None = None) -> Backend:
    name = (cfg.get("backend") or {}).get("name")
    if name == "echo":
        return EchoBackend(cfg, tokenizer)
    if name not in BACKENDS:
        raise ValueError(f"backend.name must be one of {BACKENDS}, got {name!r}")
    if tokenizer is None or model_path is None or not stop_ids:
        raise ValueError(f"backend {name!r} needs a tokenizer, a model path and stop token ids")
    cls = HFBackend if name == "hf" else VLLMBackend
    return cls(cfg, tokenizer, model_path, stop_ids)


def max_new_tokens_for(n_src_tokens: int, direction: str, gen: dict[str, Any]) -> int:
    try:
        ratio = float(gen["max_new_tokens_ratio"][direction])
        margin = int(gen["max_new_tokens_margin"])
        cap = int(gen["max_new_tokens_cap"])
    except (KeyError, TypeError) as exc:
        raise ValueError("generation needs max_new_tokens_ratio.{en2vi,vi2en}, max_new_tokens_margin and "
                         "max_new_tokens_cap") from exc
    return max(1, min(cap, math.ceil(ratio * n_src_tokens) + margin))
