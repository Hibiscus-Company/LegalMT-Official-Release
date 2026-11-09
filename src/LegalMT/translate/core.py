"""Document translator: layout units -> chunks -> batched generation -> line check -> fallback -> rendered document."""
from __future__ import annotations

import copy
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

from .. import paths
from ..chunking import Chunk, PIECE_SEP, base_uid, chunk_stats, merge_pieces, plan_chunks, split_chunk_output
from ..langid import detect_lang, direction_of, other_lang
from ..layout import parse_document, render_document, translation_units
from ..prompts import (
    CONTEXT_TEMPLATES,
    MODES,
    TEMPLATES,
    TERM_JOINERS,
    build_prompt_text,
    encode,
    encode_batch,
    render_user_prompt,
    stop_token_ids,
)
from ..textnorm import normalize_text
from .backends import BACKEND_KEYS, GENERATION_KEYS, Backend, GenOutput, GenRequest, make_backend, max_new_tokens_for

log = logging.getLogger("LegalMT.translate")

INFER_CONFIG_DIR = paths.CONFIG_DIR / "infer"
DIRECTIONS = ("en2vi", "vi2en")

STRUCTURAL_DEFAULTS: dict[str, Any] = {
    "prompt": {"template": "default", "mode": "generation_prompt", "context_mode": "none",
               "context_template": "contextual", "lang_names": "en"},
    "chunking": {"cut_before_heading": True, "min_fill_before_heading_cut": 0.5,
                 "max_unit_tokens": None, "max_units_per_chunk": None, "force_line_count": False},
    "render": {"blank_between_paragraphs": "source", "rewrap_width": None},
    "layout": None,
    "lora_adapter": None,
}


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_path(name_or_path: str | Path) -> Path:
    p = Path(name_or_path)
    if p.is_file():
        return p.resolve()
    cand = INFER_CONFIG_DIR / (p.name if p.suffix == ".yaml" else f"{p.name}.yaml")
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"no inference config at {p} or {cand}")


def load_infer_config(name_or_path: str | Path, overrides: dict[str, Any] | None = None) -> tuple[dict[str, Any], Path]:
    path = config_path(name_or_path)
    seen: list[Path] = []

    def load(p: Path) -> dict[str, Any]:
        if p in seen:
            raise ValueError(f"config extends cycle: {[str(s) for s in seen + [p]]}")
        seen.append(p)
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        parent = raw.pop("extends", None)
        if parent:
            return deep_merge(load((p.parent / parent).resolve()), raw)
        return raw

    cfg = deep_merge(load(path), overrides or {})
    cfg["config_name"] = cfg.get("config_name") or path.stem
    return resolve_config(cfg), path


def resolve_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = deep_merge(STRUCTURAL_DEFAULTS, cfg)
    backend = (out.get("backend") or {}).get("name")
    if backend not in ("hf", "vllm", "echo"):
        raise ValueError(f"backend.name must be hf, vllm or echo, got {backend!r}")
    unknown = sorted(set(out["backend"]) - BACKEND_KEYS[backend])
    unknown += sorted(f"generation.{k}" for k in set(out.get("generation") or {}) - GENERATION_KEYS)
    if unknown:
        raise ValueError(f"unknown config keys for backend {backend!r}: {unknown} (a misspelled key would be ignored "
                         f"silently); known backend keys: {sorted(BACKEND_KEYS[backend])}")
    budgets = (out.get("chunking") or {}).get("budget_tokens") or {}
    missing = [d for d in DIRECTIONS if not budgets.get(d)]
    if missing:
        raise ValueError(f"chunking.budget_tokens needs {missing}")
    fam = (out.get("model") or {}).get("family")
    if fam not in ("hymt", "qwen"):
        raise ValueError(f"model.family must be hymt or qwen, got {fam!r}")
    if backend != "echo" or out["backend"].get("tokenizer") == "model":
        m = out["model"]
        if not m.get("local_path") and not (m.get("repo_id") and m.get("revision")):
            raise ValueError("model needs repo_id AND a pinned revision (or local_path); the base model has no default, "
                             "so name it explicitly")
    p = out["prompt"]
    if p["template"] not in TEMPLATES or p["template"] in TERM_JOINERS or p["template"] in CONTEXT_TEMPLATES:
        raise ValueError(f"prompt.template must be a plain template ({sorted(set(TEMPLATES) - set(TERM_JOINERS) - CONTEXT_TEMPLATES)}); "
                         f"contextual templates go in prompt.context_template, and terminology templates need a glossary "
                         f"source the Translator does not have yet")
    if p["lang_names"] not in ("en", "zh") or p["mode"] not in MODES:
        raise ValueError(f"prompt.lang_names must be en|zh and prompt.mode one of {MODES}")
    if out["prompt"]["context_mode"] not in ("none", "prev_source"):
        raise ValueError(f"prompt.context_mode must be none or prev_source, got {out['prompt']['context_mode']!r}")
    if out["prompt"]["context_mode"] == "prev_source" and out["prompt"]["context_template"] not in CONTEXT_TEMPLATES:
        raise ValueError(f"prompt.context_template must be one of {sorted(CONTEXT_TEMPLATES)}")
    return out


def parse_set_overrides(items: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def registry_revision(repo_id: str) -> str | None:
    try:
        reg = paths.load_config("models")
    except FileNotFoundError:
        return None
    for m in reg.get("models") or []:
        if m.get("repo_id") == repo_id:
            return m.get("revision")
    return None


def resolve_model_path(model_cfg: dict[str, Any]) -> str:
    if model_cfg.get("local_path"):
        p = Path(model_cfg["local_path"]).expanduser()
        if not p.is_dir():
            raise FileNotFoundError(f"model.local_path {p} is not a directory")
        return str(p)
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(paths.load_config("paths")["hf_cache_default"])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from huggingface_hub import snapshot_download

    return snapshot_download(model_cfg["repo_id"], revision=model_cfg["revision"], local_files_only=True)


@dataclass
class DocResult:
    doc_id: str
    src_lang: str
    tgt_lang: str
    text: str
    units: list[dict[str, Any]]
    stats: dict[str, Any]

    @property
    def direction(self) -> str:
        return f"{self.src_lang}2{self.tgt_lang}"


@dataclass
class _Doc:
    doc_id: str
    src: str
    tgt: str
    skel: Any
    units: list[Any]
    chunks: list[Chunk]
    detected: str
    outputs: dict[str, str] = field(default_factory=dict)
    chunk_of: dict[str, int] = field(default_factory=dict)
    fallback: set[str] = field(default_factory=set)
    copied: set[str] = field(default_factory=set)
    failed_chunks: list[int] = field(default_factory=list)
    truncated_chunks: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    n_requests: int = 0
    truncated: int = 0
    seconds: float = 0.0
    max_prompt_tokens: int = 0
    max_request_tokens: int = 0

    @property
    def direction(self) -> str:
        return f"{self.src}2{self.tgt}"

    def add(self, out: GenOutput, req: GenRequest) -> None:
        self.max_prompt_tokens = max(self.max_prompt_tokens, len(req.prompt_ids))
        self.max_request_tokens = max(self.max_request_tokens, len(req.prompt_ids) + req.max_new_tokens)
        self.in_tokens += out.n_in
        self.out_tokens += out.n_out
        self.n_requests += 1
        self.truncated += 0 if out.finished else 1
        self.seconds += out.seconds


class Translator:
    def __init__(self, cfg: dict[str, Any], *, load: bool = True) -> None:
        self.cfg = resolve_config(cfg)
        self.family: str = self.cfg["model"]["family"]
        self.backend_name: str = self.cfg["backend"]["name"]
        self.tokenizer: Any = None
        self.model_path: str | None = None
        self.backend: Backend | None = None
        self.load_seconds: float | None = None
        self._warnings: list[str] = []
        if load:
            self.load()


    def load(self) -> None:
        t0 = time.monotonic()
        stop = None
        want_tok = self.backend_name != "echo" or (self.cfg["backend"].get("tokenizer") == "model")
        if want_tok:
            m = self.cfg["model"]
            if m.get("repo_id") and m.get("revision"):
                reg = registry_revision(m["repo_id"])
                if reg and reg != m["revision"]:
                    self._warnings.append(f"{m['repo_id']}: config revision {m['revision']} differs from "
                                          f"models.yaml {reg}")
            self.model_path = resolve_model_path(m)
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            stop = stop_token_ids(self.tokenizer, self.family)
        self.backend = make_backend(self.cfg, self.tokenizer, self.model_path, stop)
        self.backend.load()
        self.load_seconds = time.monotonic() - t0
        for w in self._warnings:
            log.warning(w)

    def info(self) -> dict[str, Any]:
        m = self.cfg["model"]
        return {
            "config_name": self.cfg.get("config_name"),
            "family": self.family,
            "repo_id": m.get("repo_id"), "revision": m.get("revision"), "local_path": m.get("local_path"),
            "model_path": self.model_path,
            "tokenizer": type(self.tokenizer).__name__ if self.tokenizer is not None else "whitespace",
            "prompt": dict(self.cfg["prompt"]),
            "load_seconds": None if self.load_seconds is None else round(self.load_seconds, 3),
            "warnings": list(self._warnings),
            **(self.backend.info() if self.backend is not None else {}),
        }

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()


    def count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text.split())
        return len(encode(self.tokenizer, text))

    def _count_many(self, texts: Sequence[str]) -> dict[str, int]:
        uniq = list(dict.fromkeys(texts))
        if self.tokenizer is None:
            return {t: len(t.split()) for t in uniq}
        return {t: len(ids) for t, ids in zip(uniq, encode_batch(self.tokenizer, uniq))}

    def _encode(self, prompt_text: str) -> list[int]:
        if self.tokenizer is None:
            return [0] * len(prompt_text.split())
        return encode(self.tokenizer, prompt_text)


    def _request(self, src_text: str, direction: str, tgt: str, context: str | None,
                 n_lines: int | None = None) -> GenRequest:
        p = self.cfg["prompt"]
        template = p["context_template"] if context else p["template"]
        user = render_user_prompt(self.family, src_text, tgt, context=context, template=template,
                                  lang_names=p["lang_names"])
        text = build_prompt_text(self.tokenizer, self.family, user, mode=p["mode"])
        n_src = self.count_tokens(src_text)
        gen = self.cfg.get("generation") or {}
        if self.backend_name == "echo" and "max_new_tokens_ratio" not in gen:
            max_new = 10**9
        else:
            max_new = max_new_tokens_for(n_src, direction, gen)
        return GenRequest(prompt_ids=self._encode(text), src_text=src_text, max_new_tokens=max_new, prompt_text=text,
                          n_lines=n_lines)

    def _context(self, d: _Doc, chunk_idx: int) -> str | None:
        if self.cfg["prompt"]["context_mode"] == "prev_source" and chunk_idx > 0:
            return d.chunks[chunk_idx - 1].text
        return None

    def _run(self, requests: list[GenRequest]) -> list[GenOutput]:
        if not requests:
            return []
        order = sorted(range(len(requests)), key=lambda i: len(requests[i].prompt_ids), reverse=True)
        outs = self.backend.generate([requests[i] for i in order])
        if len(outs) != len(requests):
            raise RuntimeError(f"backend returned {len(outs)} outputs for {len(requests)} requests")
        res: list[GenOutput | None] = [None] * len(requests)
        for k, i in enumerate(order):
            res[i] = outs[k]
        return res


    def _prepare(self, doc_id: str, text: str, src_lang: str | None) -> tuple[_Doc, float]:
        t0 = time.monotonic()
        detected = detect_lang(text)
        src = src_lang or detected
        tgt = other_lang(src)
        direction = direction_of(src)
        skel = parse_document(text, src, self.cfg.get("layout"))
        units = translation_units(skel)
        counts = self._count_many([u.text.strip() for u in units])
        ck = self.cfg["chunking"]

        def count(t: str) -> int:
            n = counts.get(t)
            return n if n is not None else self.count_tokens(t)

        chunks = plan_chunks(units, count, int(ck["budget_tokens"][direction]),
                             cut_before_heading=bool(ck["cut_before_heading"]),
                             min_fill_before_heading_cut=float(ck["min_fill_before_heading_cut"]),
                             max_unit_tokens=ck.get("max_unit_tokens"), max_units_per_chunk=ck.get("max_units_per_chunk"))
        d = _Doc(doc_id=doc_id, src=src, tgt=tgt, skel=skel, units=units, chunks=chunks, detected=detected)
        for ci, ch in enumerate(chunks):
            for uid in ch.uids:
                d.chunk_of[uid] = ci
        return d, time.monotonic() - t0

    def translate_documents(self, docs: Sequence[tuple[str, str, str | None]]) -> list[DocResult]:
        if self.backend is None:
            self.load()
        own: list[float] = []
        states: list[_Doc] = []
        for doc_id, text, src_lang in docs:
            d, secs = self._prepare(doc_id, text, src_lang)
            states.append(d)
            own.append(secs)

        keys, reqs = [], []
        force = bool(self.cfg["chunking"].get("force_line_count"))
        for di, d in enumerate(states):
            for ci, ch in enumerate(d.chunks):
                keys.append((di, ci))
                reqs.append(self._request(ch.text, d.direction, d.tgt, self._context(d, ci),
                                          n_lines=len(ch.uids) if force else None))
        outs = self._run(reqs)
        retry: list[tuple[int, int, str, str]] = []
        for (di, ci), req, out in zip(keys, reqs, outs):
            d = states[di]
            d.add(out, req)
            ch = d.chunks[ci]
            lines = split_chunk_output(ch, normalize_text(out.text)) if out.finished else None
            if lines is None:
                d.failed_chunks.append(ci)
                d.truncated_chunks += 0 if out.finished else 1
                retry.extend((di, ci, uid, src) for uid, src in zip(ch.uids, ch.lines))
            else:
                d.outputs.update(zip(ch.uids, lines))
        n_pass1 = len(reqs)

        reqs2 = [self._request(src, states[di].direction, states[di].tgt, self._context(states[di], ci))
                 for di, ci, _, src in retry]
        outs2 = self._run(reqs2)
        for (di, ci, uid, src), req, out in zip(retry, reqs2, outs2):
            d = states[di]
            d.add(out, req)
            text = " ".join(ln.strip() for ln in normalize_text(out.text).splitlines() if ln.strip())
            if not text:
                text = src
                d.copied.add(uid)
            d.outputs[uid] = text
            d.fallback.add(uid)
        log.info("translated %d docs: %d chunk requests, %d unit fallback requests", len(states), n_pass1, len(reqs2))

        return [self._finish(d, secs) for d, secs in zip(states, own)]

    def translate_text(self, text: str, src_lang: str | None = None, doc_id: str = "text") -> DocResult:
        return self.translate_documents([(doc_id, text, src_lang)])[0]


    def _finish(self, d: _Doc, own_seconds: float) -> DocResult:
        t0 = time.monotonic()
        translations = merge_pieces(d.outputs)
        for u in d.units:
            translations.setdefault(u.uid, u.text)
        r = self.cfg["render"]
        rstats: dict[str, Any] = {}
        text = render_document(d.skel, translations, blank_between_paragraphs=r["blank_between_paragraphs"],
                               rewrap_width=r.get("rewrap_width"), stats=rstats)
        n_pieces: dict[str, int] = {}
        for uid in d.outputs:
            if PIECE_SEP in uid:
                n_pieces[base_uid(uid)] = n_pieces.get(base_uid(uid), 0) + 1
        fallback_units = {base_uid(u) for u in d.fallback}
        copied_units = {base_uid(u) for u in d.copied}
        units = []
        for u in d.units:
            first = f"{u.uid}{PIECE_SEP}0" if u.uid in n_pieces else u.uid
            units.append({
                "uid": u.uid, "element_idx": u.element_idx, "cell_idx": u.cell_idx, "kind": u.kind,
                "src": u.text, "hyp": translations.get(u.uid, u.text),
                "chunk": d.chunk_of.get(first), "fallback": u.uid in fallback_units,
                "source_copied": u.uid in copied_units, "pieces": n_pieces.get(u.uid, 0),
            })
        cs = chunk_stats(d.chunks)
        stats = {
            "direction": d.direction,
            "detected_src_lang": d.detected,
            "layout_family": d.skel.layout_family,
            "n_units": len(d.units),
            "n_chunks": len(d.chunks),
            "n_pieces": cs["n_pieces"],
            "chunk_cuts": cs["cuts"],
            "max_chunk_tokens": cs["max_chunk_tokens"],
            "src_tokens": cs["src_tokens"],
            "src_words": sum(len(u.text.split()) for u in d.units),
            "in_tokens": d.in_tokens,
            "out_tokens": d.out_tokens,
            "n_requests": d.n_requests,
            "failed_chunks": len(d.failed_chunks),
            "truncated_chunks": d.truncated_chunks,
            "fallback_units": len(fallback_units),
            "source_copied_units": len(copied_units),
            "truncated_requests": d.truncated,
            "max_prompt_tokens": d.max_prompt_tokens,
            "max_request_tokens": d.max_request_tokens,
            "render_missing": rstats.get("missing", 0),
        }
        stats["seconds"] = round(own_seconds + d.seconds + (time.monotonic() - t0), 4)
        return DocResult(doc_id=d.doc_id, src_lang=d.src, tgt_lang=d.tgt, text=text, units=units, stats=stats)
