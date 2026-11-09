"""Document translation: Translator turns whole documents into translated documents (backends: hf, vllm, echo)."""
from __future__ import annotations

from .backends import BACKENDS, GenOutput, GenRequest, make_backend
from .core import DocResult, Translator, load_infer_config, parse_set_overrides, resolve_config

__all__ = [
    "BACKENDS", "DocResult", "GenOutput", "GenRequest", "Translator", "load_infer_config", "make_backend",
    "parse_set_overrides", "resolve_config",
]
