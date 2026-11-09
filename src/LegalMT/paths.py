"""Path resolution for LegalMT."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_THIS = Path(__file__).resolve()

REPO_ROOT: Path = _THIS.parents[2]
if not (REPO_ROOT / "configs" / "LegalMT").is_dir():
    for cand in _THIS.parents:
        if (cand / "configs" / "LegalMT").is_dir() and (cand / "src" / "LegalMT").is_dir():
            REPO_ROOT = cand
            break
    else:
        raise RuntimeError(f"Could not locate the LegalMT repo root from {_THIS}")

CONFIG_DIR: Path = REPO_ROOT / "configs" / "LegalMT"
DOCS_DIR: Path = REPO_ROOT / "docs" / "LegalMT"


def load_config(name: str = "paths") -> dict[str, Any]:
    path = CONFIG_DIR / (name if name.endswith(".yaml") else f"{name}.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"No config at {path}")
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if isinstance(cfg, dict) and cfg.get("hf_cache_default"):
        cfg["hf_cache_default"] = str(Path(cfg["hf_cache_default"]).expanduser())
    return cfg


def data_dir() -> Path:
    cfg = load_config("paths")
    return REPO_ROOT / cfg["data_root"]


def split_path(split: str) -> Path:
    cfg = load_config("paths")
    rel = cfg["splits"][split]
    return data_dir() / rel


def _artifacts_root() -> Path:
    cfg = load_config("paths")
    root = Path(os.environ.get(cfg["artifacts_root_env"], cfg["artifacts_root_default"])).expanduser()
    return (root if root.is_absolute() else REPO_ROOT / root).resolve()


ARTIFACTS_ROOT: Path = _artifacts_root()


def artifacts(*parts: str) -> Path:
    return ARTIFACTS_ROOT.joinpath(*parts)
