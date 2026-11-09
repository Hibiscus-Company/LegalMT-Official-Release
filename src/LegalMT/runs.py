"""Run directories: one folder per experiment run, with a manifest that says exactly what ran."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

from . import paths

MANIFEST = "manifest.json"
RESULTS_INDEX = "results_index.jsonl"
STATUSES = ("running", "done", "failed")
_VERSIONED_PACKAGES = (
    "torch", "transformers", "vllm", "peft", "trl", "accelerate", "bitsandbytes",
    "sacrebleu", "numpy", "pandas",
)
_TAG_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def runs_root() -> Path:
    return Path(paths.ARTIFACTS_ROOT) / "runs"


def results_index_path() -> Path:
    return Path(paths.ARTIFACTS_ROOT) / RESULTS_INDEX


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sanitize_tag(tag: str) -> str:
    clean = _TAG_UNSAFE.sub("-", tag.strip()).strip("-")
    if not clean:
        raise ValueError(f"tag {tag!r} has no usable characters")
    return clean


def code_fingerprint(repo_root: Path | str | None = None) -> str:
    root = Path(repo_root) if repo_root is not None else paths.REPO_ROOT
    files: list[Path] = sorted((root / "src").rglob("*.py")) if (root / "src").is_dir() else []
    if (root / "configs").is_dir():
        files += sorted(p for p in (root / "configs").rglob("*") if p.is_file())
    files = sorted({p for p in files if "__pycache__" not in p.parts}, key=lambda p: p.relative_to(root).as_posix())
    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode("utf-8") + b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def env_versions() -> dict[str, Any]:
    out: dict[str, Any] = {"python": platform.python_version(), "executable": sys.executable}
    for pkg in _VERSIONED_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda is not None:
        out["CUDA_VISIBLE_DEVICES"] = cuda
    return out


def _json_default(o: Any) -> Any:
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def _dumps(obj: Any, indent: int | None = None) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=indent, default=_json_default)


def _write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(_dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def read_jsonl_tolerant(path: Path | str) -> tuple[list[dict[str, Any]], bool]:
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").split("\n")]
    lines = [ln for ln in lines if ln]
    out: list[dict[str, Any]] = []
    for k, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if k == len(lines) - 1:
                return out, True
            raise
    return out, False


def _torn_tail_start(fh: Any, end: int, block: int = 1 << 20) -> int:
    pos = end
    while pos > 0:
        start = max(0, pos - block)
        fh.seek(start)
        chunk = fh.read(pos - start)
        k = chunk.rfind(b"\n")
        if k >= 0:
            return start + k + 1
        pos = start
    return 0


def append_records(path: Path | str, records: Iterable[dict[str, Any]]) -> Path:
    import fcntl
    import warnings

    p = Path(path)
    with p.open("a+b") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            end = fh.seek(0, os.SEEK_END)
            if end:
                fh.seek(end - 1)
                if fh.read(1) != b"\n":
                    cut = _torn_tail_start(fh, end)
                    fh.truncate(cut)
                    warnings.warn(f"{p}: dropped a half-written last line ({end - cut} bytes) before appending",
                                  RuntimeWarning, stacklevel=3)
            for rec in records:
                fh.write((_dumps(rec) + "\n").encode("utf-8"))
                fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return p


@dataclass
class Run:
    dir: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    t0: float | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.manifest.get("run_name", self.dir.name)

    @property
    def manifest_path(self) -> Path:
        return self.dir / MANIFEST

    def path(self, name: str) -> Path:
        return self.dir / name

    def write_json(self, name: str, obj: Any) -> Path:
        p = self.dir / name
        _write_json_atomic(p, obj)
        return p

    def write_jsonl(self, name: str, records: Iterable[dict[str, Any]]) -> Path:
        p = self.dir / name
        tmp = p.with_name(p.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(_dumps(rec) + "\n")
        os.replace(tmp, p)
        return p

    def append_jsonl(self, name: str, records: dict[str, Any] | Iterable[dict[str, Any]]) -> Path:
        if isinstance(records, dict):
            records = [records]
        return append_records(self.dir / name, records)

    def read_jsonl(self, name: str) -> list[dict[str, Any]]:
        return read_jsonl(self.dir / name)

    def update_manifest(self, **fields: Any) -> dict[str, Any]:
        if self.manifest_path.is_file():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.manifest.update(fields)
        _write_json_atomic(self.manifest_path, self.manifest)
        return self.manifest

    def finish(self, status: str = "done", **fields: Any) -> dict[str, Any]:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
        extra: dict[str, Any] = {"status": status, "finished": now_iso()}
        if self.t0 is not None:
            extra["wall_seconds"] = round(time.monotonic() - self.t0, 3)
        extra.update(fields)
        return self.update_manifest(**extra)


def _unique_run_dir(root: Path, stamp: str, tag: str) -> Path:
    base = f"{stamp}_{tag}"
    cand = root / base
    k = 2
    while True:
        try:
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        except FileExistsError:
            cand = root / f"{base}-{k}"
            k += 1


def new_run(
    tag: str,
    config: dict[str, Any] | None,
    config_path: str | Path | None,
    split: str | None,
    *,
    argv: list[str] | None = None,
    repo_root: Path | str | None = None,
    extra: dict[str, Any] | None = None,
) -> Run:
    tag = sanitize_tag(tag)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = _unique_run_dir(runs_root(), stamp, tag)
    manifest: dict[str, Any] = {
        "run_name": run_dir.name,
        "tag": tag,
        "created": now_iso(),
        "argv": list(sys.argv if argv is None else argv),
        "host": socket.gethostname(),
        "config": config or {},
        "config_path": str(config_path) if config_path is not None else None,
        "split": split,
        "code_fingerprint": code_fingerprint(repo_root),
        "env": env_versions(),
        "status": "running",
    }
    if extra:
        manifest.update(extra)
    run = Run(dir=run_dir, manifest=manifest, t0=time.monotonic())
    _write_json_atomic(run.manifest_path, manifest)
    return run


def resolve_run_dir(path_or_name: str | Path) -> Path:
    p = Path(path_or_name)
    if (p / MANIFEST).is_file():
        return p
    cand = runs_root() / str(path_or_name)
    if (cand / MANIFEST).is_file():
        return cand
    raise FileNotFoundError(f"no run directory with a {MANIFEST} at {p} or {cand}")


def load_run(path: str | Path) -> Run:
    d = resolve_run_dir(path)
    manifest = json.loads((d / MANIFEST).read_text(encoding="utf-8"))
    return Run(dir=d, manifest=manifest)


def append_results_index(record: dict[str, Any]) -> Path:
    missing = [k for k in ("run_name", "created", "split", "tag", "metrics") if k not in record]
    if missing:
        raise ValueError(f"results index record is missing {missing}")
    p = results_index_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    return append_records(p, [record])
