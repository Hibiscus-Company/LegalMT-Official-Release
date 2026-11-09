#!/usr/bin/env python3
"""LegalMT translation server: the process inside the VLSP 2026 Docker submission image."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import re
import signal
import socket
import sys
import threading
import time
from bisect import bisect_right
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import parse_qs, urlsplit

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from LegalMT import paths
from LegalMT.langid import detect_lang, other_lang
from LegalMT.profiling import GpuMemorySampler, memory_report
from LegalMT.translate.core import deep_merge, load_infer_config, parse_set_overrides

log = logging.getLogger("LegalMT.serve")

SERVE_CONFIG_ENV = "LEGALMT_SERVE_CONFIG"
DEFAULT_SERVE_CONFIG = REPO_ROOT / "configs" / "LegalMT" / "serve.yaml"
INFER_DIR = REPO_ROOT / "configs" / "LegalMT" / "infer"

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
}
ENV_DEFAULTS = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}

SERVER_DEFAULTS: dict[str, Any] = {
    "host": "0.0.0.0",
    "port": 8000,
    "max_body_bytes": 64 * 2**20,
    "warmup": True,
    "newline": "auto",
    "memory_sample_interval": 1.0,
    "shutdown_grace_seconds": 30,
    "socket_timeout_seconds": 300,
    "log_level": "INFO",
}
SERVE_KEYS = {"infer_config", "overrides", "server", "image"}
SRC_LANGS = ("auto", "en", "vi")

WARMUP_DOCS: list[tuple[str, str, str]] = [
    ("warmup-en", "Article 1. Scope\n\nThis regulation applies to the upkeep of public gardens in the city.\n", "en"),
    ("warmup-vi", "Điều 1. Phạm vi\n\nQuy chế này áp dụng cho việc chăm sóc vườn hoa công cộng trong thành phố.\n", "vi"),
]


def serve_config_path(explicit: str | None = None) -> Path:
    ref = explicit or os.environ.get(SERVE_CONFIG_ENV) or str(DEFAULT_SERVE_CONFIG)
    p = Path(ref).expanduser()
    if not p.is_absolute() and not p.is_file():
        p = REPO_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"no serving config at {ref}")
    return p.resolve()


def load_serve_config(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = sorted(set(raw) - SERVE_KEYS)
    unknown += sorted(f"server.{k}" for k in set(raw.get("server") or {}) - set(SERVER_DEFAULTS))
    if unknown:
        raise ValueError(f"{path}: unknown serving config keys {unknown}; known top level {sorted(SERVE_KEYS)}, "
                         f"server keys {sorted(SERVER_DEFAULTS)}")
    if not raw.get("infer_config"):
        raise ValueError(f"{path}: infer_config is required (there is no default model: name it explicitly)")
    if not isinstance(raw.get("overrides") or {}, dict):
        raise ValueError(f"{path}: overrides must be a mapping")
    raw.setdefault("overrides", {})
    raw["overrides"] = raw["overrides"] or {}
    raw["server"] = {**SERVER_DEFAULTS, **(raw.get("server") or {})}
    return raw


def resolve_infer_path(ref: str | Path, serve_path: str | Path | None = None) -> Path:
    p = Path(ref).expanduser()
    cands = [p] if p.is_absolute() else [REPO_ROOT / p] + ([Path(serve_path).parent / p] if serve_path else [])
    name = p.name if p.suffix == ".yaml" else f"{p.name}.yaml"
    cands.append(INFER_DIR / name)
    for c in cands:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"inference config {ref!s} not found (tried {[str(c) for c in cands]})")


def infer_config_for(serve_cfg: dict[str, Any], serve_path: str | Path | None = None, infer_ref: str | None = None,
                     set_items: Sequence[str] = ()) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    path = resolve_infer_path(infer_ref or serve_cfg["infer_config"], serve_path)
    overrides = deep_merge(serve_cfg.get("overrides") or {}, parse_set_overrides(list(set_items)))
    cfg, path = load_infer_config(path, overrides)
    return cfg, path, overrides


def hf_snapshot_dir(repo_id: str, revision: str, hf_home: str | None = None) -> Path:
    hub = os.environ.get("HF_HUB_CACHE")
    if not hub:
        home = hf_home or os.environ.get("HF_HOME") or paths.load_config("paths")["hf_cache_default"]
        hub = str(Path(home) / "hub")
    base = Path(hub) / f"models--{repo_id.replace('/', '--')}"
    ref = base / "refs" / revision
    commit = ref.read_text(encoding="utf-8").strip() if ref.is_file() else revision
    return base / "snapshots" / commit


def describe(cfg: dict[str, Any], infer_path: Path, serve_path: Path | None) -> dict[str, Any]:
    m = cfg.get("model") or {}
    model_dir = m.get("local_path")
    if not model_dir and m.get("repo_id") and m.get("revision"):
        model_dir = str(hf_snapshot_dir(m["repo_id"], m["revision"]))
    return {
        "serve_config": str(serve_path) if serve_path else None,
        "infer_config": str(infer_path),
        "config_name": cfg.get("config_name"),
        "backend": (cfg.get("backend") or {}).get("name"),
        "repo_id": m.get("repo_id"),
        "revision": m.get("revision"),
        "local_path": m.get("local_path"),
        "model_dir": model_dir,
        "lora_adapter": cfg.get("lora_adapter"),
        "config": cfg,
    }


def write_image_config(serve_cfg: dict[str, Any], serve_path: Path | None, out_path: str | Path, *,
                       model_local_path: str, infer_ref: str | None = None, set_items: Sequence[str] = (),
                       adapter_path: str | None = None, keep_model_id: bool = True,
                       meta: dict[str, Any] | None = None) -> dict[str, Any]:
    infer_path = resolve_infer_path(infer_ref or serve_cfg["infer_config"], serve_path)
    try:
        rel = infer_path.relative_to(INFER_DIR.resolve())
    except ValueError:
        raise ValueError(f"{infer_path} is outside {INFER_DIR}: the image carries only configs/LegalMT/infer/, "
                         f"so put the inference config (and anything it extends) there") from None
    overrides = deep_merge(serve_cfg.get("overrides") or {}, parse_set_overrides(list(set_items)))
    model: dict[str, Any] = {"local_path": model_local_path}
    if not keep_model_id:
        model.update(repo_id=None, revision=None)
    overrides = deep_merge(overrides, {"model": model})
    if adapter_path:
        overrides["lora_adapter"] = adapter_path
    load_infer_config(infer_path, overrides)
    out = {
        "infer_config": str(Path("configs") / "LegalMT" / "infer" / rel),
        "overrides": overrides,
        "server": dict(serve_cfg["server"]),
        "image": dict(meta or {}),
    }
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    source = serve_path
    if serve_path is not None:
        try:
            source = Path(serve_path).resolve().relative_to(REPO_ROOT)
        except ValueError:
            source = Path(serve_path).name
    header = (f"# Serving config INSIDE the Docker image, written {stamp} by docker/server.py --write-image-config\n"
              f"# (called from docker/build_image.sh). Source serving config: {source}.\n"
              f"# The weights live at {model_local_path} in the image; nothing is read from an HF cache.\n")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(header + yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8")
    load_serve_config(out_path)
    return out


def apply_offline_env() -> None:
    os.environ.update(OFFLINE_ENV)
    for k, v in ENV_DEFAULTS.items():
        os.environ.setdefault(k, v)


_SENT_END = re.compile(r"[.!?…]+[\"'”’»)\]]*(?=\s|$)")
_TOKEN = re.compile(r"\S+")
_ENUMERATOR = re.compile(r"^\(?(\d+([.,]\d+)*|[a-zđ]|[ivxlcdm]+)\)?$", re.IGNORECASE)
_ABBREV = frozenset({"no", "nos", "art", "arts", "cl", "para", "e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr",
                     "st", "co", "ltd", "inc", "tp", "q", "p"})


def approx_sentences(texts: Iterable[str]) -> int:
    n = 0
    for t in texts:
        t = t.strip()
        if not t:
            continue
        stops = list(_SENT_END.finditer(t))
        if not stops:
            n += 1
            continue
        spans = [m.span() for m in _TOKEN.finditer(t)]
        starts = [s for s, _ in spans]
        ends, last = 0, -1
        for m in stops:
            k = bisect_right(starts, m.start()) - 1
            if k >= 0 and spans[k][0] == m.start():
                k -= 1
                word = t[spans[k][0]:spans[k][1]] if k >= 0 else ""
            else:
                word = t[spans[k][0]:m.start()] if k >= 0 else ""
            word = word.strip("(\"'“‘")
            if not word or _ENUMERATOR.match(word) or word.lower().rstrip(".") in _ABBREV:
                continue
            ends, last = ends + 1, m.end()
        n += ends if ends and last == len(t) else ends + 1
    return n


def uses_crlf(text: str) -> bool:
    n = text.count("\n")
    return n > 0 and text.count("\r\n") == n


def to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


class Unavailable(RuntimeError):
    pass


@dataclass
class Outcome:
    text: str
    src_lang: str
    tgt_lang: str
    words: int
    sentences: int
    seconds: float
    queued: float
    stats: dict[str, Any] = field(default_factory=dict)
    request_id: int = 0

    @property
    def direction(self) -> str:
        return f"{self.src_lang}2{self.tgt_lang}"

    def headers(self) -> dict[str, str]:
        return {
            "X-Src-Lang": self.src_lang, "X-Tgt-Lang": self.tgt_lang, "X-Direction": self.direction,
            "X-Seconds": f"{self.seconds:.3f}", "X-Queue-Seconds": f"{self.queued:.3f}",
            "X-Words": str(self.words), "X-Sentences": str(self.sentences),
            "X-Units": str(self.stats.get("n_units", 0)), "X-Chunks": str(self.stats.get("n_chunks", 0)),
            "X-Fallback-Units": str(self.stats.get("fallback_units", 0)),
            "X-Source-Copied-Units": str(self.stats.get("source_copied_units", 0)),
        }


class ServeState:
    def __init__(self, cfg: dict[str, Any], *, server_opts: dict[str, Any] | None = None,
                 serve_config: str | Path | None = None, infer_config: str | Path | None = None,
                 image: dict[str, Any] | None = None, translator: Any = None) -> None:
        self.cfg = cfg
        self.opts = {**SERVER_DEFAULTS, **(server_opts or {})}
        if self.opts["newline"] not in ("auto", "lf"):
            raise ValueError(f"server.newline must be auto or lf, got {self.opts['newline']!r}")
        self.serve_config = str(serve_config) if serve_config else None
        self.infer_config = str(infer_config) if infer_config else None
        self.image = image or {}
        self.translator = translator
        self.status = "ok" if translator is not None else "loading"
        self.lock = threading.Lock()
        self._counters = threading.Lock()
        self.busy = False
        self.started = time.monotonic()
        self.load_seconds: float | None = None
        self.warmup_seconds: float | None = None
        self.n_requests = 0
        self.n_failed = 0
        self.n_rejected = 0
        self.n_words = 0
        self.n_source_copied = 0
        self.busy_seconds = 0.0
        self.last: dict[str, Any] | None = None
        self.sampler: GpuMemorySampler | None = None


    def load(self, warmup: bool | None = None) -> None:
        from LegalMT.translate import Translator

        t0 = time.monotonic()
        self.translator = Translator(self.cfg)
        self.load_seconds = time.monotonic() - t0
        if self.opts["warmup"] if warmup is None else bool(warmup):
            t1 = time.monotonic()
            self.translator.translate_documents(WARMUP_DOCS)
            self.warmup_seconds = time.monotonic() - t1
        self.status = "ok"

    def start_sampler(self) -> None:
        interval = self.opts.get("memory_sample_interval")
        if interval:
            self.sampler = GpuMemorySampler(interval=float(interval)).start()

    def wait_idle(self, timeout: float) -> bool:
        ok = self.lock.acquire(timeout=max(0.0, timeout))
        if ok:
            self.lock.release()
        return ok

    def close(self) -> None:
        if self.sampler is not None:
            self.sampler.stop()
        if self.translator is not None and hasattr(self.translator, "close"):
            self.translator.close()


    def translate(self, text: str, src_lang: str | None) -> Outcome:
        t_enter = time.monotonic()
        with self.lock:
            if self.status != "ok":
                raise Unavailable(f"server status is {self.status}")
            t0 = time.monotonic()
            self.busy = True
            try:
                with self._counters:
                    self.n_requests += 1
                    rid = self.n_requests
                if not text.strip():
                    src = src_lang or detect_lang(text)
                    out = Outcome(text=text, src_lang=src, tgt_lang=other_lang(src), words=0, sentences=0,
                                  seconds=0.0, queued=t0 - t_enter, request_id=rid)
                else:
                    res = self.translator.translate_text(to_lf(text), src_lang, doc_id=f"request-{rid}")
                    body = res.text
                    if self.opts["newline"] == "auto" and uses_crlf(text):
                        body = body.replace("\r\n", "\n").replace("\n", "\r\n")
                    out = Outcome(text=body, src_lang=res.src_lang, tgt_lang=res.tgt_lang, words=len(text.split()),
                                  sentences=approx_sentences(u["src"] for u in res.units), seconds=0.0,
                                  queued=t0 - t_enter, stats=dict(res.stats), request_id=rid)
            except Exception:
                with self._counters:
                    self.n_failed += 1
                raise
            finally:
                self.busy = False
            out.seconds = time.monotonic() - t0
            with self._counters:
                self.n_words += out.words
                self.n_source_copied += int(out.stats.get("source_copied_units", 0) or 0)
                self.busy_seconds += out.seconds
                self.last = {"request": rid, "direction": out.direction, "words": out.words,
                             "seconds": round(out.seconds, 3)}
        return out

    def count_rejected(self) -> None:
        with self._counters:
            self.n_rejected += 1


    def model_info(self) -> dict[str, Any]:
        if self.translator is not None and hasattr(self.translator, "info"):
            try:
                return self.translator.info()
            except Exception as exc:
                return {"error": f"info() failed: {exc}"}
        m = self.cfg.get("model") or {}
        return {"config_name": self.cfg.get("config_name"), "repo_id": m.get("repo_id"),
                "revision": m.get("revision"), "local_path": m.get("local_path"),
                "backend": (self.cfg.get("backend") or {}).get("name")}

    def health(self) -> tuple[int, dict[str, Any]]:
        with self._counters:
            counters = {"served": self.n_requests - self.n_failed, "failed": self.n_failed,
                        "rejected": self.n_rejected, "words": self.n_words,
                        "source_copied_units": self.n_source_copied,
                        "busy_seconds": round(self.busy_seconds, 3), "last": self.last}
        body = {
            "status": self.status,
            "model": self.model_info(),
            "config": {"serve_config": self.serve_config, "infer_config": self.infer_config,
                       "config_name": self.cfg.get("config_name")},
            "busy": self.busy,
            "load_seconds": None if self.load_seconds is None else round(self.load_seconds, 3),
            "warmup_seconds": None if self.warmup_seconds is None else round(self.warmup_seconds, 3),
            "uptime_seconds": round(time.monotonic() - self.started, 3),
            "requests": counters,
            "memory": memory_report(self.sampler) if self.sampler is not None else None,
            "image": self.image or None,
        }
        return (200 if self.status == "ok" else 503), body


class HttpError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "LegalMT/1"
    server: "TranslationServer"

    def setup(self) -> None:
        self.timeout = self.server.state.opts.get("socket_timeout_seconds")
        super().setup()

    def log_message(self, format: str, *args: Any) -> None:
        log.debug("%s %s", self.address_string(), format % args)


    def _send(self, status: int, body: bytes, content_type: str, headers: dict[str, str] | None = None,
              close: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, obj: dict[str, Any]) -> None:
        self._send(status, (json.dumps(obj, ensure_ascii=False, default=str) + "\n").encode("utf-8"),
                   "application/json; charset=utf-8")

    def _error(self, status: int, message: str, close: bool = False) -> None:
        self._send(status, f"error: {message}\n".encode("utf-8"), "text/plain; charset=utf-8", close=close)

    def _read_body(self, limit: int) -> bytes:
        te = (self.headers.get("Transfer-Encoding") or "").strip().lower()
        if te:
            if te != "chunked":
                raise HttpError(501, f"Transfer-Encoding {te!r} is not supported")
            return self._read_chunked(limit)
        length = self.headers.get("Content-Length")
        if length is None:
            raise HttpError(411, "Content-Length (or chunked transfer) required")
        try:
            n = int(length)
        except ValueError:
            raise HttpError(400, f"bad Content-Length {length!r}") from None
        if n < 0:
            raise HttpError(400, f"bad Content-Length {length!r}")
        if n > limit:
            self._drain(n)
            raise HttpError(413, f"document of {n} bytes exceeds server.max_body_bytes {limit}")
        data = self.rfile.read(n)
        if len(data) != n:
            raise HttpError(400, f"body ended after {len(data)} of {n} bytes")
        return data

    def _drain(self, n: int, cap: int = 2**30) -> None:
        left = min(n, cap)
        while left > 0:
            got = self.rfile.read(min(left, 2**16))
            if not got:
                return
            left -= len(got)

    def _read_chunked(self, limit: int) -> bytes:
        parts: list[bytes] = []
        total = 0
        while True:
            line = self.rfile.readline(65537)
            if not line:
                raise HttpError(400, "chunked body ended early")
            try:
                size = int(line.split(b";", 1)[0].strip(), 16)
            except ValueError:
                raise HttpError(400, "bad chunk size line") from None
            if size == 0:
                while self.rfile.readline(65537) not in (b"\r\n", b"\n", b""):
                    pass
                return b"".join(parts)
            total += size
            if total > limit:
                raise HttpError(413, f"document exceeds server.max_body_bytes {limit}")
            chunk = self.rfile.read(size)
            if len(chunk) != size:
                raise HttpError(400, "chunked body ended early")
            parts.append(chunk)
            self.rfile.readline(3)


    def do_GET(self) -> None:
        route = urlsplit(self.path).path.rstrip("/") or "/"
        if route == "/health":
            code, body = self.server.state.health()
            self._json(code, body)
        elif route == "/":
            self._json(200, {"service": "LegalMT translation server", "endpoints": {
                "GET /health": "status, model and config",
                "POST /translate?src_lang=auto|en|vi": "body: raw UTF-8 document; response: the translation"}})
        else:
            self._error(404, f"no route {route}")

    def do_POST(self) -> None:
        state = self.server.state
        parts = urlsplit(self.path)
        route = parts.path.rstrip("/")
        t0 = time.monotonic()
        try:
            if route != "/translate":
                raise HttpError(404, f"no route {route or '/'}")
            src = (parse_qs(parts.query).get("src_lang") or ["auto"])[0].strip().lower()
            if src not in SRC_LANGS:
                raise HttpError(400, f"src_lang must be one of {'|'.join(SRC_LANGS)}, got {src!r}")
            if state.status != "ok":
                raise HttpError(503, f"server status is {state.status}")
            raw = self._read_body(int(state.opts["max_body_bytes"]))
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise HttpError(400, f"the document is not valid UTF-8 ({exc.reason} at byte {exc.start})") from None
        except HttpError as e:
            state.count_rejected()
            log.warning("request rejected: HTTP %d %s", e.status, e.message)
            self._error(e.status, e.message, close=True)
            return
        try:
            res = state.translate(text, None if src == "auto" else src)
        except Unavailable as exc:
            state.count_rejected()
            log.warning("request rejected: HTTP 503 %s", exc)
            self._error(503, str(exc), close=True)
            return
        except Exception as exc:
            log.exception("translation failed")
            self._error(500, f"translation failed: {type(exc).__name__}: {exc}")
            return
        body = res.text.encode("utf-8")
        self._send(200, body, "text/plain; charset=utf-8", res.headers())
        st = res.stats
        wps = res.words / res.seconds if res.seconds > 0 else 0.0
        log.info("request %s: %s (%s) words=%d sentences~%d units=%s chunks=%s failed_chunks=%s fallback_units=%s "
                 "source_copied_units=%s render_missing=%s truncated_requests=%s seconds=%.3f queued=%.3f total=%.3f "
                 "words/s=%.1f bytes_in=%d bytes_out=%d",
                 res.request_id, res.direction, src, res.words, res.sentences, st.get("n_units", 0),
                 st.get("n_chunks", 0), st.get("failed_chunks", 0), st.get("fallback_units", 0),
                 st.get("source_copied_units", 0), st.get("render_missing", 0), st.get("truncated_requests", 0),
                 res.seconds, res.queued, time.monotonic() - t0, wps, len(raw), len(body))


class TranslationServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: ServeState) -> None:
        self.state = state
        super().__init__(address, Handler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError)):
            log.debug("client %s went away: %s", client_address, exc)
            return
        log.exception("unhandled error while serving %s", client_address)


def make_server(state: ServeState, host: str, port: int) -> TranslationServer:
    return TranslationServer((host, port), state)


def check_port_free(host: str, port: int) -> None:
    if port == 0:
        return
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LegalMT offline translation server (Docker submission).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=(
                                     "examples:\n"
                                     "  python docker/server.py                      # serve per configs/LegalMT/serve.yaml\n"
                                     "  python docker/server.py --infer-config echo_smoke --host 127.0.0.1 --port 0 "
                                     "--port-file /tmp/port\n"
                                     "  python docker/server.py --dump-config-field model_dir"))
    ap.add_argument("--config", default=None,
                    help=f"serving config (default ${SERVE_CONFIG_ENV}, else configs/LegalMT/serve.yaml)")
    ap.add_argument("--infer-config", default=None, help="replace the serving config's infer_config (name or path)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted override of the inference config, e.g. backend.gpu_memory_utilization=0.5")
    ap.add_argument("--host", default=None, help="bind address (default: server.host)")
    ap.add_argument("--port", type=int, default=None, help="port, 0 = any free port (default: server.port)")
    ap.add_argument("--port-file", default=None, help="write the bound port here once the server is ready")
    ap.add_argument("--pid-file", default=None, help="write this process id here at startup")
    ap.add_argument("--no-warmup", action="store_true", help="skip the warm-up translation")
    ap.add_argument("--log-level", default=None, help="logging level (default: server.log_level)")
    g = ap.add_argument_group("helper modes (no model is loaded)")
    g.add_argument("--dump-config", action="store_true", help="print the resolved configs as JSON and exit")
    g.add_argument("--dump-config-field", default=None, metavar="FIELD",
                   help="print one field of --dump-config as plain text (config_name, model_dir, repo_id, ...)")
    g.add_argument("--write-image-config", default=None, metavar="OUT", help="write the image serving config and exit")
    g.add_argument("--model-local-path", default=None, help="with --write-image-config: model dir inside the image")
    g.add_argument("--adapter-path", default=None, help="with --write-image-config: LoRA adapter dir inside the image")
    g.add_argument("--merged-model", action="store_true",
                   help="with --write-image-config: the weights are a fine-tuned model, blank repo_id/revision")
    g.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE",
                   help="with --write-image-config: build metadata stored under image:")
    return ap.parse_args(argv)


def _write_small(path: str | None, text: str) -> None:
    if path:
        Path(path).write_text(text + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    serve_path = serve_config_path(args.config)
    serve_cfg = load_serve_config(serve_path)

    if args.write_image_config:
        if not args.model_local_path:
            raise SystemExit("--write-image-config needs --model-local-path")
        meta = dict(item.split("=", 1) for item in args.meta if "=" in item)
        write_image_config(serve_cfg, serve_path, args.write_image_config, model_local_path=args.model_local_path,
                           infer_ref=args.infer_config, set_items=args.set, adapter_path=args.adapter_path,
                           keep_model_id=not args.merged_model, meta=meta)
        print(f"wrote {args.write_image_config}")
        return 0

    cfg, infer_path, _ = infer_config_for(serve_cfg, serve_path, args.infer_config, args.set)
    if args.dump_config or args.dump_config_field:
        info = describe(cfg, infer_path, serve_path)
        if args.dump_config_field:
            if args.dump_config_field not in info:
                raise SystemExit(f"unknown field {args.dump_config_field!r}; known: {sorted(info)}")
            val = info[args.dump_config_field]
            print("" if val is None else (json.dumps(val, default=str) if isinstance(val, (dict, list)) else val))
        else:
            print(json.dumps({**info, "server": serve_cfg["server"], "image": serve_cfg.get("image")}, indent=1,
                             ensure_ascii=False, default=str))
        return 0

    apply_offline_env()
    opts = dict(serve_cfg["server"])
    host = args.host or str(opts["host"])
    port = int(opts["port"] if args.port is None else args.port)
    logging.basicConfig(level=(args.log_level or str(opts["log_level"])).upper(), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _write_small(args.pid_file, str(os.getpid()))
    state = ServeState(cfg, server_opts=opts, serve_config=serve_path, infer_config=infer_path,
                       image=serve_cfg.get("image"))
    stop = threading.Event()

    def on_signal(signum: int, _frame: Any) -> None:
        if stop.is_set():
            os._exit(128 + signum)
        stop.set()
        log.info("signal %d received, shutting down", signum)
        if state.status == "loading":
            raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    try:
        check_port_free(host, port)
    except OSError as exc:
        log.error("cannot bind %s:%d: %s", host, port, exc)
        return 1
    log.info("loading %s (%s backend) from %s", cfg.get("config_name"), cfg["backend"]["name"], infer_path)
    try:
        state.load(warmup=False if args.no_warmup else None)
    except KeyboardInterrupt:
        log.warning("interrupted while loading the model")
        state.close()
        return 1
    except Exception:
        log.exception("model load failed")
        return 1
    if stop.is_set():
        state.close()
        return 0

    try:
        server = make_server(state, host, port)
    except OSError as exc:
        log.error("cannot bind %s:%d after loading: %s", host, port, exc)
        state.close()
        return 1
    bound = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.5}, name="http", daemon=True)
    thread.start()
    state.start_sampler()
    _write_small(args.port_file, str(bound))
    log.info("ready on http://%s:%d (load %.1f s, warm-up %s s)", host, bound, state.load_seconds or 0.0,
             "skipped" if state.warmup_seconds is None else f"{state.warmup_seconds:.1f}")

    while not stop.wait(0.5):
        pass
    state.status = "stopping"
    server.shutdown()
    if not state.wait_idle(float(opts["shutdown_grace_seconds"])):
        log.warning("a translation was still running after %s s; closing anyway", opts["shutdown_grace_seconds"])
    state.close()
    server.server_close()
    log.info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
