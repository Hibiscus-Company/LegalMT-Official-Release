"""Time and GPU-memory measurement for translation runs (two of the task's ranking axes)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def descendant_pids(root: int | None = None) -> set[int]:
    root = os.getpid() if root is None else root
    children: dict[int, list[int]] = {}
    proc = Path("/proc")
    try:
        entries = [p for p in proc.iterdir() if p.name.isdigit()]
    except OSError:
        return {root}
    for p in entries:
        try:
            stat = (p / "stat").read_text()
        except OSError:
            continue
        try:
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(p.name))
    out, stack = set(), [root]
    while stack:
        pid = stack.pop()
        if pid in out:
            continue
        out.add(pid)
        stack.extend(children.get(pid, []))
    return out


def _query(nvsmi: str, args: list[str], timeout: float = 5.0) -> list[list[str]]:
    try:
        res = subprocess.run([nvsmi, *args, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    if res.returncode != 0:
        return []
    return [[f.strip() for f in line.split(",")] for line in res.stdout.splitlines() if line.strip()]


def _to_float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


class GpuMemorySampler:
    def __init__(self, interval: float = 0.2, nvidia_smi: str | None = None, root_pid: int | None = None) -> None:
        self.interval = interval
        self.nvidia_smi = nvidia_smi if nvidia_smi is not None else shutil.which("nvidia-smi")
        self.root_pid = os.getpid() if root_pid is None else root_pid
        self.peak_mib: float | None = None
        self.peak_device_mib: float | None = None
        self.n_samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return bool(self.nvidia_smi)

    def sample(self) -> None:
        if not self.nvidia_smi:
            return
        rows = _query(self.nvidia_smi, ["--query-compute-apps=pid,used_memory"])
        pids = descendant_pids(self.root_pid)
        mine = 0.0
        for row in rows:
            if len(row) >= 2 and row[0].isdigit() and int(row[0]) in pids:
                mine += _to_float(row[1]) or 0.0
        dev_rows = _query(self.nvidia_smi, ["--query-gpu=memory.used"])
        dev = sum(_to_float(r[0]) or 0.0 for r in dev_rows if r) if dev_rows else None
        if rows or dev_rows:
            self.n_samples += 1
            self.peak_mib = max(self.peak_mib or 0.0, mine)
        if dev is not None:
            self.peak_device_mib = max(self.peak_device_mib or 0.0, dev)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    def start(self) -> "GpuMemorySampler":
        if self.available and self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="gpu-mem-sampler", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> "GpuMemorySampler":
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=10)
            self._thread = None
            self.sample()
        return self

    def __enter__(self) -> "GpuMemorySampler":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def torch_peak_mib(reset: bool = False) -> float | None:
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return None
        peak = torch.cuda.max_memory_allocated() / 2**20
        if reset:
            torch.cuda.reset_peak_memory_stats()
        return float(peak)
    except Exception:
        return None


def memory_report(sampler: GpuMemorySampler | None) -> dict[str, Any]:
    return {
        "peak_mib_nvidia_smi": None if sampler is None or sampler.peak_mib is None else round(sampler.peak_mib, 1),
        "peak_mib_torch_allocated": None if (p := torch_peak_mib()) is None else round(p, 1),
        "peak_mib_device_total": (None if sampler is None or sampler.peak_device_mib is None
                                  else round(sampler.peak_device_mib, 1)),
        "nvidia_smi_samples": 0 if sampler is None else sampler.n_samples,
        "sampler_interval_s": None if sampler is None else sampler.interval,
    }


class Stopwatch:
    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.seconds: dict[str, float] = {}

    @contextmanager
    def lap(self, name: str) -> Iterator[None]:
        t = time.monotonic()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + (time.monotonic() - t)

    def total(self) -> float:
        return time.monotonic() - self.t0

    def as_dict(self, ndigits: int = 3) -> dict[str, float]:
        out = {f"{k}_seconds": round(v, ndigits) for k, v in self.seconds.items()}
        out["total_seconds"] = round(self.total(), ndigits)
        return out
