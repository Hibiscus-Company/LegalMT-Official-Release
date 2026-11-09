"""Tests for LegalMT.profiling: no-op path without nvidia-smi, process-tree attribution, stopwatch."""
from __future__ import annotations

import os
import stat
import subprocess
import sys
import time

from LegalMT.profiling import GpuMemorySampler, Stopwatch, descendant_pids, memory_report, torch_peak_mib


def test_noop_without_nvidia_smi():
    s = GpuMemorySampler(nvidia_smi="")
    assert not s.available
    with s:
        time.sleep(0.05)
    assert s.peak_mib is None and s.peak_device_mib is None and s.n_samples == 0
    rep = memory_report(s)
    assert rep["peak_mib_nvidia_smi"] is None and rep["peak_mib_device_total"] is None
    assert set(rep) >= {"peak_mib_nvidia_smi", "peak_mib_torch_allocated"}
    assert memory_report(None)["peak_mib_nvidia_smi"] is None


def test_descendant_pids_includes_children():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        pids = descendant_pids()
        assert os.getpid() in pids and child.pid in pids
        assert descendant_pids(child.pid) == {child.pid}
    finally:
        child.kill()
        child.wait()


def test_fake_nvidia_smi_counts_only_this_process_tree(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        fake = tmp_path / "nvidia-smi"
        fake.write_text(
            "#!/bin/sh\n"
            'case "$1" in\n'
            f'  --query-compute-apps=*) printf "{os.getpid()}, 300\\n{child.pid}, 200\\n999999999, 7000\\n" ;;\n'
            '  --query-gpu=*) printf "9100\\n" ;;\n'
            "esac\n")
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        s = GpuMemorySampler(interval=0.05, nvidia_smi=str(fake))
        with s:
            time.sleep(0.2)
        assert s.peak_mib == 500.0
        assert s.peak_device_mib == 9100.0
        assert s.n_samples >= 2
        assert memory_report(s)["peak_mib_nvidia_smi"] == 500.0
    finally:
        child.kill()
        child.wait()


def test_torch_peak_is_none_without_cuda():
    assert torch_peak_mib() is None


def test_stopwatch_laps_add_up():
    sw = Stopwatch()
    with sw.lap("load"):
        time.sleep(0.02)
    for _ in range(2):
        with sw.lap("translate"):
            time.sleep(0.01)
    d = sw.as_dict()
    assert d["load_seconds"] >= 0.02 and d["translate_seconds"] >= 0.02
    assert d["total_seconds"] >= d["load_seconds"] + d["translate_seconds"]
