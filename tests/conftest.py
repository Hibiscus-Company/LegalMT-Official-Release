"""pytest setup for LegalMT: put src/ on sys.path and register markers."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DATA_DIR = REPO_ROOT / "data" / "raw" / "Dataset"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "data: reads the organizer data under data/raw/ (skipped if absent)")
    config.addinivalue_line("markers", "gpu: needs a CUDA device (never run during CPU-only test passes)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_data = pytest.mark.skip(reason=f"organizer data not found under {DATA_DIR}")
    has_cuda = False
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        pass
    skip_gpu = pytest.mark.skip(reason="no CUDA device visible")
    for item in items:
        if "data" in item.keywords and not DATA_DIR.is_dir():
            item.add_marker(skip_data)
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
