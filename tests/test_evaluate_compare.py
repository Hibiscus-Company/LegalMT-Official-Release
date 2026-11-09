"""evaluate.py --compare pairs only the documents both runs translated."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_evaluate():
    spec = importlib.util.spec_from_file_location("evaluate_under_test", REPO_ROOT / "scripts" / "LegalMT" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _StubRun:
    def __init__(self, root: Path, name: str, records: list[dict]) -> None:
        self.dir = root / name
        self.dir.mkdir(parents=True)
        self.name = name
        self.manifest = {"status": "done"}
        with (self.dir / "hyps.jsonl").open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def path(self, name: str) -> Path:
        return self.dir / name


def _rec(sid: str, hyp: str) -> dict:
    return {"sample_id": sid, "category": "c", "direction": "en2vi", "src_lang": "en", "tgt_lang": "vi", "hyp": hyp}


def test_compare_uses_only_documents_present_in_both_runs(tmp_path: Path) -> None:
    ev = _load_evaluate()
    refs = {
        "d1": "Điều 1. Phạm vi điều chỉnh của nghị định này về xử phạt vi phạm hành chính.",
        "d2": "Điều 2. Đối tượng áp dụng là cơ quan, tổ chức và cá nhân có liên quan.",
        "d3": "Điều 3. Nghị định này có hiệu lực kể từ ngày ký ban hành.",
    }
    ref_df = pd.DataFrame({"sample_id": list(refs), "english_markdown": ["x"] * 3, "vietnamese_markdown": list(refs.values())})
    a = _StubRun(tmp_path, "run_a", [_rec("d1", refs["d1"]), _rec("d2", refs["d2"])])
    b = _StubRun(tmp_path, "run_b", [_rec("d1", refs["d1"]), _rec("d2", "Điều 2. Đối tượng áp dụng.")])
    cmp = ev.compare_runs(a, b, ref_df, n_bootstrap=50, seed=0)
    r = cmp["directions"]["en2vi"]
    assert r["n_common"] == 2
    assert r["missing_a"] == ["d3"] and r["missing_b"] == ["d3"]
    assert r["bleu"]["a"] > 99.0
    assert r["bleu"]["b"] < r["bleu"]["a"]


def test_compare_skips_a_direction_with_no_common_document(tmp_path: Path) -> None:
    ev = _load_evaluate()
    ref_df = pd.DataFrame({"sample_id": ["d1", "d2"], "english_markdown": ["x", "y"], "vietnamese_markdown": ["a b c", "d e f"]})
    a = _StubRun(tmp_path, "run_a", [_rec("d1", "a b c")])
    b = _StubRun(tmp_path, "run_b", [_rec("d2", "d e f")])
    cmp = ev.compare_runs(a, b, ref_df, n_bootstrap=10, seed=0)
    assert "en2vi" not in cmp["directions"]
    assert "en2vi" in cmp["skipped_directions"]
