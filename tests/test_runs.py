"""Tests for LegalMT.runs and the evaluation scripts on fake run directories."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from LegalMT import paths
from LegalMT import runs as R

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "LegalMT" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def artifacts_root(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    monkeypatch.setattr(paths, "ARTIFACTS_ROOT", root)
    return root


def test_new_run_manifest(artifacts_root):
    run = R.new_run("hy mt/1.8B smoke!", {"model": "x", "batch": 2}, "configs/LegalMT/infer/x.yaml", "dev",
                    argv=["prog", "--flag"])
    assert run.dir.parent == artifacts_root / "runs"
    assert re.fullmatch(r"\d{8}-\d{6}_hy-mt-1\.8B-smoke", run.dir.name)
    m = json.loads((run.dir / "manifest.json").read_text())
    for key in ("run_name", "tag", "created", "argv", "host", "config", "config_path", "split",
                "code_fingerprint", "env", "status"):
        assert key in m
    assert m["status"] == "running" and m["split"] == "dev" and m["argv"] == ["prog", "--flag"]
    assert re.fullmatch(r"[0-9a-f]{64}", m["code_fingerprint"])
    assert m["env"]["python"] and "sacrebleu" in m["env"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}", m["created"])
    other = R.new_run("hy mt/1.8B smoke!", None, None, "dev")
    assert other.dir != run.dir


def test_jsonl_manifest_and_finish(artifacts_root):
    run = R.new_run("t", {}, None, "dev_mini")
    run.write_jsonl("hyps.jsonl", [{"sample_id": "A/1", "hyp": "xin chào"}])
    run.append_jsonl("hyps.jsonl", {"sample_id": "A/2", "hyp": "b"})
    run.append_jsonl("hyps.jsonl", [{"sample_id": "A/3", "hyp": "c"}])
    assert [r["sample_id"] for r in run.read_jsonl("hyps.jsonl")] == ["A/1", "A/2", "A/3"]
    assert run.read_jsonl("hyps.jsonl")[0]["hyp"] == "xin chào"
    run.update_manifest(directions=["en2vi"])
    other = R.load_run(run.dir)
    other.update_manifest(timing={"total_seconds": 3.0})
    m = run.finish("done", memory={"peak_mib_nvidia_smi": 1234})
    assert m["status"] == "done" and m["directions"] == ["en2vi"] and m["timing"]["total_seconds"] == 3.0
    assert "finished" in m and m["wall_seconds"] >= 0 and m["memory"]["peak_mib_nvidia_smi"] == 1234
    with pytest.raises(ValueError):
        run.finish("weird")
    assert R.load_run(run.name).manifest["status"] == "done"
    with pytest.raises(FileNotFoundError):
        R.load_run("no_such_run")
    with pytest.raises(ValueError):
        R.sanitize_tag("///")


def test_code_fingerprint(tmp_path):
    (tmp_path / "src" / "pkg" / "__pycache__").mkdir(parents=True)
    (tmp_path / "configs" / "P").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "a.py").write_text("x = 1\n")
    (tmp_path / "configs" / "P" / "c.yaml").write_text("k: 1\n")
    (tmp_path / "src" / "pkg" / "notes.txt").write_text("ignored: not .py under src\n")
    f0 = R.code_fingerprint(tmp_path)
    (tmp_path / "src" / "pkg" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0\1")
    (tmp_path / "src" / "pkg" / "notes.txt").write_text("changed\n")
    assert R.code_fingerprint(tmp_path) == f0
    (tmp_path / "configs" / "P" / "c.yaml").write_text("k: 2\n")
    f1 = R.code_fingerprint(tmp_path)
    assert f1 != f0
    (tmp_path / "src" / "pkg" / "a.py").rename(tmp_path / "src" / "pkg" / "b.py")
    assert R.code_fingerprint(tmp_path) != f1
    assert re.fullmatch(r"[0-9a-f]{64}", R.code_fingerprint())


def test_append_results_index(artifacts_root):
    rec = {"run_name": "r", "created": "t", "split": "dev", "tag": "x", "metrics": {"en2vi": {"bleu": 1.0}}}
    p = R.append_results_index(rec)
    R.append_results_index({**rec, "run_name": "r2"})
    assert p == artifacts_root / "results_index.jsonl"
    assert [r["run_name"] for r in R.read_jsonl(p)] == ["r", "r2"]
    with pytest.raises(ValueError):
        R.append_results_index({"run_name": "r"})


EN_DOCS = [
    "Article 1. The court shall decide the case within ten days. The court may extend the time limit once.",
    "Decree on administrative penalty. The decree applies to every organization and the decree takes effect now.",
    "The people's court hears the dispute. The land use right is transferred by contract.",
]
VI_DOCS = [
    "Điều 1. Tòa án phải quyết định vụ việc trong mười ngày. Tòa án có thể gia hạn một lần.",
    "Nghị định về xử phạt vi phạm hành chính. Nghị định áp dụng với mọi tổ chức và nghị định có hiệu lực ngay.",
    "Tòa án nhân dân xét xử tranh chấp. Quyền sử dụng đất được chuyển nhượng bằng hợp đồng.",
]
GLOSSARY = (
    "en\tvi\tvi_variants\ten_variants\tnote\n"
    "court\ttòa án\t\t\tinstitution\n"
    "decree\tnghị định\t\t\tinstrument\n"
    "land use right\tquyền sử dụng đất\t\t\tland\n"
)


def _ref_df() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": ["Cat_a/1", "Cat_a/2", "Cat_b/7"],
        "category": ["Cat_a", "Cat_a", "Cat_b"],
        "english_path": ["", "", ""], "vietnamese_path": ["", "", ""],
        "english_markdown": EN_DOCS, "vietnamese_markdown": VI_DOCS,
    })


def _fake_run(tag: str, hyp_fn, *, drop: str | None = None) -> R.Run:
    run = R.new_run(tag, {"backend": "echo"}, None, "dev")
    recs = []
    for i, (sid, en, vi) in enumerate(zip(["Cat_a/1", "Cat_a/2", "Cat_b/7"], EN_DOCS, VI_DOCS)):
        for d, src, ref in (("en2vi", en, vi), ("vi2en", vi, en)):
            if sid == drop:
                continue
            recs.append({"sample_id": sid, "category": sid.split("/")[0], "direction": d, "src_lang": d[:2],
                         "tgt_lang": d[3:], "hyp": hyp_fn(src, ref), "seconds": 1.5 + i, "n_units": 2,
                         "n_chunks": 1, "in_tokens": 10, "out_tokens": 12, "fallback_units": 0})
    run.write_jsonl("hyps.jsonl", recs)
    run.finish("done", directions=["en2vi", "vi2en"],
               timing={"load_seconds": 1.0, "translate_seconds": 9.0, "total_seconds": 10.0},
               memory={"peak_mib_nvidia_smi": 4321, "peak_mib_torch_allocated": 3000},
               throughput={"words_per_second": 50.0, "out_tokens_per_second": 80.0})
    return run


def test_evaluate_and_results_table(artifacts_root, tmp_path, monkeypatch, capsys):
    ev = _load_script("evaluate")
    monkeypatch.setattr(ev, "references", lambda split, version: _ref_df())
    gl = tmp_path / "gl.tsv"
    gl.write_text(GLOSSARY, encoding="utf-8")

    oracle = _fake_run("oracle", lambda src, ref: ref)
    partial = _fake_run("copy", lambda src, ref: src, drop="Cat_b/7")
    assert ev.main([str(oracle.dir), "--glossary", str(gl), "--n-bootstrap", "50", "--split", "dev_x"]) == 0
    assert ev.main([partial.name, "--glossary", str(gl), "--n-bootstrap", "50", "--split", "dev_x"]) == 0

    res = json.loads((oracle.dir / "results.json").read_text())
    en2vi = res["directions"]["en2vi"]
    assert en2vi["bleu"] == pytest.approx(100.0) and en2vi["chrf"] == pytest.approx(100.0)
    assert en2vi["n_docs"] == 3 and en2vi["n_missing"] == 0
    assert en2vi["terms"]["term_recall"] == 1.0
    assert en2vi["seconds_per_doc_mean"] == pytest.approx(2.5)
    assert set(en2vi["per_category"]) == {"Cat_a", "Cat_b"}
    assert res["cost"]["peak_mib_nvidia_smi"] == 4321 and res["split"] == "dev_x"
    assert "PROVISIONAL" in res["metric_note"]

    res2 = json.loads((partial.dir / "results.json").read_text())
    for d in ("en2vi", "vi2en"):
        assert res2["directions"][d]["missing_hyps"] == ["Cat_b/7"]
        assert res2["directions"][d]["per_doc_bleu"]["Cat_b/7"] == 0.0
        assert res2["directions"][d]["bleu"] < 50

    index = R.read_jsonl(artifacts_root / "results_index.jsonl")
    assert [r["run_name"] for r in index] == [oracle.name, partial.name]
    assert index[0]["metrics"]["en2vi"]["bleu"] == pytest.approx(100.0)
    assert index[0]["metrics"]["run"]["peak_mib_nvidia_smi"] == 4321
    out = capsys.readouterr().out
    assert "| en2vi | 3 | 0 |" in out

    table = _load_script("make_results_table")
    assert table.main([]) == 0
    out = capsys.readouterr().out
    assert f"| {oracle.name} | dev_x | en2vi | 100.00 | 100.00 | 100.00 | 1.000 |" in out
    assert "4321" in out and out.count(partial.name) == 2

    assert ev.main([str(oracle.dir), "--glossary", str(gl), "--n-bootstrap", "20", "--split", "dev_x"]) == 0
    capsys.readouterr()
    table.main([])
    assert capsys.readouterr().out.count(oracle.name) == 2
    table.main(["--all"])
    assert capsys.readouterr().out.count(oracle.name) == 4

    out_json = tmp_path / "cmp.json"
    assert ev.main(["--compare", partial.name, oracle.name, "--n-bootstrap", "100", "--out", str(out_json)]) == 0
    cmp = json.loads(out_json.read_text())
    for d in ("en2vi", "vi2en"):
        assert cmp["directions"][d]["bleu"]["delta"] > 50
        assert cmp["directions"][d]["missing_a"] == ["Cat_b/7"] and cmp["directions"][d]["unit"] == "document"
    assert "B - A" in capsys.readouterr().out


def test_evaluate_rejects_bad_direction(artifacts_root, monkeypatch):
    ev = _load_script("evaluate")
    run = R.new_run("bad", {}, None, "dev")
    run.write_jsonl("hyps.jsonl", [{"sample_id": "Cat_a/1", "direction": "en2fr", "hyp": "x"}])
    with pytest.raises(ValueError):
        ev.load_hyps(run)


def test_json_writes_accept_numpy_and_paths(artifacts_root):
    import numpy as np

    run = R.new_run("np", {"path": Path("/x/y"), "lr": np.float32(0.5)}, Path("cfg.yaml"), "dev")
    assert run.manifest["config"]["path"] == Path("/x/y")
    m = json.loads(run.manifest_path.read_text())
    assert m["config"] == {"path": "/x/y", "lr": 0.5} and m["config_path"] == "cfg.yaml"
    run.append_jsonl("hyps.jsonl", {"seconds": np.float64(1.25), "ids": np.arange(2)})
    assert run.read_jsonl("hyps.jsonl") == [{"seconds": 1.25, "ids": [0, 1]}]


def test_read_jsonl_tolerant_drops_only_a_truncated_tail(tmp_path):
    p = tmp_path / "h.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "hyp": "half wri', encoding="utf-8")
    recs, truncated = R.read_jsonl_tolerant(p)
    assert recs == [{"a": 1}, {"a": 2}] and truncated
    with pytest.raises(json.JSONDecodeError):
        R.read_jsonl(p)
    p.write_text('{"a": 1}\n{"a": \n{"a": 3}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        R.read_jsonl_tolerant(p)
    p.write_text('{"a": 1}\n\n', encoding="utf-8")
    assert R.read_jsonl_tolerant(p) == ([{"a": 1}], False)


def test_evaluate_flags_partial_wrong_language_and_direction_bugs(artifacts_root, tmp_path, monkeypatch, capsys):
    ev = _load_script("evaluate")
    monkeypatch.setattr(ev, "references", lambda split, version: _ref_df())
    gl = tmp_path / "gl.tsv"
    gl.write_text(GLOSSARY, encoding="utf-8")
    ids = ["Cat_a/1", "Cat_a/2", "Cat_b/7"]

    crashed = R.new_run("crashed", {}, None, "dev")
    crashed.write_jsonl("hyps.jsonl", [
        {"sample_id": ids[0], "direction": "en2vi", "src_lang": "en", "tgt_lang": "vi", "hyp": VI_DOCS[0], "seconds": 1.0},
        {"sample_id": ids[1], "direction": "en2vi", "src_lang": "en", "tgt_lang": "vi", "hyp": "  \n", "seconds": 1.0},
    ])
    with crashed.path("hyps.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"sample_id": "Cat_b/7", "direction": "en2vi", "hyp": "Tòa án nh')
    assert ev.main([crashed.name, "--glossary", str(gl), "--n-bootstrap", "10", "--split", "dev_x"]) == 0
    res = json.loads((crashed.dir / "results.json").read_text())
    r = res["directions"]["en2vi"]
    assert res["run_status"] == "running" and res["hyps_truncated_last_line"] is True
    assert r["missing_hyps"] == ["Cat_b/7"] and r["empty_hyps"] == ["Cat_a/2"] and r["n_wrong_language"] == 0
    out = capsys.readouterr().out
    assert "WARNING: run status is 'running'" in out and "half-written line" in out

    wrongdir = R.new_run("wrongdir", {}, None, "dev")
    wrongdir.write_jsonl("hyps.jsonl", [
        {"sample_id": s, "direction": "en2vi", "src_lang": "vi", "tgt_lang": "en", "hyp": en, "seconds": 1.0}
        for s, en in zip(ids, EN_DOCS)
    ])
    wrongdir.finish("done", directions=["en2vi"])
    assert ev.main([wrongdir.name, "--glossary", str(gl), "--n-bootstrap", "10", "--split", "dev_x"]) == 0
    res = json.loads((wrongdir.dir / "results.json").read_text())
    assert res["direction_field_mismatch"] == {"en2vi": ids}
    assert res["directions"]["en2vi"]["wrong_language_hyps"] == ids
    out = capsys.readouterr().out
    assert "not in the target language" in out and "contradict the direction label" in out

    clean = _fake_run("clean", lambda src, ref: ref)
    assert ev.main([clean.name, "--glossary", str(gl), "--n-bootstrap", "10", "--split", "dev_x"]) == 0
    assert "WARNING" not in capsys.readouterr().out

    table = _load_script("make_results_table")
    assert table.main([]) == 0
    rows = {ln.split(" | ")[0].lstrip("| "): ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("| 2")}
    assert "status running" in rows[crashed.name] and "missing 1/3" in rows[crashed.name] and "empty 1/3" in rows[crashed.name]
    assert "wrong-lang 3/3" in rows[wrongdir.name] and "dir-field 3/3" in rows[wrongdir.name]
    assert rows[clean.name].rstrip().endswith("|  |")


def test_evaluate_reads_split_version_from_manifest(artifacts_root, tmp_path, monkeypatch, capsys):
    ev = _load_script("evaluate")
    seen = []
    monkeypatch.setattr(ev, "references", lambda split, version: seen.append((split, version)) or _ref_df())
    monkeypatch.setattr(ev, "split_fingerprint", lambda split, version: {"split_spec_sha256": "f" * 64})
    gl = tmp_path / "gl.tsv"
    gl.write_text(GLOSSARY, encoding="utf-8")
    run = _fake_run("versioned", lambda src, ref: ref)
    run.update_manifest(split_version="v7", split="dev_x")
    assert ev.main([run.name, "--glossary", str(gl), "--n-bootstrap", "5", "--no-index"]) == 0
    assert seen == [("dev_x", "v7")]
    res = json.loads((run.dir / "results.json").read_text())
    assert res["split_version"] == "v7" and res["references"] == {"split_spec_sha256": "f" * 64}
    assert ev.main([run.name, "--glossary", str(gl), "--n-bootstrap", "5", "--no-index", "--split-version", "v0"]) == 0
    assert seen[-1] == ("dev_x", "v0")
    capsys.readouterr()


def test_compare_guards(artifacts_root, monkeypatch, capsys):
    ev = _load_script("evaluate")
    monkeypatch.setattr(ev, "references", lambda split, version: _ref_df())
    a = _fake_run("a", lambda src, ref: src)
    b = R.new_run("b_en2vi_only", {}, None, "dev")
    b.write_jsonl("hyps.jsonl", [{"sample_id": s, "direction": "en2vi", "hyp": vi} for s, vi in
                                 zip(["Cat_a/1", "Cat_a/2", "Cat_b/7"], VI_DOCS)])
    with pytest.raises(SystemExit):
        ev.main(["--compare", a.name, b.name, "--n-bootstrap", "0"])
    capsys.readouterr()
    assert ev.main(["--compare", a.name, b.name, "--n-bootstrap", "20"]) == 0
    out = capsys.readouterr().out
    assert "not compared: ['vi2en']" in out and "run B has status 'running'" in out


def test_append_jsonl_repairs_a_torn_tail_before_resuming(artifacts_root):
    run = R.new_run("resume", {}, None, "dev")
    run.append_jsonl("hyps.jsonl", {"sample_id": "A/1", "hyp": "một"})
    with run.path("hyps.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"sample_id": "A/2", "hyp": "hai b')
    with pytest.warns(RuntimeWarning, match="half-written"):
        run.append_jsonl("hyps.jsonl", {"sample_id": "A/2", "hyp": "hai"})
    run.append_jsonl("hyps.jsonl", [{"sample_id": "A/3", "hyp": "ba"}])
    recs = run.read_jsonl("hyps.jsonl")
    assert [r["sample_id"] for r in recs] == ["A/1", "A/2", "A/3"] and recs[1]["hyp"] == "hai"
    assert R.read_jsonl_tolerant(run.path("hyps.jsonl")) == (recs, False)
    p = run.path("frag.jsonl")
    p.write_text('{"a": ', encoding="utf-8")
    with pytest.warns(RuntimeWarning):
        R.append_records(p, [{"a": 1}])
    assert R.read_jsonl(p) == [{"a": 1}]
    p.write_bytes(b'{"a": 1}\n' + b"x" * 5000)
    with pytest.warns(RuntimeWarning):
        R.append_records(p, [{"a": 2}])
    assert R.read_jsonl(p) == [{"a": 1}, {"a": 2}]


def test_torn_tail_scan_crosses_block_boundaries(tmp_path):
    p = tmp_path / "t.bin"
    p.write_bytes(b"a" * 10 + b"\n" + b"b" * 37)
    with p.open("rb") as fh:
        for block in (1, 3, 7, 1 << 20):
            assert R._torn_tail_start(fh, 48, block=block) == 11
    p.write_bytes(b"no newline at all")
    with p.open("rb") as fh:
        assert R._torn_tail_start(fh, 17, block=4) == 0


def test_unicode_line_separators_inside_hypotheses_round_trip(artifacts_root):
    run = R.new_run("seps", {}, None, "dev")
    hyp = "dòng một dòng hai đoạn\u0085cuối\x0bvt\x0cff"
    run.write_jsonl("hyps.jsonl", [{"sample_id": "A/1", "hyp": hyp}])
    run.append_jsonl("hyps.jsonl", {"sample_id": "A/2", "hyp": hyp})
    recs, truncated = R.read_jsonl_tolerant(run.path("hyps.jsonl"))
    assert not truncated and [r["hyp"] for r in recs] == [hyp, hyp]
    assert run.read_jsonl("hyps.jsonl") == recs


def test_concurrent_appends_never_interleave(tmp_path):
    import subprocess
    import sys

    p = tmp_path / "idx.jsonl"
    code = ("import sys; sys.path.insert(0, %r); from LegalMT import runs as R\n"
            "for i in range(25): R.append_records(%r, [{'worker': int(sys.argv[1]), 'i': i, 'pad': 'x' * 200000}])"
            ) % (str(REPO_ROOT / "src"), str(p))
    procs = [subprocess.Popen([sys.executable, "-c", code, str(k)]) for k in range(4)]
    assert all(pr.wait(timeout=120) == 0 for pr in procs)
    recs = R.read_jsonl(p)
    assert len(recs) == 100 and sorted((r["worker"], r["i"]) for r in recs) == [(k, i) for k in range(4) for i in range(25)]


def test_results_table_skips_a_torn_index_tail(artifacts_root, capsys):
    rec = {"run_name": "r", "created": "t", "split": "dev", "tag": "x",
           "metrics": {"en2vi": {"bleu": 12.5, "n_docs": 3}}, "run_status": "done"}
    p = R.append_results_index(rec)
    with p.open("a", encoding="utf-8") as fh:
        fh.write('{"run_name": "r2", "metr')
    table = _load_script("make_results_table")
    assert table.main([]) == 0
    out = capsys.readouterr().out
    assert "| r | dev | en2vi | 12.50 |" in out and "half-written" in out
    with pytest.warns(RuntimeWarning):
        R.append_results_index({**rec, "run_name": "r3"})
    assert [r["run_name"] for r in R.read_jsonl(p)] == ["r", "r3"]


def test_evaluate_scores_undeclared_directions_flags_duplicates_and_records_provenance(
        artifacts_root, tmp_path, monkeypatch, capsys):
    ev = _load_script("evaluate")
    monkeypatch.setattr(ev, "references", lambda split, version: _ref_df())
    gl = tmp_path / "gl.tsv"
    gl.write_text(GLOSSARY, encoding="utf-8")
    ids = ["Cat_a/1", "Cat_a/2", "Cat_b/7"]
    run = R.new_run("stale-manifest", {}, None, "dev")
    recs = [{"sample_id": s, "direction": "en2vi", "src_lang": "en", "tgt_lang": "vi", "hyp": vi, "seconds": 1.0}
            for s, vi in zip(ids, VI_DOCS)]
    recs += [{"sample_id": s, "direction": "vi2en", "src_lang": "vi", "tgt_lang": "en", "hyp": en, "seconds": 1.0}
             for s, en in zip(ids, EN_DOCS)]
    recs.append({"sample_id": ids[0], "direction": "en2vi", "src_lang": "en", "tgt_lang": "vi", "hyp": VI_DOCS[0],
                 "seconds": 1.0})
    recs.insert(0, {"sample_id": ids[0], "direction": "en2vi", "src_lang": "en", "tgt_lang": "vi", "hyp": "x",
                    "seconds": 1.0})
    run.write_jsonl("hyps.jsonl", recs)
    run.finish("done", directions=["en2vi"])
    assert ev.main([run.name, "--glossary", str(gl), "--n-bootstrap", "10", "--split", "dev_x"]) == 0
    res = json.loads((run.dir / "results.json").read_text())
    assert set(res["directions"]) == {"en2vi", "vi2en"}
    assert res["directions"]["en2vi"]["bleu"] == pytest.approx(100.0)
    assert res["duplicate_hyp_records"] == {"en2vi": 2}
    assert "duplicate hypothesis records" in capsys.readouterr().out
    import hashlib

    assert res["glossary"]["sha256"] == hashlib.sha256(gl.read_bytes()).hexdigest()
    assert re.fullmatch(r"[0-9a-f]{64}", res["eval_code_fingerprint"])
    idx = R.read_jsonl(artifacts_root / "results_index.jsonl")[-1]
    assert idx["glossary_sha256"] == res["glossary"]["sha256"]


def test_evaluate_refuses_to_index_an_empty_run(artifacts_root, tmp_path, monkeypatch, capsys):
    ev = _load_script("evaluate")
    monkeypatch.setattr(ev, "references", lambda split, version: _ref_df())
    gl = tmp_path / "gl.tsv"
    gl.write_text(GLOSSARY, encoding="utf-8")
    run = R.new_run("empty", {}, None, "dev")
    run.write_jsonl("hyps.jsonl", [])
    run.finish("done")
    assert ev.main([run.name, "--glossary", str(gl), "--n-bootstrap", "10", "--split", "dev_x"]) == 1
    assert "no hypotheses for any direction" in capsys.readouterr().out
    assert not (artifacts_root / "results_index.jsonl").exists()


def test_results_table_flags_rows_scored_under_different_glossaries_or_split_versions(artifacts_root, capsys):
    base = {"created": "t", "split": "dev", "tag": "x", "run_status": "done", "split_version": "v0",
            "metrics": {"en2vi": {"bleu": 10.0, "n_docs": 40}}}
    R.append_results_index({**base, "run_name": "a", "glossary_sha256": "a" * 64})
    R.append_results_index({**base, "run_name": "b", "glossary_sha256": "a" * 64})
    table = _load_script("make_results_table")
    table.main([])
    rows = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("| a ") or ln.startswith("| b ")]
    assert len(rows) == 2 and all(r.rstrip().endswith("|  |") for r in rows)
    R.append_results_index({**base, "run_name": "c", "glossary_sha256": "b" * 64, "split_version": "v1"})
    table.main([])
    rows = {ln.split(" | ")[0].lstrip("| "): ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("| ")}
    assert "glossary aaaaaaaa" in rows["a"] and "glossary bbbbbbbb" in rows["c"]
    assert "split version v0" in rows["a"] and "split version v1" in rows["c"]


@pytest.mark.data
def test_evaluate_end_to_end_on_real_dev_mini_references(artifacts_root, capsys):
    from LegalMT import splits as S

    ev = _load_script("evaluate")
    df = S.docs_for("dev_mini")
    def fake(tag, fn):
        run = R.new_run(tag, {}, None, "dev_mini")
        run.write_jsonl("hyps.jsonl", [
            {"sample_id": s, "direction": d, "src_lang": d[:2], "tgt_lang": d[3:], "hyp": fn(en, vi, d), "seconds": 1.0}
            for s, en, vi in zip(df["sample_id"], df["english_markdown"], df["vietnamese_markdown"])
            for d in ("en2vi", "vi2en")])
        run.finish("done", directions=["en2vi", "vi2en"])
        return run

    oracle = fake("oracle", lambda en, vi, d: vi if d == "en2vi" else en)
    copied = fake("copied", lambda en, vi, d: en if d == "en2vi" else vi)
    assert ev.main([oracle.name, "--n-bootstrap", "20"]) == 0
    assert ev.main([copied.name, "--n-bootstrap", "20"]) == 0
    capsys.readouterr()
    res = json.loads((oracle.dir / "results.json").read_text())
    assert res["references"]["split_spec_sha256"] and res["glossary"]["n_entries"] >= 150
    for d in ("en2vi", "vi2en"):
        r = res["directions"][d]
        assert r["n_docs"] == 12 and r["bleu"] == pytest.approx(100.0) and r["terms"]["term_recall"] == 1.0
        assert sum(v["n_docs"] for v in r["by_near_relative"].values()) == 12
        assert sum(v["n_docs"] for v in r["per_category"].values()) == 12 and len(r["per_category"]) == 7
    bad = json.loads((copied.dir / "results.json").read_text())
    for d in ("en2vi", "vi2en"):
        assert bad["directions"][d]["n_wrong_language"] == 12 and bad["directions"][d]["bleu"] < 20
