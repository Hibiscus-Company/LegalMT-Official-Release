"""Tests of LegalMT.segments and evaluate_segments.py on synthetic documents."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import textwrap
import tomllib
import unicodedata
from pathlib import Path

import pytest

from LegalMT import segments as SEG
from LegalMT.layout import Element, parse_document, translation_units

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "LegalMT" / "evaluate_segments.py"
COMET_SCRIPT = REPO_ROOT / "scripts" / "LegalMT" / "comet_score.py"


EN_BODY = [
    ("h", "Article 1. Scope of application"),
    ("p", "This charter governs how the harbour council of Sampletown registers small fishing boats, keeps "
          "the register open to the public and settles disputes about mooring places, as set out in resolution 4417."),
    ("p", "A boat owner shall apply for registration within 36 days after buying the boat and shall pay "
          "the fee of 2250 credits to the council treasury."),
    ("h", "Article 2. Documents to be submitted"),
    ("p", "The application file shall contain the following items, numbered as in form 7391:"),
    ("p", "a) the signed application form with the name and address of the owner, code 5820;"),
    ("p", "b) a copy of the purchase contract dated in the year 2034."),
]
VI_BODY = [
    ("h", "Điều 1. Phạm vi áp dụng"),
    ("p", "Điều lệ này quy định việc hội đồng cảng Thị trấn Mẫu đăng ký thuyền đánh cá nhỏ, công khai sổ "
          "đăng ký cho người dân và giải quyết tranh chấp về chỗ neo đậu, theo nghị quyết 4417."),
    ("p", "Chủ thuyền phải nộp đơn đăng ký trong thời hạn 36 ngày kể từ ngày mua thuyền và nộp lệ phí "
          "2250 đồng tín dụng vào ngân quỹ của hội đồng."),
    ("h", "Điều 2. Hồ sơ phải nộp"),
    ("p", "Hồ sơ đề nghị gồm các giấy tờ sau đây, đánh số theo mẫu 7391:"),
    ("p", "a) đơn đề nghị có chữ ký, ghi họ tên và địa chỉ của chủ thuyền, mã số 5820;"),
    ("p", "b) bản sao hợp đồng mua bán lập trong năm 2034."),
]
EN_HEAD = ["HARBOUR COUNCIL OF SAMPLETOWN                   REPUBLIC OF EXAMPLELAND",
           "        -------                                Order - Progress - Harmony",
           "      No. 88/2034/HC                                   Sampletown, May 9, 2034",
           "", " " * 36 + "CHARTER", ""]
VI_HEAD = ["HỘI ĐỒNG CẢNG THỊ TRẤN MẪU                      CỘNG HÒA VÍ DỤ",
           "        -------                                Trật tự - Tiến bộ - Hài hòa",
           "      Số: 88/2034/HĐC                                   Thị trấn Mẫu, ngày 9 tháng 5 năm 2034",
           "", " " * 36 + "ĐIỀU LỆ", ""]
EN_SIG = ["", " " * 66 + "FOR THE COUNCIL", " " * 72 + "CHAIRPERSON", "", "", " " * 72 + "Robin Example"]
VI_SIG = ["", " " * 66 + "TM. HỘI ĐỒNG", " " * 72 + "CHỦ TỊCH", "", "", " " * 72 + "Nguyễn Văn Mẫu"]


def _wrap(text: str) -> list[str]:
    return textwrap.wrap(text, width=64, initial_indent="  ", subsequent_indent="  ",
                         break_long_words=False, break_on_hyphens=False)


def make_doc(lang: str, *, signature: bool = True, body: list[tuple[str, str]] | None = None) -> str:
    head, sig = (EN_HEAD, EN_SIG) if lang == "en" else (VI_HEAD, VI_SIG)
    lines = list(head)
    for kind, t in body or (EN_BODY if lang == "en" else VI_BODY):
        lines += (["  " + t] if kind == "h" else _wrap(t)) + [""]
    if signature:
        lines += sig
    return "\n".join(lines) + "\n"


def marker(uid: str, direction: str) -> str:
    return f"H<{direction}:{uid}>"


def fake_run(tmp_path: Path, docs: dict[str, tuple[str, str]], *, directions=("en2vi", "vi2en"),
             hyp=marker, only: dict[str, set[str]] | None = None) -> tuple[Path, list[dict]]:
    run = tmp_path / "runs" / "20260925-000000_fake"
    run.mkdir(parents=True)
    manifest = {"run_name": run.name, "tag": "fake", "created": "2026-09-25T00:00:00+07:00", "split": "dev_mini",
                "split_version": "v0", "status": "done", "config": {"layout": None}, "directions": list(directions)}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    hyps, units = [], []
    for sid, (en, vi) in docs.items():
        for d in directions:
            if only is not None and sid not in only.get(d, set()):
                continue
            src_lang, text = ("en", en) if d == "en2vi" else ("vi", vi)
            for u in translation_units(parse_document(text, src_lang)):
                units.append({"sample_id": sid, "direction": d, "uid": u.uid, "element_idx": u.element_idx,
                              "cell_idx": u.cell_idx, "kind": u.kind, "src": u.text, "hyp": hyp(u.uid, d)})
            hyps.append({"sample_id": sid, "category": "Synthetic", "direction": d, "src_lang": src_lang,
                         "tgt_lang": "vi" if src_lang == "en" else "en", "hyp": "(rendered document)"})
    (run / "hyps.jsonl").write_text(SEG.dumps_jsonl(hyps), encoding="utf-8")
    (run / "units.jsonl").write_text(SEG.dumps_jsonl(units), encoding="utf-8")
    refs = [{"sample_id": sid, "category": "Synthetic", "english_markdown": en, "vietnamese_markdown": vi}
            for sid, (en, vi) in docs.items()]
    return run, refs


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d{2,}", text))


def expected_mt(sid: str, direction: str, src_text: str, src_lang: str, elems: list[int]) -> str:
    by_elem: dict[int, list] = {}
    for u in translation_units(parse_document(src_text, src_lang)):
        by_elem.setdefault(u.element_idx, []).append(u)
    return " ".join(marker(u.uid, direction) for i in elems for u in sorted(by_elem.get(i, []),
                    key=lambda u: -1 if u.cell_idx is None else u.cell_idx))


def test_segments_map_each_source_group_to_its_own_hypotheses(tmp_path):
    docs = {"Synthetic/1": (make_doc("en"), make_doc("vi"))}
    run, refs = fake_run(tmp_path, docs)
    seg = SEG.build_segments(run, refs=refs)
    assert [d["status"] for d in seg.docs] == ["ok", "ok"]
    by_dir = {d: [t for t in seg.triples if t["direction"] == d] for d in ("en2vi", "vi2en")}
    en, vi = docs["Synthetic/1"]
    en_sk, vi_sk = parse_document(en, "en"), parse_document(vi, "vi")
    for d, ts in by_dir.items():
        src_text, src_lang, src_sk, ref_sk = (en, "en", en_sk, vi_sk) if d == "en2vi" else (vi, "vi", vi_sk, en_sk)
        assert len(ts) == 14 and all(t["link"] == "1-1" for t in ts)
        assert [t["seg_id"] for t in ts] == [f"{d}:Synthetic/1:{k:04d}" for k in range(14)]
        for t in ts:
            assert t["src_elems"] and t["ref_elems"]
            assert t["mt"] == expected_mt("Synthetic/1", d, src_text, src_lang, t["src_elems"])
            assert t["src"] == " ".join(SEG.element_text(src_sk.elements[i]) for i in t["src_elems"])
            assert t["ref"] == " ".join(SEG.element_text(ref_sk.elements[i]) for i in t["ref_elems"])
            assert numbers(t["src"]) == numbers(t["ref"])
            assert t["mt_missing_units"] == 0 and t["trimmed_rows"] == 0
        motto = next(t for t in ts if t["src_elems"] == [1])
        assert motto["mt"] == marker("e1c1", d) and "---" not in motto["src"]
        assert {t["kind_group"] for t in ts} == {"header_row", "text", "signature_row"}
    assert {(tuple(t["src_elems"]), tuple(t["ref_elems"])) for t in by_dir["en2vi"]} == \
        {(tuple(t["ref_elems"]), tuple(t["src_elems"])) for t in by_dir["vi2en"]}
    cov = SEG.coverage([x for x in seg.docs if x["direction"] == "en2vi"])
    assert cov["src_coverage"] == 1.0 and cov["ref_coverage"] == 1.0 and cov["src_chars_dropped"] == {}
    assert SEG.build_triples(run, refs=refs) == seg.triples


def test_row_hypothesis_joins_cells_in_column_order(tmp_path):
    run, refs = fake_run(tmp_path, {"Synthetic/1": (make_doc("en"), make_doc("vi"))}, directions=("en2vi",),
                         hyp=lambda uid, d: f"T{uid}")
    t = next(t for t in SEG.build_triples(run, refs=refs) if t["src_elems"] == [2])
    assert t["mt"] == "Te2c0 Te2c1" and t["src"] == "No. 88/2034/HC Sampletown, May 9, 2034"


def test_directions_name_the_source_language_first(tmp_path):
    run, refs = fake_run(tmp_path, {"Synthetic/1": (make_doc("en"), make_doc("vi"))})
    seen = []
    for t in SEG.build_triples(run, refs=refs):
        if t["src_kinds"] == ["heading"] and "1" in t["src"]:
            en_side, vi_side = (t["src"], t["ref"]) if t["direction"] == "en2vi" else (t["ref"], t["src"])
            assert en_side.startswith("Article 1") and vi_side.startswith("Điều 1")
            assert (t["src_lang"], t["tgt_lang"]) == (("en", "vi") if t["direction"] == "en2vi" else ("vi", "en"))
            seen.append(t["direction"])
    assert sorted(seen) == ["en2vi", "vi2en"]


def test_hypotheses_are_nfc_and_one_line_like_source_and_reference(tmp_path):
    nfc = "Chủ thuyền phải nộp đơn đăng ký"
    run, refs = fake_run(tmp_path, {"Synthetic/1": (make_doc("en"), make_doc("vi"))}, directions=("en2vi",),
                         hyp=lambda uid, d: unicodedata.normalize("NFD", nfc) + "\n\t" + uid + "​")
    ts = SEG.build_triples(run, refs=refs)
    assert ts and all(unicodedata.is_normalized("NFC", t["mt"]) and "\n" not in t["mt"] for t in ts)
    para = next(t for t in ts if t["src_kinds"] == ["paragraph"])
    assert para["mt"] == f"{nfc} e{para['src_elems'][0]}"
    assert SEG.score_triples([dict(para, ref=para["mt"])], n_bootstrap=0)[0]["bleu"] == pytest.approx(100.0)


def test_zone_rows_enter_only_when_aligned_to_the_same_zone(tmp_path):
    run, refs = fake_run(tmp_path, {"Synthetic/1": (make_doc("en"), make_doc("vi", signature=False))},
                         directions=("en2vi",))
    seg = SEG.build_segments(run, refs=refs)
    kinds = {k for t in seg.triples for k in t["src_kinds"]}
    assert "signature_row" not in kinds and {"header_row", "heading", "paragraph"} <= kinds
    last = max(seg.triples, key=lambda t: t["src_elems"])
    assert last["src_elems"] == [10] and numbers(last["src"]) == numbers(last["ref"]) == {"2034"}
    dropped = seg.docs[0]["src_chars_dropped"]
    sig_chars = sum(len(s.replace(" ", "")) for s in ("FOR THE COUNCIL", "CHAIRPERSON", "Robin Example"))
    assert dropped.get("unaligned", 0) + dropped.get("unaligned_row", 0) == sig_chars


def test_trim_zone_rows_rule():
    def el(i, kind):
        return Element(idx=i, kind=kind, text="x", indent=0, lines=(i, i + 1))
    src = [el(0, "paragraph"), el(1, "signature_row"), el(2, "header_row")]
    ref = [el(0, "paragraph"), el(1, "signature_row")]
    assert SEG.trim_zone_rows([0, 1], [0], src, ref) == ([0], [0], [1], [])
    assert SEG.trim_zone_rows([1], [1], src, ref) == ([1], [1], [], [])
    assert SEG.trim_zone_rows([2], [1], src, ref) == ([], [], [2], [1])
    assert SEG.trim_zone_rows([0], [0, 1], src, ref) == ([0], [0], [], [1])


def test_unit_records_that_do_not_match_the_source_skip_the_document(tmp_path):
    docs = {"Synthetic/1": (make_doc("en"), make_doc("vi")), "Synthetic/2": (make_doc("en"), make_doc("vi"))}
    run, refs = fake_run(tmp_path, docs, directions=("en2vi",))
    lines = (run / "units.jsonl").read_text(encoding="utf-8").splitlines()
    recs = [json.loads(x) for x in lines]
    k = next(i for i, r in enumerate(recs) if r["sample_id"] == "Synthetic/2" and r["uid"] == "e5")
    recs[k]["element_idx"] = 6
    (run / "units.jsonl").write_text(SEG.dumps_jsonl(recs), encoding="utf-8")
    seg = SEG.build_segments(run, refs=refs)
    status = {d["sample_id"]: d["status"] for d in seg.docs}
    assert status == {"Synthetic/1": "ok", "Synthetic/2": "unit_mismatch"}
    assert {t["sample_id"] for t in seg.triples} == {"Synthetic/1"}
    sk = parse_document(make_doc("en"), "en")
    good = [r for r in recs if r["sample_id"] == "Synthetic/1"]
    assert SEG.check_units(sk, good) == []
    assert any("no record" in p for p in SEG.check_units(sk, good[1:]))
    bad_src = [dict(r, src="changed") if r["uid"] == "e4" else r for r in good]
    assert any("differs" in p for p in SEG.check_units(sk, bad_src))


def test_unit_uid_must_name_its_own_element(tmp_path):
    twin = "c) the closing sentence of the charter, repeated on purpose."
    body = EN_BODY + [("p", twin), ("p", twin)]
    en = make_doc("en", body=body)
    sk = parse_document(en, "en")
    i, j = [e.idx for e in sk.elements if e.text == twin]
    recs = [{"uid": u.uid, "element_idx": u.element_idx, "cell_idx": u.cell_idx, "src": u.text, "hyp": marker(u.uid, "en2vi")}
            for u in translation_units(sk)]
    assert SEG.check_units(sk, recs) == []
    moved = [dict(r, element_idx=j) if r["uid"] == f"e{i}" else r for r in recs]
    assert any("uid does not match" in p for p in SEG.check_units(sk, moved))
    no_uid = [{k: v for k, v in r.items() if k != "uid"} for r in recs]
    assert SEG.check_units(sk, no_uid) == []
    assert any("more than one record" in p for p in SEG.check_units(sk, no_uid + [no_uid[0]]))
    triples, st = SEG.doc_segments("Synthetic/1", "Synthetic", "en2vi", en, make_doc("vi"), moved)
    assert triples == [] and st["status"] == "unit_mismatch"


def test_declared_direction_without_any_hypothesis_is_reported_missing(tmp_path):
    run, refs = fake_run(tmp_path, {"Synthetic/1": (make_doc("en"), make_doc("vi"))},
                         only={"en2vi": {"Synthetic/1"}, "vi2en": set()})
    seg = SEG.build_segments(run, refs=refs)
    assert seg.info["directions"] == ["en2vi", "vi2en"]
    assert seg.info["missing_hyps"] == {"en2vi": [], "vi2en": ["Synthetic/1"]}
    with pytest.raises(ValueError):
        SEG.build_segments(run, refs=refs, directions=["en2fr"])


def test_duplicate_hypothesis_records_use_the_last_one(tmp_path):
    run, refs = fake_run(tmp_path, {"Synthetic/1": (make_doc("en"), make_doc("vi"))}, directions=("en2vi",))
    first = SEG.build_segments(run, refs=refs)
    for name in ("hyps.jsonl", "units.jsonl"):
        p = run / name
        p.write_text(p.read_text(encoding="utf-8") * 2, encoding="utf-8")
    again = SEG.build_segments(run, refs=refs)
    assert again.triples == first.triples and again.info["duplicate_hyp_records"] == {"en2vi": 1}


def test_missing_documents_subsets_and_foreign_hypotheses(tmp_path):
    docs = {"Synthetic/1": (make_doc("en"), make_doc("vi")), "Synthetic/2": (make_doc("en"), make_doc("vi"))}
    run, refs = fake_run(tmp_path, docs, only={"en2vi": {"Synthetic/1", "Synthetic/2"}, "vi2en": {"Synthetic/1"}})
    seg = SEG.build_segments(run, refs=refs)
    assert seg.info["missing_hyps"] == {"en2vi": [], "vi2en": ["Synthetic/2"]}
    assert seg.info["run_status"] == "done" and seg.info["directions"] == ["en2vi", "vi2en"]
    sub = SEG.build_segments(run, refs=refs, ids=["Synthetic/2"])
    assert {t["sample_id"] for t in sub.triples} == {"Synthetic/2"}
    assert sub.info["missing_hyps"]["vi2en"] == ["Synthetic/2"] and sub.info["hyps_not_in_refs"]["en2vi"] == []
    only_one = SEG.build_segments(run, refs=refs[:1])
    assert only_one.info["hyps_not_in_refs"]["en2vi"] == ["Synthetic/2"]
    with pytest.raises(KeyError):
        SEG.build_segments(run, refs=refs, ids=["Synthetic/9"])


def test_training_filters_drop_suspicious_links_only_when_on(tmp_path):
    vi_body = list(VI_BODY)
    vi_body[2] = ("p", vi_body[2][1].replace("36", "77").replace("2250", "9911"))
    vi_body[4] = ("p", " ".join([vi_body[4][1]] * 3))
    run, refs = fake_run(tmp_path, {"Synthetic/1": (make_doc("en"), make_doc("vi", body=vi_body))}, directions=("en2vi",))
    on = SEG.build_segments(run, refs=refs)
    off = SEG.build_segments(run, refs=refs, filters=False)
    assert len(off.triples) == 14 and len(on.triples) == 12
    kept = {tuple(t["src_elems"]) for t in on.triples}
    assert (6,) not in kept and (8,) not in kept
    assert set(on.docs[0]["src_chars_dropped"]) == {"numeric", "length_ratio"}
    assert on.info["align"]["filters_applied"] and not off.info["align"]["filters_applied"]


def _triples(pairs, kinds=None):
    return [{"seg_id": f"en2vi:D{k // 2}:{k:04d}", "sample_id": f"D{k // 2}", "direction": "en2vi", "src": "s",
             "mt": mt, "ref": ref, "kind_group": (kinds or ["text"] * len(pairs))[k]}
            for k, (mt, ref) in enumerate(pairs)]


def test_score_triples_identity_and_empty_hypotheses():
    refs = ["the council registers small boats", "owners pay a fee of 2250 credits", "form 7391 lists the items",
            "a copy of the contract"]
    same, seg = SEG.score_triples(_triples([(r, r) for r in refs], ["text", "text", "table_row", "text"]), n_bootstrap=50)
    assert same["bleu"] == pytest.approx(100.0) and same["chrf"] == pytest.approx(100.0)
    assert same["n_segments"] == 4 and same["n_docs"] == 2 and same["bootstrap"]["unit"] == "document"
    assert same["bleu_ci95"][0] == pytest.approx(100.0) and set(same["per_kind"]) == {"text", "table_row"}
    assert all(s["chrf"] == pytest.approx(100.0) and s["bleu"] == pytest.approx(100.0) for s in seg)
    empty, _ = SEG.score_triples(_triples([("", r) for r in refs]), n_bootstrap=0)
    assert empty["bleu"] == 0.0 and "bleu_ci95" not in empty
    assert SEG.score_triples([]) == ({"n_segments": 0}, [])


def test_comet_by_direction_means_and_document_bootstrap():
    ts = _triples([("a", "a")] * 4, ["text", "text", "header_row", "text"])
    scores = {t["seg_id"]: v for t, v in zip(ts, [0.9, 0.7, 0.5, 0.3])}
    res = SEG.comet_by_direction(ts, scores, n_bootstrap=200, seed=1)["en2vi"]
    assert res["n_scored"] == 4 and res["mean"] == pytest.approx(0.6)
    assert res["doc_macro_mean"] == pytest.approx((0.8 + 0.4) / 2)
    assert 0.4 <= res["ci95"][0] <= res["mean"] <= res["ci95"][1] <= 0.8
    assert res["per_kind"]["header_row"] == {"n_segments": 1, "mean": pytest.approx(0.5)}
    partial = SEG.comet_by_direction(ts, dict(list(scores.items())[:3]), n_bootstrap=0)["en2vi"]
    assert partial["n_scored"] == 3 and "ci95" not in partial


def _run_script(run: Path, refs_file: Path, *extra: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(EVAL_SCRIPT), str(run), "--refs-file", str(refs_file),
                           "--n-bootstrap", "20", *extra], capture_output=True, text=True, cwd=str(REPO_ROOT), env=env)


def _pins() -> dict:
    table = tomllib.loads((REPO_ROOT / "tools" / "comet" / "pyproject.toml").read_text(encoding="utf-8"))
    t = table["tool"]["legalmt-comet"]
    return {k: t[k] for k in ("model", "model_revision", "encoder", "encoder_revision")}


def test_evaluate_segments_script_and_comet_merge(tmp_path):
    docs = {"Synthetic/1": (make_doc("en"), make_doc("vi")), "Synthetic/2": (make_doc("en"), make_doc("vi"))}
    run, refs = fake_run(tmp_path, docs)
    refs_file = tmp_path / "refs.jsonl"
    refs_file.write_text(SEG.dumps_jsonl(refs), encoding="utf-8")
    res = _run_script(run, refs_file)
    assert res.returncode == 0, res.stderr
    assert "Robin" not in res.stdout and "Sampletown" not in res.stdout
    out = json.loads((run / SEG.RESULTS_FILE).read_text(encoding="utf-8"))
    assert set(out["directions"]) == {"en2vi", "vi2en"} and out["subset"] is None and "comet" not in out
    for d, r in out["directions"].items():
        assert r["n_segments"] == 28 and r["coverage"]["src_coverage"] == 1.0 and r["missing_hyps"] == []
        assert len(r["bleu_ci95"]) == 2 and "comet" not in r
    segs = [json.loads(x) for x in (run / SEG.SEGMENTS_FILE).read_text(encoding="utf-8").splitlines()]
    assert len(segs) == 56 and all({"chrf", "bleu", "src", "mt", "ref"} <= set(s) for s in segs)
    assert not (run / "comet_input.jsonl").exists()

    blob = SEG.dumps_jsonl(SEG.comet_input_records(segs)).encode("utf-8")
    fake = {**_pins(), "input_sha256": hashlib.sha256(blob).hexdigest(),
            "ids": [s["seg_id"] for s in segs], "scores": [0.5 if s["direction"] == "en2vi" else 0.25 for s in segs],
            "n_segments": len(segs), "system_score": 0.375}
    (run / "comet_scores.json").write_text(json.dumps(fake), encoding="utf-8")
    assert _run_script(run, refs_file).returncode == 0
    out = json.loads((run / SEG.RESULTS_FILE).read_text(encoding="utf-8"))
    assert out["comet"]["reused"] is True and out["comet"]["system_score_pooled"] == pytest.approx(0.375)
    assert out["directions"]["en2vi"]["comet"]["mean"] == pytest.approx(0.5)
    assert out["directions"]["vi2en"]["comet"]["mean"] == pytest.approx(0.25)
    segs = [json.loads(x) for x in (run / SEG.SEGMENTS_FILE).read_text(encoding="utf-8").splitlines()]
    assert {s["comet"] for s in segs} == {0.5, 0.25}

    for key, value, why in (("input_sha256", "0" * 64, "scored different segments"),
                            ("model_revision", "0" * 40, "other pins (model_revision)")):
        (run / "comet_scores.json").write_text(json.dumps({**fake, key: value}), encoding="utf-8")
        res = _run_script(run, refs_file)
        assert res.returncode == 0
        out = json.loads((run / SEG.RESULTS_FILE).read_text(encoding="utf-8"))
        assert out["comet"]["stale"] is True and "comet" not in out["directions"]["en2vi"]
        assert "not merged" in res.stdout and why in res.stdout
        assert all("comet" not in json.loads(x) for x in (run / SEG.SEGMENTS_FILE).read_text(encoding="utf-8").splitlines())

    res = _run_script(run, refs_file, "--limit", "1", "--directions", "en2vi")
    assert res.returncode == 0 and "partial evaluation" in res.stdout
    out = json.loads((run / SEG.RESULTS_FILE).read_text(encoding="utf-8"))
    assert out["subset"] == {"n_docs": 1, "ids": ["Synthetic/1"]} and list(out["directions"]) == ["en2vi"]


def test_script_is_deterministic_and_line_counts_match(tmp_path):
    docs = {"Synthetic/1": (make_doc("en"), make_doc("vi")), "Synthetic/2": (make_doc("en"), make_doc("vi", signature=False))}
    run, refs = fake_run(tmp_path, docs)
    refs_file = tmp_path / "refs.jsonl"
    refs_file.write_text(SEG.dumps_jsonl(refs), encoding="utf-8")
    outs = []
    for _ in range(2):
        assert _run_script(run, refs_file).returncode == 0
        res = json.loads((run / SEG.RESULTS_FILE).read_text(encoding="utf-8"))
        outs.append(((run / SEG.SEGMENTS_FILE).read_bytes(), {d: (r["bleu"], r["bleu_ci95"]) for d, r in res["directions"].items()}))
    assert outs[0] == outs[1]
    lines = outs[0][0].decode("utf-8").split("\n")
    assert lines[-1] == "" and len(lines) - 1 == sum(r["n_segments"] for r in res["directions"].values())
    assert all(sum(d["n_segments"] for d in res["docs"] if d["direction"] == k) == r["n_segments"]
               for k, r in res["directions"].items())


def test_empty_run_is_reported_and_exits_nonzero(tmp_path):
    run, refs = fake_run(tmp_path, {"Synthetic/1": (make_doc("en"), make_doc("vi"))}, only={"en2vi": set(), "vi2en": set()})
    refs_file = tmp_path / "refs.jsonl"
    refs_file.write_text(SEG.dumps_jsonl(refs), encoding="utf-8")
    res = _run_script(run, refs_file)
    assert res.returncode == 1, res.stderr
    assert "no segments at all" in res.stdout and "documents have no hypothesis" in res.stdout
    out = json.loads((run / SEG.RESULTS_FILE).read_text(encoding="utf-8"))
    assert out["directions"]["en2vi"]["n_segments"] == 0 and out["directions"]["en2vi"]["coverage"]["src_coverage"] is None
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    (run / "manifest.json").write_text(json.dumps({**manifest, "directions": []}), encoding="utf-8")
    res = _run_script(run, refs_file)
    assert res.returncode == 1 and "no direction to evaluate" in res.stdout


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_comet_reuse_rules():
    es = _load_script(EVAL_SCRIPT, "legalmt_evaluate_segments")
    pins = es.comet_pins()
    good = {**_pins(), "input_sha256": "a" * 64, "ids": ["x", "y"], "scores": [0.5, 0.25], "n_segments": 2}
    assert es.comet_reuse_problem(good, "a" * 64, pins) is None
    assert "different segments" in es.comet_reuse_problem(good, "b" * 64, pins)
    assert "encoder_revision" in es.comet_reuse_problem({**good, "encoder_revision": "c" * 40}, "a" * 64, pins)
    assert "inconsistent" in es.comet_reuse_problem({**good, "scores": [0.5]}, "a" * 64, pins)
    assert "NaN" in es.comet_reuse_problem({**good, "scores": [0.5, float("nan")]}, "a" * 64, pins)


def _fake_comet_env(root: Path) -> Path:
    env = root / "comet-env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").write_text("", encoding="utf-8")
    pkg = env / "lib" / "python3.12" / "site-packages" / "comet"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    return env


def test_comet_env_guard_refuses_the_gpu_env_and_unbuilt_envs(tmp_path):
    es = _load_script(EVAL_SCRIPT, "legalmt_evaluate_segments")
    base = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", str(tmp_path))}
    gpu = es.gpu_env_dir(base)
    assert gpu is not None
    assert es.gpu_env_dir({**base, "UV_PROJECT_ENVIRONMENT": str(tmp_path / "x")}) == gpu
    with pytest.raises(SystemExit, match="GPU env"):
        es.check_comet_env(gpu, base)
    with pytest.raises(SystemExit, match="GPU env"):
        es.check_comet_env(Path(str(gpu) + "/"), base)
    with pytest.raises(SystemExit, match="no COMET env"):
        es.check_comet_env(tmp_path / "empty", base)
    es.check_comet_env(_fake_comet_env(tmp_path), base)


FAKE_UV = '''#!{python}
"""Stands in for 'uv run --project tools/comet ... comet_score.py': records how it was called and
writes a comet_scores.json the way comet_score.py does (FAKE_UV_MODE: ok, fail, badsha)."""
import hashlib, json, os, sys
argv = sys.argv[1:]
mode = os.environ.get("FAKE_UV_MODE", "ok")
log = os.environ["FAKE_UV_LOG"]
with open(log, "w", encoding="utf-8") as fh:
    json.dump({{"argv": argv, "UV_PROJECT_ENVIRONMENT": os.environ.get("UV_PROJECT_ENVIRONMENT"),
               "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR")}}, fh)
if mode == "fail":
    sys.exit(3)
inp, outp = argv[argv.index("--input") + 1], argv[argv.index("--output") + 1]
data = open(inp, "rb").read()
recs = [json.loads(x) for x in data.decode("utf-8").split("\\n") if x.strip()]
scores = [0.75 if r["seg_id"].startswith("en2vi") else 0.5 for r in recs]
out = {{**json.loads(os.environ["FAKE_UV_PINS"]), "input_sha256": "0" * 64 if mode == "badsha" else hashlib.sha256(data).hexdigest(),
       "ids": [r["seg_id"] for r in recs], "scores": scores, "n_segments": len(recs),
       "system_score": sum(scores) / len(scores), "truncation": {{"max_tokens": 510, "n_segments": 1, "ids": [recs[0]["seg_id"]]}}}}
open(outp, "w", encoding="utf-8").write(json.dumps(out))
'''


def test_evaluate_segments_comet_path_with_a_fake_uv(tmp_path):
    docs = {"Synthetic/1": (make_doc("en"), make_doc("vi"))}
    run, refs = fake_run(tmp_path, docs)
    refs_file = tmp_path / "refs.jsonl"
    refs_file.write_text(SEG.dumps_jsonl(refs), encoding="utf-8")
    uv = tmp_path / "fake_uv"
    uv.write_text(FAKE_UV.format(python=sys.executable), encoding="utf-8")
    uv.chmod(0o755)
    comet_env = _fake_comet_env(tmp_path)
    log = tmp_path / "uv_call.json"
    env = {**os.environ, "FAKE_UV_LOG": str(log), "FAKE_UV_PINS": json.dumps(_pins())}
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    args = ("--comet", "--comet-gpus", "0", "--comet-batch-size", "4", "--comet-env", str(comet_env), "--uv", str(uv))

    res = _run_script(run, refs_file, *args, env=env)
    assert res.returncode == 0, res.stderr
    call = json.loads(log.read_text(encoding="utf-8"))
    assert call["UV_PROJECT_ENVIRONMENT"] == str(comet_env) and call["UV_CACHE_DIR"]
    assert call["argv"][:5] == ["run", "--project", str(REPO_ROOT / "tools" / "comet"), "--locked", "python"]
    assert call["argv"][call["argv"].index("--gpus") + 1] == "0" and call["argv"][call["argv"].index("--batch-size") + 1] == "4"
    out = json.loads((run / SEG.RESULTS_FILE).read_text(encoding="utf-8"))
    assert out["comet"]["reused"] is False and out["comet"]["model_revision"] == _pins()["model_revision"]
    assert out["directions"]["en2vi"]["comet"]["mean"] == pytest.approx(0.75)
    assert out["directions"]["vi2en"]["comet"]["mean"] == pytest.approx(0.5)
    assert out["directions"]["en2vi"]["comet"]["n_truncated"] == 1 and "510-token window" in res.stdout
    written = [json.loads(x) for x in (run / "comet_input.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(set(r) == {"seg_id", "src", "mt", "ref"} for r in written) and len(written) == 28

    before = (run / SEG.RESULTS_FILE).read_bytes(), (run / SEG.SEGMENTS_FILE).read_bytes()
    for mode, msg in (("fail", "COMET scoring failed"), ("badsha", "does not match")):
        res = _run_script(run, refs_file, *args, env={**env, "FAKE_UV_MODE": mode})
        assert res.returncode != 0 and msg in res.stderr
        assert ((run / SEG.RESULTS_FILE).read_bytes(), (run / SEG.SEGMENTS_FILE).read_bytes()) == before


def test_comet_subprocess_command_and_environment(tmp_path):
    es = _load_script(EVAL_SCRIPT, "legalmt_evaluate_segments")
    cmd = es.comet_command("uv", tmp_path / "in.jsonl", tmp_path / "out.json", gpus=0, batch_size=8, precision="auto")
    assert cmd[:6] == ["uv", "run", "--project", str(REPO_ROOT / "tools" / "comet"), "--locked", "python"]
    assert cmd[cmd.index("--gpus") + 1] == "0" and cmd[cmd.index("--batch-size") + 1] == "8"
    env = es.comet_subprocess_env(Path("/nonexistent/comet-env"), base={"PATH": os.environ["PATH"]})
    assert env["UV_PROJECT_ENVIRONMENT"] == "/nonexistent/comet-env"
    assert env["UV_CACHE_DIR"] and env["HF_HOME"] and env["TMPDIR"]
    keep = es.comet_subprocess_env(Path("/x"), base={"PATH": os.environ["PATH"], "HF_HOME": "/my/hf"})
    assert keep["HF_HOME"] == "/my/hf"


def test_comet_score_script_helpers_without_comet(tmp_path):
    cs = _load_script(COMET_SCRIPT, "legalmt_comet_score")
    pins = cs.load_pins()
    assert pins["model"] == "Unbabel/wmt22-comet-da" and re.fullmatch(r"[0-9a-f]{40}", pins["model_revision"])
    assert re.fullmatch(r"[0-9a-f]{40}", pins["encoder_revision"]) and pins["encoder_hparam"] == "xlm-roberta-large"
    good = tmp_path / "good.jsonl"
    good.write_text(SEG.dumps_jsonl([{"seg_id": "a", "src": "x", "mt": "", "ref": "z", "extra": 1}]), encoding="utf-8")
    samples, ids = cs.read_segments(good, "seg_id")
    assert samples == [{"src": "x", "mt": "", "ref": "z"}] and ids == ["a"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(SEG.dumps_jsonl([{"src": "x", "mt": "y"}]), encoding="utf-8")
    with pytest.raises(SystemExit):
        cs.read_segments(bad, "seg_id")
    odd = tmp_path / "odd.jsonl"
    odd.write_text(SEG.dumps_jsonl([{"seg_id": "a", "src": "x y", "mt": "p\x85q", "ref": "z"}]), encoding="utf-8")
    assert cs.read_segments(odd, "seg_id") == ([{"src": "x y", "mt": "p\x85q", "ref": "z"}], ["a"])
    assert cs.nonfinite_problem([0.5, 0.25], ["a", "b"], "fp16") is None
    msg = cs.nonfinite_problem([0.5, float("nan"), float("inf")], ["a", "b", "c"], "fp16")
    assert msg.startswith("2 of 3") and "['b', 'c']" in msg and "--precision fp32" in msg
    res = subprocess.run([sys.executable, str(COMET_SCRIPT), "--help"], capture_output=True, text=True)
    assert res.returncode == 0 and "--gpus" in res.stdout


SETUP_SCRIPT = REPO_ROOT / "scripts" / "LegalMT" / "setup_comet_env.sh"


def test_setup_script_help_and_gpu_env_guard(tmp_path):
    base = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", str(tmp_path)), "MIN_FREE_GB": "100000000"}
    res = subprocess.run(["bash", str(SETUP_SCRIPT), "--help"], capture_output=True, text=True, env=base)
    assert res.returncode == 0 and "COMET" in res.stdout and "set -euo" not in res.stdout
    es = _load_script(EVAL_SCRIPT, "legalmt_evaluate_segments")
    gpu = es.gpu_env_dir(base)
    comet_env = str(tmp_path / "comet-env")
    res = subprocess.run(["bash", str(SETUP_SCRIPT)], capture_output=True, text=True,
                         env={**base, "UV_PROJECT_ENVIRONMENT": comet_env, "LEGALMT_COMET_ENV": comet_env})
    assert res.returncode == 1 and "Aborting" in res.stderr, res.stderr
    res = subprocess.run(["bash", str(SETUP_SCRIPT)], capture_output=True, text=True,
                         env={**base, "UV_PROJECT_ENVIRONMENT": str(tmp_path / "other"), "LEGALMT_COMET_ENV": f"{gpu}/"})
    assert res.returncode == 2 and "refusing" in res.stderr, res.stderr


def test_comet_project_pins_reuse_the_gpu_env_torch():
    lock = tomllib.loads((REPO_ROOT / "tools" / "comet" / "uv.lock").read_text(encoding="utf-8"))
    comet = {p["name"]: p["version"] for p in lock["package"]}
    main = {}
    for p in tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]:
        main.setdefault(p["name"], set()).add(p["version"])
    assert comet["unbabel-comet"] == "2.2.7" and comet["transformers"].startswith("4.")
    assert comet["torch"] in main["torch"]
    assert int(comet["setuptools"].split(".")[0]) < 81


def _echo_records(docs: list[tuple[str, str, str, str]]):
    from LegalMT.translate import Translator, load_infer_config

    cfg, _ = load_infer_config("echo_smoke")
    tr = Translator(cfg)
    hyps, units = [], []
    for direction, src_lang in (("en2vi", "en"), ("vi2en", "vi")):
        results = tr.translate_documents([(sid, en if src_lang == "en" else vi, src_lang) for sid, _c, en, vi in docs])
        for (sid, cat, _en, _vi), res in zip(docs, results):
            hyps.append({"sample_id": sid, "category": cat, "direction": direction, "hyp": res.text})
            units.extend({"sample_id": sid, "direction": direction, **u} for u in res.units)
    return hyps, units


def test_echo_translator_units_give_identity_segments():
    docs = [("Synthetic/1", "Synthetic", make_doc("en"), make_doc("vi"))]
    hyps, units = _echo_records(docs)
    refs = [{"sample_id": s, "category": c, "english_markdown": en, "vietnamese_markdown": vi} for s, c, en, vi in docs]
    seg = SEG.build_segments(hyps=hyps, units=units, refs=refs, layout_cfg=None)
    assert [d["status"] for d in seg.docs] == ["ok", "ok"] and len(seg.triples) == 28
    assert all(t["mt"] == t["src"] for t in seg.triples)


@pytest.mark.data
def test_echo_segments_on_two_dev_mini_documents():
    from LegalMT.splits import docs_for

    df = docs_for("dev_mini").head(2)
    docs = list(zip(df["sample_id"], df["category"], df["english_markdown"], df["vietnamese_markdown"]))
    hyps, units = _echo_records(docs)
    seg = SEG.build_segments(hyps=hyps, units=units, refs=df, layout_cfg=None)
    assert all(d["status"] == "ok" for d in seg.docs) and len(seg.docs) == 4
    assert seg.triples and all(t["mt"] == t["src"] for t in seg.triples)
    for d in ("en2vi", "vi2en"):
        cov = SEG.coverage([x for x in seg.docs if x["direction"] == d])
        assert cov["src_coverage"] > 0.9 and cov["ref_coverage"] > 0.9 and cov["mt_missing_units"] == 0
