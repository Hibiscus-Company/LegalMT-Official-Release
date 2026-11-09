"""End-to-end tests of LegalMT.translate with the echo backend, config checks and a tiny random model."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from LegalMT import paths
from LegalMT import runs as R
from LegalMT.chunking import plan_chunks
from LegalMT.layout import parse_document, render_document, translation_units
from LegalMT.translate import Translator, load_infer_config, parse_set_overrides
from LegalMT.translate.backends import EchoBackend, GenRequest, HFBackend, decoding_params, max_new_tokens_for
from LegalMT.translate.core import INFER_CONFIG_DIR, registry_revision

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "LegalMT" / "translate.py"


def _wrap(text: str, width: int = 64, indent: int = 2) -> list[str]:
    return textwrap.wrap(text, width=width, initial_indent=" " * indent, subsequent_indent=" " * indent,
                         break_long_words=False, break_on_hyphens=False)


EN_PARAS = [
    "This regulation sets out how the city board receives, reviews and answers written requests from "
    "residents about the maintenance of public gardens and small parks.",
    "Each request shall be reviewed within 15 working days from the date of receipt, and the applicant "
    "shall be informed of the outcome in writing.",
    "a) the name and address of the applicant;",
    "b) a short description of the garden.",
]
VI_PARAS = [
    "Quy chế này quy định việc tiếp nhận, xem xét và trả lời các yêu cầu bằng văn bản của người dân về "
    "việc duy trì vườn hoa và công viên nhỏ trong thành phố.",
    "Mỗi yêu cầu phải được xem xét trong thời hạn 15 ngày làm việc kể từ ngày nhận, và người nộp đơn phải "
    "được thông báo kết quả bằng văn bản.",
    "a) tên và địa chỉ của người nộp đơn;",
    "b) mô tả ngắn gọn về khu vườn.",
]


def en_doc() -> str:
    lines = [
        "CITY BOARD OF TESTVILLE                         REPUBLIC OF EXAMPLELAND",
        "        -------                                Order - Progress - Harmony",
        "      No. 12/2031/QD-TB                                   Testville, March 3, 2031",
        "", " " * 36 + "REGULATION", "", "  Article 1. Scope", ""]
    for p in EN_PARAS[:2]:
        lines += _wrap(p) + [""]
    lines += ["  Article 2. Content of a request", ""]
    for p in EN_PARAS[2:]:
        lines += _wrap(p) + [""]
    lines += ["", " " * 66 + "ON BEHALF OF THE BOARD", " " * 72 + "CHAIRPERSON", "", "", " " * 72 + "Alex Sample"]
    return "\n".join(lines) + "\n"


def vi_doc() -> str:
    lines = [
        "ỦY BAN THÀNH PHỐ THỬ NGHIỆM                    CỘNG HÒA XÃ HỘI VÍ DỤ",
        "        -------                                Trật tự - Tiến bộ - Hài hòa",
        "      Số: 12/2031/QĐ-TN                                   Thử Nghiệm, ngày 3 tháng 3 năm 2031",
        "", " " * 36 + "QUY CHẾ", "", "  Điều 1. Phạm vi", ""]
    for p in VI_PARAS[:2]:
        lines += _wrap(p) + [""]
    lines += ["  Điều 2. Nội dung yêu cầu", ""]
    for p in VI_PARAS[2:]:
        lines += _wrap(p) + [""]
    lines += ["", " " * 66 + "TM. ỦY BAN", " " * 72 + "CHỦ TỊCH", "", "", " " * 72 + "Nguyễn Văn Mẫu"]
    return "\n".join(lines) + "\n"


def identity_render(text: str, lang: str) -> str:
    skel = parse_document(text, lang)
    return render_document(skel, {u.uid: u.text for u in translation_units(skel)})


def echo_cfg(**over) -> dict:
    cfg, _ = load_infer_config("echo_smoke", over)
    return cfg


class RecordingEcho(EchoBackend):
    def __init__(self, cfg, empty: bool = False):
        super().__init__(cfg)
        self.calls: list[list[GenRequest]] = []
        self.empty = empty

    def generate(self, requests):
        self.calls.append(list(requests))
        outs = super().generate(requests)
        if self.empty:
            for o in outs:
                o.text = ""
        return outs


@pytest.mark.parametrize("lang,make", [("en", en_doc), ("vi", vi_doc)])
def test_echo_roundtrip_both_directions(lang, make):
    doc = make()
    tr = Translator(echo_cfg())
    res = tr.translate_text(doc, lang)
    assert (res.src_lang, res.tgt_lang) == (lang, "vi" if lang == "en" else "en")
    ident = identity_render(doc, lang)
    assert res.text == ident
    assert len(res.text.splitlines()) == len(ident.splitlines())
    units = translation_units(parse_document(doc, lang))
    assert [u["uid"] for u in res.units] == [u.uid for u in units]
    assert all(u["hyp"] == u["src"] for u in res.units)
    assert {"header_row", "heading", "paragraph", "signature_row"} <= {u["kind"] for u in res.units}
    st = res.stats
    assert st["direction"] == f"{res.src_lang}2{res.tgt_lang}" and st["detected_src_lang"] == lang
    assert st["n_units"] == len(units) and st["n_chunks"] >= 1 and st["failed_chunks"] == 0
    assert st["fallback_units"] == 0 and st["render_missing"] == 0 and st["in_tokens"] > 0 and st["out_tokens"] > 0
    assert tr.translate_text(doc).src_lang == lang


def test_echo_prefix_marks_every_unit_and_keeps_line_structure():
    doc = en_doc()
    res = Translator(echo_cfg(backend={"prefix": "[vi] "})).translate_text(doc, "en")
    assert all(u["hyp"] == "[vi] " + u["src"] for u in res.units)
    ident = identity_render(doc, "en").splitlines()
    out = res.text.splitlines()
    assert len(out) == len(ident)
    for a, b in zip(ident, out):
        assert (a.strip() == "") == (b.strip() == "")
        if a.strip() and not a.startswith("    "):
            assert len(a) - len(a.lstrip()) == len(b) - len(b.lstrip())
    rows = lambda text: [e.kind for e in parse_document(text, "en").elements if e.kind.endswith("_row")]
    assert rows(res.text) == rows(doc) and "header_row" in rows(doc) and "signature_row" in rows(doc)


@pytest.mark.parametrize("fail_mode", ["drop_last_line", "merge_lines"])
def test_fallback_retranslates_failed_chunks_unit_by_unit(fail_mode):
    doc = en_doc()
    tr = Translator(echo_cfg(backend={"fail_mode": fail_mode}, chunking={"budget_tokens": {"en2vi": 30}}))
    rec = RecordingEcho(tr.cfg)
    tr.backend = rec
    res = tr.translate_text(doc, "en")
    chunks = plan_chunks(translation_units(parse_document(doc, "en")), lambda t: len(t.split()), 30)
    multi = [c for c in chunks if len(c.uids) > 1]
    assert multi, "the test document must produce multi-line chunks"
    assert res.stats["failed_chunks"] == len(multi)
    assert res.stats["fallback_units"] == sum(len(c.uids) for c in multi)
    assert len(rec.calls) == 2 and all(len(r.src_text.split("\n")) == 1 for r in rec.calls[1])
    assert res.text == identity_render(doc, "en")
    fb = {u["uid"] for u in res.units if u["fallback"]}
    assert fb == {uid for c in multi for uid in c.uids}


def test_empty_output_falls_back_to_source_text():
    doc = vi_doc()
    tr = Translator(echo_cfg())
    tr.backend = RecordingEcho(tr.cfg, empty=True)
    res = tr.translate_text(doc, "vi")
    assert res.stats["source_copied_units"] == res.stats["n_units"] == res.stats["fallback_units"]
    assert res.text == identity_render(doc, "vi") and all(u["source_copied"] for u in res.units)


def test_truncation_is_counted_and_document_stays_complete():
    doc = en_doc()
    cfg = echo_cfg(generation={"max_new_tokens_ratio": {"en2vi": 1.0, "vi2en": 1.0}, "max_new_tokens_margin": 0,
                               "max_new_tokens_cap": 3})
    res = Translator(cfg).translate_text(doc, "en")
    assert res.stats["truncated_requests"] > 0 and res.stats["render_missing"] == 0
    assert all(len(u["hyp"].split()) <= 3 for u in res.units)


class CutLastLineEcho(EchoBackend):
    def generate(self, requests):
        outs = super().generate(requests)
        for r, o in zip(requests, outs):
            lines = r.src_text.split("\n")
            if len(lines) > 1:
                lines[-1] = lines[-1].split()[0]
                o.text, o.finished, o.n_out = "\n".join(lines), False, r.max_new_tokens
        return outs


def test_length_stopped_chunk_with_matching_line_count_is_retranslated():
    doc = en_doc()
    tr = Translator(echo_cfg(chunking={"budget_tokens": {"en2vi": 30}}))
    tr.backend = CutLastLineEcho(tr.cfg)
    res = tr.translate_text(doc, "en")
    multi = [c for c in plan_chunks(translation_units(parse_document(doc, "en")), lambda t: len(t.split()), 30)
             if len(c.uids) > 1]
    assert multi and res.stats["truncated_chunks"] == len(multi) == res.stats["failed_chunks"]
    assert res.stats["fallback_units"] == sum(len(c.uids) for c in multi)
    assert res.text == identity_render(doc, "en")


class NFDEcho(EchoBackend):
    def generate(self, requests):
        import unicodedata

        outs = super().generate(requests)
        for o in outs:
            o.text = unicodedata.normalize("NFD", o.text).replace(" ", " ​", 1)
        return outs


@pytest.mark.parametrize("fail_mode", [None, "merge_lines"])
def test_model_output_is_nfc_normalised(fail_mode):
    import unicodedata

    doc = vi_doc()
    tr = Translator(echo_cfg(backend={"fail_mode": fail_mode}, chunking={"budget_tokens": {"vi2en": 30}}))
    tr.backend = NFDEcho(tr.cfg)
    res = tr.translate_text(doc, "vi")
    assert res.text == unicodedata.normalize("NFC", res.text) and "​" not in res.text
    assert res.text == identity_render(doc, "vi")
    assert all(u["hyp"] == unicodedata.normalize("NFC", u["hyp"]) for u in res.units)


def test_over_budget_unit_pieces_are_merged_back():
    long_para = " ".join(f"Sentence number {i} explains one more condition of the garden rule." for i in range(12))
    doc = "\n".join(_wrap(long_para, width=70)) + "\n\n" + "\n".join(_wrap(EN_PARAS[1])) + "\n"
    res = Translator(echo_cfg(chunking={"budget_tokens": {"en2vi": 25}})).translate_text(doc, "en")
    assert res.stats["n_pieces"] >= 4 and res.stats["max_chunk_tokens"] <= 25
    long_unit = next(u for u in res.units if u["pieces"])
    assert long_unit["hyp"] == long_unit["src"] and long_unit["chunk"] == 0
    assert res.text == identity_render(doc, "en")


def test_all_documents_go_to_the_backend_in_one_length_sorted_call():
    tr = Translator(echo_cfg(chunking={"budget_tokens": {"en2vi": 30, "vi2en": 30}}))
    rec = RecordingEcho(tr.cfg)
    tr.backend = rec
    out = tr.translate_documents([("a", en_doc(), "en"), ("b", vi_doc(), None), ("c", en_doc(), "en")])
    assert [r.doc_id for r in out] == ["a", "b", "c"] and [r.direction for r in out] == ["en2vi", "vi2en", "en2vi"]
    assert len(rec.calls) == 1
    lens = [len(r.prompt_ids) for r in rec.calls[0]]
    assert lens == sorted(lens, reverse=True)
    assert len(rec.calls[0]) == sum(r.stats["n_chunks"] for r in out)


def test_prev_source_context_uses_the_contextual_template():
    doc = en_doc()
    tr = Translator(echo_cfg(prompt={"context_mode": "prev_source"}, chunking={"budget_tokens": {"en2vi": 30}}))
    rec = RecordingEcho(tr.cfg)
    tr.backend = rec
    tr.translate_text(doc, "en")
    chunks = plan_chunks(translation_units(parse_document(doc, "en")), lambda t: len(t.split()), 30)
    by_src = {r.src_text: r.prompt_text for r in rec.calls[0]}
    first = by_src[chunks[0].text]
    assert "Translate the following segment into Vietnamese" in first and "参考上面的信息" not in first
    second = by_src[chunks[1].text]
    assert second.index(chunks[0].text) < second.index("参考上面的信息") < second.index(chunks[1].text)
    assert first.startswith("<｜hy_begin▁of▁sentence｜><｜hy_User｜>") and first.endswith("<｜hy_Assistant｜>")


def test_empty_document():
    res = Translator(echo_cfg()).translate_text("", "en")
    assert res.stats["n_units"] == 0 and res.stats["n_chunks"] == 0 and res.text.strip() == ""
    assert Translator(echo_cfg()).translate_documents([]) == []


def test_script_single_file_mode(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text(vi_doc(), encoding="utf-8")
    out = tmp_path / "out" / "hyp.txt"
    res = subprocess.run([sys.executable, str(SCRIPT), "--config", "echo_smoke", "--input-file", str(src),
                          "--output-file", str(out)], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    assert "vi2en (langid)" in res.stdout and "words/s" in res.stdout
    assert out.read_text(encoding="utf-8") == identity_render(vi_doc(), "vi")
    assert subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, timeout=60).returncode == 0


RUNNABLE = sorted(p.stem for p in INFER_CONFIG_DIR.glob("*.yaml") if not p.name.startswith("_"))


def test_expected_configs_exist():
    assert {"echo_smoke", "hymt15_hf_greedy", "hymt15_vllm_greedy", "hymt2_hf_greedy", "hymt2_vllm_greedy",
            "hymt15_gptq_vllm", "hymt15_hf_cardsampling"} <= set(RUNNABLE)


@pytest.mark.parametrize("name", RUNNABLE)
def test_every_config_resolves_with_pinned_registry_revision(name):
    cfg, path = load_infer_config(name)
    assert path.parent == INFER_CONFIG_DIR and cfg["config_name"] == name
    assert cfg["chunking"]["budget_tokens"] == {"en2vi": 1000, "vi2en": 1600}
    assert cfg["render"]["blank_between_paragraphs"] == "source"
    m = cfg["model"]
    if m.get("repo_id"):
        assert m["revision"] == registry_revision(m["repo_id"])
    b = cfg["backend"]
    if b["name"] != "echo":
        decoding_params(cfg["generation"])
    if b["name"] == "vllm":
        assert b["gpu_memory_utilization"] == 0.6 and b["generation_config"] == "vllm"
        assert b["dtype"] == "bfloat16" and b["max_model_len"] == 8192
        if b["enforce_eager"] is not True:
            assert b["hf_overrides"] == {"rope_parameters": {"rope_type": "default", "rope_theta": 11158839.92507748}}


def test_specific_config_values():
    greedy, _ = load_infer_config("hymt15_hf_greedy")
    assert greedy["generation"]["do_sample"] is False and greedy["generation"]["repetition_penalty"] == 1.0
    card, _ = load_infer_config("hymt15_hf_cardsampling")
    g = card["generation"]
    assert (g["do_sample"], g["temperature"], g["top_k"], g["top_p"], g["repetition_penalty"]) == (True, 0.7, 20, 0.6, 1.05)
    gptq, _ = load_infer_config("hymt15_gptq_vllm")
    assert gptq["backend"]["quantization"] == "gptq_marlin" and gptq["model"]["repo_id"].endswith("GPTQ-Int4")
    hymt2, _ = load_infer_config("hymt2_vllm_greedy")
    assert hymt2["model"]["revision"] == "9a341cd1b679d3efd23b46e847b01745a71ed792"
    assert hymt2["backend"]["enforce_eager"] is True
    graphs, _ = load_infer_config("hymt2_vllm_graphs")
    assert graphs["backend"]["enforce_eager"] is False and graphs["model"] == hymt2["model"]
    assert {k: v for k, v in graphs.items() if k not in ("backend", "config_name")} == \
        {k: v for k, v in hymt2.items() if k not in ("backend", "config_name")}


def test_overrides_and_validation():
    assert parse_set_overrides(["generation.do_sample=true", "backend.batch_size=4", "prompt.template=hymt2_default"]) == {
        "generation": {"do_sample": True}, "backend": {"batch_size": 4}, "prompt": {"template": "hymt2_default"}}
    cfg, _ = load_infer_config("hymt15_hf_greedy", parse_set_overrides(["backend.batch_size=2"]))
    assert cfg["backend"]["batch_size"] == 2 and cfg["generation"]["do_sample"] is False
    with pytest.raises(ValueError):
        load_infer_config("hymt15_hf_greedy", {"model": {"revision": None}})
    with pytest.raises(ValueError):
        load_infer_config("echo_smoke", {"prompt": {"context_mode": "whole_doc"}})
    for bad in ({"template": "terminology"}, {"template": "contextual"}, {"template": "nope"}, {"lang_names": "de"},
                {"mode": "chat"}):
        with pytest.raises(ValueError):
            load_infer_config("echo_smoke", {"prompt": bad})
    with pytest.raises(ValueError):
        parse_set_overrides(["no_equals_sign"])
    with pytest.raises(FileNotFoundError):
        load_infer_config("does_not_exist")


def test_decoding_params_are_explicit():
    with pytest.raises(ValueError):
        decoding_params({"do_sample": False})
    with pytest.raises(ValueError):
        decoding_params({"do_sample": True, "repetition_penalty": 1.0, "temperature": 0.7})
    assert decoding_params({"do_sample": False, "repetition_penalty": 1.0}) == {
        "do_sample": False, "repetition_penalty": 1.0, "seed": 0}


def test_max_new_tokens_for():
    gen = {"max_new_tokens_ratio": {"en2vi": 2.7, "vi2en": 1.3}, "max_new_tokens_margin": 64, "max_new_tokens_cap": 4096}
    assert max_new_tokens_for(1000, "en2vi", gen) == 2764
    assert max_new_tokens_for(1600, "vi2en", gen) == 2144
    assert max_new_tokens_for(5000, "en2vi", gen) == 4096
    with pytest.raises(ValueError):
        max_new_tokens_for(10, "en2vi", {})


def test_hf_batches_respect_size_and_token_budget():
    cfg, _ = load_infer_config("hymt15_hf_greedy", {"backend": {"batch_size": 2, "max_batch_tokens": 100}})
    be = HFBackend(cfg, tokenizer=None, model_path="unused", stop_ids=[120020])
    reqs = [GenRequest([0] * n, "", 10) for n in (40, 30, 10, 5, 80)]
    order = sorted(range(len(reqs)), key=lambda i: len(reqs[i].prompt_ids), reverse=True)
    assert be._batches(order, reqs) == [[4], [0, 1], [2, 3]]


@pytest.fixture(scope="module")
def tiny_hy_model(tmp_path_factory):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    os.environ.setdefault("HF_HOME", str(paths.load_config("paths")["hf_cache_default"]))
    hub = pytest.importorskip("huggingface_hub")
    try:
        snap = hub.snapshot_download("tencent/HY-MT1.5-1.8B", revision="dbad03788f49709801014c95d481a514c272ca52",
                                     local_files_only=True)
    except Exception as exc:
        pytest.skip(f"HY-MT1.5 snapshot not cached: {exc}")
    config = transformers.AutoConfig.from_pretrained(snap)
    for k, v in {"num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
                 "num_key_value_heads": 2, "head_dim": 16, "attention_head_dim": 16}.items():
        setattr(config, k, v)
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config)
    out = tmp_path_factory.mktemp("tiny_hy")
    model.save_pretrained(out)
    transformers.AutoTokenizer.from_pretrained(snap).save_pretrained(out)
    (out / "generation_config.json").write_bytes((Path(snap) / "generation_config.json").read_bytes())
    return out


def _tiny_cfg(path: Path, **gen) -> dict:
    cfg, _ = load_infer_config("hymt15_hf_greedy", {
        "model": {"local_path": str(path), "repo_id": None, "revision": None},
        "backend": {"device": "cpu", "batch_size": 3, "max_batch_tokens": 100000},
        "chunking": {"budget_tokens": {"en2vi": 40, "vi2en": 40}},
        "generation": {"max_new_tokens_ratio": {"en2vi": 0.1, "vi2en": 0.1}, "max_new_tokens_margin": 3,
                       "max_new_tokens_cap": 8, **gen}})
    return cfg


def test_hf_backend_tiny_model_end_to_end(tiny_hy_model):
    tr = Translator(_tiny_cfg(tiny_hy_model))
    assert tr.backend.dtype_name == "float32" and tr.backend.device == "cpu"
    res = tr.translate_text(en_doc(), "en")
    st = res.stats
    assert st["n_requests"] >= st["n_chunks"] and st["render_missing"] == 0
    assert len(res.units) == st["n_units"] and all(isinstance(u["hyp"], str) for u in res.units)
    assert 0 < st["out_tokens"] <= 11 * st["n_requests"]
    info = tr.info()
    assert info["decoding"] == {"do_sample": False, "repetition_penalty": 1.0, "seed": 0}
    assert info["stop_token_ids"] == [120020]


def test_hf_left_padding_matches_unbatched_greedy(tiny_hy_model):
    from LegalMT import prompts as P

    tr = Translator(_tiny_cfg(tiny_hy_model))
    texts = ["Short line.", "A somewhat longer line with several more words in it.", "Mid length line here."]
    reqs = [GenRequest(P.encode(tr.tokenizer, P.build_prompt_text(tr.tokenizer, "hymt",
                                                                 P.render_user_prompt("hymt", t, "vi"))), t, 6)
            for t in texts]
    batched = tr.backend.generate(reqs)
    single = [tr.backend.generate([r])[0] for r in reqs]
    assert all(o.token_ids for o in batched)
    assert [o.token_ids for o in batched] == [o.token_ids for o in single]
    assert [o.text for o in batched] == [o.text for o in single]
    assert all(o.n_in == len(r.prompt_ids) for o, r in zip(batched, reqs))


def test_hf_greedy_ignores_the_models_sampling_generation_config(tiny_hy_model):
    import torch

    from LegalMT import prompts as P

    tr = Translator(_tiny_cfg(tiny_hy_model))
    assert tr.backend.model.generation_config.do_sample is True
    req = GenRequest(P.encode(tr.tokenizer, P.build_prompt_text(tr.tokenizer, "hymt", P.render_user_prompt(
        "hymt", "One more synthetic line to translate.", "vi"))), "", 8)
    runs = []
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        runs.append(tr.backend.generate([req])[0].token_ids)
    assert runs[0] and runs[0] == runs[1] == runs[2]
    with torch.inference_mode():
        logits = tr.backend.model(torch.tensor([req.prompt_ids])).logits[0, -1]
    assert runs[0][0] == int(logits.argmax())


def test_hf_sampling_path_runs(tiny_hy_model):
    cfg = _tiny_cfg(tiny_hy_model, do_sample=True, temperature=0.7, top_k=20, top_p=0.6, repetition_penalty=1.05)
    tr = Translator(cfg)
    a = tr.translate_text(vi_doc(), "vi").text
    b = tr.translate_text(vi_doc(), "vi").text
    assert a == b


def test_vllm_sampling_params_are_explicit():
    pytest.importorskip("vllm")
    from LegalMT.translate.backends import VLLMBackend

    cfg, _ = load_infer_config("hymt15_vllm_greedy")
    sp = VLLMBackend(cfg, None, "unused", [120020])._sampling(100)
    assert (sp.temperature, sp.repetition_penalty, sp.max_tokens, sp.stop_token_ids, sp.skip_special_tokens) == (
        0.0, 1.0, 100, [120020], True)
    card, _ = load_infer_config("hymt15_vllm_greedy", {"generation": {
        "do_sample": True, "temperature": 0.7, "top_p": 0.6, "top_k": 20, "repetition_penalty": 1.05, "seed": 3}})
    sp = VLLMBackend(card, None, "unused", [120020])._sampling(50)
    assert (sp.temperature, sp.top_p, sp.top_k, sp.repetition_penalty, sp.seed) == (0.7, 0.6, 20, 1.05, 3)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"legalmt_{name}", REPO_ROOT / "scripts" / "LegalMT" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def artifacts_root(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    monkeypatch.setattr(paths, "ARTIFACTS_ROOT", root)
    return root


@pytest.mark.data
def test_translate_script_echo_dev_mini_and_evaluate(artifacts_root, capsys):
    tmod = _load_script("translate")
    assert tmod.main(["--config", "echo_smoke", "--split", "dev_mini", "--limit", "2", "--evaluate",
                      "--n-bootstrap", "20", "--tag", "echo_test"]) == 0
    run_dir = next((artifacts_root / "runs").iterdir())
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["status"] == "done" and m["split"] == "dev_mini" and m["directions"] == ["en2vi", "vi2en"]
    assert m["subset"]["n_docs"] == 2 and m["subset"]["of"] == 12
    for block, keys in (("timing", ("load_seconds", "translate_seconds", "total_seconds")),
                        ("memory", ("peak_mib_nvidia_smi", "peak_mib_torch_allocated")),
                        ("throughput", ("words_per_second", "out_tokens_per_second"))):
        assert all(k in m[block] for k in keys)
    hyps = R.read_jsonl(run_dir / "hyps.jsonl")
    assert len(hyps) == 4 and {h["direction"] for h in hyps} == {"en2vi", "vi2en"}
    need = {"sample_id", "category", "direction", "src_lang", "tgt_lang", "hyp", "seconds", "n_units", "n_chunks",
            "in_tokens", "out_tokens", "fallback_units"}
    assert all(need <= set(h) for h in hyps)
    assert all((h["src_lang"], h["tgt_lang"]) == (("en", "vi") if h["direction"] == "en2vi" else ("vi", "en"))
               for h in hyps)
    units = R.read_jsonl(run_dir / "units.jsonl")
    assert len(units) == sum(h["n_units"] for h in hyps)
    assert {"sample_id", "direction", "uid", "element_idx", "cell_idx", "kind", "src", "hyp"} <= set(units[0])
    res = json.loads((run_dir / "results.json").read_text())
    assert set(res["directions"]) == {"en2vi", "vi2en"}
    assert all(d["n_docs"] == 2 and d["n_missing"] == 0 for d in res["directions"].values())
    assert res["subset"]["n_docs"] == 2
    assert not (artifacts_root / "results_index.jsonl").exists()
    ev = _load_script("evaluate")
    assert ev.main([str(run_dir), "--no-index", "--n-bootstrap", "20"]) == 0
    res2 = json.loads((run_dir / "results.json").read_text())
    assert res2["directions"]["en2vi"]["n_docs"] == 12 and res2["directions"]["en2vi"]["n_missing"] == 10
    capsys.readouterr()


@pytest.mark.data
def test_translate_script_resume_after_crash(artifacts_root, capsys):
    tmod = _load_script("translate")
    assert tmod.main(["--config", "echo_smoke", "--split", "dev_mini", "--limit", "2", "--docs-per-call", "1",
                      "--directions", "en2vi", "--tag", "resume_test"]) == 0
    run_dir = next((artifacts_root / "runs").iterdir())
    hyps_path = run_dir / "hyps.jsonl"
    lines = hyps_path.read_text(encoding="utf-8").split("\n")
    hyps_path.write_text(lines[0] + "\n" + lines[1][:50], encoding="utf-8")
    assert tmod.main(["--resume", str(run_dir)]) == 0
    hyps = R.read_jsonl(hyps_path)
    assert sorted(h["sample_id"] for h in hyps) == sorted(json.loads(ln)["sample_id"] for ln in lines[:2])
    units = R.read_jsonl(run_dir / "units.jsonl")
    assert len(units) == sum(h["n_units"] for h in hyps)
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["status"] == "done" and len(m["resumed"]) == 1 and m["resumed"][0]["n_done"] == 1
    capsys.readouterr()


def _first_run(tmod, artifacts_root, tag: str) -> Path:
    assert tmod.main(["--config", "echo_smoke", "--split", "dev_mini", "--limit", "2", "--docs-per-call", "1",
                      "--directions", "en2vi", "--tag", tag]) == 0
    return next(p for p in (artifacts_root / "runs").iterdir() if p.name.endswith(tag))


@pytest.mark.data
def test_resume_after_exception_merges_sessions(artifacts_root, monkeypatch, capsys):
    tmod = _load_script("translate")

    class CrashOnSecondCall(tmod.Translator):
        calls = 0

        def translate_documents(self, docs):
            type(self).calls += 1
            if type(self).calls == 2:
                raise RuntimeError("simulated crash")
            return super().translate_documents(docs)

    monkeypatch.setattr(tmod, "Translator", CrashOnSecondCall)
    with pytest.raises(RuntimeError):
        tmod.main(["--config", "echo_smoke", "--split", "dev_mini", "--limit", "2", "--docs-per-call", "1",
                   "--directions", "en2vi", "--tag", "crash"])
    run_dir = next((artifacts_root / "runs").iterdir())
    m1 = json.loads((run_dir / "manifest.json").read_text())
    assert m1["status"] == "failed" and m1["translate_stats"]["en2vi"]["n_docs"] == 1
    monkeypatch.setattr(tmod, "Translator", Translator)
    assert tmod.main(["--resume", str(run_dir)]) == 0
    m = json.loads((run_dir / "manifest.json").read_text())
    hyps = R.read_jsonl(run_dir / "hyps.jsonl")
    assert m["status"] == "done" and len(hyps) == 2 and m["resumed"][0]["earlier_sessions_consistent"] is True
    st = m["translate_stats"]["en2vi"]
    assert st["n_docs"] == 2 and st["n_chunks"] == sum(h["n_chunks"] for h in hyps)
    assert m["throughput"]["src_words"] == st["src_words"] > m1["translate_stats"]["en2vi"]["src_words"]
    assert len(m["sessions"]) == 2 and m["translate_stats_partial"] is False
    for k in ("load_seconds", "translate_seconds", "total_seconds"):
        assert m["timing"][k] == pytest.approx(sum(s["timing"][k] for s in m["sessions"]), abs=0.01)
        assert m["timing"][k] >= m1["timing"][k]
    capsys.readouterr()


@pytest.mark.data
def test_resume_after_hard_kill_records_unknown_timing(artifacts_root, capsys):
    tmod = _load_script("translate")
    run_dir = _first_run(tmod, artifacts_root, "killed")
    hyps_path = run_dir / "hyps.jsonl"
    hyps_path.write_text(hyps_path.read_text(encoding="utf-8").split("\n")[0] + "\n", encoding="utf-8")
    m = json.loads((run_dir / "manifest.json").read_text())
    for k in ("timing", "throughput", "translate_stats", "memory", "status", "finished"):
        m.pop(k, None)
    (run_dir / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    assert tmod.main(["--resume", str(run_dir)]) == 0
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["status"] == "done" and m["resumed"][0]["earlier_sessions_consistent"] is False
    assert m["timing"]["incomplete"] is True and m["timing"]["translate_seconds"] is None
    assert m["throughput"]["words_per_second"] is None
    assert m["translate_stats_partial"] is True and m["translate_stats"]["en2vi"]["n_docs"] == 1
    assert len(R.read_jsonl(hyps_path)) == 2
    capsys.readouterr()


@pytest.mark.data
def test_resume_refuses_a_code_change(artifacts_root, capsys):
    tmod = _load_script("translate")
    run_dir = _first_run(tmod, artifacts_root, "codechange")
    m = json.loads((run_dir / "manifest.json").read_text())
    m["code_fingerprint"] = "0" * 64
    (run_dir / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(SystemExit):
        tmod.main(["--resume", str(run_dir)])
    assert tmod.main(["--resume", str(run_dir), "--allow-code-change"]) == 0
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["resumed"][-1]["code_fingerprint"] == R.code_fingerprint()
    capsys.readouterr()


def test_misspelled_backend_or_generation_keys_are_rejected():
    for over in ({"backend": {"max_num_seq": 32}}, {"backend": {"swap_space": 4}}, {"generation": {"min_p": 0.1}}):
        with pytest.raises(ValueError, match="unknown config keys"):
            load_infer_config("hymt15_vllm_greedy", over)
    with pytest.raises(ValueError, match="unknown config keys"):
        load_infer_config("hymt15_hf_greedy", {"backend": {"gpu_memory_utilization": 0.5}})


def test_vllm_engine_keys_are_engine_args_fields():
    pytest.importorskip("vllm")
    import dataclasses

    from vllm.engine.arg_utils import EngineArgs

    from LegalMT.translate.backends import VLLMBackend

    fields = {f.name for f in dataclasses.fields(EngineArgs)}
    passed = {"model", "tokenizer", "dtype", "seed", "generation_config", "trust_remote_code", "disable_log_stats",
              "enable_lora", "max_lora_rank", "max_loras"}
    assert (set(VLLMBackend.ENGINE_KEYS) | passed) - fields == set()


def test_vllm_backend_with_a_fake_engine(monkeypatch, capsys):
    vllm = pytest.importorskip("vllm")
    import types

    from LegalMT.translate import backends as B

    seen: dict = {}

    class FakeLLM:
        def __init__(self, **kw):
            seen["kw"], seen["env"] = kw, os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")
            self.model_config = types.SimpleNamespace(max_model_len=kw["max_model_len"])

        def generate(self, prompts, params, use_tqdm=False, lora_request=None):
            seen["prompts"], seen["params"], seen["lora"] = prompts, params, lora_request
            out = types.SimpleNamespace(token_ids=[5, 6, 120020], text="xy", finish_reason="stop")
            return [types.SimpleNamespace(outputs=[out]) for _ in prompts]

    monkeypatch.setattr(vllm, "LLM", FakeLLM)
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    monkeypatch.setattr(B, "gpu_free_fraction", lambda: (3000.0, 12000.0))
    cfg, _ = load_infer_config("hymt15_vllm_greedy", {"backend": {"max_model_len": 100}})
    be = B.VLLMBackend(cfg, tokenizer=None, model_path="/fake/snapshot", stop_ids=[120020])
    be.load()
    kw = seen["kw"]
    assert seen["env"] == "0"
    assert (kw["enforce_eager"], kw["gpu_memory_utilization"], kw["generation_config"], kw["dtype"], kw["seed"]) == (
        True, 0.6, "vllm", "bfloat16", 0)
    assert kw["max_model_len"] == 100 and "enable_lora" not in kw and "kv_cache_memory_bytes" not in kw
    assert "Lower backend.gpu_memory_utilization below 0.25" in capsys.readouterr().err
    ids = [120000, 120006, 11, 12, 120007]
    outs = be.generate([GenRequest(ids, "a", 500), GenRequest([1] * 99, "b", 10), GenRequest(ids, "c", 20)])
    assert len(seen["prompts"]) == 2 and seen["prompts"][0]["prompt_token_ids"] == ids
    assert [p.max_tokens for p in seen["params"]] == [95, 20] and seen["lora"] is None
    assert outs[1].text == "" and outs[1].finished is False and outs[1].n_out == 0
    assert outs[0].finished and outs[0].token_ids == [5, 6] and outs[0].n_out == 3
    lcfg, _ = load_infer_config("hymt15_vllm_greedy", {"lora_adapter": "/fake/adapter"})
    lb = B.VLLMBackend(lcfg, tokenizer=None, model_path="/fake/snapshot", stop_ids=[120020])
    lb.load()
    assert (seen["kw"]["enable_lora"], seen["kw"]["max_lora_rank"], seen["kw"]["max_loras"]) == (True, 64, 1)
    lb.generate([GenRequest(ids, "a", 5)])
    assert seen["lora"].lora_path == "/fake/adapter"


def test_hf_greedy_builds_no_penalty_or_sampling_processors(tiny_hy_model, monkeypatch):
    import transformers.generation.utils as GU

    from LegalMT import prompts as P

    made: list = []
    for name in ("RepetitionPenaltyLogitsProcessor", "TemperatureLogitsWarper", "TopKLogitsWarper", "TopPLogitsWarper"):
        base = getattr(GU, name)
        spy = type(name, (base,), {"__init__": lambda self, *a, _b=base, **k: (_b.__init__(self, *a, **k),
                                                                                 made.append(self))[0]})
        monkeypatch.setattr(GU, name, spy)
    tr = Translator(_tiny_cfg(tiny_hy_model))
    req = GenRequest(P.encode(tr.tokenizer, P.build_prompt_text(tr.tokenizer, "hymt", P.render_user_prompt(
        "hymt", "alpha beta alpha beta gamma", "vi"))), "", 6)
    assert tr.backend.generate([req])[0].token_ids
    assert made == []
    card = Translator(_tiny_cfg(tiny_hy_model, do_sample=True, temperature=0.7, top_k=20, top_p=0.6,
                                repetition_penalty=1.05))
    card.backend.generate([req])
    got = {type(p).__name__: p for p in made}
    assert got["RepetitionPenaltyLogitsProcessor"].penalty == 1.05
    assert got["TemperatureLogitsWarper"].temperature == 0.7
    assert got["TopKLogitsWarper"].top_k == 20 and got["TopPLogitsWarper"].top_p == 0.6


def test_static_rope_override_reproduces_hunyuan_dynamic_rope():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import copy
    import os

    from LegalMT import paths

    os.environ.setdefault("HF_HOME", str(paths.load_config("paths")["hf_cache_default"]))
    hub = pytest.importorskip("huggingface_hub")
    try:
        snap = hub.snapshot_download("tencent/Hy-MT2-1.8B", revision="9a341cd1b679d3efd23b46e847b01745a71ed792",
                                     local_files_only=True, allow_patterns=["config.json"])
    except Exception as exc:
        pytest.skip(f"Hy-MT2 config not cached: {exc}")
    from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import HunYuanDenseV1RotaryEmbedding

    cfg = transformers.AutoConfig.from_pretrained(snap)
    graphs, _ = load_infer_config("hymt2_vllm_graphs")
    static = copy.deepcopy(cfg)
    static.update(graphs["backend"]["hf_overrides"])
    a, b = HunYuanDenseV1RotaryEmbedding(cfg), HunYuanDenseV1RotaryEmbedding(static)
    assert a.rope_type == "dynamic" and b.rope_type == "default"
    x = torch.zeros(1, 1, 8, dtype=torch.bfloat16)
    pos = torch.arange(0, graphs["backend"]["max_model_len"]).unsqueeze(0)
    (ca, sa), (cb, sb) = a(x, pos), b(x, pos)
    assert torch.equal(ca, cb) and torch.equal(sa, sb)
