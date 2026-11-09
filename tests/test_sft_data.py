"""Tests for LegalMT.sft_data: packing, filters, role split, determinism and prompt equality with inference."""
from __future__ import annotations

import importlib.util
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

from LegalMT import paths
from LegalMT import sft_data as S
from LegalMT.chunking import JOINER
from LegalMT.layout import parse_document, translation_units
from LegalMT.prompts import HY_ASSISTANT, HY_BOS, HY_EOS, HY_NO8
from LegalMT.translate import Translator, load_infer_config

CHUNKING = {"budget_tokens": {"en2vi": 20, "vi2en": 30}, "cut_before_heading": True,
            "min_fill_before_heading_cut": 0.5, "max_unit_tokens": None, "max_units_per_chunk": None}
PROMPT = {"template": "default", "mode": "generation_prompt", "context_mode": "none",
          "context_template": "contextual", "lang_names": "en"}
WS = S.TokenCounter(None)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_legalmt_{name}", REPO_ROOT / "scripts" / "LegalMT" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def link(sid: str, order: int, n_en: int, n_vi: int | None = None, *, kind: str = "paragraph", **kw) -> dict:
    n_vi = n_en if n_vi is None else n_vi
    rec = {"pair_id": f"{sid}#{order:05d}", "sample_id": sid, "category": sid.split("/")[0], "order": order,
           "en": " ".join(f"w{order}e{k}" for k in range(n_en)),
           "vi": " ".join(f"từ{order}v{k}" for k in range(n_vi)),
           "en_units": [order], "vi_units": [order], "link": "1-1", "method": "zip", "anchor": None,
           "len_ratio": 1.0, "num_f1": None, "cost": 0.0, "kind_en": kind, "kind_vi": kind}
    rec.update(kw)
    return rec


def settings(**kw) -> S.BuildSettings:
    base = {"family": "hymt", "chunking": CHUNKING, "prompt": PROMPT}
    base.update(kw)
    return S.BuildSettings(**base)


def test_pack_equal_lines_budget_and_order():
    links = [link("Cat/1", k, 3 + k % 5, 4 + k % 3) for k in range(30)]
    for direction, src, tgt in (("en2vi", "en", "vi"), ("vi2en", "vi", "en")):
        packed, st = S.pack_links(links, direction, WS, CHUNKING)
        budget = CHUNKING["budget_tokens"][direction]
        assert st["links_used"] == 30 and not st["oversize_links"]
        assert [o for p in packed for o in p.orders] == list(range(30))
        for p in packed:
            assert len(p.src_lines) == len(p.tgt_lines)
            assert p.src_tokens <= budget
            assert p.src_tokens == sum(len(ln.split()) for ln in p.src_lines) + len(p.src_lines) - 1
            by_order = {r["order"]: r for r in links}
            assert p.src_lines == [by_order[o][src] for o in p.orders]
            assert p.tgt_lines == [by_order[o][tgt] for o in p.orders]
            assert JOINER not in "".join(p.src_lines + p.tgt_lines)


def test_pack_uses_order_not_element_index():
    links = [link("Cat/1", k, 2) for k in range(4)]
    for rec, elem in zip(links, (0, 2, 1, 3)):
        rec["en_units"] = rec["vi_units"] = [elem]
    packed, _ = S.pack_links(links, "en2vi", WS, CHUNKING)
    assert [o for p in packed for o in p.orders] == [0, 1, 2, 3]


def test_pack_cuts_before_heading_once_half_full():
    links = [link("Cat/1", k, 3) for k in range(3)] + [link("Cat/1", 3, 2, kind="heading")] + \
            [link("Cat/1", k, 3) for k in range(4, 6)]
    packed, _ = S.pack_links(links, "en2vi", WS, CHUNKING)
    assert packed[0].orders == [0, 1, 2] and packed[0].cut == "heading"
    assert packed[1].orders[0] == 3
    few = [link("Cat/1", 0, 3), link("Cat/1", 1, 2, kind="heading"), link("Cat/1", 2, 3)]
    assert [p.orders for p in S.pack_links(few, "en2vi", WS, CHUNKING)[0]] == [[0, 1, 2]]


def test_oversize_links_dropped_or_alone_never_split():
    links = [link("Cat/1", 0, 3), link("Cat/1", 1, 25), link("Cat/1", 2, 3)]
    packed, st = S.pack_links(links, "en2vi", WS, CHUNKING)
    assert st["oversize_links"] == 1 and [p.orders for p in packed] == [[0, 2]]
    packed, st = S.pack_links(links, "en2vi", WS, CHUNKING, oversize="alone")
    assert [p.orders for p in packed] == [[0], [1], [2]] and packed[1].cut == "oversize"
    capped = {**CHUNKING, "max_unit_tokens": 2}
    packed, st = S.pack_links(links, "en2vi", WS, capped)
    assert st["oversize_links"] == 3 and packed == []


def test_break_at_gaps():
    links = [link("Cat/1", k, 2) for k in (0, 1, 3, 4)]
    assert [p.orders for p in S.pack_links(links, "en2vi", WS, CHUNKING)[0]] == [[0, 1, 3, 4]]
    assert [p.orders for p in S.pack_links(links, "en2vi", WS, CHUNKING, break_at_gaps=True)[0]] == [[0, 1], [3, 4]]


def test_pack_rejects_bad_arguments():
    with pytest.raises(ValueError):
        S.pack_links([], "en2fr", WS, CHUNKING)
    with pytest.raises(ValueError):
        S.pack_links([], "en2vi", WS, CHUNKING, oversize="split")


def test_filters():
    links = [link("Cat/1", 0, 3), link("Cat/1", 1, 3, method="gc", link="2-1"),
             link("Cat/1", 2, 3, len_ratio=3.0), link("Cat/1", 3, 3, link="header", method="header", len_ratio=3.0),
             link("Cat/1", 4, 3, num_f1=0.2), link("Cat/1", 5, 3, en="   ")]
    kept, dropped = S.filter_links(links, None)
    assert [r["order"] for r in kept] == [0, 1, 2, 3, 4] and dropped == Counter({"empty_side": 1})
    kept, dropped = S.filter_links(links, {"methods": ["zip", "header"], "len_ratio": [0.5, 2.0], "min_num_f1": 0.34})
    assert [r["order"] for r in kept] == [0, 3]
    assert dropped == Counter({"method": 1, "len_ratio": 1, "num_f1": 1, "empty_side": 1})
    kept, _ = S.filter_links(links, {"links": ["1-1"]})
    assert [r["order"] for r in kept] == [0, 2, 4]
    with pytest.raises(ValueError):
        S.filter_links(links, {"lenratio": [0, 1]})
    with pytest.raises(ValueError):
        S.filter_links(links, {"len_ratio": [2.0, 1.0]})


def _docs() -> dict[str, list[dict]]:
    return {f"Cat{c}/{d}": [link(f"Cat{c}/{d}", k, 2 + (k * d) % 6, 3 + (k + c) % 4,
                                 kind="heading" if k % 9 == 0 else "paragraph") for k in range(25)]
            for c in range(2) for d in range(1, 4)}


def test_build_examples_invariants():
    exs, counts = S.build_examples(_docs(), settings())
    assert counts["docs"] == 6 and counts["links_in"] == 150
    assert {ex["direction"] for ex in exs} == {"en2vi", "vi2en"}
    for ex in exs:
        assert ex["prompt"].startswith(HY_BOS) and ex["prompt"].count(HY_BOS) == 1
        assert ex["prompt"].endswith(HY_ASSISTANT) and HY_NO8 not in ex["prompt"] + ex["completion"]
        assert ex["completion"].endswith(HY_EOS) and ex["completion"].count(HY_EOS) == 1
        body = ex["completion"][: -len(HY_EOS)]
        src_block = ex["prompt"].split("\n\n", 1)[1][: -len(HY_ASSISTANT)]
        assert len(body.split(JOINER)) == len(src_block.split(JOINER)) == ex["n_lines"] == len(ex["orders"])
        assert ex["src_tokens"] <= CHUNKING["budget_tokens"][ex["direction"]]
        assert ex["prompt_tokens"] == len(ex["prompt"].split())
        assert ex["completion_tokens"] == len(ex["completion"].split())
        assert not ex["context"]
    for d in ("en2vi", "vi2en"):
        used = Counter((ex["sample_id"], o) for ex in exs if ex["direction"] == d for o in ex["orders"])
        assert len(used) == 150 and set(used.values()) == {1}


def _long_en_doc() -> str:
    paras = []
    for a in range(1, 5):
        paras.append(f"Article {a}. Rules for garden number {a}")
        for p in range(4):
            paras.append(f"The keeper of garden {a} shall water the plants in row {p} every morning and "
                         f"record the amount of water used in the logbook kept at the gate.")
    return "\n\n".join(paras) + "\n"


@pytest.mark.parametrize("context_mode", ["none", "prev_source"])
def test_prompts_equal_the_translator_requests(context_mode):
    over = {"chunking": {"budget_tokens": {"en2vi": 60, "vi2en": 60}}, "prompt": {"context_mode": context_mode}}
    cfg, _ = load_infer_config("echo_smoke", over)
    tr = Translator(cfg)
    seen = []
    real_generate = tr.backend.generate

    def spy(reqs):
        seen.extend(reqs)
        return real_generate(reqs)

    tr.backend.generate = spy
    doc = _long_en_doc()
    tr.translate_documents([("Syn/1", doc, "en")])
    units = translation_units(parse_document(doc, "en"))
    links = [{"order": k, "en": u.text, "vi": f"dòng {k}", "kind_en": u.kind, "kind_vi": u.kind,
              "sample_id": "Syn/1", "category": "Syn", "link": "1-1", "method": "zip"} for k, u in enumerate(units)]
    s = settings(chunking=cfg["chunking"], prompt=cfg["prompt"], directions=("en2vi",),
                 context_p=1.0 if context_mode == "prev_source" else 0.0)
    exs, _ = S.build_examples({"Syn/1": links}, s)
    assert len(exs) > 2
    assert sorted(ex["prompt"] for ex in exs) == sorted(r.prompt_text for r in seen)
    assert any(ex["context"] for ex in exs) == (context_mode == "prev_source")


def test_prompts_equal_the_translator_requests_with_the_real_tokenizer():
    pytest.importorskip("transformers")
    over = {"chunking": {"budget_tokens": {"en2vi": 80, "vi2en": 80}}}
    try:
        cfg, _ = load_infer_config("echo_hytok_smoke", over)
        tr = Translator(cfg)
    except Exception as exc:
        pytest.skip(f"HY tokenizer not available: {exc}")
    seen = []
    real_generate = tr.backend.generate
    tr.backend.generate = lambda reqs: (seen.extend(reqs), real_generate(reqs))[1]
    en_doc = _long_en_doc()
    vi_doc = "\n\n".join(f"Điều {a}. Quy định về khu vườn số {a}\n\n" + "\n\n".join(
        f"Người giữ vườn {a} phải tưới cây ở hàng {p} mỗi sáng và ghi lượng nước đã dùng vào sổ đặt ở cổng." for p in range(4))
        for a in range(1, 5)) + "\n"
    counter = S.TokenCounter(tr.tokenizer)
    cuts: Counter = Counter()
    for doc, lang, direction in ((en_doc, "en", "en2vi"), (vi_doc, "vi", "vi2en")):
        seen.clear()
        tr.translate_documents([("Syn/1", doc, lang)])
        units = translation_units(parse_document(doc, lang))
        other = "vi" if lang == "en" else "en"
        links = [{"order": k, lang: u.text, other: f"line {k}", f"kind_{lang}": u.kind, f"kind_{other}": "paragraph",
                  "sample_id": "Syn/1", "category": "Syn", "link": "1-1", "method": "zip"} for k, u in enumerate(units)]
        s = settings(chunking=cfg["chunking"], prompt=cfg["prompt"], directions=(direction,))
        exs, _ = S.build_examples({"Syn/1": links}, s, tr.tokenizer, counter)
        assert len(exs) > 2
        cuts.update(ex["cut"] for ex in exs)
        assert sorted(ex["prompt"] for ex in exs) == sorted(r.prompt_text for r in seen)
        ids = {tuple(r.prompt_ids) for r in seen}
        assert all(tuple(tr._encode(ex["prompt"])) in ids for ex in exs)
    assert cuts["heading"] and cuts["budget"]


def test_context_probability():
    exs0, _ = S.build_examples(_docs(), settings(context_p=0.0))
    assert not any(ex["context"] for ex in exs0)
    exs1, _ = S.build_examples(_docs(), settings(context_p=1.0))
    first = [ex for ex in exs1 if ex["id"].endswith("|0000")]
    assert not any(ex["context"] for ex in first)
    assert all(ex["context"] for ex in exs1 if not ex["id"].endswith("|0000"))
    half, _ = S.build_examples(_docs(), settings(context_p=0.5))
    n = sum(ex["context"] for ex in half)
    assert 0 < n < sum(ex["context"] for ex in exs1)
    with pytest.raises(ValueError):
        settings(context_p=1.5)
    with pytest.raises(ValueError):
        S.build_examples(_docs(), settings(context_p=0.5, prompt={**PROMPT, "context_template": "default"}))


def test_determinism_and_order_independence():
    a, _ = S.build_examples(_docs(), settings(context_p=0.5, seed=7))
    b, _ = S.build_examples(dict(reversed(list(_docs().items()))), settings(context_p=0.5, seed=7))
    assert json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)
    one = {"Cat1/2": _docs()["Cat1/2"]}
    c, _ = S.build_examples(one, settings(context_p=0.5, seed=7))
    assert c == [ex for ex in a if ex["sample_id"] == "Cat1/2"]
    s1 = S.shuffle_examples(a, 1)
    assert s1 == S.shuffle_examples(list(reversed(a)), 1)
    assert [ex["id"] for ex in s1] != [ex["id"] for ex in S.shuffle_examples(a, 2)]
    assert sorted(ex["id"] for ex in s1) == sorted(ex["id"] for ex in a)


def test_max_total_tokens_and_cap():
    exs, counts = S.build_examples(_docs(), settings(packing={"max_total_tokens": 30}))
    assert all(ex["prompt_tokens"] + ex["completion_tokens"] <= 30 for ex in exs)
    assert sum(counts["per_direction"][d]["over_max_total_tokens"] for d in ("en2vi", "vi2en")) > 0
    exs, _ = S.build_examples(_docs(), settings())
    total = [ex["prompt_tokens"] + ex["completion_tokens"] for ex in exs]
    capped = S.cap_tokens(exs, total[0] + total[1] + total[2] - 1)
    assert capped == exs[:2]
    assert S.cap_tokens(exs, None) == exs
    with pytest.raises(ValueError):
        settings(packing={"max_tokens": 5})


def test_split_roles_is_leak_free():
    roles = S.split_roles(["A/2", "A/10", "B/1"], ["C/1", "A/3"])
    assert roles == {"train": ["A/2", "A/10", "B/1"], "dev": ["A/3", "C/1"]}
    assert S.split_roles(["A/2", "A/10"], ["C/1"], limit=1) == {"train": ["A/2"], "dev": ["C/1"]}
    with pytest.raises(ValueError):
        S.split_roles(["A/1", "B/1"], ["B/1"])
    docs = _docs()
    roles = S.split_roles(["Cat0/1", "Cat0/2", "Cat1/1", "Cat1/3"], ["Cat0/3", "Cat1/2"])
    built = {r: S.build_examples({s: docs[s] for s in ids}, settings())[0] for r, ids in roles.items()}
    assert {ex["sample_id"] for ex in built["train"]}.isdisjoint(roles["dev"])
    assert {ex["sample_id"] for ex in built["dev"]} == set(roles["dev"])


def test_final_roles_put_dev_and_public_in_train():
    roles = S.final_roles(["A/2", "A/10"], ["C/1"], ["P/1"])
    assert roles == {"train": ["A/2", "A/10", "C/1", "P/1"], "dev": ["C/1"]}
    assert S.final_roles(["A/2"], ["C/1"]) == {"train": ["A/2", "C/1"], "dev": ["C/1"]}
    with pytest.raises(ValueError, match="both"):
        S.final_roles(["A/1", "B/1"], ["B/1"])
    with pytest.raises(ValueError, match="public-test ids"):
        S.final_roles(["A/1"], ["C/1"], ["A/1"])
    assert S.validate_final(None) is None
    assert S.validate_final({}) == {"public_test": False}
    assert S.validate_final({"public_test": True}) == {"public_test": True}
    for bad in ([], {"public_test": "yes"}, {"dev": True}):
        with pytest.raises(ValueError):
            S.validate_final(bad)


def test_check_example_catches_broken_records():
    ex = S.build_examples(_docs(), settings())[0][0]
    S.check_example(ex, "hymt", HY_EOS)
    with pytest.raises(AssertionError):
        S.check_example({**ex, "completion": ex["completion"] + HY_NO8}, "hymt", HY_EOS)
    with pytest.raises(AssertionError):
        S.check_example({**ex, "n_lines": ex["n_lines"] + 1}, "hymt", HY_EOS)
    with pytest.raises(AssertionError):
        S.check_example({**ex, "prompt": HY_BOS + ex["prompt"]}, "hymt", HY_EOS)
    with pytest.raises(AssertionError):
        S.check_example({**ex, "completion": ex["completion"][: -len(HY_EOS)]}, "hymt", HY_EOS)


EN_DOC = """Article 1. Scope

This regulation applies to the small gardens of the example city.

Article 2. Duties

The board shall review each written request within 15 working days.
"""
VI_DOC = """Điều 1. Phạm vi

Quy chế này áp dụng cho các vườn hoa nhỏ của thành phố ví dụ.

Điều 2. Nhiệm vụ

Hội đồng phải xem xét mỗi yêu cầu bằng văn bản trong thời hạn 15 ngày làm việc.
"""


def test_attach_kinds_from_the_parser():
    en_units = translation_units(parse_document(EN_DOC, "en"))
    vi_units = translation_units(parse_document(VI_DOC, "vi"))
    assert len(en_units) == len(vi_units) == 4
    links = [{"order": k, "en": e.text, "vi": v.text, "en_uids": [e.uid], "vi_uids": [v.uid],
              "en_units": [e.element_idx], "vi_units": [v.element_idx]} for k, (e, v) in enumerate(zip(en_units, vi_units))]
    st = S.attach_kinds(links, EN_DOC, VI_DOC)
    assert st == {"missing_uid": 0, "no_uids": 0, "text_mismatch_en": 0, "text_mismatch_vi": 0}
    assert [r["kind_en"] for r in links] == [u.kind for u in en_units] == ["heading", "paragraph"] * 2
    assert [r["kind_vi"] for r in links] == ["heading", "paragraph"] * 2
    links[1]["en_uids"] = ["e99"]
    links[3]["vi"] = "khác"
    st = S.attach_kinds(links, EN_DOC, VI_DOC)
    assert st["missing_uid"] == 1 and st["text_mismatch_vi"] == 1


def test_links_without_uids_cannot_pass_the_stale_check_vacuously():
    en_units = translation_units(parse_document(EN_DOC, "en"))
    vi_units = translation_units(parse_document(VI_DOC, "vi"))
    links = [{"order": k, "en": "edited " + e.text, "vi": v.text, "en_units": [e.element_idx], "vi_units": [v.element_idx]}
             for k, (e, v) in enumerate(zip(en_units, vi_units))]
    st = S.attach_kinds(links, EN_DOC, VI_DOC)
    assert st["no_uids"] == 2 * len(links) and st["missing_uid"] == 0 and st["text_mismatch_en"] == 0
    assert [r["kind_en"] for r in links] == ["heading", "paragraph"] * 2
    reason = _load_script("build_sft").stale_alignment(st)
    assert reason and "no_uids" in reason


def test_read_aligned_groups_sorts_and_filters(tmp_path):
    p = tmp_path / "train.jsonl"
    recs = [link("A/1", 2, 2), link("B/1", 0, 2), link("A/1", 0, 2), link("A/1", 1, 2)]
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    docs = S.read_aligned(p)
    assert [r["order"] for r in docs["A/1"]] == [0, 1, 2] and len(docs["B/1"]) == 1
    assert list(S.read_aligned(p, keep_ids={"B/1"})) == ["B/1"]
    p.write_text(p.read_text(encoding="utf-8") + json.dumps(link("A/1", 1, 2)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        S.read_aligned(p)


class _FakeTok:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, texts, add_special_tokens=True):
        self.calls.append({"n": len(texts) if isinstance(texts, list) else 1, "special": add_special_tokens})
        if isinstance(texts, list):
            return {"input_ids": [[1] * len(t) for t in texts]}
        return {"input_ids": [1] * len(texts)}


def test_token_counter_batches_and_never_adds_special_tokens():
    tok = _FakeTok()
    c = S.TokenCounter(tok, batch_size=2)
    c.prime(["ab", "cde", "ab", "f"])
    assert c("cde") == 3 and c("xyzw") == 4
    assert [k["n"] for k in tok.calls] == [2, 1, 1] and not any(k["special"] for k in tok.calls)


def test_percentiles_summary_fingerprint_and_jsonl(tmp_path):
    assert S.percentiles([]) == {"p50": 0, "p90": 0, "p99": 0, "mean": 0.0, "max": 0}
    pc = S.percentiles(list(range(1, 101)))
    assert (pc["p50"], pc["p90"], pc["p99"], pc["max"], pc["mean"]) == (50, 90, 99, 100, 50.5)
    exs, _ = S.build_examples(_docs(), settings())
    sm = S.summarize(exs)
    assert sm["examples"] == len(exs) == sum(b["examples"] for b in sm["per_direction"].values())
    assert sm["total_tokens"] == sm["prompt_tokens"] + sm["completion_tokens"]
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    fp1 = S.tokenizer_fingerprint(tmp_path)
    (tmp_path / "chat_template.jinja").write_text("x", encoding="utf-8")
    fp2 = S.tokenizer_fingerprint(tmp_path)
    assert fp1["files"] == ["tokenizer.json"] and fp2["sha256"] != fp1["sha256"]
    assert S.write_jsonl(tmp_path / "x.jsonl", exs) == len(exs)
    assert [json.loads(ln) for ln in (tmp_path / "x.jsonl").read_text(encoding="utf-8").splitlines()] == exs


def test_direction_naming_sources_and_targets():
    docs = {"Cat0/1": [link("Cat0/1", k, 3, 4) for k in range(6)]}
    recs = {r["order"]: r for r in docs["Cat0/1"]}
    exs, _ = S.build_examples(docs, settings())
    assert {ex["direction"] for ex in exs} == set(S.DIRECTIONS)
    for ex in exs:
        src, tgt = S.SRC_TGT[ex["direction"]]
        assert (ex["src_lang"], ex["tgt_lang"]) == (src, tgt)
        name = "Vietnamese" if tgt == "vi" else "English"
        assert ex["prompt"].split("\n\n", 1)[0].endswith(f"into {name}, without additional explanation.")
        src_block = ex["prompt"].split("\n\n", 1)[1][: -len(HY_ASSISTANT)]
        assert src_block.split(JOINER) == [recs[o][src] for o in ex["orders"]]
        assert ex["completion"][: -len(HY_EOS)].split(JOINER) == [recs[o][tgt] for o in ex["orders"]]


def test_line_breaks_inside_a_link_never_change_line_counts():
    links = [link("Cat/1", 0, 3), link("Cat/1", 1, 3, en="first half\r\nsecond half", vi="nửa đầu\nnửa sau"),
             link("Cat/1", 2, 3)]
    for direction in S.DIRECTIONS:
        packed, _ = S.pack_links(links, direction, WS, CHUNKING)
        assert [len(p.src_lines) for p in packed] == [len(p.tgt_lines) for p in packed] == [3]
        assert packed[0].src_lines[1] in ("first half second half", "nửa đầu nửa sau")
    exs, _ = S.build_examples({"Cat/1": links}, settings())
    for ex in exs:
        body = ex["completion"][: -len(HY_EOS)]
        assert len(body.split(JOINER)) == len(body.splitlines()) == ex["n_lines"] == 3


def test_nfd_aligned_text_is_flagged_and_refused():
    assert all(unicodedata.is_normalized("NFC", u.text)
               for u in translation_units(parse_document(unicodedata.normalize("NFD", VI_DOC), "vi")))
    en_units = translation_units(parse_document(EN_DOC, "en"))
    vi_units = translation_units(parse_document(VI_DOC, "vi"))
    links = [{"order": k, "sample_id": "Cat/1", "category": "Cat", "link": "1-1", "method": "zip",
              "en": e.text, "vi": unicodedata.normalize("NFD", v.text), "en_uids": [e.uid], "vi_uids": [v.uid],
              "en_units": [e.element_idx], "vi_units": [v.element_idx]} for k, (e, v) in enumerate(zip(en_units, vi_units))]
    st = S.attach_kinds(links, EN_DOC, VI_DOC)
    assert st["missing_uid"] == 0 and st["text_mismatch_vi"] == len(links) and st["text_mismatch_en"] == 0
    build_sft = _load_script("build_sft")
    assert build_sft.stale_alignment(st) and "text_mismatch_vi" in build_sft.stale_alignment(st)
    assert build_sft.stale_alignment({"missing_uid": 0, "text_mismatch_en": 0, "text_mismatch_vi": 0}) is None
    with pytest.raises(AssertionError, match="NFC"):
        S.build_examples({"Cat/1": links}, settings())
    ok = [{**r, "vi": unicodedata.normalize("NFC", r["vi"])} for r in links]
    exs, _ = S.build_examples({"Cat/1": ok}, settings())
    assert exs and all(unicodedata.is_normalized("NFC", ex["prompt"] + ex["completion"]) for ex in exs)


def test_empty_inputs():
    assert S.build_examples({}, settings()) == ([], {"docs": 0, "links_in": 0, "links_dropped": {},
                                                     "per_direction": {"en2vi": {}, "vi2en": {}}})
    blank = {"Cat/1": [link("Cat/1", k, 2, en=" ") for k in range(3)]}
    exs, counts = S.build_examples(blank, settings())
    assert exs == [] and counts["links_dropped"] == {"empty_side": 3}
    assert S.pack_links([], "en2vi", WS, CHUNKING) == ([], Counter())
    sm = S.summarize([])
    assert sm["examples"] == 0 and sm["per_direction"] == {} and sm["total_tokens_pct"]["max"] == 0
    assert S.cap_tokens([], 10) == [] and S.shuffle_examples([], 1) == []


@pytest.mark.data
@pytest.mark.parametrize("name", ["sft_v1", "sft_v1_final"])
def test_real_build_is_leak_free_and_parallel(name):
    out = paths.artifacts("sft", name)
    if not (out / "manifest.json").is_file():
        pytest.skip(f"no SFT build at {out}")
    from LegalMT.splits import load_split_spec

    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    spec = load_split_spec(man["inputs"]["split_version"])
    budgets = man["infer_config"]["resolved"]["chunking"]["budget_tokens"]
    allowed = {"train": set(spec.train_fit_ids), "dev": set(spec.dev_ids)}
    final = man.get("final")
    if final:
        allowed["train"] |= allowed["dev"]
        if final["public_test"]:
            from LegalMT.data import load_split

            allowed["train"] |= set(load_split("public_test").df["sample_id"])
    for role in ("train", "dev"):
        n = 0
        with (out / f"{role}.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                ex = json.loads(line)
                n += 1
                assert ex["sample_id"] in allowed[role]
                assert ex["src_tokens"] <= budgets[ex["direction"]]
                assert ex["completion"].count(JOINER) + 1 == ex["n_lines"] == len(ex["orders"])
                assert ex["prompt"].startswith(HY_BOS) and ex["prompt"].count(HY_BOS) == 1
                assert ex["completion"].endswith(HY_EOS) and HY_NO8 not in ex["completion"]
        assert n > 0
    assert final or allowed["train"].isdisjoint(allowed["dev"])


@pytest.mark.data
@pytest.mark.parametrize("name", ["sft_v1", "sft_v1_final"])
def test_real_build_reconstructs_from_the_aligned_links(name):
    from LegalMT.chunking import one_line
    from LegalMT.splits import file_sha256

    out = paths.artifacts("sft", name)
    if not (out / "manifest.json").is_file():
        pytest.skip(f"no SFT build at {out}")
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    aligned_path = Path(man["inputs"]["aligned"])
    if not aligned_path.is_file():
        pytest.skip(f"aligned input {aligned_path} is gone")
    if file_sha256(aligned_path) != man["inputs"]["aligned_sha256"]:
        pytest.skip(f"the aligned file changed after the SFT build: rebuild {name} (scripts/LegalMT/build_sft.py --force)")
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    prompt_cfg = man["infer_config"]["resolved"]["prompt"]
    aligned = S.read_aligned(aligned_path)
    if man["inputs"].get("aligned_public_test"):
        aligned.update(S.read_aligned(Path(man["inputs"]["aligned_public_test"])))
    for role in ("train", "dev"):
        used = {d: Counter() for d in S.DIRECTIONS}
        docs: set[str] = set()
        with (out / f"{role}.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                ex = json.loads(line)
                src, tgt = S.SRC_TGT[ex["direction"]]
                recs = {r["order"]: r for r in aligned[ex["sample_id"]]}
                src_lines = [one_line(recs[o][src]) for o in ex["orders"]]
                assert ex["completion"][: -len(HY_EOS)].split(JOINER) == [one_line(recs[o][tgt]) for o in ex["orders"]]
                assert ex["prompt"] == S.render_prompt(None, "hymt", JOINER.join(src_lines), tgt, prompt_cfg)
                assert ex["orders"] == sorted(ex["orders"])
                used[ex["direction"]].update((ex["sample_id"], o) for o in ex["orders"])
                docs.add(ex["sample_id"])
        links = {(s, r["order"]) for s in docs for r in aligned[s]}
        build = stats["roles"][role]["build"]["per_direction"]
        for d in S.DIRECTIONS:
            assert set(used[d].values()) <= {1}
            assert len(links) - len(used[d]) == build[d].get("oversize_links", 0)
