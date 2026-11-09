"""Tests for the extra sentence-level sources of the SFT build."""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

from LegalMT import paths
from LegalMT import sft_data as S
from LegalMT.chunking import JOINER
from LegalMT.prompts import HY_EOS

CHUNKING = {"budget_tokens": {"en2vi": 60, "vi2en": 90}, "cut_before_heading": True,
            "min_fill_before_heading_cut": 0.5, "max_unit_tokens": None, "max_units_per_chunk": None}
PROMPT = {"template": "default", "mode": "generation_prompt", "context_mode": "none",
          "context_template": "contextual", "lang_names": "en"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_legalmt_{name}", REPO_ROOT / "scripts" / "LegalMT" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def pairs(n: int, *, seed: int = 0, lo: int = 3, hi: int = 9) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for k in range(n):
        ne, nv = rng.randint(lo, hi), rng.randint(lo, hi)
        out.append({"id": f"syn:{k:06d}", "en": " ".join(f"e{k}w{j}" for j in range(ne)),
                    "vi": " ".join(f"từ{k}v{j}" for j in range(nv))})
    return out


def settings(**kw) -> S.BuildSettings:
    base = {"family": "hymt", "chunking": CHUNKING, "prompt": PROMPT,
            "filters": {"methods": ["zip", "gc", "header"]},
            "packing": {"max_total_tokens": 400}, "seed": 7}
    base.update(kw)
    return S.BuildSettings(**base)


def build(prs, **kw):
    args = {"name": "syn", "seed": 11, "max_tokens": None, "max_lines": 40, "block_pairs": 50}
    args.update(kw)
    return S.build_extra_examples(prs, settings(), **args)


def test_extra_packing_lines_budgets_both_directions_and_source():
    prs = pairs(300)
    exs, st = build(prs)
    by_id = {p["id"]: p for p in prs}
    stream = S.shuffle_pairs(prs, 11)
    assert {ex["direction"] for ex in exs} == set(S.DIRECTIONS)
    used = {d: [] for d in S.DIRECTIONS}
    suffix = S.completion_suffix(None, "hymt")
    for ex in exs:
        assert ex["source"] == "syn" and ex["sample_id"].startswith("extra:syn/")
        assert ex["src_tokens"] <= CHUNKING["budget_tokens"][ex["direction"]]
        assert ex["n_lines"] == len(ex["orders"]) <= 40
        assert ex["prompt_tokens"] + ex["completion_tokens"] <= 400
        assert ex["context"] is False
        src, tgt = S.SRC_TGT[ex["direction"]]
        body = ex["completion"][: -len(HY_EOS)].split(JOINER)
        assert body == [stream[o][tgt] for o in ex["orders"]]
        assert ex["prompt"] == S.render_prompt(None, "hymt", JOINER.join(stream[o][src] for o in ex["orders"]),
                                               tgt, PROMPT)
        S.check_example(ex, "hymt", suffix)
        used[ex["direction"]].extend(ex["orders"])
    for d in S.DIRECTIONS:
        assert sorted(used[d]) == list(range(len(prs)))
    assert st["pairs_taken"] == 300 and st["blocks_taken"] == 6
    assert {st["per_direction"][d]["links_used"] for d in S.DIRECTIONS} == {300}
    assert set(by_id) == {stream[k]["id"] for k in range(300)}


def test_extra_line_cap_of_40_binds():
    prs = pairs(200, lo=1, hi=1)
    chunking = {**CHUNKING, "budget_tokens": {"en2vi": 200, "vi2en": 200}}
    exs, _ = S.build_extra_examples(prs, settings(chunking=chunking), name="syn", seed=1, max_tokens=None,
                                    max_lines=40, block_pairs=200)
    lines = Counter(ex["n_lines"] for ex in exs)
    assert max(lines) == 40 and lines[40] == 10
    assert sum(ex["cut"] == "max_units" for ex in exs) >= 8


def test_extra_drops_empty_and_oversize_pairs():
    prs = pairs(40)
    prs[3]["vi"] = "   "
    prs[5]["en"] = " ".join(["long"] * 80)
    exs, st = build(prs, block_pairs=40)
    assert st["links_dropped"] == {"empty_side": 1}
    assert st["per_direction"]["en2vi"]["oversize_links"] == 1
    assert st["per_direction"]["en2vi"]["links_used"] == 38 and st["per_direction"]["vi2en"]["links_used"] == 39
    assert all("long long" not in ex["prompt"] for ex in exs if ex["direction"] == "en2vi")


def test_extra_token_cap_takes_whole_blocks():
    prs = pairs(400)
    full, _ = build(prs)
    per_block = Counter()
    for ex in full:
        per_block[ex["sample_id"]] += ex["prompt_tokens"] + ex["completion_tokens"]
    blocks = sorted(per_block)
    cap = sum(per_block[b] for b in blocks[:3]) + per_block[blocks[3]] // 2
    exs, st = build(prs, max_tokens=cap)
    assert sorted({ex["sample_id"] for ex in exs}) == blocks[:3]
    assert st["total_tokens"] == sum(ex["prompt_tokens"] + ex["completion_tokens"] for ex in exs) <= cap
    assert st["pairs_taken"] == 150 and st["blocks_taken"] == 3 and st["stopped_by"] == "max_tokens"
    for d in S.DIRECTIONS:
        assert sorted(o for ex in exs if ex["direction"] == d for o in ex["orders"]) == list(range(150))


def test_extra_is_deterministic_and_input_order_independent():
    prs = pairs(250)
    a, sa = build(prs)
    rev = list(reversed(prs))
    b, sb = build(rev)
    assert json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False) and sa == sb
    c, _ = build(prs, seed=12)
    assert [ex["prompt"] for ex in c] != [ex["prompt"] for ex in a]
    mixed = S.shuffle_examples(a + [], 3)
    assert json.dumps(mixed) == json.dumps(S.shuffle_examples(list(reversed(a)), 3))


def test_pair_shuffle_is_subset_stable():
    full = pairs(500)
    removed = {p["id"] for p in full[::7]}
    sub = [p for p in full if p["id"] not in removed]
    a = [p["id"] for p in S.shuffle_pairs(sub, 5)]
    b = [p["id"] for p in S.shuffle_pairs(full, 5)]
    assert a == [i for i in b if i not in removed]
    assert b != sorted(b) and [p["id"] for p in S.shuffle_pairs(full, 6)] != b


def test_read_pairs_and_config_validation(tmp_path):
    p = tmp_path / "pairs.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pairs(5)), encoding="utf-8")
    assert [r["id"] for r in S.read_pairs(p)] == [f"syn:{k:06d}" for k in range(5)]
    p.write_text(json.dumps({"id": "x", "en": "a b c", "vi": "d e f"}) + "\n" +
                 json.dumps({"id": "x", "en": "g h i", "vi": "j k l"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate pair id"):
        S.read_pairs(p)
    p.write_text(json.dumps({"id": "x", "en": "a b c"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="needs"):
        S.read_pairs(p)
    ok = S.validate_extra([{"name": "m", "path": "/x.jsonl", "seed": 1, "max_tokens": 10}])
    assert ok == [{"name": "m", "path": "/x.jsonl", "seed": 1, "max_tokens": 10, "max_lines": 40, "block_pairs": 1000}]
    assert S.validate_extra(None) == [] and S.validate_extra([]) == []
    for bad, msg in (([{"name": "m", "path": "p", "seed": 1, "typo": 1}], "unknown"),
                     ([{"name": "m", "path": "p"}], "seed"),
                     ([{"name": "m", "path": "p", "seed": 1}] * 2, "duplicate"),
                     ([{"name": S.IN_DOMAIN_SOURCE, "path": "p", "seed": 1}], "reserved"),
                     ([{"name": "m", "path": "p", "seed": 1, "max_lines": 0}], "max_lines"),
                     ({"name": "m"}, "list")):
        with pytest.raises(ValueError, match=msg):
            S.validate_extra(bad)


def test_summarize_without_extra_is_unchanged_and_counts_sources_with_it():
    exs, _ = build(pairs(120))
    in_domain = [{k: v for k, v in ex.items() if k != "source"} for ex in exs[:10]]
    plain = S.summarize(in_domain)
    assert "per_source" not in plain
    mixed = S.summarize(in_domain + exs[10:])
    assert set(mixed["per_source"]) == {S.IN_DOMAIN_SOURCE, "syn"}
    assert mixed["per_source"][S.IN_DOMAIN_SOURCE]["examples"] == 10
    assert mixed["per_source"]["syn"]["examples"] == len(exs) - 10
    assert set(mixed["per_source"]["syn"]["per_direction"]) == set(S.DIRECTIONS)
    assert {k: v for k, v in mixed.items() if k != "per_source"}.keys() == plain.keys()


def test_build_sft_configs_load_with_and_without_extra():
    build_sft = _load_script("build_sft")
    cfg, _ = build_sft.load_sft_config("sft_v0")
    assert "extra" not in cfg and S.validate_extra(cfg.get("extra")) == []
    for name in ("sft_v0_mtetA", "sft_v0_mtetB"):
        cfg, _ = build_sft.load_sft_config(name)
        (ex,) = S.validate_extra(cfg["extra"])
        assert ex["name"] == "mtet_legal" and ex["max_tokens"] == 20_000_000 and ex["seed"] == 20260926
        assert ex["max_lines"] == 40
        base, _ = build_sft.load_sft_config("sft_v0")
        same = ("aligned_version", "split_version", "infer_config", "infer_overrides", "directions", "filters",
                "packing", "context", "seed", "max_train_tokens")
        assert {k: cfg.get(k) for k in same} == {k: base.get(k) for k in same}


def test_final_configs_equal_their_base_plus_final():
    build_sft = _load_script("build_sft")
    for base in ("sft_v1", "sft_v1_mtetA", "sft_v1_mtetQ"):
        b, _ = build_sft.load_sft_config(base)
        f, _ = build_sft.load_sft_config(f"{base}_final")
        assert f["name"] == f"{base}_final" and S.validate_final(f["final"]) == {"public_test": True}
        assert "final" not in b
        drop = {"name", "description", "final"}
        assert {k: v for k, v in f.items() if k not in drop} == {k: v for k, v in b.items() if k not in drop}
    a, _ = build_sft.load_sft_config("sft_v1_mtetA")
    q, _ = build_sft.load_sft_config("sft_v1_mtetQ")
    assert q["extra"][0]["path"].endswith("mtet_legal_clean_q75_pool300k.jsonl")
    q["extra"][0]["path"] = a["extra"][0]["path"]
    assert {k: v for k, v in q.items() if k not in {"name", "description"}} == \
        {k: v for k, v in a.items() if k not in {"name", "description"}}


def test_training_guard_accepts_extra_in_train_only():
    train_lora = _load_script("train_lora")
    from LegalMT.splits import load_split_spec

    spec = load_split_spec("v0")
    man = {"inputs": {"split_version": "v0"}, "extra": [{"name": "syn"}]}
    tr = [{"sample_id": spec.train_fit_ids[0]}, {"sample_id": "extra:syn/00000", "source": "syn"}]
    dv = [{"sample_id": spec.dev_ids[0]}]
    got = train_lora.sft_input_checks(man, tr, dv)
    assert got["split_check"] == "ok: split v0" and got["extra_train_examples"] == {"syn": 1}
    with pytest.raises(ValueError, match="dev file"):
        train_lora.sft_input_checks(man, tr, dv + [{"sample_id": "extra:syn/00001", "source": "syn"}])
    with pytest.raises(ValueError, match="not listed"):
        train_lora.sft_input_checks(man, tr + [{"sample_id": "extra:zz/00000", "source": "zz"}], dv)
    with pytest.raises(ValueError, match="split document"):
        train_lora.sft_input_checks(man, tr + [{"sample_id": spec.dev_ids[1], "source": "syn"}], dv)
    plain = train_lora.sft_input_checks({"inputs": {"split_version": "v0"}}, tr[:1], dv)
    assert plain == {"split_check": "ok: split v0", "aligned_current": None}


def test_training_guard_final_build_accepts_dev_in_train_only_when_marked():
    train_lora = _load_script("train_lora")
    from LegalMT.splits import load_split_spec

    spec = load_split_spec("v0")
    tr = [{"sample_id": spec.train_fit_ids[0]}, {"sample_id": spec.dev_ids[0]}]
    dv = [{"sample_id": spec.dev_ids[0]}]
    with pytest.raises(ValueError, match="dev documents"):
        train_lora.sft_input_checks({"inputs": {"split_version": "v0"}}, tr, dv)
    man = {"inputs": {"split_version": "v0"}, "final": {"public_test": False}}
    got = train_lora.sft_input_checks(man, tr, dv)
    assert got["split_check"].startswith("FINAL build on split v0: dev documents are in train")
    with pytest.raises(ValueError, match="neither split nor public-test"):
        train_lora.sft_input_checks(man, tr + [{"sample_id": "Nowhere/1"}], dv)
    with pytest.raises(ValueError, match="dev-file documents are not dev"):
        train_lora.sft_input_checks(man, tr, dv + [{"sample_id": spec.train_fit_ids[0]}])


@pytest.mark.data
def test_training_guard_final_build_with_public_test():
    train_lora = _load_script("train_lora")
    from LegalMT.data import load_split
    from LegalMT.splits import load_split_spec

    spec = load_split_spec("v0")
    pub = load_split("public_test").df["sample_id"].iloc[0]
    tr = [{"sample_id": spec.train_fit_ids[0]}, {"sample_id": spec.dev_ids[0]}, {"sample_id": pub}]
    dv = [{"sample_id": spec.dev_ids[0]}]
    with pytest.raises(ValueError, match="neither split nor public-test"):
        train_lora.sft_input_checks({"inputs": {"split_version": "v0"}, "final": {"public_test": False}}, tr, dv)
    got = train_lora.sft_input_checks({"inputs": {"split_version": "v0"}, "final": {"public_test": True}}, tr, dv)
    assert "dev and public-test documents are in train" in got["split_check"]


@pytest.mark.data
@pytest.mark.parametrize("name", ["sft_v1_mtetA", "sft_v1_mtetA_final"])
def test_real_mixed_build_keeps_extra_out_of_dev(name):
    out = paths.artifacts("sft", name)
    if not (out / "manifest.json").is_file():
        pytest.skip(f"no SFT build at {out}")
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    names = {e["name"] for e in man["extra"]}
    with (out / "dev.jsonl").open(encoding="utf-8") as fh:
        assert not any("source" in json.loads(line) for line in fh)
    n = Counter()
    with (out / "train.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            ex = json.loads(line)
            if "source" in ex:
                assert ex["source"] in names and ex["sample_id"].startswith(f"extra:{ex['source']}/")
                assert ex["n_lines"] <= 40
            n[S.example_source(ex)] += 1
    for e in man["extra"]:
        assert n[e["name"]] == e["examples"] > 0
