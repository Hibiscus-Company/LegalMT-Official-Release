"""Tests for LegalMT.splits: protocol pieces on synthetic inputs and frozen-split invariants."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from LegalMT import splits as S

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_COUNTS = {"Bat_dong_san": 78, "Dich_vu_phap_ly": 78, "Quyen_dan_su": 197, "So_huu_tri_tue": 92,
                "Thu_tuc_to_tung": 77, "Trach_nhiem_hinh_su": 119, "Vi_pham_hanh_chinh": 213}
V0_QUOTAS = {"Bat_dong_san": 4, "Dich_vu_phap_ly": 4, "Quyen_dan_su": 9, "So_huu_tri_tue": 4,
             "Thu_tuc_to_tung": 4, "Trach_nhiem_hinh_su": 5, "Vi_pham_hanh_chinh": 10}


def test_largest_remainder_quotas():
    assert S.largest_remainder_quotas(TRAIN_COUNTS, 40) == V0_QUOTAS
    mini = S.largest_remainder_quotas(TRAIN_COUNTS, 12, minimum=1)
    assert sum(mini.values()) == 12 and min(mini.values()) >= 1
    assert S.largest_remainder_quotas({"a": 100, "b": 1, "c": 1}, 5, minimum=1) == {"a": 3, "b": 1, "c": 1}
    assert S.largest_remainder_quotas({"a": 1, "b": 1}, 3) in ({"a": 2, "b": 1}, {"a": 1, "b": 2})
    with pytest.raises(ValueError):
        S.largest_remainder_quotas({"a": 1, "b": 1, "c": 1}, 2, minimum=1)


def test_natural_sort():
    assert S.sort_ids(["C/10", "C/9", "B/100", "C/1"]) == ["B/100", "C/1", "C/9", "C/10"]


def test_length_stratified_draw():
    ids = [f"K/{i}" for i in range(10)]
    words = {s: 100 * (10 - i) for i, s in enumerate(ids)}
    picks = S.length_stratified_draw(ids, words, 3, S.category_rng(1, "K"))
    assert [k for _, k in picks] == [0, 1, 2]
    ordered = sorted(ids, key=lambda s: words[s])
    strata = [ordered[0:4], ordered[4:7], ordered[7:10]]
    for (s, k) in picks:
        assert s in strata[k]
    assert picks == S.length_stratified_draw(ids, words, 3, S.category_rng(1, "K"))
    assert S.length_stratified_draw(ids[:2], words, 5, S.category_rng(1, "K")) == [("K/1", 0), ("K/0", 1)]


def synthetic_inputs(seed: int = 0) -> S.SplitInputs:
    rng = np.random.default_rng(seed)
    ids, cats = [], []
    for c, n in {"A": 30, "B": 25, "C": 60, "D": 20}.items():
        for i in range(n):
            ids.append(f"{c}/{i}")
            cats.append(c)
    n = len(ids)
    en_w = [int(x) for x in rng.lognormal(8, 1, n)]
    vi_w = [int(x * 1.3) for x in en_w]
    C_en = rng.uniform(0, 0.4, (n, n))
    C_en = np.maximum(C_en, C_en.T)
    C_vi = C_en.copy()
    near = [(3, 40), (60, 61), (100, 5)]
    for a, b in near:
        C_vi[a, b] = C_vi[b, a] = 0.85
    C_en[7, 8] = C_en[8, 7] = 0.6
    np.fill_diagonal(C_en, 0)
    np.fill_diagonal(C_vi, 0)
    return S.SplitInputs(
        sample_ids=ids, categories=cats, en_words=en_w, vi_words=vi_w,
        broken={"swapped": ["A/1"], "both_vietnamese": ["C/2"], "both_english": []},
        identical_pairs=[("B/1", "C/10"), ("D/3", "D/4")],
        containment={"exact": (C_en, C_vi), "filtered": (C_en * 0.9, C_vi * 0.9)},
        en_chars=[w * 6 for w in en_w], vi_chars=[w * 5 for w in vi_w],
    )


PARAMS = {
    "seed": 20260925, "dev_size": 12,
    "dev_quotas": {"A": 3, "B": 2, "C": 5, "D": 2},
    "dev_mini_size": 5,
    "near_duplicates": {"threshold": 0.8, "near_relative_threshold": 0.5},
    "tokens_per_word_estimate": {"en": 1.4, "vi": 1.7},
    "length_ratio_flag": [0.6, 1.3],
}


def test_build_split_spec_invariants():
    inp = synthetic_inputs()
    spec = S.build_split_spec(inp, PARAMS, "vtest", {"train_sha256": "x"})
    ids = set(inp.sample_ids)
    dev, mini, tf = set(spec.dev_ids), set(spec.dev_mini_ids), set(spec.train_fit_ids)
    cats = dict(zip(inp.sample_ids, inp.categories))
    assert len(spec.dev_ids) == 12
    assert {c: sum(1 for s in dev if cats[s] == c) for c in PARAMS["dev_quotas"]} == PARAMS["dev_quotas"]
    banned = {"A/1", "C/2", "B/1", "C/10", "D/3", "D/4", "A/3", "B/10", "C/5", "C/6", "C/45", "A/5"}
    assert not dev & banned
    assert spec.excluded == {"broken": ["A/1", "C/2"], "duplicate_dropped": ["C/10", "D/4"]}
    assert tf == ids - dev - {"A/1", "C/2", "C/10", "D/4"}
    assert mini <= dev and len(mini) == 5
    assert set(cats[s] for s in mini) == set(PARAMS["dev_quotas"])
    words = inp.words
    median = float(np.median([words[s] for s in dev]))
    assert all(words[s] <= median for s in mini)
    for s in dev:
        info = spec.doc_info[s]
        assert info["near_relative"] == (info["max_containment_into_train_fit"] >= 0.5)
        assert info["words"] == words[s] and info["category"] == cats[s]
    assert spec.summary["n_train_fit"] == len(tf)
    assert spec.eligibility["n_eligible"] == len(ids) - len(banned)


def test_split_json_is_deterministic_and_round_trips(tmp_path):
    a = S.build_split_spec(synthetic_inputs(), PARAMS, "vtest").to_json()
    b = S.build_split_spec(synthetic_inputs(), PARAMS, "vtest").to_json()
    assert a == b
    spec = S.SplitSpec.from_dict(json.loads(a))
    assert spec.to_json() == a


def test_split_json_is_identical_across_processes():
    code = (
        "import sys, json; sys.path.insert(0, %r); sys.path.insert(0, %r); "
        "import test_splits as t; from LegalMT import splits as S; "
        "print(S.build_split_spec(t.synthetic_inputs(), t.PARAMS, 'vtest').to_json())"
    ) % (str(REPO_ROOT / "src"), str(REPO_ROOT / "tests"))
    outs = set()
    for seed in ("1", "999"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                check=True, cwd=str(REPO_ROOT)).stdout)
    assert len(outs) == 1


def test_docs_for_with_synthetic_spec(tmp_path, monkeypatch):
    spec = S.build_split_spec(synthetic_inputs(), PARAMS, "vtest")
    p = tmp_path / "dev_split_vtest.json"
    p.write_text(spec.to_json(), encoding="utf-8")
    monkeypatch.setattr(S, "spec_path", lambda version="v0": p)
    inp = synthetic_inputs()
    df = pd.DataFrame({"sample_id": inp.sample_ids, "category": inp.categories,
                       "english_markdown": ["en"] * len(inp.sample_ids), "vietnamese_markdown": ["vi"] * len(inp.sample_ids)})
    monkeypatch.setattr(S, "load_split", lambda name: type("X", (), {"df": df})())
    dev = S.docs_for("dev", "vtest", check_data=False)
    assert dev["sample_id"].tolist() == spec.dev_ids
    assert len(S.docs_for("train_fit", "vtest", check_data=False)) == len(spec.train_fit_ids)
    assert S.load_split_spec("vtest").dev_mini_ids == spec.dev_mini_ids
    with pytest.raises(ValueError):
        S.docs_for("test", "vtest")


@pytest.mark.data
def test_frozen_v0_split_invariants():
    spec = S.load_split_spec("v0")
    dev, mini, tf = spec.dev_ids, spec.dev_mini_ids, spec.train_fit_ids
    assert len(dev) == 40 and len(set(dev)) == 40
    assert len(mini) == 12 and set(mini) <= set(dev)
    cats = {s: spec.doc_info[s]["category"] for s in dev}
    assert {c: sum(1 for s in dev if cats[s] == c) for c in V0_QUOTAS} == V0_QUOTAS
    assert {cats[s] for s in mini} == set(V0_QUOTAS)
    med = float(np.median([spec.doc_info[s]["words"] for s in dev]))
    assert all(spec.doc_info[s]["words"] <= med for s in mini)
    excluded = set(spec.excluded["broken"]) | set(spec.excluded["duplicate_dropped"])
    assert len(spec.excluded["broken"]) == 9 and len(spec.excluded["duplicate_dropped"]) == 18
    assert not set(dev) & set(tf) and not excluded & set(tf) and not excluded & set(dev)
    assert len(tf) == 854 - 40 - 27
    inelig = set(spec.eligibility["ineligible_broken"]) | set(spec.eligibility["ineligible_identical_pair_member"]) \
        | set(spec.eligibility["ineligible_near_duplicate"])
    assert not inelig & set(dev)
    for s in dev:
        assert spec.doc_info[s]["near_relative"] == (spec.doc_info[s]["max_containment_into_train_fit"] >= 0.5)
    assert spec.seed == 20260925 and spec.protocol["dev_quotas"] == V0_QUOTAS


@pytest.mark.data
def test_docs_for_real_roles():
    spec = S.load_split_spec("v0")
    dev = S.docs_for("dev")
    assert dev["sample_id"].tolist() == spec.dev_ids
    assert len(S.docs_for("dev_mini")) == 12
    assert len(S.docs_for("train_fit")) == len(spec.train_fit_ids)
    assert len(S.docs_for("public_test")) == 50


@pytest.mark.data
def test_make_dev_split_reproduces_frozen_json():
    r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "LegalMT" / "make_dev_split.py"), "--check", "--no-public"],
                       capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "is up to date" in r.stdout
    assert "match the re-derivation exactly" in r.stdout


def _load_script(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "LegalMT" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_words(rng: np.random.Generator, n: int, vi: bool) -> str:
    letters = "abcdeghiklmnopqrstuvxy"
    words = []
    for _ in range(n):
        w = "".join(rng.choice(list(letters), size=4))
        words.append(w + ("ơ" if vi else ""))
    return " ".join(words)


@pytest.mark.parametrize("sample_k", [1, 4])
def test_public_overlap_samples_queries_like_the_pool(monkeypatch, sample_k):
    mds = _load_script("make_dev_split")
    rng = np.random.default_rng(3)
    ids = [f"K/{i}" for i in range(4)]
    en = [_synthetic_words(rng, 6000, vi=False) for _ in ids]
    vi = [_synthetic_words(rng, 6000, vi=True) for _ in ids]
    train = pd.DataFrame({"sample_id": ids, "category": ["K"] * 4, "english_markdown": en, "vietnamese_markdown": vi})
    pub = pd.DataFrame({"sample_id": ["P/1", "P/2"], "category": ["K", "K"],
                        "english_markdown": [" ".join(en[2].split()[1000:1600]), _synthetic_words(rng, 600, vi=False)],
                        "vietnamese_markdown": [" ".join(vi[2].split()[2000:2600]), _synthetic_words(rng, 600, vi=True)]})
    params = {"language_vi_share_threshold": 0.2,
              "near_duplicates": {"shingle_n": 5, "sample_k": sample_k, "measures": {"exact": None, "df2": 2}}}
    derived = mds.derive_quality(train, params, verbose=False)
    spec = S.SplitSpec("vtest", 0, {}, [], [], ids, {"broken": [], "duplicate_dropped": []}, {},
                       summary={"n_near_relative": 0, "n_dev": 0})
    monkeypatch.setattr(mds, "load_split", lambda name: type("X", (), {"df": pub})())
    out = mds.public_overlap(spec, derived, params)
    assert out["exact"][0] > 0.9 and out["max"][0] > 0.9
    assert out["exact"][1] < 0.1


@pytest.mark.data
def test_frozen_dev_has_no_near_copy_anywhere_in_train_brute_force():
    import re

    from LegalMT.data import load_split
    from LegalMT.paths import load_config
    from LegalMT.textnorm import normalize_text

    spec = S.load_split_spec("v0")
    dq = load_config("data_quality_v0")["broken_pairs"]
    df = load_split("train").df
    word = re.compile(r"\w+")

    def grams(t: str) -> set:
        w = word.findall(normalize_text(t).lower())
        return set(zip(*[w[i:] for i in range(5)]))

    cols = {"en": [], "vi": []}
    for s, e, v in zip(df["sample_id"], df["english_markdown"], df["vietnamese_markdown"]):
        if s in dq["swapped"]:
            e, v = v, e
        elif s in dq["both_vietnamese"]:
            e = ""
        elif s in dq["both_english"]:
            v = ""
        cols["en"].append(e)
        cols["vi"].append(v)
    ids = df["sample_id"].tolist()
    pos = {s: i for i, s in enumerate(ids)}
    tf = set(spec.train_fit_ids)
    G = {lang: [grams(t) for t in texts] for lang, texts in cols.items()}
    for s in spec.dev_ids:
        i = pos[s]
        best_tf = {"en": 0.0, "vi": 0.0}
        for lang in ("en", "vi"):
            a = G[lang][i]
            for j, o in enumerate(ids):
                b = G[lang][j]
                if j == i or not a or not b:
                    continue
                c = len(a & b) / min(len(a), len(b))
                assert c < 0.8, f"{s} vs {o} ({lang}) containment {c:.3f}"
                if o in tf:
                    best_tf[lang] = max(best_tf[lang], c)
        detail = spec.doc_info[s]["containment_detail"]["exact"]
        assert detail["en"] == pytest.approx(best_tf["en"], abs=1e-4)
        assert detail["vi"] == pytest.approx(best_tf["vi"], abs=1e-4)


def test_make_dev_split_never_silently_replaces_a_frozen_split(tmp_path, monkeypatch, capsys):
    mds = _load_script("make_dev_split")
    spec = S.build_split_spec(synthetic_inputs(), PARAMS, "v0")
    monkeypatch.setattr(mds, "load_config", lambda name: {"v0": PARAMS} if name == "splits" else {})
    monkeypatch.setattr(mds, "split_path", lambda name: REPO_ROOT / "configs" / "LegalMT" / "splits.yaml")
    monkeypatch.setattr(mds, "load_split", lambda name: type("X", (), {"df": pd.DataFrame({"sample_id": ["A/1"]})})())
    monkeypatch.setattr(mds, "derive_quality", lambda df, params: {})
    monkeypatch.setattr(mds, "compare_quality", lambda dq, derived: [])
    monkeypatch.setattr(mds, "print_quality", lambda dq, derived, diffs: None)
    monkeypatch.setattr(mds, "build_inputs", lambda df, dq, derived: None)
    monkeypatch.setattr(mds.S, "build_split_spec", lambda inputs, params, version, data=None: spec)
    out = tmp_path / "dev_split_v0.json"
    assert mds.main(["--out", str(out), "--no-public"]) == 0 and out.read_text() == spec.to_json()
    assert mds.main(["--out", str(out), "--no-public"]) == 0
    assert "(unchanged)" in capsys.readouterr().out
    out.write_text("{}\n")
    assert mds.main(["--out", str(out), "--no-public"]) == 2
    assert out.read_text() == "{}\n" and "REFUSED" in capsys.readouterr().out
    assert mds.main(["--out", str(out), "--no-public", "--check"]) == 1
    assert mds.main(["--out", str(out), "--no-public", "--overwrite"]) == 0 and out.read_text() == spec.to_json()


def test_pick_dev_mini_falls_back_when_a_category_has_no_short_dev_doc():
    dev = ["A/1", "A/2", "B/1", "B/2", "C/1", "C/2", "C/3"]
    cats = {s: s[0] for s in dev}
    words = {"A/1": 10, "A/2": 20, "B/1": 30, "B/2": 40, "C/1": 900, "C/2": 1000, "C/3": 950}
    notes: list[str] = []
    mini = S._pick_dev_mini(dev, cats, words, {"A": 10, "B": 10, "C": 10}, 4, 7, notes)
    assert len(mini) == 4 and {cats[s] for s in mini} == {"A", "B", "C"}
    assert "C/1" in mini and any("no C dev doc" in n for n in notes)
