"""Tests for LegalMT.quality: synthetic strings only; data-marked tests assert counts and ids."""
from __future__ import annotations

import os
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from LegalMT import quality as Q

REPO_ROOT = Path(__file__).resolve().parents[1]

EN = "The applicant shall submit the form to the competent office within thirty days of the decision."
VI = "Người nộp đơn phải nộp tờ khai cho cơ quan có thẩm quyền trong thời hạn ba mươi ngày kể từ ngày quyết định."
VI_LOW = "Danh mục chất cấm: methanol ethanol propanol butanol pentanol hexanol toluene."


def test_column_language_threshold():
    assert Q.column_language(EN)[0] == "en"
    lang, share = Q.column_language(VI)
    assert lang == "vi" and share > 0.5
    lang, share = Q.column_language(VI_LOW)
    assert lang == "vi" and 0.2 < share < 0.35


def test_language_checks_and_broken_pairs():
    ids = ["A/1", "A/2", "B/3", "B/4"]
    en = [EN, VI, VI, EN]
    vi = [VI, EN, VI, EN]
    checks = Q.language_checks(ids, en, vi)
    assert [c.status for c in checks] == ["ok", "swapped", "both_vietnamese", "both_english"]
    assert Q.broken_pairs(checks) == {"swapped": ["A/2"], "both_vietnamese": ["B/3"], "both_english": ["B/4"]}
    ce, cv = Q.corrected_columns(checks, en, vi)
    assert ce == [EN, EN, "", EN] and cv == [VI, VI, VI, ""]


def test_exact_duplicate_groups_ignore_whitespace_and_form():
    import unicodedata

    ids = ["X/2", "X/1", "Y/1"]
    en = [EN, "  " + EN.replace(" ", "   ") + "\n", EN + " extra"]
    vi = [VI, unicodedata.normalize("NFD", VI), VI]
    assert Q.exact_duplicate_groups(ids, en, vi) == [["X/1", "X/2"]]


def test_shingles_basic_properties():
    assert len(Q.shingles("one two three four")) == 0
    s = Q.shingles("one two three four five six")
    assert len(s) == 2 and s.dtype == np.uint64 and (np.diff(s.astype(np.float64)) > 0).all()
    assert np.array_equal(Q.shingles("One, two; THREE four five!"), Q.shingles("one two three four five"))
    assert not np.array_equal(Q.shingles("one two three four fives"), Q.shingles("one two three four five"))
    long = " ".join(f"w{i}" for i in range(400))
    full, sub = Q.shingles(long), Q.shingles(long, sample_k=4)
    assert set(sub.tolist()) <= set(full.tolist()) and 0 < len(sub) < len(full)
    assert all(int(h) % 4 == 0 for h in sub)


def test_shingle_hash_is_process_independent():
    code = (
        "import sys; sys.path.insert(0, %r); from LegalMT.quality import shingles; "
        "print(','.join(str(int(x)) for x in shingles('alpha beta gamma delta epsilon zeta eta')))"
    ) % str(REPO_ROOT / "src")
    outs = set()
    for seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout)
    assert len(outs) == 1


def _synthetic_corpus(n_docs: int = 14, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    blocks = [" ".join(f"b{k}w{i}" for i in range(12)) for k in range(6)]
    docs = []
    for d in range(n_docs):
        parts = [blocks[k] for k in range(6) if rng.random() < 0.5]
        parts.append(" ".join(f"d{d}w{i}" for i in range(int(rng.integers(5, 20)))))
        docs.append(" ".join(parts))
    docs.append(docs[0])
    docs.append(docs[1] + " " + " ".join(f"tail{i}" for i in range(30)))
    return docs


def _brute_force(sets: list[set], max_df: int | None) -> tuple[np.ndarray, np.ndarray]:
    from collections import Counter

    df = Counter(x for s in sets for x in s)
    kept = [{x for x in s if max_df is None or df[x] <= max_df} for s in sets]
    n = len(sets)
    inter = np.zeros((n, n), dtype=np.int64)
    for a, b in combinations(range(n), 2):
        inter[a, b] = inter[b, a] = len(kept[a] & kept[b])
    return inter, np.array([len(s) for s in kept])


@pytest.mark.parametrize("max_df", [None, 2, 3, 5])
@pytest.mark.parametrize("dense_above", [2, 50])
def test_pairwise_intersections_match_brute_force(max_df, dense_above):
    sets = Q.shingle_sets(_synthetic_corpus())
    idx = Q.ShingleIndex.build(sets)
    inter = Q.pairwise_intersections(idx, max_df, dense_above=dense_above, max_keys_per_batch=7, dense_block=3)
    bf_inter, bf_sizes = _brute_force([set(s.tolist()) for s in sets], max_df)
    assert np.array_equal(inter, bf_inter)
    assert np.array_equal(idx.doc_sizes(max_df), bf_sizes)
    C = Q.containment(inter, idx.doc_sizes(max_df))
    assert C.shape == inter.shape and float(C.max()) <= 1.0 + 1e-12


def test_query_intersections_match_brute_force():
    docs = _synthetic_corpus()
    pool, queries = Q.shingle_sets(docs[:10]), Q.shingle_sets(docs[10:])
    idx = Q.ShingleIndex.build(pool)
    for max_df in (None, 3):
        inter, qsizes = Q.query_intersections(idx, queries, max_df=max_df)
        from collections import Counter

        df = Counter(x for s in pool for x in s.tolist())
        for qi, q in enumerate(queries):
            qk = {x for x in q.tolist() if max_df is None or df.get(x, 0) <= max_df}
            assert qsizes[qi] == len(qk)
            for pj, p in enumerate(pool):
                pk = {x for x in p.tolist() if max_df is None or df[x] <= max_df}
                assert inter[qi, pj] == len(qk & pk)


def test_near_duplicates_groups_and_identical_pairs():
    docs = _synthetic_corpus()
    ids = [f"C/{i}" for i in range(len(docs))]
    en_sets = Q.shingle_sets(docs)
    vi_sets = Q.shingle_sets([d.replace("w", "v") for d in docs])
    res = Q.near_duplicates(ids, en_sets, vi_sets, max_df=None)
    n = len(docs)
    assert res.containment["en"][0, n - 2] == pytest.approx(1.0)
    assert res.containment["en"][1, n - 1] == pytest.approx(1.0)
    groups = res.groups(0.99)
    assert any({"C/0", f"C/{n - 2}"} <= set(g) for g in groups)
    assert any({"C/1", f"C/{n - 1}"} <= set(g) for g in groups)
    assert Q.group_stats(groups)["docs_in_multi_doc_groups"] >= 4
    assert Q.identical_pairs(ids, en_sets, vi_sets) == sorted([("C/0", f"C/{n - 2}")])


def test_diff_lists():
    assert Q.diff_lists(["a", "b"], ["b", "c"]) == (["a"], ["c"])


@pytest.mark.data
def test_real_train_quality_matches_data_quality_yaml():
    from LegalMT.data import load_split
    from LegalMT.paths import load_config

    df = load_split("train").df
    dq = load_config("data_quality_v0")
    ids = df["sample_id"].tolist()
    en, vi = df["english_markdown"].tolist(), df["vietnamese_markdown"].tolist()
    checks = Q.language_checks(ids, en, vi)
    broken = Q.broken_pairs(checks)
    for key in ("swapped", "both_vietnamese", "both_english"):
        assert broken[key] == sorted(dq["broken_pairs"][key])
    ce, cv = Q.corrected_columns(checks, en, vi)
    pairs = Q.identical_pairs(ids, Q.shingle_sets(ce), Q.shingle_sets(cv))
    assert sorted(pairs) == sorted(tuple(p) for p in dq["identical_pairs"])
    assert len(pairs) == 18


@pytest.mark.parametrize("max_df", [None, 1, 50])
def test_near_duplicates_degenerate_pools(max_df):
    docs = ["", "a b c", "one two three four five six", "one two three four five six", ""]
    ids = [f"X/{i}" for i in range(len(docs))]
    sets = Q.shingle_sets(docs)
    res = Q.near_duplicates(ids, sets, sets, max_df=max_df)
    C = res.max_either()
    assert np.isfinite(C).all() and C[0].max() == 0.0 and C[1].max() == 0.0
    linked = max_df is None or max_df >= 2
    assert (C[2, 3] == 1.0) == linked
    assert any(set(g) == {"X/2", "X/3"} for g in res.groups(0.8)) == linked
    empty_idx = Q.ShingleIndex.build([np.zeros(0, dtype=np.uint64)] * 2)
    inter, sizes = Q.query_intersections(empty_idx, [Q.shingles("one two three four five six")], max_df=max_df)
    assert inter.shape == (1, 2) and not inter.any() and sizes.tolist() == [2]


def test_language_check_on_nfd_and_invisible_characters():
    import unicodedata

    nfd = unicodedata.normalize("NFD", VI) + "​" * 20
    assert Q.column_language(nfd) == Q.column_language(VI)


@pytest.mark.data
def test_real_train_near_duplicate_groups_reproduce_research_counts():
    import time

    from LegalMT.data import load_split

    df = load_split("train").df
    ids = df["sample_id"].tolist()
    t0 = time.time()
    checks = Q.language_checks(ids, df["english_markdown"].tolist(), df["vietnamese_markdown"].tolist())
    en, vi = Q.corrected_columns(checks, df["english_markdown"].tolist(), df["vietnamese_markdown"].tolist())
    en_sets, vi_sets = Q.shingle_sets(en), Q.shingle_sets(vi)
    exact = Q.near_duplicates(ids, en_sets, vi_sets, max_df=None)
    filtered = Q.near_duplicates(ids, en_sets, vi_sets, max_df=50)
    assert time.time() - t0 < 120
    g8, g5, g3 = (Q.group_stats(exact.groups(t)) for t in (0.8, 0.5, 0.3))
    assert (g8["n_groups"], g8["docs_in_multi_doc_groups"]) == (792, 110)
    assert (g5["n_groups"], g5["docs_in_multi_doc_groups"]) == (701, 249)
    assert g3["largest_group"] == 290
    assert Q.group_stats(filtered.groups(0.3))["largest_group"] < 100
    for res in (exact, filtered):
        for C in res.containment.values():
            assert np.allclose(C, C.T) and not np.diag(C).any() and float(C.max()) <= 1.0 + 1e-12
