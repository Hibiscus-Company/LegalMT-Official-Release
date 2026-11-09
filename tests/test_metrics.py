"""Tests for LegalMT.metrics (synthetic strings only, no organizer text)."""
from __future__ import annotations

import unicodedata

import numpy as np
import pytest
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

from LegalMT import metrics as M

REFS = [
    "the court shall decide the case within ten days of receiving the file",
    "article 5 of this law applies to all contracts signed after the effective date",
    "fines range from one million to two million dong for each violation",
    "the applicant submits one set of documents to the provincial office",
    "a b c d e f g h i j k l",
]
HYP_A = [
    "the court decides the case in ten days after receiving the file",
    "article 5 of the law applies to contracts signed after the date",
    "fines from one to two million dong for every violation",
    "the applicant submits documents to the office of the province",
    "a b c d x f g h i j k l",
]
HYP_B = [
    "the court shall decide the case within 10 days of receiving the file",
    "article 5 of this law applies to every contract signed after the effective date",
    "fines range from one million to two million dong for each violation",
    "the applicant submits a set of documents to the provincial office",
    "a b c",
]


def test_legacy_api_kept():
    s = M.self_test("một văn bản thử nghiệm ngắn gồm vài từ để kiểm tra")
    assert isinstance(s, M.DocScore)
    assert s.bleu > 99.9 and s.chrf > 99.9
    d = M.score_documents(HYP_A, REFS)
    assert d.bleu == pytest.approx(sacrebleu.corpus_bleu(HYP_A, [REFS]).score)
    with pytest.raises(ValueError):
        M.score_documents(["a"], [])


def test_prepare_for_bleu_line_final_hyphen():
    ref = "the rule protects all co-\nowners of the land plot\n\n-----\nsigned"
    hyp = "the rule protects all co- owners of the land plot ----- signed"
    raw = sacrebleu.corpus_bleu([hyp], [[ref]]).score
    assert raw < 100.0
    assert M.prepare_for_bleu(ref) == hyp
    assert M.corpus_scores([hyp], [ref])["bleu"] == pytest.approx(100.0)


def test_prepare_for_bleu_unicode_and_whitespace():
    composed = "Điều 5. Phạm vi điều chỉnh"
    decomposed = unicodedata.normalize("NFD", composed) + "​"
    assert M.prepare_for_bleu(decomposed) == composed
    assert M.prepare_for_bleu("  a\t b\n\n  c  ") == "a b c"
    assert M.prepare_for_bleu(None) == "" and M.prepare_for_bleu("") == ""
    assert M.corpus_scores([decomposed], [composed])["bleu"] == pytest.approx(100.0)


def test_corpus_scores_match_sacrebleu_on_prepared_text():
    res = M.corpus_scores(HYP_A, REFS)
    ph = [M.prepare_for_bleu(h) for h in HYP_A]
    pr = [M.prepare_for_bleu(r) for r in REFS]
    assert res["bleu"] == pytest.approx(sacrebleu.corpus_bleu(ph, [pr]).score)
    assert res["chrf"] == pytest.approx(sacrebleu.corpus_chrf(ph, [pr]).score)
    assert res["chrfpp"] == pytest.approx(sacrebleu.corpus_chrf(ph, [pr], word_order=2).score)
    assert "tok:13a" in res["bleu_signature"] and "nrefs:1" in res["bleu_signature"]
    assert "nw:0" in res["chrf_signature"] and "nw:2" in res["chrfpp_signature"]


def test_doc_macro_and_per_document_bleu():
    per_doc = M.per_document_bleu(HYP_A, REFS)
    expected = [sacrebleu.corpus_bleu([h], [[r]]).score for h, r in zip(HYP_A, REFS)]
    assert per_doc == pytest.approx(expected)
    assert M.doc_macro_bleu(HYP_A, REFS) == pytest.approx(np.mean(expected))
    assert M.per_document_bleu([""], ["some reference words here"]) == [0.0]


def test_per_category_scores_equal_subset_corpus_scores():
    cats = ["x", "y", "x", "y", "x"]
    out = M.per_category_scores(HYP_A, REFS, cats)
    assert set(out) == {"x", "y"}
    idx = [0, 2, 4]
    sub = M.corpus_scores([HYP_A[i] for i in idx], [REFS[i] for i in idx])
    assert out["x"]["n_docs"] == 3
    assert out["x"]["bleu"] == pytest.approx(sub["bleu"])
    assert out["x"]["chrfpp"] == pytest.approx(sub["chrfpp"])
    with pytest.raises(ValueError):
        M.per_category_scores(HYP_A, REFS, ["x"])


def test_doc_statistics_sum_to_corpus_score():
    ds = M.doc_statistics(HYP_A, REFS)
    assert ds.n_docs == 5
    assert ds.stats["bleu"].shape == (5, 10)
    assert ds.stats["chrf"].shape == (5, 18) and ds.stats["chrfpp"].shape == (5, 24)
    assert ds.score("bleu") == pytest.approx(M.corpus_scores(HYP_A, REFS)["bleu"])


def test_paired_bootstrap_matches_sacrebleu_paired_bs():
    from sacrebleu.significance import Result, _paired_bs_test

    ph_a = [M.prepare_for_bleu(h) for h in HYP_A]
    ph_b = [M.prepare_for_bleu(h) for h in HYP_B]
    pr = [M.prepare_for_bleu(r) for r in REFS]
    n, seed = 300, 7
    ours = M.paired_bootstrap(HYP_A, HYP_B, REFS, n=n, seed=seed)
    assert ours["unit"] == "document" and ours["n_docs"] == 5
    for name, metric_cls, kw in (("bleu", BLEU, {}), ("chrf", CHRF, {}), ("chrfpp", CHRF, {"word_order": 2})):
        m = metric_cls(**kw)
        bl_stats = m._extract_corpus_statistics(ph_a, [pr])
        bl = m._aggregate_and_compute(bl_stats)
        _, res = _paired_bs_test({name: (bl_stats, Result(bl.score, None))}, "b", ph_b, [pr],
                                 {name: metric_cls(**kw)}, n_samples=n, seed=seed)
        r = res[name]
        assert ours[name]["b"] == pytest.approx(r.score)
        assert ours[name]["p_value"] == pytest.approx(r.p_value)
        counts = M._resample_counts(5, n, seed)
        ds_b = M.doc_statistics(HYP_B, REFS)
        mean_b = np.mean(M._resampled_scores(ds_b, counts)[name])
        assert mean_b == pytest.approx(r.mean, rel=1e-6)


def test_paired_bootstrap_identical_systems_and_ci():
    out = M.paired_bootstrap(HYP_A, HYP_A, REFS, n=200, seed=1)
    for m in M.METRICS:
        assert out[m]["delta"] == 0.0
        assert out[m]["delta_ci95"] == [0.0, 0.0]
        assert out[m]["p_value"] == 1.0
    ci = M.bootstrap_ci(HYP_A, REFS, n=200, seed=1)
    for m in M.METRICS:
        assert 0.0 <= ci[m][0] < ci[m][1] <= 100.0
    assert M.bootstrap_ci(HYP_A, REFS, n=200, seed=1) == ci


def test_resample_counts_rows_sum_to_n_docs():
    c = M._resample_counts(7, 50, 3)
    assert c.shape == (50, 7)
    assert (c.sum(axis=1) == 7).all()
    rng = np.random.default_rng(3)
    idxs = rng.choice(7, size=(50, 7), replace=True)
    assert (np.stack([np.bincount(r, minlength=7) for r in idxs]) == c).all()


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        M.corpus_scores(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        M.paired_bootstrap(["a"], ["a", "b"], ["a"])


def test_bootstrap_rejects_degenerate_inputs():
    ds_5 = M.doc_statistics(HYP_A, REFS)
    ds_4 = M.doc_statistics(HYP_A[:4], REFS[:4])
    with pytest.raises(ValueError):
        M.paired_bootstrap(HYP_A, HYP_B, REFS, n=10, stats_a=ds_5, stats_b=ds_4)
    with pytest.raises(ValueError):
        M.paired_bootstrap(HYP_A, HYP_B, REFS, n=0)
    with pytest.raises(ValueError):
        M.bootstrap_ci(HYP_A, REFS, n=0)


def test_missing_documents_count_against_the_system():
    full = M.corpus_scores(HYP_A, REFS)["bleu"]
    dropped = M.corpus_scores(HYP_A[:-1] + [""], REFS)["bleu"]
    assert dropped < full
    assert M.corpus_scores(HYP_A[:-1], REFS[:-1])["bleu"] != dropped


def test_prepare_for_bleu_is_idempotent_and_layout_blind():
    doc = "Điều 1.\n\n  Phạm vi   điều chỉnh\n\n| a |  b |\n-----\nKý tên"
    once = M.prepare_for_bleu(doc)
    assert M.prepare_for_bleu(once) == once and "\n" not in once
    reflowed = " ".join(doc.split())
    assert M.corpus_scores([reflowed], [doc])["bleu"] == pytest.approx(100.0)
