"""Dev-loop evaluation metrics for LegalMT: SacreBLEU BLEU, chrF and chrF++ over document pairs."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

from .textnorm import normalize_text

METRICS = ("bleu", "chrf", "chrfpp")
PREPROCESS_NOTE = "prepare_for_bleu: normalize_text (NFC, invisible chars dropped), newlines to spaces, whitespace collapsed"
PROVISIONAL_NOTE = (
    "PROVISIONAL: each whole document is one SacreBLEU segment after prepare_for_bleu; "
    "the organizers' document-level BLEU protocol is unpublished"
)


@dataclass(frozen=True)
class DocScore:
    bleu: float
    chrf: float


def score_documents(hypotheses: list[str], references: list[str]) -> DocScore:
    if len(hypotheses) != len(references):
        raise ValueError(
            f"hypotheses ({len(hypotheses)}) and references ({len(references)}) must be the same length"
        )
    if not hypotheses:
        raise ValueError("cannot score an empty list of documents")

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return DocScore(bleu=bleu.score, chrf=chrf.score)


def self_test(sample_text: str) -> DocScore:
    return score_documents([sample_text], [sample_text])


def prepare_for_bleu(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(normalize_text(text).split())


def _metric_objects() -> dict[str, BLEU | CHRF]:
    return {"bleu": BLEU(), "chrf": CHRF(), "chrfpp": CHRF(word_order=2)}


def _check_lengths(hyps: Sequence[str], refs: Sequence[str]) -> None:
    if len(hyps) != len(refs):
        raise ValueError(f"hypotheses ({len(hyps)}) and references ({len(refs)}) must be the same length")
    if not refs:
        raise ValueError("cannot score an empty list of documents")


@dataclass
class DocStats:
    stats: dict[str, np.ndarray]
    metrics: dict[str, BLEU | CHRF]

    @property
    def n_docs(self) -> int:
        return int(next(iter(self.stats.values())).shape[0])

    def score(self, metric: str, rows: np.ndarray | None = None) -> float:
        s = self.stats[metric] if rows is None else self.stats[metric][rows]
        return score_from_summed(self.metrics[metric], s.sum(axis=0))

    def scores(self, rows: np.ndarray | None = None) -> dict[str, float]:
        return {m: self.score(m, rows) for m in self.stats}

    def per_doc(self, metric: str) -> list[float]:
        return [score_from_summed(self.metrics[metric], row) for row in self.stats[metric]]


def score_from_summed(metric: BLEU | CHRF, summed: np.ndarray | Sequence[int]) -> float:
    vals = [int(x) for x in np.asarray(summed).tolist()]
    return float(metric._compute_score_from_stats(vals).score)


def doc_statistics(hyps: Sequence[str], refs: Sequence[str], *, prepared: bool = False) -> DocStats:
    _check_lengths(hyps, refs)
    if not prepared:
        hyps = [prepare_for_bleu(h) for h in hyps]
        refs = [prepare_for_bleu(r) for r in refs]
    metrics = _metric_objects()
    stats = {}
    for name, metric in metrics.items():
        rows = metric._extract_corpus_statistics(list(hyps), [list(refs)])
        stats[name] = np.asarray(rows, dtype=np.int64)
    return DocStats(stats=stats, metrics=metrics)


def signatures() -> dict[str, str]:
    out = {}
    for name, metric in _metric_objects().items():
        metric.corpus_score(["a b c d"], [["a b c d"]])
        out[name] = metric.get_signature().format()
    return out


def corpus_scores(hyps: Sequence[str], refs: Sequence[str]) -> dict[str, float | str]:
    ds = doc_statistics(hyps, refs)
    sig = signatures()
    out: dict[str, float | str] = ds.scores()
    out["bleu_signature"] = sig["bleu"]
    out["chrf_signature"] = sig["chrf"]
    out["chrfpp_signature"] = sig["chrfpp"]
    out["preprocess"] = PREPROCESS_NOTE
    return out


def per_document_bleu(hyps: Sequence[str], refs: Sequence[str]) -> list[float]:
    return doc_statistics(hyps, refs).per_doc("bleu")


def doc_macro_bleu(hyps: Sequence[str], refs: Sequence[str]) -> float:
    scores = per_document_bleu(hyps, refs)
    return float(np.mean(scores))


def per_category_scores(
    hyps: Sequence[str], refs: Sequence[str], categories: Sequence[str], *, stats: DocStats | None = None
) -> dict[str, dict[str, float]]:
    if len(categories) != len(refs):
        raise ValueError("categories must align with references")
    ds = stats if stats is not None else doc_statistics(hyps, refs)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(categories):
        groups[c].append(i)
    out = {}
    for cat in sorted(groups):
        rows = np.asarray(groups[cat])
        entry: dict[str, float] = {"n_docs": len(rows)}
        entry.update(ds.scores(rows))
        per_doc = [score_from_summed(ds.metrics["bleu"], ds.stats["bleu"][r]) for r in rows]
        entry["doc_macro_bleu"] = float(np.mean(per_doc))
        out[cat] = entry
    return out


def _resample_counts(n_docs: int, n: int, seed: int) -> np.ndarray:
    if n < 1 or n_docs < 1:
        raise ValueError(f"bootstrap needs n >= 1 resamples and at least one document (n={n}, n_docs={n_docs})")
    rng = np.random.default_rng(seed)
    idxs = rng.choice(n_docs, size=(n, n_docs), replace=True)
    counts = np.zeros((n, n_docs), dtype=np.int64)
    rows = np.repeat(np.arange(n), n_docs)
    np.add.at(counts, (rows, idxs.ravel()), 1)
    return counts


def _resampled_scores(ds: DocStats, counts: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for m, s in ds.stats.items():
        sums = counts @ s
        out[m] = np.array([score_from_summed(ds.metrics[m], row) for row in sums])
    return out


def _ci95(values: np.ndarray) -> list[float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_ci(
    hyps: Sequence[str], refs: Sequence[str], n: int = 1000, seed: int = 12345, *, stats: DocStats | None = None
) -> dict[str, list[float]]:
    ds = stats if stats is not None else doc_statistics(hyps, refs)
    counts = _resample_counts(ds.n_docs, n, seed)
    return {m: _ci95(v) for m, v in _resampled_scores(ds, counts).items()}


def paired_bootstrap(
    hyps_a: Sequence[str],
    hyps_b: Sequence[str],
    refs: Sequence[str],
    n: int = 1000,
    seed: int = 12345,
    *,
    stats_a: DocStats | None = None,
    stats_b: DocStats | None = None,
) -> dict[str, object]:
    if len(hyps_a) != len(hyps_b):
        raise ValueError("both systems must cover the same documents")
    ds_a = stats_a if stats_a is not None else doc_statistics(hyps_a, refs)
    ds_b = stats_b if stats_b is not None else doc_statistics(hyps_b, refs)
    if ds_a.n_docs != ds_b.n_docs:
        raise ValueError(f"statistics cover {ds_a.n_docs} vs {ds_b.n_docs} documents")
    counts = _resample_counts(ds_a.n_docs, n, seed)
    res_a = _resampled_scores(ds_a, counts)
    res_b = _resampled_scores(ds_b, counts)
    out: dict[str, object] = {"n_docs": ds_a.n_docs, "n_resamples": n, "seed": seed, "unit": "document"}
    for m in ds_a.stats:
        a, b = ds_a.score(m), ds_b.score(m)
        d = res_b[m] - res_a[m]
        sample_diffs = np.abs(d)
        centred = sample_diffs - sample_diffs.mean()
        p = (int(np.sum(centred > abs(b - a))) + 1) / (n + 1)
        if not np.any(d) and a == b:
            p = 1.0
        out[m] = {
            "a": a,
            "b": b,
            "delta": b - a,
            "delta_ci95": _ci95(d),
            "b_better_share": float(np.mean(d > 0)),
            "p_value": p,
            "a_ci95": _ci95(res_a[m]),
            "b_ci95": _ci95(res_b[m]),
        }
    return out
