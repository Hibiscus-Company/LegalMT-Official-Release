"""Data-quality checks: column language, broken pairs, exact duplicates and near-duplicates."""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .textnorm import normalize_text, vietnamese_word_share

VI_SHARE_THRESHOLD = 0.2
SHINGLE_N = 5
_WORD = re.compile(r"\w+")
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_MUL = np.uint64(0x9E3779B97F4A7C15)
_FIN1 = np.uint64(0xBF58476D1CE4E5B9)
_FIN2 = np.uint64(0x94D049BB133111EB)


def column_language(text: str, threshold: float = VI_SHARE_THRESHOLD) -> tuple[str, float]:
    share = vietnamese_word_share(text)
    return ("vi" if share > threshold else "en"), share


@dataclass(frozen=True)
class LangCheck:
    sample_id: str
    en_col_lang: str
    en_col_share: float
    vi_col_lang: str
    vi_col_share: float

    @property
    def status(self) -> str:
        pair = (self.en_col_lang, self.vi_col_lang)
        return {
            ("en", "vi"): "ok",
            ("vi", "en"): "swapped",
            ("vi", "vi"): "both_vietnamese",
            ("en", "en"): "both_english",
        }[pair]


def language_checks(
    sample_ids: Sequence[str],
    en_texts: Sequence[str],
    vi_texts: Sequence[str],
    threshold: float = VI_SHARE_THRESHOLD,
) -> list[LangCheck]:
    out = []
    for sid, en, vi in zip(sample_ids, en_texts, vi_texts):
        le, se = column_language(en, threshold)
        lv, sv = column_language(vi, threshold)
        out.append(LangCheck(sid, le, se, lv, sv))
    return out


def broken_pairs(checks: Iterable[LangCheck]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"swapped": [], "both_vietnamese": [], "both_english": []}
    for c in checks:
        if c.status != "ok":
            out[c.status].append(c.sample_id)
    return {k: sorted(v) for k, v in out.items()}


def corrected_columns(
    checks: Sequence[LangCheck], en_texts: Sequence[str], vi_texts: Sequence[str]
) -> tuple[list[str], list[str]]:
    en_out, vi_out = [], []
    for c, en, vi in zip(checks, en_texts, vi_texts):
        s = c.status
        if s == "ok":
            en_out.append(en); vi_out.append(vi)
        elif s == "swapped":
            en_out.append(vi); vi_out.append(en)
        elif s == "both_vietnamese":
            en_out.append(""); vi_out.append(vi)
        else:
            en_out.append(en); vi_out.append("")
    return en_out, vi_out


def _ws_digest(text: str) -> str:
    return hashlib.sha256(" ".join(normalize_text(text).split()).encode("utf-8")).hexdigest()


def exact_duplicate_groups(
    sample_ids: Sequence[str], en_texts: Sequence[str], vi_texts: Sequence[str]
) -> list[list[str]]:
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sid, en, vi in zip(sample_ids, en_texts, vi_texts):
        buckets[(_ws_digest(en), _ws_digest(vi))].append(sid)
    return sorted(sorted(g) for g in buckets.values() if len(g) > 1)


_word_hash_cache: dict[str, int] = {}


def _word_hash(word: str) -> int:
    h = _word_hash_cache.get(word)
    if h is None:
        h = int.from_bytes(hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest(), "little")
        _word_hash_cache[word] = h
    return h


def shingle_words(text: str) -> list[str]:
    return _WORD.findall(normalize_text(text).lower())


def shingles(text: str, n: int = SHINGLE_N, sample_k: int = 1) -> np.ndarray:
    words = shingle_words(text)
    L = len(words) - n + 1
    if L <= 0:
        return np.zeros(0, dtype=np.uint64)
    wh = np.fromiter((_word_hash(w) for w in words), dtype=np.uint64, count=len(words))
    with np.errstate(over="ignore"):
        h = wh[:L].copy()
        for k in range(1, n):
            h = h * _MUL + wh[k : L + k]
        h ^= h >> np.uint64(30)
        h *= _FIN1
        h ^= h >> np.uint64(27)
        h *= _FIN2
        h ^= h >> np.uint64(31)
    h = np.unique(h)
    if sample_k > 1:
        h = h[h % np.uint64(sample_k) == 0]
    return h


def shingle_sets(texts: Sequence[str], n: int = SHINGLE_N, sample_k: int = 1) -> list[np.ndarray]:
    return [shingles(t, n=n, sample_k=sample_k) if t else np.zeros(0, dtype=np.uint64) for t in texts]


def set_digest(sh: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(sh).tobytes()).hexdigest()


def identical_pairs(
    sample_ids: Sequence[str], en_sets: Sequence[np.ndarray], vi_sets: Sequence[np.ndarray]
) -> list[tuple[str, str]]:
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sid, e, v in zip(sample_ids, en_sets, vi_sets):
        if len(e) and len(v):
            buckets[(set_digest(e), set_digest(v))].append(sid)
    pairs = []
    for g in buckets.values():
        if len(g) > 1:
            g = sorted(g)
            pairs += [(g[0], other) for other in g[1:]]
    return sorted(pairs)


@dataclass
class ShingleIndex:
    n_docs: int
    keys: np.ndarray
    df: np.ndarray
    offsets: np.ndarray
    postings: np.ndarray
    doc_sizes_all: np.ndarray = field(repr=False)

    @classmethod
    def build(cls, sets: Sequence[np.ndarray]) -> "ShingleIndex":
        n = len(sets)
        lens = np.array([len(s) for s in sets], dtype=np.int64)
        if lens.sum() == 0:
            z = np.zeros(0, dtype=np.int64)
            return cls(n, np.zeros(0, dtype=np.uint64), z, z, z, lens)
        allsh = np.concatenate([np.asarray(s, dtype=np.uint64) for s in sets])
        docs = np.repeat(np.arange(n, dtype=np.int64), lens)
        order = np.argsort(allsh, kind="stable")
        allsh = allsh[order]
        postings = docs[order]
        keys, offsets, df = np.unique(allsh, return_index=True, return_counts=True)
        return cls(n, keys, df.astype(np.int64), offsets.astype(np.int64), postings, lens)

    def kept_mask(self, max_df: int | None) -> np.ndarray:
        return np.ones(len(self.keys), dtype=bool) if max_df is None else self.df <= max_df

    def doc_sizes(self, max_df: int | None) -> np.ndarray:
        if max_df is None:
            return self.doc_sizes_all.copy()
        kept_entries = np.repeat(self.kept_mask(max_df), self.df)
        return np.bincount(self.postings[kept_entries], minlength=self.n_docs).astype(np.int64)


def pairwise_intersections(
    index: ShingleIndex,
    max_df: int | None = 50,
    *,
    dense_above: int = 50,
    max_keys_per_batch: int = 20_000_000,
    dense_block: int = 4096,
) -> np.ndarray:
    N = index.n_docs
    upper = np.zeros(N * N, dtype=np.int64)
    limit = index.df.max(initial=0) if max_df is None else max_df
    pair_cap = min(limit, dense_above)
    for m in range(2, int(pair_cap) + 1):
        sel = np.nonzero(index.df == m)[0]
        if len(sel) == 0:
            continue
        ii, jj = np.triu_indices(m, 1)
        rows_per_batch = max(1, max_keys_per_batch // len(ii))
        for b in range(0, len(sel), rows_per_batch):
            starts = index.offsets[sel[b : b + rows_per_batch]]
            P = index.postings[starts[:, None] + np.arange(m)[None, :]]
            keys = (P[:, ii] * N + P[:, jj]).ravel()
            upper += np.bincount(keys, minlength=N * N)
    I = upper.reshape(N, N)
    I = I + I.T
    if limit > dense_above:
        hi = np.nonzero((index.df > dense_above) & (index.df <= limit))[0]
        for b in range(0, len(hi), dense_block):
            cols = hi[b : b + dense_block]
            X = np.zeros((N, len(cols)), dtype=np.float32)
            for c, k in enumerate(cols):
                o = index.offsets[k]
                X[index.postings[o : o + index.df[k]], c] = 1.0
            I += np.rint(X @ X.T).astype(np.int64)
    np.fill_diagonal(I, 0)
    return I


def query_intersections(
    index: ShingleIndex, query_sets: Sequence[np.ndarray], max_df: int | None = 50
) -> tuple[np.ndarray, np.ndarray]:
    N = index.n_docs
    kept = index.kept_mask(max_df)
    out = np.zeros((len(query_sets), N), dtype=np.int64)
    sizes = np.zeros(len(query_sets), dtype=np.int64)
    for q, sh in enumerate(query_sets):
        sh = np.asarray(sh, dtype=np.uint64)
        if len(sh) == 0 or len(index.keys) == 0:
            sizes[q] = len(sh)
            continue
        pos = np.searchsorted(index.keys, sh)
        pos_c = np.minimum(pos, len(index.keys) - 1)
        found = index.keys[pos_c] == sh
        k = pos_c[found]
        boiler = ~kept[k]
        sizes[q] = len(sh) - int(boiler.sum())
        k = k[~boiler]
        if len(k) == 0:
            continue
        starts, counts = index.offsets[k], index.df[k]
        total = int(counts.sum())
        rel = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        docs = index.postings[np.repeat(starts, counts) + rel]
        out[q] = np.bincount(docs, minlength=N)
    return out, sizes


def containment(I: np.ndarray, sizes_rows: np.ndarray, sizes_cols: np.ndarray | None = None) -> np.ndarray:
    sizes_cols = sizes_rows if sizes_cols is None else sizes_cols
    denom = np.minimum(sizes_rows[:, None], sizes_cols[None, :]).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(denom > 0, I / np.maximum(denom, 1), 0.0)
    return C


def union_find_groups(n: int, edges: Iterable[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def near_duplicate_groups(
    containments: Sequence[np.ndarray], threshold: float
) -> list[list[int]]:
    n = containments[0].shape[0]
    edges: list[tuple[int, int]] = []
    for C in containments:
        a, b = np.nonzero(np.triu(C >= threshold, 1))
        edges += list(zip(a.tolist(), b.tolist()))
    return union_find_groups(n, edges)


def group_stats(groups: Sequence[Sequence[int]]) -> dict[str, int]:
    multi = [g for g in groups if len(g) > 1]
    return {
        "n_groups": len(groups),
        "docs_in_multi_doc_groups": sum(len(g) for g in multi),
        "largest_group": max((len(g) for g in groups), default=0),
    }


@dataclass
class NearDupResult:
    sample_ids: list[str]
    containment: dict[str, np.ndarray]
    sizes: dict[str, np.ndarray]
    indexes: dict[str, ShingleIndex]
    max_df: int | None

    def max_either(self) -> np.ndarray:
        return np.maximum(self.containment["en"], self.containment["vi"])

    def groups(self, threshold: float) -> list[list[str]]:
        idx_groups = near_duplicate_groups(list(self.containment.values()), threshold)
        return [[self.sample_ids[i] for i in g] for g in idx_groups]


def near_duplicates(
    sample_ids: Sequence[str],
    en_sets: Sequence[np.ndarray],
    vi_sets: Sequence[np.ndarray],
    max_df: int | None = 50,
) -> NearDupResult:
    containment_by_lang, sizes, indexes = {}, {}, {}
    for lang, sets in (("en", en_sets), ("vi", vi_sets)):
        index = ShingleIndex.build(sets)
        I = pairwise_intersections(index, max_df)
        sz = index.doc_sizes(max_df)
        containment_by_lang[lang] = containment(I, sz)
        sizes[lang] = sz
        indexes[lang] = index
    return NearDupResult(list(sample_ids), containment_by_lang, sizes, indexes, max_df)


def diff_lists(expected: Iterable, derived: Iterable) -> tuple[list, list]:
    e, d = set(expected), set(derived)
    return sorted(e - d), sorted(d - e)
