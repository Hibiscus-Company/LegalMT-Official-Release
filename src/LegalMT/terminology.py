"""Terminology metrics: glossary loading, term matching, term recall and consistency."""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from . import paths
from .textnorm import normalize_text

GLOSSARY_COLUMNS = ("en", "vi", "vi_variants", "en_variants", "note")
DROPPED_PREFIX = "#DROPPED"
_WORD = re.compile(r"\w+")
DIRECTIONS = {"en2vi": ("en", "vi"), "vi2en": ("vi", "en")}

_TONE_FOLD = {
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
}
_TONE_RE = re.compile("|".join(_TONE_FOLD))


def _en_fold(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("sses"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def canonical_text(text: str) -> str:
    t = normalize_text(text).lower().replace("ð", "đ")
    return _TONE_RE.sub(lambda m: _TONE_FOLD[m.group(0)], t)


def canonical_tokens(text: str, lang: str) -> tuple[str, ...]:
    toks = _WORD.findall(canonical_text(text))
    if lang == "en":
        toks = [_en_fold(t) for t in toks]
    return tuple(toks)


@dataclass(frozen=True)
class GlossaryEntry:
    en: str
    vi: str
    vi_variants: tuple[str, ...] = ()
    en_variants: tuple[str, ...] = ()
    note: str = ""
    dropped: bool = False

    def forms(self, lang: str) -> tuple[str, ...]:
        raw = (self.en, *self.en_variants) if lang == "en" else (self.vi, *self.vi_variants)
        seen, out = set(), []
        for f in raw:
            key = canonical_tokens(f, lang)
            if f and key and key not in seen:
                seen.add(key)
                out.append(f)
        return tuple(out)

    def form_tokens(self, lang: str) -> tuple[tuple[str, ...], ...]:
        return tuple(canonical_tokens(f, lang) for f in self.forms(lang))

    @property
    def key(self) -> str:
        return f"{self.en} => {self.vi}"


def default_glossary_path() -> Path:
    return paths.CONFIG_DIR / "glossary_seed_v0.tsv"


def _split_variants(field_value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in (field_value or "").split("|") if v.strip())


def load_glossary(path: Path | str | None = None, *, include_dropped: bool = False) -> list[GlossaryEntry]:
    p = Path(path) if path is not None else default_glossary_path()
    entries: list[GlossaryEntry] = []
    header: list[str] | None = None
    with p.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if not row or not any(c.strip() for c in row):
                continue
            dropped = False
            if row[0].startswith("#"):
                if row[0].strip() == DROPPED_PREFIX and include_dropped:
                    dropped, row = True, row[1:]
                else:
                    continue
            if header is None:
                header = [c.strip() for c in row]
                missing = [c for c in ("en", "vi") if c not in header]
                if missing:
                    raise ValueError(f"{p}: header lacks {missing}")
                continue
            rec = {h: (row[i].strip() if i < len(row) else "") for i, h in enumerate(header)}
            if not rec.get("en") or not rec.get("vi"):
                raise ValueError(f"{p}: entry without en or vi: {row}")
            entries.append(GlossaryEntry(
                en=rec["en"], vi=rec["vi"],
                vi_variants=_split_variants(rec.get("vi_variants", "")),
                en_variants=_split_variants(rec.get("en_variants", "")),
                note=rec.get("note", ""), dropped=dropped,
            ))
    return entries


def format_entry(e: GlossaryEntry) -> str:
    cells = [e.en, e.vi, "|".join(e.vi_variants), "|".join(e.en_variants), e.note]
    line = "\t".join(cells)
    return f"{DROPPED_PREFIX}\t{line}" if e.dropped else line


class TermIndex:
    def __init__(self, text: str, lang: str):
        self.lang = lang
        self.tokens = canonical_tokens(text or "", lang)
        pos: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(self.tokens):
            pos[t].append(i)
        self.positions = pos

    def find(self, form: Sequence[str]) -> list[int]:
        if not form:
            return []
        n, toks, form = len(form), self.tokens, tuple(form)
        return [i for i in self.positions.get(form[0], ()) if toks[i : i + n] == form]

    def has(self, form: Sequence[str]) -> bool:
        if not form:
            return False
        n, toks, form = len(form), self.tokens, tuple(form)
        return any(toks[i : i + n] == form for i in self.positions.get(form[0], ()))

    def contains_any(self, forms: Iterable[Sequence[str]]) -> bool:
        return any(self.has(f) for f in forms)

    def count_forms(self, forms: Sequence[Sequence[str]]) -> list[int]:
        spans = []
        for k, f in enumerate(forms):
            for i in self.find(f):
                spans.append((i, -len(f), k))
        spans.sort()
        counts = [0] * len(forms)
        taken_until = -1
        for i, neg_len, k in spans:
            if i < taken_until:
                continue
            counts[k] += 1
            taken_until = i - neg_len
        return counts


def _contains_seq(longer: Sequence[str], shorter: Sequence[str]) -> bool:
    n = len(shorter)
    return any(tuple(longer[i : i + n]) == tuple(shorter) for i in range(len(longer) - n + 1))


def form_groups(forms: Sequence[Sequence[str]]) -> list[int]:
    parent = list(range(len(forms)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(forms)):
        for b in range(len(forms)):
            if a != b and len(forms[a]) >= len(forms[b]) and _contains_seq(forms[a], forms[b]):
                parent[find(a)] = find(b)
    return [find(i) for i in range(len(forms))]


def distinct_choices(counts: Sequence[int], groups: Sequence[int]) -> int:
    return len({g for c, g in zip(counts, groups) if c > 0})


@dataclass
class TermCounts:
    recall_eligible: int = 0
    recall_hits: int = 0
    consistency_eligible: int = 0
    consistent: int = 0
    inconsistent_zero: int = 0
    inconsistent_multi: int = 0
    ref_consistent: int = 0
    per_entry: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "term_recall": self.recall_hits / self.recall_eligible if self.recall_eligible else None,
            "term_recall_hits": self.recall_hits,
            "term_recall_eligible": self.recall_eligible,
            "term_consistency": self.consistent / self.consistency_eligible if self.consistency_eligible else None,
            "term_consistency_eligible": self.consistency_eligible,
            "term_consistent": self.consistent,
            "term_inconsistent_absent": self.inconsistent_zero,
            "term_inconsistent_multiple_forms": self.inconsistent_multi,
            "ref_term_consistency": self.ref_consistent / self.consistency_eligible if self.consistency_eligible else None,
        }


def term_metrics(
    entries: Sequence[GlossaryEntry],
    srcs: Sequence[str],
    refs: Sequence[str],
    hyps: Sequence[str],
    direction: str,
    *,
    min_occurrences: int = 2,
) -> TermCounts:
    if not (len(srcs) == len(refs) == len(hyps)):
        raise ValueError("srcs, refs and hyps must have the same length")
    src_lang, tgt_lang = DIRECTIONS[direction]
    active = [e for e in entries if not e.dropped]
    src_forms = [e.form_tokens(src_lang) for e in active]
    tgt_forms = [e.form_tokens(tgt_lang) for e in active]
    tgt_groups = [form_groups(tf) for tf in tgt_forms]
    tc = TermCounts()
    for src, ref, hyp in zip(srcs, refs, hyps):
        si, ri, hi = TermIndex(src, src_lang), TermIndex(ref, tgt_lang), TermIndex(hyp, tgt_lang)
        for e, sf, tf, tg in zip(active, src_forms, tgt_forms, tgt_groups):
            src_n = sum(si.count_forms(sf))
            if src_n == 0:
                continue
            pe = tc.per_entry.setdefault(e.key, {"recall_eligible": 0, "recall_hits": 0,
                                                 "consistency_eligible": 0, "consistent": 0})
            hyp_counts = hi.count_forms(tf)
            if ri.contains_any(tf):
                tc.recall_eligible += 1
                pe["recall_eligible"] += 1
                if any(hyp_counts):
                    tc.recall_hits += 1
                    pe["recall_hits"] += 1
            if src_n >= min_occurrences:
                tc.consistency_eligible += 1
                pe["consistency_eligible"] += 1
                used = distinct_choices(hyp_counts, tg)
                if used == 1:
                    tc.consistent += 1
                    pe["consistent"] += 1
                elif used == 0:
                    tc.inconsistent_zero += 1
                else:
                    tc.inconsistent_multi += 1
                if distinct_choices(ri.count_forms(tf), tg) == 1:
                    tc.ref_consistent += 1
    return tc


def cooccurrence_stats(
    entries: Sequence[GlossaryEntry], en_texts: Sequence[str], vi_texts: Sequence[str]
) -> list[dict[str, object]]:
    en_idx = [TermIndex(t, "en") for t in en_texts]
    vi_idx = [TermIndex(t, "vi") for t in vi_texts]
    rows = []
    for e in entries:
        ef, vf = e.form_tokens("en"), e.form_tokens("vi")
        has_en = [ix.contains_any(ef) for ix in en_idx]
        has_vi = [ix.contains_any(vf) for ix in vi_idx]
        n_en, n_vi = sum(has_en), sum(has_vi)
        both = sum(1 for a, b in zip(has_en, has_vi) if a and b)
        rows.append({
            "en": e.en, "vi": e.vi, "note": e.note,
            "n_en_docs": n_en, "n_vi_docs": n_vi, "n_both_docs": both,
            "cooc_en2vi": both / n_en if n_en else 0.0,
            "cooc_vi2en": both / n_vi if n_vi else 0.0,
            "en_occurrences": sum(sum(ix.count_forms(ef)) for ix in en_idx),
            "vi_occurrences": sum(sum(ix.count_forms(vf)) for ix in vi_idx),
        })
    return rows
