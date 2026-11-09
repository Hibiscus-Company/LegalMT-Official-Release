"""Segment-level evaluation triples for LegalMT translation runs: one (source, hypothesis, reference)"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sacrebleu.metrics import BLEU

from . import metrics as M
from .align import align_elements, load_align_config, numeric_f1, numeric_tokens, resolve_cfg
from .layout import ROW_KINDS, TEXT_KINDS, DocSkeleton, Element, cell_is_translatable, parse_document, translation_units
from .textnorm import normalize_text

DIRECTIONS: dict[str, tuple[str, str]] = {"en2vi": ("en", "vi"), "vi2en": ("vi", "en")}
COLUMNS = {"en": "english_markdown", "vi": "vietnamese_markdown"}
SEGMENTS_FILE = "segments.jsonl"
RESULTS_FILE = "results_segments.json"
NOTE = ("PROVISIONAL: segments = source paragraphs aligned to reference paragraphs with align.align_elements "
        "(EN-VI length model of align.yaml), hypothesis mapped through units.jsonl; segment BLEU/chrF are "
        "sacrebleu corpus scores over one-line segments after prepare_for_bleu; CI by document bootstrap")
ZONE_KINDS = frozenset({"header_row", "signature_row"})


def _one_line(text: str | None) -> str:
    return " ".join((text or "").split())


def _nonws(text: str) -> int:
    return sum(not ch.isspace() for ch in text)


def element_text(e: Element) -> str:
    if e.kind in TEXT_KINDS:
        return e.text
    if e.kind in ROW_KINDS and e.cells:
        return " ".join(c.text for c in e.cells if cell_is_translatable(c.text))
    return ""


def trim_zone_rows(
    s_idx: Sequence[int], r_idx: Sequence[int], src_elems: Sequence[Element], ref_elems: Sequence[Element],
) -> tuple[list[int], list[int], list[int], list[int]]:
    sk = {src_elems[i].kind for i in s_idx}
    rk = {ref_elems[i].kind for i in r_idx}
    ks = [i for i in s_idx if src_elems[i].kind not in ZONE_KINDS or src_elems[i].kind in rk]
    kr = [i for i in r_idx if ref_elems[i].kind not in ZONE_KINDS or ref_elems[i].kind in sk]
    return ks, kr, [i for i in s_idx if i not in ks], [i for i in r_idx if i not in kr]


def kind_group(kinds: Iterable[str]) -> str:
    ks = set(kinds)
    if ks <= TEXT_KINDS:
        return "text"
    if len(ks) == 1:
        return next(iter(ks))
    return "mixed"


def check_units(skel: DocSkeleton, records: Sequence[Mapping[str, Any]]) -> list[str]:
    problems: list[str] = []
    elems = skel.elements
    seen: set[str] = set()
    slots: Counter = Counter()
    for r in records:
        idx, cell = r.get("element_idx"), r.get("cell_idx")
        own = f"e{idx}" + ("" if cell is None else f"c{cell}")
        uid = r.get("uid") or own
        seen.add(uid)
        slots[own] += 1
        if uid != own:
            problems.append(f"{uid}: uid does not match element_idx {idx!r} / cell_idx {cell!r}")
            continue
        if not isinstance(idx, int) or not 0 <= idx < len(elems):
            problems.append(f"{uid}: element_idx {idx!r} not in the re-parsed source ({len(elems)} elements)")
            continue
        e = elems[idx]
        if cell is None:
            expected = e.text if e.kind in TEXT_KINDS else None
        elif e.cells and isinstance(cell, int) and 0 <= cell < len(e.cells):
            expected = e.cells[cell].text
        else:
            expected = None
        if expected is None:
            problems.append(f"{uid}: no {'text' if cell is None else 'cell'} unit at element {idx} ({e.kind})")
        elif r.get("src") != expected:
            problems.append(f"{uid}: unit src differs from the re-parsed source text")
    dup = sorted(k for k, n in slots.items() if n > 1)
    if dup:
        problems.append(f"{len(dup)} elements/cells have more than one record, e.g. {dup[:3]}")
    missing = [u.uid for u in translation_units(skel) if u.uid not in seen]
    if missing:
        problems.append(f"{len(missing)} translation units have no record, e.g. {missing[:3]}")
    return problems


def element_hypothesis(e: Element, by_cell: Mapping[int | None, Mapping[str, Any]]) -> tuple[str, int, int, int]:
    parts: list[str] = []
    missing = fallback = copied = 0
    keys: list[int | None]
    if e.kind in TEXT_KINDS:
        keys = [None]
    elif e.kind in ROW_KINDS and e.cells:
        keys = [k for k, c in enumerate(e.cells) if cell_is_translatable(c.text)]
    else:
        keys = []
    for k in keys:
        r = by_cell.get(k)
        if r is None:
            missing += 1
            continue
        parts.append(_one_line(normalize_text(r.get("hyp") or "")))
        fallback += bool(r.get("fallback"))
        copied += bool(r.get("source_copied"))
    return " ".join(p for p in parts if p), missing, fallback, copied


@dataclass
class _Filters:
    on: bool
    c: float
    ratio_factor: float
    min_num_f1: float
    min_numbers: int
    min_digits: int

    @classmethod
    def from_cfg(cls, cfg: dict, on: bool) -> "_Filters":
        f = cfg["filters"]
        return cls(on=on, c=float(cfg["gc"]["c"]), ratio_factor=float(f["ratio_factor"]),
                   min_num_f1=float(f["min_num_f1"]), min_numbers=int(f["min_numbers"]),
                   min_digits=int(cfg["numbers"]["min_digits"]))

    def reason(self, en: str, vi: str, rows_only: bool) -> tuple[str | None, float, float | None]:
        ne, nv = numeric_tokens(en, self.min_digits), numeric_tokens(vi, self.min_digits)
        f1 = numeric_f1(ne, nv)
        ratio = len(vi) / len(en) if en else 0.0
        if not self.on:
            return None, ratio, f1
        if not rows_only and not (self.c / self.ratio_factor <= ratio <= self.c * self.ratio_factor):
            return "length_ratio", ratio, f1
        if (sum(ne.values()) >= self.min_numbers and sum(nv.values()) >= self.min_numbers
                and f1 is not None and f1 < self.min_num_f1):
            return "numeric", ratio, f1
        return None, ratio, f1


def doc_segments(
    sample_id: str,
    category: str,
    direction: str,
    src_text: str,
    ref_text: str,
    unit_records: Sequence[Mapping[str, Any]],
    *,
    layout_cfg: dict | None = None,
    align_cfg: dict | None = None,
    filters: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {sorted(DIRECTIONS)}, got {direction!r}")
    t0 = time.perf_counter()
    cfg = resolve_cfg(align_cfg)
    flt = _Filters.from_cfg(cfg, filters)
    src_lang, tgt_lang = DIRECTIONS[direction]
    src_skel = parse_document(src_text, src_lang, layout_cfg)
    ref_skel = parse_document(ref_text, tgt_lang, cfg.get("layout") or None)
    stats: dict[str, Any] = {
        "sample_id": sample_id, "category": category, "direction": direction, "status": "ok",
        "n_segments": 0, "links": Counter(), "kept_links": Counter(),
        "src_chars": 0, "ref_chars": 0, "src_chars_kept": 0, "ref_chars_kept": 0,
        "src_chars_dropped": Counter(), "ref_chars_dropped": Counter(),
        "mt_missing_units": 0, "fallback_units": 0, "source_copied_units": 0,
    }
    problems = check_units(src_skel, unit_records)
    if problems:
        stats.update(status="unit_mismatch", problems=problems[:5], n_problems=len(problems))
        return [], _finish_stats(stats, t0)

    by_elem: dict[int, dict[int | None, Mapping[str, Any]]] = defaultdict(dict)
    for r in unit_records:
        by_elem[r["element_idx"]][r.get("cell_idx")] = r
    src_elems, ref_elems = src_skel.elements, ref_skel.elements
    src_txt = {e.idx: element_text(e) for e in src_elems}
    ref_txt = {e.idx: element_text(e) for e in ref_elems}
    stats["src_chars"] = sum(_nonws(t) for t in src_txt.values())
    stats["ref_chars"] = sum(_nonws(t) for t in ref_txt.values())

    if src_lang == "en":
        links = [(s, r, cost) for s, r, _link, cost in align_elements(src_elems, ref_elems, cfg)]
    else:
        links = [(s, r, cost) for r, s, _link, cost in align_elements(ref_elems, src_elems, cfg)]

    triples: list[dict[str, Any]] = []
    for s_idx, r_idx, cost in links:
        stats["links"][f"{len(s_idx)}-{len(r_idx)}"] += 1
        trimmed = 0
        if s_idx and r_idx:
            s_idx, r_idx, s_cut, r_cut = trim_zone_rows(s_idx, r_idx, src_elems, ref_elems)
            for i in s_cut:
                stats["src_chars_dropped"]["unaligned_row"] += _nonws(src_txt[i])
            for i in r_cut:
                stats["ref_chars_dropped"]["unaligned_row"] += _nonws(ref_txt[i])
            trimmed = len(s_cut) + len(r_cut)
        link = f"{len(s_idx)}-{len(r_idx)}"
        src = " ".join(src_txt[i] for i in s_idx)
        ref = " ".join(ref_txt[i] for i in r_idx)
        if not s_idx or not r_idx:
            if src:
                stats["src_chars_dropped"]["unaligned"] += _nonws(src)
            if ref:
                stats["ref_chars_dropped"]["unaligned"] += _nonws(ref)
            continue
        kinds = [src_elems[i].kind for i in s_idx]
        rows_only = all(k in ROW_KINDS for k in kinds) and all(ref_elems[i].kind in ROW_KINDS for i in r_idx)
        en, vi = (src, ref) if src_lang == "en" else (ref, src)
        reason, ratio, f1 = flt.reason(en, vi, rows_only)
        if reason:
            stats["src_chars_dropped"][reason] += _nonws(src)
            stats["ref_chars_dropped"][reason] += _nonws(ref)
            continue
        parts, missing, fallback, copied = [], 0, 0, 0
        for i in s_idx:
            h, m, f, c = element_hypothesis(src_elems[i], by_elem.get(i, {}))
            if h:
                parts.append(h)
            missing, fallback, copied = missing + m, fallback + f, copied + c
        stats["kept_links"][link] += 1
        stats["src_chars_kept"] += _nonws(src)
        stats["ref_chars_kept"] += _nonws(ref)
        stats["mt_missing_units"] += missing
        stats["fallback_units"] += fallback
        stats["source_copied_units"] += copied
        triples.append({
            "seg_id": f"{direction}:{sample_id}:{len(triples):04d}",
            "sample_id": sample_id, "category": category, "direction": direction,
            "src_lang": src_lang, "tgt_lang": tgt_lang,
            "src": src, "mt": " ".join(parts), "ref": ref,
            "src_elems": list(s_idx), "ref_elems": list(r_idx),
            "src_kinds": kinds, "kind_group": kind_group(kinds), "link": link,
            "cost": cost, "len_ratio": round(ratio, 4), "num_f1": None if f1 is None else round(f1, 4),
            "mt_missing_units": missing, "fallback_units": fallback, "source_copied_units": copied,
            "trimmed_rows": trimmed,
        })
    stats["n_segments"] = len(triples)
    return triples, _finish_stats(stats, t0)


def _finish_stats(stats: dict[str, Any], t0: float) -> dict[str, Any]:
    for k in ("links", "kept_links", "src_chars_dropped", "ref_chars_dropped"):
        stats[k] = dict(stats[k])
    stats["seconds"] = round(time.perf_counter() - t0, 4)
    return stats


@dataclass
class SegmentSet:
    triples: list[dict[str, Any]]
    docs: list[dict[str, Any]]
    info: dict[str, Any] = field(default_factory=dict)


def _read_jsonl_tolerant(path: Path) -> tuple[list[dict[str, Any]], bool]:
    from .runs import read_jsonl_tolerant

    return read_jsonl_tolerant(path)


def load_run_records(run_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from .runs import load_run

    run = load_run(run_dir)
    hyps, trunc_h = _read_jsonl_tolerant(run.path("hyps.jsonl"))
    units_path = run.path("units.jsonl")
    if not units_path.is_file():
        raise FileNotFoundError(f"{units_path} not found: segment scoring needs units.jsonl")
    units, trunc_u = _read_jsonl_tolerant(units_path)
    info = {"run_dir": str(run.dir), "run_name": run.name, "hyps_truncated_last_line": trunc_h,
            "units_truncated_last_line": trunc_u}
    return run.manifest, hyps, units, info


def reference_table(refs: Any) -> dict[str, dict[str, str]]:
    if hasattr(refs, "to_dict") and hasattr(refs, "columns"):
        rows: Iterable[Mapping[str, Any]] = refs.to_dict("records")
    elif isinstance(refs, Mapping):
        return {str(k): dict(v) for k, v in refs.items()}
    else:
        rows = refs
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        out[str(r["sample_id"])] = {"category": str(r.get("category", "")),
                                    COLUMNS["en"]: r[COLUMNS["en"]], COLUMNS["vi"]: r[COLUMNS["vi"]]}
    return out


_UNSET: Any = object()


def build_segments(
    run: str | Path | None = None,
    split: str | None = None,
    *,
    hyps: Sequence[Mapping[str, Any]] | None = None,
    units: Sequence[Mapping[str, Any]] | None = None,
    refs: Any = None,
    split_version: str | None = None,
    directions: Sequence[str] | None = None,
    layout_cfg: dict | None = _UNSET,
    align_cfg: dict | str | None = "align",
    filters: bool = True,
    expected_ids: Sequence[str] | None = None,
    ids: Sequence[str] | None = None,
) -> SegmentSet:
    info: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    if run is not None and (hyps is None or units is None):
        manifest, h, u, info = load_run_records(run)
        hyps = h if hyps is None else hyps
        units = u if units is None else units
    if hyps is None or units is None:
        raise ValueError("give a run directory or both hyps and units records")
    split = split or manifest.get("split")
    split_version = split_version or manifest.get("split_version") or "v0"
    if layout_cfg is _UNSET:
        layout_cfg = (manifest.get("config") or {}).get("layout")
    if isinstance(align_cfg, str):
        align_name = align_cfg
        align_cfg = load_align_config(align_cfg)
    else:
        align_name = "dict" if align_cfg else "defaults"
        align_cfg = resolve_cfg(align_cfg)
    if refs is None:
        if not split:
            raise ValueError("no split: pass split=... or refs=...")
        from .splits import docs_for

        refs = docs_for(split, split_version)
    table = reference_table(refs)
    all_ref_ids = set(table)
    if ids is not None:
        keep = set(ids)
        unknown = sorted(keep - set(table))
        if unknown:
            raise KeyError(f"{len(unknown)} ids are not in the references, e.g. {unknown[:3]}")
        table = {s: v for s, v in table.items() if s in keep}
        expected_ids = [s for s in (expected_ids or table) if s in keep]

    latest: dict[tuple[str, str], Mapping[str, Any]] = {}
    dup = Counter()
    for r in hyps:
        key = (str(r["sample_id"]), r["direction"])
        if r["direction"] not in DIRECTIONS:
            raise ValueError(f"hyps record for {key[0]} has direction {r['direction']!r}")
        dup[key[1]] += key in latest
        latest[key] = r
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for r in units:
        grouped[(str(r["sample_id"]), r["direction"])][r.get("uid") or f"e{r['element_idx']}c{r.get('cell_idx')}"] = r

    if directions:
        dirs = list(directions)
        bad = [d for d in dirs if d not in DIRECTIONS]
        if bad:
            raise ValueError(f"unknown directions {bad}; expected some of {sorted(DIRECTIONS)}")
    else:
        declared = set(manifest.get("directions") or [])
        present = {k[1] for k in latest}
        dirs = [d for d in DIRECTIONS if d in declared or d in present]
    if expected_ids is None:
        subset = manifest.get("subset")
        expected_ids = subset["ids"] if subset else list(table)
    triples: list[dict[str, Any]] = []
    docs: list[dict[str, Any]] = []
    missing: dict[str, list[str]] = {}
    not_in_refs: dict[str, list[str]] = {}
    for d in dirs:
        src_lang, tgt_lang = DIRECTIONS[d]
        missing[d] = [s for s in expected_ids if (s, d) not in latest]
        not_in_refs[d] = sorted(s for (s, dd) in latest if dd == d and s not in all_ref_ids)
        for sid in (s for s in table if (s, d) in latest):
            ref_row = table[sid]
            recs = list(grouped.get((sid, d), {}).values())
            if not recs:
                docs.append({"sample_id": sid, "category": ref_row["category"], "direction": d,
                             "status": "no_units", "n_segments": 0})
                continue
            t, st = doc_segments(sid, ref_row["category"], d, ref_row[COLUMNS[src_lang]], ref_row[COLUMNS[tgt_lang]],
                                 recs, layout_cfg=layout_cfg, align_cfg=align_cfg, filters=filters)
            triples.extend(t)
            docs.append(st)
    info.update({
        "split": split, "split_version": split_version, "directions": dirs,
        "layout_cfg": layout_cfg, "align_config": align_name,
        "align": {"c": align_cfg["gc"]["c"], "s2": align_cfg["gc"]["s2"], "filters": dict(align_cfg["filters"]),
                  "filters_applied": filters},
        "missing_hyps": missing, "hyps_not_in_refs": not_in_refs, "duplicate_hyp_records": dict(dup),
        "restricted_to_ids": None if ids is None else len(table),
        "run_status": manifest.get("status"), "run_name": info.get("run_name", manifest.get("run_name")),
    })
    return SegmentSet(triples=triples, docs=docs, info=info)


def build_triples(run: str | Path | None = None, split: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
    return build_segments(run, split, **kwargs).triples


def _doc_rows(triples: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, list[str]]:
    order: dict[str, int] = {}
    idx = [order.setdefault(t["sample_id"], len(order)) for t in triples]
    return np.asarray(idx, dtype=np.int64), list(order)


def _resample_counts(n_docs: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.choice(n_docs, size=(n, n_docs), replace=True)
    counts = np.zeros((n, n_docs), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(n), n_docs), draws.ravel()), 1)
    return counts


def _ci95(values: np.ndarray) -> list[float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def score_triples(
    triples: Sequence[Mapping[str, Any]], *, n_bootstrap: int = 1000, seed: int = 12345, per_segment: bool = True,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    if not triples:
        return {"n_segments": 0}, []
    hyps = [t["mt"] for t in triples]
    refs = [t["ref"] for t in triples]
    ds = M.doc_statistics(hyps, refs)
    out: dict[str, Any] = {"n_segments": len(triples), **ds.scores()}
    seg_chrf = ds.per_doc("chrf")
    out["segment_mean_chrf"] = float(np.mean(seg_chrf))
    doc_idx, doc_ids = _doc_rows(triples)
    out["n_docs"] = len(doc_ids)
    if n_bootstrap > 1:
        summed = {}
        for m, rows in ds.stats.items():
            acc = np.zeros((len(doc_ids), rows.shape[1]), dtype=np.int64)
            np.add.at(acc, doc_idx, rows)
            summed[m] = acc
        doc_ds = M.DocStats(stats=summed, metrics=ds.metrics)
        for m, ci in M.bootstrap_ci([], [], n=n_bootstrap, seed=seed, stats=doc_ds).items():
            out[f"{m}_ci95"] = ci
        out["bootstrap"] = {"n": n_bootstrap, "seed": seed, "unit": "document"}
    groups: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(triples):
        groups[t.get("kind_group", "text")].append(i)
    out["per_kind"] = {g: {"n_segments": len(rows), **ds.scores(np.asarray(rows))}
                       for g, rows in sorted(groups.items())}
    seg: list[dict[str, float]] = []
    if per_segment:
        sb = BLEU(effective_order=True)
        prepared = [(M.prepare_for_bleu(h), M.prepare_for_bleu(r)) for h, r in zip(hyps, refs)]
        seg = [{"chrf": round(c, 4), "bleu": round(sb.sentence_score(h, [r]).score, 4)}
               for c, (h, r) in zip(seg_chrf, prepared)]
    return out, seg


def coverage(docs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ok = [d for d in docs if d.get("status") == "ok"]
    src = sum(d["src_chars"] for d in ok)
    ref = sum(d["ref_chars"] for d in ok)
    src_kept = sum(d["src_chars_kept"] for d in ok)
    ref_kept = sum(d["ref_chars_kept"] for d in ok)
    dropped_src: Counter = Counter()
    dropped_ref: Counter = Counter()
    links: Counter = Counter()
    kept: Counter = Counter()
    for d in ok:
        dropped_src.update(d["src_chars_dropped"])
        dropped_ref.update(d["ref_chars_dropped"])
        links.update(d["links"])
        kept.update(d["kept_links"])
    return {
        "n_docs_scored": len(ok),
        "n_docs_skipped": {s: n for s, n in Counter(d.get("status") for d in docs if d.get("status") != "ok").items()},
        "src_chars": src, "src_chars_kept": src_kept, "src_coverage": round(src_kept / src, 4) if src else None,
        "ref_chars": ref, "ref_chars_kept": ref_kept, "ref_coverage": round(ref_kept / ref, 4) if ref else None,
        "src_chars_dropped": dict(dropped_src), "ref_chars_dropped": dict(dropped_ref),
        "links": dict(links), "kept_links": dict(kept),
        "mt_missing_units": sum(d.get("mt_missing_units", 0) for d in ok),
        "fallback_units_in_segments": sum(d.get("fallback_units", 0) for d in ok),
        "source_copied_units_in_segments": sum(d.get("source_copied_units", 0) for d in ok),
    }


def comet_by_direction(
    triples: Sequence[Mapping[str, Any]], scores: Mapping[str, float], *, n_bootstrap: int = 1000, seed: int = 12345,
    truncated: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    cut = set(truncated)
    for d in DIRECTIONS:
        ts = [t for t in triples if t["direction"] == d]
        if not ts:
            continue
        have = [t for t in ts if t["seg_id"] in scores]
        res: dict[str, Any] = {"n_segments": len(ts), "n_scored": len(have),
                               "n_truncated": sum(t["seg_id"] in cut for t in ts)}
        if have:
            vals = np.asarray([scores[t["seg_id"]] for t in have], dtype=np.float64)
            res["mean"] = float(vals.mean())
            doc_idx, doc_ids = _doc_rows(have)
            sums = np.bincount(doc_idx, weights=vals, minlength=len(doc_ids))
            ns = np.bincount(doc_idx, minlength=len(doc_ids)).astype(np.float64)
            res["doc_macro_mean"] = float(np.mean(sums / ns))
            if n_bootstrap > 1:
                counts = _resample_counts(len(doc_ids), n_bootstrap, seed)
                res["ci95"] = _ci95((counts @ sums) / (counts @ ns))
            groups: dict[str, list[float]] = defaultdict(list)
            for t, v in zip(have, vals):
                groups[t.get("kind_group", "text")].append(float(v))
            res["per_kind"] = {g: {"n_segments": len(v), "mean": float(np.mean(v))} for g, v in sorted(groups.items())}
        out[d] = res
    return out


def comet_input_records(triples: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [{"seg_id": t["seg_id"], "src": t["src"], "mt": t["mt"], "ref": t["ref"]} for t in triples]


def dumps_jsonl(records: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
