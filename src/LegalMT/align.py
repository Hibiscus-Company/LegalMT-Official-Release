"""Paragraph-level alignment of EN/VI document pairs and of hypothesis vs reference documents."""
from __future__ import annotations

import copy
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Iterable

from .layout import (
    ROW_KINDS,
    TEXT_KINDS,
    Anchor,
    DocSkeleton,
    Element,
    cell_is_translatable,
    parse_document,
)
from .textnorm import normalize_text

DEFAULT_ALIGN_CFG: dict = {
    "gc": {
        "c": 0.8952,
        "s2": 3.994,
        "priors": {"1-1": 0.89, "1-0": 0.005, "0-1": 0.005, "2-1": 0.0445, "1-2": 0.0445, "2-2": 0.011},
        "band": 60,
        "band_slack": 20,
        "skip_cost": 6.0,
        "num_weight": 3.87,
        "num_mode": "table",
        "num_table": {"one_sided": 4.337, "f1_lt_034": 4.152, "f1_lt_08": 2.667, "f1_ge_08": -3.048},
    },
    "filters": {
        "ratio_factor": 1.8,
        "min_num_f1": 0.34,
        "min_numbers": 2,
        "drop_unaligned": True,
    },
    "numbers": {"min_digits": 2},
    "layout": {},
    "keep_dropped": False,
}

_MOVES = ((1, 1), (1, 0), (0, 1), (2, 1), (1, 2), (2, 2))


def _deep_merge(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_cfg(cfg: dict | None) -> dict:
    out = _deep_merge(DEFAULT_ALIGN_CFG, cfg)
    for key in ("ratio", "c"):
        if cfg and key in cfg and not isinstance(cfg[key], dict):
            out["gc"]["c"] = float(cfg[key])
    if cfg and "s2" in cfg:
        out["gc"]["s2"] = float(cfg["s2"])
    return out


def load_align_config(name: str = "align") -> dict:
    from .paths import load_config

    return resolve_cfg(load_config(name))


_NUM = re.compile(r"\d+(?:[.,]\d{3})*(?:[.,]\d+)?")
_FOOTNOTE_MARK = re.compile(r"\[\d+\]")


def numeric_tokens(text: str, min_digits: int = 2) -> Counter:
    out: Counter = Counter()
    for m in _NUM.finditer(_FOOTNOTE_MARK.sub(" ", text)):
        tok = m.group(0).replace(".", "").replace(",", "").lstrip("0")
        if len(tok) >= min_digits:
            out[tok] += 1
    return out


def numeric_f1(a: Counter, b: Counter) -> float | None:
    na, nb = sum(a.values()), sum(b.values())
    if not na and not nb:
        return None
    if not na or not nb:
        return 0.0
    inter = sum((a & b).values())
    if not inter:
        return 0.0
    p, r = inter / na, inter / nb
    return 2 * p * r / (p + r)


NUMERIC_OUTCOMES = ("none", "one_sided", "f1_lt_034", "f1_lt_08", "f1_ge_08")


def numeric_outcome(a: Counter, b: Counter) -> str:
    if not a and not b:
        return "none"
    if not a or not b:
        return "one_sided"
    f1 = numeric_f1(a, b) or 0.0
    return "f1_lt_034" if f1 < 0.34 else ("f1_lt_08" if f1 < 0.8 else "f1_ge_08")


@dataclass
class AlignedPair:
    pair_id: str
    sample_id: str
    category: str
    order: int
    en: str
    vi: str
    en_units: list[int]
    vi_units: list[int]
    link: str
    method: str
    anchor: str | None
    len_ratio: float
    num_f1: float | None
    cost: float | None
    en_uids: list[str] = field(default_factory=list)
    vi_uids: list[str] = field(default_factory=list)
    drop_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _AUnit:
    text: str
    elem: int
    uid: str
    anchor: Anchor | None
    nums: Counter
    cell: int | None = None


def _element_units(skel: DocSkeleton, kinds: frozenset[str], min_digits: int, *, cells: bool) -> list[_AUnit]:
    units: list[_AUnit] = []
    for e in skel.elements:
        if e.kind not in kinds:
            continue
        if e.kind in TEXT_KINDS:
            units.append(_AUnit(e.text, e.idx, f"e{e.idx}", e.anchor, numeric_tokens(e.text, min_digits)))
        elif e.kind in ROW_KINDS and e.cells:
            if cells:
                for k, cell in enumerate(e.cells):
                    if cell_is_translatable(cell.text):
                        units.append(_AUnit(cell.text, e.idx, f"e{e.idx}c{k}", None, numeric_tokens(cell.text, min_digits), k))
            else:
                txt = " ".join(c.text for c in e.cells if cell_is_translatable(c.text))
                if txt:
                    units.append(_AUnit(txt, e.idx, f"e{e.idx}", None, numeric_tokens(txt, min_digits)))
    return units


def lcs_pairs(a: list, b: list) -> list[tuple[int, int]]:
    n, m = len(a), len(b)
    if not n or not m:
        return []
    if n * m > 25_000_000:
        import difflib

        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        return [(bl.a + k, bl.b + k) for bl in sm.get_matching_blocks() for k in range(bl.size)]
    rows: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        ai, row, nxt = a[i], rows[i], rows[i + 1]
        for j in range(m - 1, -1, -1):
            if ai == b[j]:
                row[j] = nxt[j + 1] + 1
            else:
                x, y = nxt[j], row[j + 1]
                row[j] = x if x >= y else y
    out: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            out.append((i, j))
            i += 1
            j += 1
        elif rows[i + 1][j] >= rows[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def anchor_chunks(src: list[_AUnit], tgt: list[_AUnit]) -> tuple[list[tuple[int, int, int, int, str | None]], int]:
    sa = [(k, u.anchor.key()) for k, u in enumerate(src) if u.anchor is not None]
    ta = [(k, u.anchor.key()) for k, u in enumerate(tgt) if u.anchor is not None]
    matched = lcs_pairs([x[1] for x in sa], [x[1] for x in ta])
    cuts = [(sa[i][0], ta[j][0], src[sa[i][0]].anchor.label) for i, j in matched]
    bounds = [(0, 0, None)] + cuts + [(len(src), len(tgt), None)]
    chunks = []
    for (s0, t0, label), (s1, t1, _) in zip(bounds, bounds[1:]):
        if s1 > s0 or t1 > t0:
            chunks.append((s0, s1, t0, t1, label))
    return chunks, len(matched)


_SQRT2 = math.sqrt(2.0)
_LOG_SQRT_PI = 0.5 * math.log(math.pi)


def _neglog_tail(z: float) -> float:
    x = abs(z) / _SQRT2
    if x < 25.0:
        return -math.log(max(math.erfc(x), 1e-300))
    return x * x + math.log(x) + _LOG_SQRT_PI


class _GC:
    def __init__(self, gc: dict):
        self.c = float(gc["c"])
        self.s2 = float(gc["s2"])
        self.band = int(gc.get("band", 60))
        self.band_slack = int(gc.get("band_slack", 20))
        self.num_weight = float(gc.get("num_weight", 0.0))
        sc = gc.get("skip_cost")
        self.skip_cost = float("inf") if sc is None else float(sc)
        self.num_mode = gc.get("num_mode", "either")
        if self.num_mode not in ("either", "both", "table"):
            raise ValueError(f"gc.num_mode must be either|both|table, got {self.num_mode!r}")
        self.num_both = self.num_mode == "both"
        self.num_table = {k: float(v) for k, v in (gc.get("num_table") or {}).items()}
        pri = gc["priors"]
        self.move_cost = {mv: -math.log(float(pri[f"{mv[0]}-{mv[1]}"])) for mv in _MOVES}

    def length_cost(self, le: int, lv: int) -> float:
        mean = (le + lv / self.c) / 2.0
        if mean <= 0:
            return 0.0
        z = (lv - self.c * le) / math.sqrt(self.s2 * mean)
        return _neglog_tail(z)

    def num_cost(self, a: Counter | None, b: Counter | None) -> float:
        if not a and not b:
            return 0.0
        if self.num_mode == "table":
            return self.num_table[numeric_outcome(a or Counter(), b or Counter())]
        if not self.num_weight or (self.num_both and not (a and b)):
            return 0.0
        return self.num_weight * (1.0 - (numeric_f1(a or Counter(), b or Counter()) or 0.0))

    def link_cost(self, mv: tuple[int, int], le: int, lv: int, a: Counter | None = None, b: Counter | None = None) -> float:
        lc = self.length_cost(le, lv)
        if not (mv[0] and mv[1]):
            return self.move_cost.get(mv, self.move_cost[(1, 0)]) + min(lc, self.skip_cost)
        return self.move_cost.get(mv, self.move_cost[(1, 1)]) + lc + self.num_cost(a, b)

    def align(self, src: list[_AUnit], tgt: list[_AUnit]) -> list[tuple[int, int, int, int, float]]:
        n, m = len(src), len(tgt)
        if n == 0:
            return [(0, 0, j, j + 1, self.link_cost((0, 1), 0, len(tgt[j].text))) for j in range(m)]
        if m == 0:
            return [(i, i + 1, 0, 0, self.link_cost((1, 0), len(src[i].text), 0)) for i in range(n)]
        ps = [0] * (n + 1)
        for i, u in enumerate(src):
            ps[i + 1] = ps[i] + len(u.text)
        pt = [0] * (m + 1)
        for j, u in enumerate(tgt):
            pt[j + 1] = pt[j] + len(u.text)
        sn = {1: [u.nums or None for u in src], 2: [None] + [((src[i - 1].nums + src[i].nums) or None) for i in range(1, n)]}
        tn = {1: [u.nums or None for u in tgt], 2: [None] + [((tgt[j - 1].nums + tgt[j].nums) or None) for j in range(1, m)]}
        band = max(self.band, abs(n - m) + self.band_slack)
        lo = [0] * (n + 1)
        hi = [0] * (n + 1)
        for i in range(n + 1):
            centre = i * m / n
            lo[i] = max(0, int(centre) - band)
            hi[i] = min(m, int(math.ceil(centre)) + band)
        inf = float("inf")
        cost: list[list[float]] = []
        back: list[list[int]] = []
        mv_cost = [self.move_cost[mv] for mv in _MOVES]
        c, s2, skip_cost = self.c, self.s2, self.skip_cost
        use_num = self.num_mode == "table" or self.num_weight != 0.0
        num_cost = self.num_cost
        for i in range(n + 1):
            width = hi[i] - lo[i] + 1
            crow = [inf] * width
            brow = [-1] * width
            for j in range(lo[i], hi[i] + 1):
                if i == 0 and j == 0:
                    crow[0] = 0.0
                    continue
                best, arg = inf, -1
                for k, (di, dj) in enumerate(_MOVES):
                    pi, pj = i - di, j - dj
                    if pi < 0 or pj < 0 or pj < lo[pi] or pj > hi[pi]:
                        continue
                    prev = (crow if di == 0 else cost[pi])[pj - lo[pi]]
                    if prev == inf:
                        continue
                    le = ps[i] - ps[pi]
                    lv = pt[j] - pt[pj]
                    mean = (le + lv / c) / 2.0
                    x = abs(lv - c * le) / math.sqrt(s2 * mean) / _SQRT2 if mean > 0 else 0.0
                    lc = -math.log(max(math.erfc(x), 1e-300)) if x < 25.0 else x * x + math.log(x) + _LOG_SQRT_PI
                    if not (di and dj) and lc > skip_cost:
                        lc = skip_cost
                    total = prev + mv_cost[k] + lc
                    if di and dj and use_num:
                        a, b = sn[di][i - 1], tn[dj][j - 1]
                        if a is not None or b is not None:
                            total += num_cost(a, b)
                    if total < best:
                        best, arg = total, k
                crow[j - lo[i]] = best
                brow[j - lo[i]] = arg
            cost.append(crow)
            back.append(brow)
        links: list[tuple[int, int, int, int, float]] = []
        i, j = n, m
        if cost[n][m - lo[n]] == inf:
            raise RuntimeError("Gale-Church DP found no path")
        while i or j:
            k = back[i][j - lo[i]]
            di, dj = _MOVES[k]
            pi, pj = i - di, j - dj
            links.append((pi, i, pj, j, cost[i][j - lo[i]] - cost[pi][pj - lo[pi]]))
            i, j = pi, pj
        links.reverse()
        return links


def _chunk_links(src: list[_AUnit], tgt: list[_AUnit], gc: _GC) -> tuple[list[tuple[int, int, int, int, float]], str]:
    if len(src) == len(tgt):
        links = []
        for k, (a, b) in enumerate(zip(src, tgt)):
            links.append((k, k + 1, k, k + 1, gc.link_cost((1, 1), len(a.text), len(b.text), a.nums, b.nums)))
        return links, "zip"
    return gc.align(src, tgt), "gc"


def _zone_rows(skel: DocSkeleton, kind: str) -> list[Element]:
    return [e for e in skel.elements if e.kind == kind and e.cells and any(cell_is_translatable(c.text) for c in e.cells)]


_HEADER_CELL_KEYS = (
    ("docno", re.compile(r"^(?:No|NO)\b"), re.compile(r"^(?:Số|SỐ)\b")),
    ("motto", re.compile(r"^Independence\b", re.I), re.compile(r"^Độc lập\b", re.I)),
    ("country", re.compile(r"^(?:THE\s+)?SOCIALIST\s+REPUBLIC\b", re.I), re.compile(r"^CỘNG\s+H(?:ÒA|OÀ)\b", re.I)),
)


def _zone_cell_anchor(u: _AUnit, lang: str, header: bool) -> Anchor | None:
    if header:
        for key, rx_en, rx_vi in _HEADER_CELL_KEYS:
            if (rx_en if lang == "en" else rx_vi).match(u.text):
                return Anchor(key, "", False)
    if u.nums:
        return Anchor("num", ",".join(sorted(u.nums.elements())), False)
    return None


def _zone_sides(rows: list[Element], min_digits: int) -> list[list[tuple[str, _AUnit]]]:
    def centre(c) -> float:
        return c.col + len(c.text) / 2.0

    lefts = [centre(e.cells[0]) for e in rows if len(e.cells) >= 2]
    rights = [centre(c) for e in rows if len(e.cells) >= 2 for c in e.cells[1:]]
    ml = sum(lefts) / len(lefts) if lefts else None
    mr = sum(rights) / len(rights) if rights else None
    out: list[list[tuple[str, _AUnit]]] = []
    for e in rows:
        row: list[tuple[str, _AUnit]] = []
        for k, cell in enumerate(e.cells):
            if not cell_is_translatable(cell.text):
                continue
            if len(e.cells) >= 2:
                side = "L" if k == 0 else "R"
            elif ml is None or mr is None:
                side = "L"
            else:
                side = "L" if abs(centre(cell) - ml) <= abs(centre(cell) - mr) else "R"
            row.append((side, _AUnit(cell.text, e.idx, f"e{e.idx}c{k}", None, numeric_tokens(cell.text, min_digits), k)))
        out.append(row)
    return out


def _align_zone(sk_en: DocSkeleton, sk_vi: DocSkeleton, kind: str, gc: _GC, min_digits: int):
    rows_en, rows_vi = _zone_rows(sk_en, kind), _zone_rows(sk_vi, kind)
    if not rows_en and not rows_vi:
        return [], "none"
    sides_en, sides_vi = _zone_sides(rows_en, min_digits), _zone_sides(rows_vi, min_digits)
    zipped_link, zipped_method = ("header", "header") if kind == "header_row" else ("1-1", "zip")
    out = []
    if [[sd for sd, _ in r] for r in sides_en] == [[sd for sd, _ in r] for r in sides_vi]:
        for re_, rv in zip(sides_en, sides_vi):
            for (_, a), (_, b) in zip(re_, rv):
                out.append(([a], [b], zipped_link, zipped_method, gc.link_cost((1, 1), len(a.text), len(b.text), a.nums, b.nums), False))
        return out, "cells"

    def columns(sides: list[list[tuple[str, _AUnit]]]) -> list[list[_AUnit]]:
        cols = [[u for r in sides for sd, u in r if sd == side] for side in ("L", "R")]
        return [c for c in cols if c]

    cols_en, cols_vi = columns(sides_en), columns(sides_vi)
    if len(cols_en) != len(cols_vi):
        cols_en = [[u for col in cols_en for u in col]]
        cols_vi = [[u for col in cols_vi for u in col]]
    ratio_filter = kind != "header_row"
    for ce, cv in zip(cols_en, cols_vi):
        for lang, col in (("en", ce), ("vi", cv)):
            for u in col:
                u.anchor = _zone_cell_anchor(u, lang, kind == "header_row")
        chunks, _ = anchor_chunks(ce, cv)
        for s0, s1, t0, t1, _label in chunks:
            se, sv = ce[s0:s1], cv[t0:t1]
            links, method = _chunk_links(se, sv, gc)
            for i0, i1, j0, j1, cost in links:
                if method == "zip":
                    out.append((se[i0:i1], sv[j0:j1], zipped_link, zipped_method, cost, False))
                else:
                    out.append((se[i0:i1], sv[j0:j1], f"{i1 - i0}-{j1 - j0}", "gc", cost, ratio_filter))
    return out, "columns"


def _row_shape(cells: list[_AUnit]) -> list[int]:
    shape: list[int] = []
    last = None
    for u in cells:
        if u.elem != last:
            shape.append(0)
            last = u.elem
        shape[-1] += 1
    return shape


def _table_links(te: list[_AUnit], tv: list[_AUnit], gc: _GC) -> tuple[list[tuple], bool]:
    if te and tv and _row_shape(te) == _row_shape(tv):
        return [([a], [b], "1-1", "zip", gc.link_cost((1, 1), len(a.text), len(b.text), a.nums, b.nums), False, None)
                for a, b in zip(te, tv)], True
    out = [([a], [], "1-0", "table", None, False, "table_mismatch") for a in te]
    out += [([], [b], "0-1", "table", None, False, "table_mismatch") for b in tv]
    return out, False


_BODY_KINDS = frozenset({"heading", "paragraph", "table_row"})


def align_documents(
    en_text: str, vi_text: str, *, sample_id: str = "", category: str = "", cfg: dict | None = None,
) -> tuple[list[AlignedPair], dict]:
    t0 = time.perf_counter()
    cfg = resolve_cfg(cfg)
    gc = _GC(cfg["gc"])
    flt = cfg["filters"]
    min_digits = int(cfg["numbers"]["min_digits"])
    en_norm, vi_norm = normalize_text(en_text), normalize_text(vi_text)
    sk_en = parse_document(en_norm, "en", cfg["layout"])
    sk_vi = parse_document(vi_norm, "vi", cfg["layout"])

    raw_links: list[tuple] = []
    header_links, header_mode = _align_zone(sk_en, sk_vi, "header_row", gc, min_digits)
    for s, t, link, method, cost, rf in header_links:
        raw_links.append((s, t, link, method, "header", cost, rf, False, None))

    body_en = _element_units(sk_en, _BODY_KINDS, min_digits, cells=True)
    body_vi = _element_units(sk_vi, _BODY_KINDS, min_digits, cells=True)
    chunks, n_matched = anchor_chunks(body_en, body_vi)
    n_equal = 0
    tables = Counter()
    for s0, s1, t0_, t1, label in chunks:
        ce = [u for u in body_en[s0:s1] if u.cell is None]
        cv = [u for u in body_vi[t0_:t1] if u.cell is None]
        links, method = _chunk_links(ce, cv, gc)
        n_equal += method == "zip"
        chunk_links = []
        key = -1
        for i0, i1, j0, j1, cost in links:
            if i1 > i0:
                key = ce[i0].elem
            chunk_links.append((key, 0, (ce[i0:i1], cv[j0:j1], f"{i1 - i0}-{j1 - j0}", method, label, cost, True,
                                         method == "gc", None)))
        te = [u for u in body_en[s0:s1] if u.cell is not None]
        tv = [u for u in body_vi[t0_:t1] if u.cell is not None]
        if te or tv:
            tlinks, matched = _table_links(te, tv, gc)
            tables["matched" if matched else "mismatched"] += 1
            for s, t, link, tmethod, cost, rf, forced in tlinks:
                tkey = s[0].elem if (s and not forced) else 10**9
                chunk_links.append((tkey, 1, (s, t, link, tmethod, label, cost, rf, False, forced)))
            chunk_links.sort(key=lambda x: (x[0], x[1]))
        raw_links.extend(x[2] for x in chunk_links)

    sig_links, sig_mode = _align_zone(sk_en, sk_vi, "signature_row", gc, min_digits)
    for s, t, link, method, cost, rf in sig_links:
        raw_links.append((s, t, link, method, "signature", cost, rf, False, None))

    c = gc.c
    lo_ratio, hi_ratio = c / float(flt["ratio_factor"]), c * float(flt["ratio_factor"])
    stats: dict = {
        "sample_id": sample_id, "category": category,
        "family_en": sk_en.layout_family, "family_vi": sk_vi.layout_family,
        "chars_en": len(en_norm), "chars_vi": len(vi_norm),
        "nonws_en": sum(not ch.isspace() for ch in en_norm), "nonws_vi": sum(not ch.isspace() for ch in vi_norm),
        "units_en": len(body_en), "units_vi": len(body_vi),
        "anchors_en": sum(u.anchor is not None for u in body_en), "anchors_vi": sum(u.anchor is not None for u in body_vi),
        "anchors_matched": n_matched, "chunks": len(chunks), "chunks_equal": n_equal,
        "header_mode": header_mode, "signature_mode": sig_mode,
        "table_chunks": dict(tables),
        "table_cells_en": sum(u.cell is not None for u in body_en), "table_cells_vi": sum(u.cell is not None for u in body_vi),
        "links": Counter(), "kept_links": Counter(),
        "kept_en_chars": Counter(), "kept_vi_chars": Counter(), "kept_nonws_en": 0, "kept_nonws_vi": 0,
        "kept_en_chars_by_link": Counter(),
        "dropped_en_chars": Counter(), "dropped_vi_chars": Counter(),
        "gc_en_chars": 0, "gc_nm_en_chars": 0,
        "nm_unequal": Counter(),
        "gc11_unequal": Counter(),
    }
    pairs: list[AlignedPair] = []
    for order, (src, tgt, link, method, anchor, cost, ratio_filter, unequal, forced) in enumerate(raw_links):
        en = " ".join(u.text for u in src)
        vi = " ".join(u.text for u in tgt)
        ne: Counter = Counter()
        nv: Counter = Counter()
        for u in src:
            ne.update(u.nums)
        for u in tgt:
            nv.update(u.nums)
        f1 = numeric_f1(ne, nv)
        ratio = len(vi) / len(en) if en else 0.0
        stats["links"][link] += 1
        if unequal and src and tgt:
            stats["gc_en_chars"] += len(en)
            bucket = stats["gc11_unequal"] if link == "1-1" else stats["nm_unequal"]
            if link != "1-1":
                stats["gc_nm_en_chars"] += len(en)
            if f1 is not None:
                bucket["with_numbers"] += 1
                bucket["f1_ge_0.8"] += f1 >= 0.8
                bucket["f1_eq_0"] += f1 == 0.0
                if ne and nv:
                    bucket["both_with_numbers"] += 1
                    bucket["both_f1_ge_0.8"] += f1 >= 0.8
                    bucket["both_f1_eq_0"] += f1 == 0.0
        reason = forced
        if reason:
            pass
        elif not src or not tgt:
            if flt.get("drop_unaligned", True):
                reason = "unaligned"
        elif ratio_filter and not (lo_ratio <= ratio <= hi_ratio):
            reason = "length_ratio"
        elif (sum(ne.values()) >= int(flt["min_numbers"]) and sum(nv.values()) >= int(flt["min_numbers"])
              and f1 is not None and f1 < float(flt["min_num_f1"])):
            reason = "numeric"
        if reason:
            stats["dropped_en_chars"][reason] += len(en)
            stats["dropped_vi_chars"][reason] += len(vi)
        else:
            stats["kept_links"][link] += 1
            stats["kept_en_chars"][method] += len(en)
            stats["kept_vi_chars"][method] += len(vi)
            stats["kept_en_chars_by_link"][link] += len(en)
            stats["kept_nonws_en"] += sum(not ch.isspace() for ch in en)
            stats["kept_nonws_vi"] += sum(not ch.isspace() for ch in vi)
        if reason and not cfg.get("keep_dropped"):
            continue
        pairs.append(AlignedPair(
            pair_id=f"{sample_id}#{order:05d}", sample_id=sample_id, category=category, order=order,
            en=en, vi=vi,
            en_units=sorted({u.elem for u in src}), vi_units=sorted({u.elem for u in tgt}),
            link=link, method=method, anchor=anchor, len_ratio=round(ratio, 4),
            num_f1=None if f1 is None else round(f1, 4), cost=None if cost is None else round(cost, 4),
            en_uids=[u.uid for u in src], vi_uids=[u.uid for u in tgt], drop_reason=reason,
        ))
    for k in ("links", "kept_links", "kept_en_chars", "kept_vi_chars", "kept_en_chars_by_link",
              "dropped_en_chars", "dropped_vi_chars", "nm_unequal", "gc11_unequal"):
        stats[k] = dict(stats[k])
    stats["seconds"] = round(time.perf_counter() - t0, 4)
    return pairs, stats


def align_elements(src_elems: list[Element], tgt_elems: list[Element], cfg: dict | None = None) -> list[tuple[list[int], list[int], str, float]]:
    cfg = resolve_cfg(cfg)
    gc = _GC(cfg["gc"])
    min_digits = int(cfg["numbers"]["min_digits"])
    kinds = frozenset(k for k in ("header_row", "heading", "paragraph", "table_row", "signature_row"))
    su = _element_units(DocSkeleton(list(src_elems), "", "", False), kinds, min_digits, cells=False)
    tu = _element_units(DocSkeleton(list(tgt_elems), "", "", False), kinds, min_digits, cells=False)
    out: list[tuple[list[int], list[int], str, float]] = []
    chunks, _ = anchor_chunks(su, tu)
    for s0, s1, t0, t1, _label in chunks:
        ce, cv = su[s0:s1], tu[t0:t1]
        links, _method = _chunk_links(ce, cv, gc)
        for i0, i1, j0, j1, cost in links:
            out.append(([u.elem for u in ce[i0:i1]], [u.elem for u in cv[j0:j1]], f"{i1 - i0}-{j1 - j0}", round(cost, 4)))
    return out


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if not n:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def fit_length_model(pairs: Iterable[tuple[int, int]], *, clip_sigmas: float = 5.0) -> dict:
    le_l, lv_l = [], []
    for le, lv in pairs:
        if le > 0 and lv > 0:
            le_l.append(le)
            lv_l.append(lv)
    if not le_l:
        raise ValueError("no usable length pairs")
    c = sum(lv_l) / sum(le_l)
    r = [(lv - c * le) / math.sqrt((le + lv / c) / 2.0) for le, lv in zip(le_l, lv_l)]
    n = len(r)
    mean = sum(r) / n
    var_raw = sum((x - mean) ** 2 for x in r) / n
    med = _median(r)
    mad = _median([abs(x - med) for x in r])
    sigma = 1.4826 * mad
    kept = [x for x in r if abs(x - med) <= clip_sigmas * sigma] if sigma > 0 else r
    km = sum(kept) / len(kept)
    var_clip = sum((x - km) ** 2 for x in kept) / len(kept)
    return {
        "c": c, "s2": var_clip, "n_pairs": n, "clip_sigmas": clip_sigmas, "clipped_share": 1 - len(kept) / n,
        "residual_mean": km, "s2_unclipped": var_raw, "s2_mad": sigma ** 2,
        "median_sq_residual_prototype": _median([x * x for x in r]),
    }


def fit_numeric_weight(f1_values: Iterable[float], *, hi: float = 0.8) -> dict:
    vals = list(f1_values)
    if not vals:
        raise ValueError("no F1 values")
    p_hi = sum(v >= hi for v in vals) / len(vals)
    p_zero = sum(v == 0.0 for v in vals) / len(vals)
    w = math.log(max(p_hi, 1e-6) / max(p_zero, 1e-6))
    return {"num_weight": w, "p_f1_ge_hi": p_hi, "p_f1_eq_0": p_zero, "n": len(vals), "hi": hi}


def fit_numeric_table(aligned: Iterable[str], shifted: Iterable[str]) -> dict:
    ca, cs = Counter(aligned), Counter(shifted)
    na, ns = sum(ca.values()), sum(cs.values())
    if not na or not ns:
        raise ValueError("need aligned and shifted outcomes")

    def llr(o: str) -> float:
        pa = max(ca[o], 1) / na
        ps = max(cs[o], 1) / ns
        return math.log(pa / ps)

    base = llr("none")
    table = {o: round(-(llr(o) - base), 3) for o in NUMERIC_OUTCOMES if o != "none"}
    return {"num_table": table, "aligned_counts": dict(ca), "shifted_counts": dict(cs),
            "n_aligned": na, "n_shifted": ns}
