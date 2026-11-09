"""Layout parser for LegalMT documents, shared by alignment, SFT chunking and inference."""
from __future__ import annotations

import re
import textwrap
from collections import Counter
from dataclasses import dataclass, field

from .textnorm import normalize_text

DEFAULT_LAYOUT_CFG: dict = {
    "min_gap": 2,
    "row_min_gap": 3,
    "header_min_gap": 5,
    "header_max_lines": 12,
    "header_line_max_chars": 90,
    "signature_max_lines": 15,
    "signature_line_max_chars": 60,
    "signature_min_indent": 20,
    "centred_indent": 8,
    "page_number_min_indent": 20,
    "blanksep_blank_ratio": 0.25,
    "dense_blank_ratio": 0.05,
    "short_line_frac": 0.85,
    "very_short_line_frac": 0.5,
    "unwrapped_long_share": 0.2,
    "unwrapped_line_chars": 200,
    "max_paragraph_chars": 3000,
    "title_max_chars": 300,
    "join_centred_caps": False,
}

KINDS = ("header_row", "heading", "paragraph", "table_row", "signature_row", "separator", "page_number")
ROW_KINDS = frozenset({"header_row", "table_row", "signature_row"})
TEXT_KINDS = frozenset({"heading", "paragraph"})
COPY_KINDS = frozenset({"separator", "page_number"})


@dataclass
class Cell:
    text: str
    col: int


@dataclass
class Anchor:
    type: str
    number: str
    quoted: bool

    @property
    def label(self) -> str:
        return f"{self.type}:{self.number}"

    def key(self) -> tuple[str, str, bool]:
        return (self.type, self.number, self.quoted)


@dataclass
class Element:
    idx: int
    kind: str
    text: str
    indent: int
    lines: tuple[int, int]
    cells: list[Cell] | None = None
    blank_before: int = 0
    anchor: Anchor | None = None


@dataclass
class DocSkeleton:
    elements: list[Element]
    lang: str
    layout_family: str
    trailing_newline: bool
    trailing_blank: int = 0
    stats: dict = field(default_factory=dict)


@dataclass
class Unit:
    uid: str
    element_idx: int
    cell_idx: int | None
    text: str
    kind: str


_WORDCHAR = re.compile(r"\w")
_INNER_GAP = re.compile(r"\S( +)(?=\S)")
_TERM = re.compile(r"[.;:!?…”\"’)\]]\s*$")
_SENT_END = re.compile(r"[.;!?]\s*$")
_LIST_START = re.compile(
    r"^\s*[“\"‘]?\s*(?:"
    r"\d{1,3}(?:\.\d{1,3})*[.)]\s"
    r"|[a-zđ]{1,2}[)/]\s"
    r"|\([a-zđ0-9]{1,3}\)\s"
    r"|[IVXLC]+\s?\.\s"
    r"|[A-ZĐ]\.\s+[A-ZĐ]"
    r"|[-\u2013\u2014+•*]\s"
    r")"
)
_PAGE_NUM = re.compile(r"(?:[-\u2013\u2014]\s*)?\d{1,4}(?:\s*[-\u2013\u2014])?")
_PAGE_WORD = re.compile(r"(?:Page|PAGE|Trang|TRANG)\s+\d{1,4}(?:\s*/\s*\d{1,4})?")
_DASHED_PAGE = re.compile(r"[-\u2013\u2014]\s*\d{1,4}\s*[-\u2013\u2014]")
_YEAR = re.compile(r"\b(?:19|20)\d\d\b")
_WORD = re.compile(r"[^\W\d_]+")
_TRAIL_FRAG = re.compile(r"([^\W\d_]+)-$")
_LEAD_WORD = re.compile(r"^([^\W\d_]+)")
_KEEP_SPACE_AFTER_HYPHEN = frozenset({"and", "or", "to", "nor", "và", "hoặc", "đến"})

_QUOTES = "“\"‘"
_Q = r"^\s*[“\"‘]?\s*"
_FOOT = r"(?:\[\d+\])?"
_SEP = r"(\.\s*[-\u2013\u2014]|\.|:|[-\u2013\u2014])?"

_ARTICLE = {
    "en": re.compile(_Q + r"(?:Article|ARTICLE)\s*(\d+[a-z]?)\s*" + _FOOT + r"\s*" + _SEP + r"\s*(.*)$"),
    "vi": re.compile(_Q + r"(?:[ĐÐ]iều|[ĐÐ]IỀU)\s*(\d+[a-zđ]?)\s*" + _FOOT + r"\s*" + _SEP + r"\s*(.*)$"),
}
_PARTWORDS_EN = r"ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten"
_OTHER_HEADS = {
    "en": [
        ("part", r"(?:Part|PART)\s+", r"(\d+|[IVXLC]+|" + _PARTWORDS_EN + r")"),
        ("chapter", r"(?:Chapter|CHAPTER)\s*", r"([IVXLC]+|\d+)"),
        ("section", r"(?:Section|SECTION)\s+", r"(\d+|[IVXLC]+)"),
        ("appendix", r"(?:Appendix|APPENDIX|Annex|ANNEX)\b\s*", r"(?:No\.?\s*)?([IVXLC]+\b|\d+[a-z]?|[A-Z]\b)?"),
    ],
    "vi": [
        ("part", r"(?:Phần|PHẦN)\s+", r"(\d+|[IVXLC]+|(?:THỨ|Thứ|thứ)\s+[^\W\d_]+)"),
        ("chapter", r"(?:Chương|CHƯƠNG)\s*", r"([IVXLC]+|\d+)"),
        ("section", r"(?:Mục|MỤC)\s+", r"(\d+|[IVXLC]+)"),
        ("appendix", r"(?:Phụ lục|PHỤ LỤC|Phụ Lục)\b\s*", r"(?:(?:số|SỐ)\s*)?([IVXLC]+\b|\d+[a-zđ]?|[A-Z]\b)?"),
    ],
}
_NUM_END = {"part": r"(?![^\W\d_])", "chapter": r"(?![^\W\d_])", "section": r"(?![^\W\d_])", "appendix": ""}
_OTHER_RE = {
    lang: [(t, re.compile(_Q + kw + num + _NUM_END[t] + r"\s*" + _FOOT + r"\s*([.:\u2013\u2014-]?)\s*(.*)$"))
           for t, kw, num in pats]
    for lang, pats in _OTHER_HEADS.items()
}
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_PARTWORD = {
    "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4", "FIVE": "5", "SIX": "6", "SEVEN": "7",
    "EIGHT": "8", "NINE": "9", "TEN": "10",
    "NHẤT": "1", "HAI": "2", "BA": "3", "TƯ": "4", "BỐN": "4", "NĂM": "5", "SÁU": "6", "BẢY": "7",
    "TÁM": "8", "CHÍN": "9", "MƯỜI": "10",
}


def _roman_to_int(s: str) -> str:
    if s and all(ch in _ROMAN for ch in s):
        total = 0
        for i, ch in enumerate(s):
            v = _ROMAN[ch]
            total += -v if i + 1 < len(s) and _ROMAN[s[i + 1]] > v else v
        return str(total)
    return s


def canonical_number(raw: str | None) -> str:
    if not raw:
        return ""
    x = raw.strip().replace("đ", "d").replace("Đ", "D").replace("Ð", "D")
    word = re.sub(r"^(?:THỨ|Thứ|thứ)\s+", "", x).upper()
    if word in _PARTWORD:
        return _PARTWORD[word]
    m = re.fullmatch(r"0*(\d+)(.*)", x)
    if m:
        return m.group(1) + m.group(2)
    return _roman_to_int(x)


def _starts_upperish(rest: str) -> bool:
    if not rest:
        return False
    if rest[0].isupper() or rest[0] in "([“":
        return True
    return rest[0] == '"' and len(rest) > 1 and not rest[1].isspace()


def match_heading(line: str, lang: str) -> tuple[Anchor, str] | None:
    if lang not in _ARTICLE:
        raise ValueError(f"unsupported language {lang!r}")
    stripped = line.strip()
    if not stripped:
        return None
    quoted = stripped[0] in _QUOTES
    m = _ARTICLE[lang].match(line)
    if m:
        num, sep, rest = m.group(1), m.group(2), m.group(3).strip()
        strength = None
        has_word = bool(_WORDCHAR.search(rest))
        if sep and has_word and _starts_upperish(rest):
            strength = "strong"
        elif sep and not has_word:
            strength = "weak"
        elif sep and rest and sep.strip() != ":" and not rest[0].isdigit():
            strength = "weak"
        elif not sep and not rest:
            strength = "weak"
        if strength:
            return Anchor("article", canonical_number(num), quoted), strength
    for typ, rx in _OTHER_RE[lang]:
        m = rx.match(line)
        if not m:
            continue
        num, rest = m.group(1), m.group(3).strip()
        if rest and not _starts_upperish(rest):
            continue
        if typ == "appendix" and not num and rest and not rest.isupper():
            continue
        return Anchor(typ, canonical_number(num), quoted), "weak"
    return None


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_separator(stripped: str) -> bool:
    return bool(stripped) and not _WORDCHAR.search(stripped)


def _max_inner_gap(stripped: str) -> int:
    gaps = [len(m.group(1)) for m in _INNER_GAP.finditer(stripped)]
    return max(gaps) if gaps else 0


def split_cells(line: str, min_gap: int) -> list[Cell]:
    cells: list[Cell] = []
    rx = re.compile(r"\S+(?: {1,%d}\S+)*" % max(1, min_gap - 1))
    for m in rx.finditer(line):
        cells.append(Cell(text=m.group(0), col=m.start()))
    return cells


def _starts_lower(stripped: str) -> bool:
    if not stripped:
        return False
    ch = stripped[0]
    if ch.isalpha():
        return ch.islower()
    if ch in ",;":
        return True
    if ch == "(" and len(stripped) > 1 and stripped[1].isalpha():
        return stripped[1].islower()
    return False


def _is_upper_title(text: str, max_chars: int) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    return 4 <= len(letters) and len(text) <= max_chars and not any(ch.islower() for ch in letters)


def _percentile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    return sorted_vals[int(q * (len(sorted_vals) - 1))]


def layout_family(n_blank: int, n_nonblank: int, cfg: dict | None = None) -> str:
    c = {**DEFAULT_LAYOUT_CFG, **(cfg or {})}
    if n_nonblank == 0:
        return "blanksep"
    ratio = n_blank / n_nonblank
    if ratio >= c["blanksep_blank_ratio"]:
        return "blanksep"
    if ratio < c["dense_blank_ratio"]:
        return "dense"
    return "mixed"


def _header_zone(lines: list[str], nb: list[int], lang: str, c: dict) -> list[int]:
    last = -1
    for pos, i in enumerate(nb[: c["header_max_lines"]]):
        s = lines[i].strip()
        if _is_separator(s) or _max_inner_gap(s) >= c["header_min_gap"]:
            last = pos
        elif (len(s) > c["header_line_max_chars"] or match_heading(lines[i], lang)
              or (any(ch.islower() for ch in s) and _indent(lines[i]) < 4)):
            break
    while last + 1 < len(nb) and last >= 0:
        i = nb[last + 1]
        s = lines[i].strip()
        if (i - nb[last] <= 2 and _indent(lines[i]) >= c["signature_min_indent"] and _YEAR.search(s)
                and len(s) <= c["header_line_max_chars"] and _max_inner_gap(s) < c["header_min_gap"]):
            last += 1
        else:
            break
    return nb[: last + 1]


def _signature_zone(lines: list[str], nb: list[int], start_pos: int, base_indent: int, lang: str, c: dict) -> list[int]:
    tail_from = max(start_pos, len(nb) - c["signature_max_lines"])
    zone: list[int] = []
    for pos in range(len(nb) - 1, tail_from - 1, -1):
        i = nb[pos]
        raw = lines[i]
        s = raw.strip()
        two_col = (_max_inner_gap(s) >= c["header_min_gap"] and not any(ch.isdigit() for ch in s)
                   and all(len(cell.text) <= c["signature_line_max_chars"] for cell in split_cells(raw, c["min_gap"])))
        short = (len(s) <= c["signature_line_max_chars"] and not _SENT_END.search(s) and not s.endswith(":")
                 and not _LIST_START.match(raw) and match_heading(raw, lang) is None)
        if two_col or _is_separator(s) or short:
            zone.append(i)
        else:
            break
    zone.reverse()

    def anchored(i: int) -> bool:
        s = lines[i].strip()
        two_col = _max_inner_gap(s) >= c["header_min_gap"] and not any(ch.isdigit() for ch in s)
        return two_col or _indent(lines[i]) - base_indent >= c["signature_min_indent"]

    while zone:
        i = zone[0]
        prev_blank = i > 0 and not lines[i - 1].strip()
        if prev_blank or anchored(i):
            break
        zone.pop(0)
    if not any(anchored(i) for i in zone):
        return []
    return zone


def parse_document(text: str, lang: str, cfg: dict | None = None) -> DocSkeleton:
    c = {**DEFAULT_LAYOUT_CFG, **(cfg or {})}
    text = normalize_text(text)
    trailing_newline = text.endswith("\n")
    lines = text.split("\n")
    if trailing_newline:
        lines = lines[:-1]
    lines = [ln.rstrip() for ln in lines]
    nb = [i for i, ln in enumerate(lines) if ln.strip()]
    family = layout_family(len(lines) - len(nb), len(nb), c)
    stats: Counter = Counter()

    cls: dict[int, str] = {}
    header = _header_zone(lines, nb, lang, c)
    for i in header:
        cls[i] = "separator" if _is_separator(lines[i].strip()) else "header_row"
    body_start = len(header)

    for i in nb[body_start:]:
        s = lines[i].strip()
        if _is_separator(s):
            cls[i] = "separator"
        elif _max_inner_gap(s) >= c["row_min_gap"]:
            cls[i] = "table_row"
        else:
            cls[i] = "text"
    text_idx = [i for i in nb if cls.get(i) == "text"]
    indent_counts = Counter(_indent(lines[i]) for i in text_idx)
    base_indent = indent_counts.most_common(1)[0][0] if indent_counts else 0
    for i in text_idx:
        s = lines[i].strip()
        if (_PAGE_WORD.fullmatch(s) or _DASHED_PAGE.fullmatch(s)
                or (_PAGE_NUM.fullmatch(s) and _indent(lines[i]) - base_indent >= c["page_number_min_indent"])):
            cls[i] = "page_number"
    for i in _signature_zone(lines, nb, body_start, base_indent, lang, c):
        if cls[i] in ("text", "table_row"):
            cls[i] = "signature_row"
    stats["header_rows"] = sum(1 for i in header if cls[i] == "header_row")
    stats["signature_rows"] = sum(1 for v in cls.values() if v == "signature_row")

    text_idx = [i for i in nb if cls[i] == "text"]
    widths = sorted(len(lines[i]) for i in text_idx)
    p95 = _percentile(widths, 0.95)
    short_limit = c["short_line_frac"] * p95
    dense_rules = family != "blanksep"
    long_share = sum(w >= short_limit for w in widths) / len(widths) if widths else 1.0
    unwrapped = dense_rules and long_share < c["unwrapped_long_share"]
    if widths and widths[-1] > c["unwrapped_line_chars"]:
        unwrapped = True

    anchors: dict[int, Anchor] = {}
    prev_i: int | None = None
    for i in nb:
        if cls[i] != "text":
            prev_i = i
            continue
        hit = match_heading(lines[i], lang)
        if hit:
            anchor, strength = hit
            if strength == "weak" and prev_i is not None and prev_i == i - 1 and cls[prev_i] == "text":
                prev = lines[prev_i]
                if (prev_i not in anchors and not _TERM.search(prev) and not prev.strip().isupper()
                        and _indent(prev) - base_indent < 10):
                    hit = None
            if hit:
                anchors[i] = anchor
        prev_i = i

    def centred(i: int) -> bool:
        return _indent(lines[i]) - base_indent >= c["centred_indent"]

    def breaks(a: int, b: int) -> bool:
        la, lb = lines[a], lines[b]
        sb = lb.strip()
        if b in anchors or _LIST_START.match(lb):
            return True
        if centred(a) or centred(b):
            if (c["join_centred_caps"] and b == a + 1 and centred(a) and centred(b) and a not in anchors
                    and la.strip().isupper() and sb.isupper()):
                stats["centred_caps_joins"] += 1
                return False
            return True
        if b > a + 1:
            joinable = (not _TERM.search(la) and _starts_lower(sb) and a not in anchors)
            if joinable:
                stats["page_break_joins"] += 1
            return not joinable
        if unwrapped:
            return bool(_TERM.search(la)) or not _starts_lower(sb)
        if a in anchors and len(la) < short_limit and (_TERM.search(la) or not _starts_lower(sb)):
            return True
        if len(la) < c["very_short_line_frac"] * p95 and not _starts_lower(sb):
            return True
        if dense_rules and _TERM.search(la) and len(la) < short_limit and not _starts_lower(sb):
            return True
        return False

    vocab: set[str] | None = None

    def join_lines(idxs: list[int]) -> str:
        nonlocal vocab
        out = ""
        for i in idxs:
            piece = " ".join(lines[i].split())
            if not out:
                out = piece
                continue
            frag = _TRAIL_FRAG.search(out)
            lead = _LEAD_WORD.match(piece)
            if frag and lead and piece[0].islower() and lead.group(1).lower() not in _KEEP_SPACE_AFTER_HYPHEN:
                if vocab is None:
                    vocab = set(_WORD.findall(text.lower()))
                if (frag.group(1) + lead.group(1)).lower() in vocab:
                    out = out[:-1] + piece
                    stats["hyphen_dropped"] += 1
                else:
                    out = out + piece
                    stats["hyphen_kept"] += 1
            else:
                out = out + " " + piece
        return out

    def cap_groups(idxs: list[int]) -> list[list[int]]:
        cap = c["max_paragraph_chars"]
        lens = [len(lines[i].strip()) + 1 for i in idxs]
        if sum(lens) <= cap or len(idxs) == 1:
            return [idxs]
        stats["capped_paragraphs"] += 1
        groups: list[list[int]] = []
        start = 0
        while start < len(idxs):
            acc, end, best = 0, start, None
            while end < len(idxs) and (end == start or acc + lens[end] <= cap):
                acc += lens[end]
                end += 1
                if _TERM.search(lines[idxs[end - 1]]) and acc >= 0.3 * cap:
                    best = end
            if end < len(idxs) and best is not None:
                end = best
            groups.append(idxs[start:end])
            start = end
        return groups

    elements: list[Element] = []
    prev_nonblank = -1

    def blank_before(first_line: int) -> int:
        return first_line - prev_nonblank - 1

    def emit(kind: str, first: int, last: int, text_: str, cells: list[Cell] | None = None,
             anchor: Anchor | None = None) -> None:
        nonlocal prev_nonblank
        elements.append(Element(
            idx=len(elements), kind=kind, text=text_, indent=_indent(lines[first]), lines=(first, last + 1),
            cells=cells, blank_before=blank_before(first), anchor=anchor,
        ))
        prev_nonblank = last

    def flush(par: list[int]) -> None:
        if not par:
            return
        groups = cap_groups(par)
        for g_pos, g in enumerate(groups):
            body = join_lines(g)
            anchor = anchors.get(g[0]) if g_pos == 0 else None
            kind = "heading" if anchor is not None or _is_upper_title(body, c["title_max_chars"]) else "paragraph"
            emit(kind, g[0], g[-1], body, anchor=anchor)

    par: list[int] = []
    for i in nb:
        k = cls[i]
        if k == "text":
            if par and breaks(par[-1], i):
                flush(par)
                par = []
            par.append(i)
            continue
        flush(par)
        par = []
        raw = lines[i]
        if k in ROW_KINDS:
            emit(k, i, i, raw, cells=split_cells(raw, c["min_gap"]))
        else:
            emit(k, i, i, raw)
    flush(par)

    trailing_blank = len(lines) - 1 - prev_nonblank if lines else 0
    stats["lines"] = len(lines)
    stats["elements"] = len(elements)
    stats["base_indent"] = base_indent
    stats["p95_width"] = p95
    stats["max_width"] = widths[-1] if widths else 0
    stats["min_gap"] = c["min_gap"]
    stats["row_min_gap"] = c["row_min_gap"]
    stats["header_min_gap"] = c["header_min_gap"]
    stats["unwrapped"] = int(unwrapped)
    return DocSkeleton(
        elements=elements, lang=lang, layout_family=family, trailing_newline=trailing_newline,
        trailing_blank=trailing_blank, stats=dict(stats),
    )


def cell_is_translatable(text: str) -> bool:
    return bool(_WORDCHAR.search(text))


def translation_units(skel: DocSkeleton) -> list[Unit]:
    units: list[Unit] = []
    for e in skel.elements:
        if e.kind in TEXT_KINDS:
            units.append(Unit(uid=f"e{e.idx}", element_idx=e.idx, cell_idx=None, text=e.text, kind=e.kind))
        elif e.kind in ROW_KINDS and e.cells:
            for k, cell in enumerate(e.cells):
                if cell_is_translatable(cell.text):
                    units.append(Unit(uid=f"e{e.idx}c{k}", element_idx=e.idx, cell_idx=k, text=cell.text, kind=e.kind))
    return units


def find_anchors(skel: DocSkeleton) -> list[tuple[int, Anchor]]:
    return [(e.idx, e.anchor) for e in skel.elements if e.anchor is not None]


def _one_line(text: str) -> str:
    return " ".join(text.split())


def render_document(
    skel: DocSkeleton,
    translations: dict[str, str],
    *,
    blank_between_paragraphs: str = "source",
    rewrap_width: int | None = None,
    stats: dict | None = None,
) -> str:
    if blank_between_paragraphs not in ("source", "always", "never"):
        raise ValueError(f"blank_between_paragraphs must be source|always|never, got {blank_between_paragraphs!r}")
    gap = max(2, int(skel.stats.get("min_gap", DEFAULT_LAYOUT_CFG["min_gap"])))
    zone_gap = int(skel.stats.get("header_min_gap", DEFAULT_LAYOUT_CFG["header_min_gap"]))
    row_gap = {"header_row": zone_gap, "signature_row": zone_gap,
               "table_row": int(skel.stats.get("row_min_gap", DEFAULT_LAYOUT_CFG["row_min_gap"]))}
    missing: list[str] = []

    def get(uid: str, fallback: str) -> str:
        t = translations.get(uid)
        if t is None:
            missing.append(uid)
            return fallback
        return _one_line(t)

    out: list[str] = []
    prev_kind: str | None = None
    for pos, e in enumerate(skel.elements):
        if blank_between_paragraphs == "source":
            nblank = e.blank_before
        elif blank_between_paragraphs == "never" or pos == 0:
            nblank = 0
        elif e.kind in TEXT_KINDS or prev_kind in TEXT_KINDS:
            nblank = 1
        else:
            nblank = min(e.blank_before, 1)
        out.extend([""] * nblank)
        if e.kind in TEXT_KINDS:
            body = get(f"e{e.idx}", e.text)
            if rewrap_width:
                pad = " " * e.indent
                wrapped = textwrap.wrap(body, width=max(rewrap_width, e.indent + 20), initial_indent=pad,
                                        subsequent_indent=pad, break_long_words=False, break_on_hyphens=False)
                out.extend(wrapped or [pad.rstrip()])
            else:
                out.append(" " * e.indent + body)
        elif e.kind in ROW_KINDS and e.cells:
            line = ""
            kind_gap = max(gap, row_gap.get(e.kind, gap))
            src_end = 0
            for k, cell in enumerate(e.cells):
                t = get(f"e{e.idx}c{k}", cell.text) if cell_is_translatable(cell.text) else cell.text
                need = max(gap, min(kind_gap, cell.col - src_end))
                if line and len(line) + need > cell.col:
                    line += " " * need
                else:
                    line += " " * (cell.col - len(line))
                line += t
                src_end = cell.col + len(cell.text)
            out.append(line)
        else:
            out.append(e.text)
        prev_kind = e.kind
    if blank_between_paragraphs == "source":
        out.extend([""] * skel.trailing_blank)
    if stats is not None:
        stats["missing"] = len(missing)
        stats["missing_uids"] = missing
    return "\n".join(out) + ("\n" if skel.trailing_newline else "")
