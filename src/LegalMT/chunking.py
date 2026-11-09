"""Chunk planning: pack translation units into model-sized requests, one unit per line, and map output lines back."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

JOINER = "\n"
PIECE_SEP = "#"

_SPLIT_LEVELS = (
    re.compile(r"(?<=[.!?…;])\s+(?=\S)"),
    re.compile(r"(?<=[,:])\s+(?=\S)"),
    re.compile(r"\s+"),
)


@dataclass
class Chunk:
    uids: list[str]
    text: str
    n_tokens: int = 0
    cut: str = "end"
    kinds: list[str] = field(default_factory=list)

    @property
    def lines(self) -> list[str]:
        return self.text.split(JOINER)


def one_line(text: str) -> str:
    return re.sub(r"[\r\n]+", " ", text).strip()


def piece_uid(uid: str, k: int) -> str:
    return f"{uid}{PIECE_SEP}{k}"


def base_uid(uid: str) -> str:
    return uid.split(PIECE_SEP, 1)[0]


def split_unit_text(text: str, count_tokens: Callable[[str], int], max_tokens: int) -> list[str]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")

    def split(t: str, level: int) -> list[str]:
        if count_tokens(t) <= max_tokens:
            return [t]
        if level >= len(_SPLIT_LEVELS):
            if len(t) <= 1:
                return [t]
            mid = len(t) // 2
            return split(t[:mid], level) + split(t[mid:], level)
        atoms = [a for a in _SPLIT_LEVELS[level].split(t) if a]
        if len(atoms) <= 1:
            return split(t, level + 1)
        out: list[str] = []
        cur = atoms[0]
        for a in atoms[1:]:
            cand = f"{cur} {a}"
            if count_tokens(cand) <= max_tokens:
                cur = cand
            else:
                out.extend(split(cur, level + 1))
                cur = a
        out.extend(split(cur, level + 1))
        return out

    return split(text.strip(), 0)


def plan_chunks(
    units: Sequence[Any],
    count_tokens: Callable[[str], int],
    budget_tokens: int,
    *,
    cut_before_heading: bool = True,
    min_fill_before_heading_cut: float = 0.5,
    max_unit_tokens: int | None = None,
    max_units_per_chunk: int | None = None,
) -> list[Chunk]:
    if budget_tokens < 1:
        raise ValueError("budget_tokens must be >= 1")
    cap = min(max_unit_tokens or budget_tokens, budget_tokens)
    items: list[tuple[str, str, str, int]] = []
    for u in units:
        line = one_line(u.text)
        if not line:
            continue
        n = count_tokens(line)
        if n <= cap:
            items.append((u.uid, line, u.kind, n))
            continue
        for k, piece in enumerate(split_unit_text(line, count_tokens, cap)):
            items.append((piece_uid(u.uid, k), piece, u.kind if k == 0 else "piece", count_tokens(piece)))

    chunks: list[Chunk] = []
    cur: list[tuple[str, str, str, int]] = []
    cur_tok = 0

    def flush(reason: str) -> None:
        nonlocal cur, cur_tok
        if cur:
            chunks.append(Chunk(uids=[c[0] for c in cur], text=JOINER.join(c[1] for c in cur),
                                n_tokens=cur_tok, cut=reason, kinds=[c[2] for c in cur]))
        cur, cur_tok = [], 0

    for item in items:
        n = item[3]
        if cur:
            if cur_tok + 1 + n > budget_tokens:
                flush("budget")
            elif cut_before_heading and item[2] == "heading" and cur_tok >= min_fill_before_heading_cut * budget_tokens:
                flush("heading")
            elif max_units_per_chunk and len(cur) >= max_units_per_chunk:
                flush("max_units")
        cur_tok += n + (1 if cur else 0)
        cur.append(item)
    flush("end")
    return chunks


def split_chunk_output(chunk: Chunk, output: str) -> list[str] | None:
    lines = [ln.strip() for ln in output.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [ln for ln in lines if ln]
    if len(lines) != len(chunk.uids):
        return None
    return lines


def merge_pieces(translations: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    pieces: dict[str, dict[int, str]] = {}
    for uid, text in translations.items():
        if PIECE_SEP in uid:
            base, k = uid.split(PIECE_SEP, 1)
            pieces.setdefault(base, {})[int(k)] = text
        else:
            out[uid] = text
    for base, parts in pieces.items():
        if sorted(parts) != list(range(len(parts))):
            raise ValueError(f"unit {base}: pieces {sorted(parts)} are not 0..{len(parts) - 1}")
        out[base] = " ".join(parts[k].strip() for k in range(len(parts)) if parts[k].strip())
    return out


def chunk_stats(chunks: Iterable[Chunk]) -> dict[str, Any]:
    chunks = list(chunks)
    cuts: dict[str, int] = {}
    for c in chunks:
        cuts[c.cut] = cuts.get(c.cut, 0) + 1
    return {
        "n_chunks": len(chunks),
        "n_lines": sum(len(c.uids) for c in chunks),
        "n_pieces": sum(1 for c in chunks for u in c.uids if PIECE_SEP in u),
        "cuts": cuts,
        "src_tokens": sum(c.n_tokens for c in chunks),
        "max_chunk_tokens": max((c.n_tokens for c in chunks), default=0),
    }
