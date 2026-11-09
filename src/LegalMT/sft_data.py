"""SFT data: aligned paragraph links packed into prompt/completion examples that match inference requests."""
from __future__ import annotations

import hashlib
import json
import math
import random
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .chunking import JOINER, one_line, plan_chunks
from .layout import parse_document, translation_units
from .prompts import (
    CONTEXT_TEMPLATES,
    HY_BOS,
    HY_NO8,
    build_prompt_text,
    completion_suffix,
    encode_batch,
    render_user_prompt,
)
from .splits import natural_key

DIRECTIONS = ("en2vi", "vi2en")
SRC_TGT = {"en2vi": ("en", "vi"), "vi2en": ("vi", "en")}
OVERSIZE_POLICIES = ("drop", "alone")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json")

DEFAULT_FILTERS: dict[str, Any] = {
    "methods": None,
    "links": None,
    "len_ratio": None,
    "min_num_f1": None,
}
IN_DOMAIN_SOURCE = "in_domain"
EXTRA_PREFIX = "extra:"
DEFAULT_EXTRA: dict[str, Any] = {
    "name": None,
    "path": None,
    "seed": None,
    "max_tokens": None,
    "max_lines": 40,
    "block_pairs": 1000,
}
DEFAULT_PACKING: dict[str, Any] = {
    "oversize_links": "drop",
    "break_at_gaps": False,
    "max_total_tokens": None,
}


class TokenCounter:
    def __init__(self, tokenizer: Any = None, batch_size: int = 1024) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.cache: dict[str, int] = {}

    def prime(self, texts: Iterable[str]) -> None:
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if self.tokenizer is None:
            self.cache.update((t, len(t.split())) for t in todo)
            return
        for k in range(0, len(todo), self.batch_size):
            part = todo[k:k + self.batch_size]
            for t, ids in zip(part, encode_batch(self.tokenizer, part)):
                self.cache[t] = len(ids)

    def __call__(self, text: str) -> int:
        n = self.cache.get(text)
        if n is None:
            self.prime([text])
            n = self.cache[text]
        return n


def tokenizer_fingerprint(model_dir: str | Path) -> dict[str, Any]:
    d = Path(model_dir)
    h = hashlib.sha256()
    files = []
    for name in TOKENIZER_FILES:
        p = d / name
        if p.is_file():
            files.append(name)
            h.update(name.encode() + b"\0" + p.read_bytes() + b"\0")
    return {"sha256": h.hexdigest(), "files": files}


def read_aligned(path: str | Path, keep_ids: set[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    docs: dict[str, list[dict[str, Any]]] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec["sample_id"]
            if keep_ids is not None and sid not in keep_ids:
                continue
            docs.setdefault(sid, []).append(rec)
    for links in docs.values():
        links.sort(key=lambda r: r["order"])
        orders = [r["order"] for r in links]
        if len(set(orders)) != len(orders):
            raise ValueError(f"{links[0]['sample_id']}: duplicate link orders in {path}")
    return docs


def split_roles(train_ids: Iterable[str], dev_ids: Iterable[str], *, limit: int | None = None) -> dict[str, list[str]]:
    train, dev = set(train_ids), set(dev_ids)
    both = sorted(train & dev, key=natural_key)
    if both:
        raise ValueError(f"{len(both)} documents are in both train_fit and dev, e.g. {both[:3]}")
    roles = {"train": sorted(train, key=natural_key), "dev": sorted(dev, key=natural_key)}
    if limit:
        roles = {r: ids[:limit] for r, ids in roles.items()}
    return roles


def validate_final(final: Any) -> dict[str, bool] | None:
    if final is None:
        return None
    if not isinstance(final, Mapping) or set(final) - {"public_test"} \
            or not isinstance(final.get("public_test", False), bool):
        raise ValueError(f"'final' must be a mapping {{public_test: true|false}}, got {final!r}")
    return {"public_test": bool(final.get("public_test", False))}


def final_roles(train_ids: Iterable[str], dev_ids: Iterable[str], public_ids: Iterable[str] = (), *,
                limit: int | None = None) -> dict[str, list[str]]:
    train, dev, public = set(train_ids), set(dev_ids), set(public_ids)
    both = sorted(train & dev, key=natural_key)
    if both:
        raise ValueError(f"{len(both)} documents are in both train_fit and dev, e.g. {both[:3]}")
    clash = sorted(public & (train | dev), key=natural_key)
    if clash:
        raise ValueError(f"{len(clash)} public-test ids equal split document ids, e.g. {clash[:3]}")
    roles = {"train": sorted(train | dev | public, key=natural_key), "dev": sorted(dev, key=natural_key)}
    if limit:
        roles = {r: ids[:limit] for r, ids in roles.items()}
    return roles


def attach_kinds(links: Sequence[dict[str, Any]], en_text: str, vi_text: str,
                 layout_cfg: dict | None = None) -> dict[str, int]:
    stats = Counter({"missing_uid": 0, "no_uids": 0, "text_mismatch_en": 0, "text_mismatch_vi": 0})
    for lang, text in (("en", en_text), ("vi", vi_text)):
        skel = parse_document(text, lang, layout_cfg)
        units = {u.uid: u for u in translation_units(skel)}
        elems = skel.elements
        for rec in links:
            uids = rec.get(f"{lang}_uids") or []
            got = [units[u] for u in uids if u in units]
            if not uids:
                stats["no_uids"] += 1
            if len(got) != len(uids):
                stats["missing_uid"] += 1
            if got:
                kind = got[0].kind
                if one_line(" ".join(u.text for u in got)) != one_line(rec[lang]):
                    stats[f"text_mismatch_{lang}"] += 1
            else:
                idx = (rec.get(f"{lang}_units") or [None])[0]
                kind = elems[idx].kind if isinstance(idx, int) and 0 <= idx < len(elems) else "paragraph"
            rec[f"kind_{lang}"] = kind
    return dict(stats)


def filter_links(links: Sequence[dict[str, Any]], filters: Mapping[str, Any] | None) -> tuple[list[dict[str, Any]], Counter]:
    f = {**DEFAULT_FILTERS, **(filters or {})}
    unknown = sorted(set(f) - set(DEFAULT_FILTERS))
    if unknown:
        raise ValueError(f"unknown filter keys {unknown}; known: {sorted(DEFAULT_FILTERS)}")
    methods = set(f["methods"]) if f["methods"] else None
    kinds = set(f["links"]) if f["links"] else None
    ratio = f["len_ratio"]
    if ratio is not None and (len(ratio) != 2 or not ratio[0] < ratio[1]):
        raise ValueError(f"filters.len_ratio must be [lo, hi] with lo < hi, got {ratio!r}")
    kept: list[dict[str, Any]] = []
    dropped: Counter = Counter()
    for rec in links:
        if not one_line(rec["en"]) or not one_line(rec["vi"]):
            dropped["empty_side"] += 1
        elif methods is not None and rec.get("method") not in methods:
            dropped["method"] += 1
        elif kinds is not None and rec.get("link") not in kinds:
            dropped["link_type"] += 1
        elif ratio is not None and rec.get("link") != "header" and not (ratio[0] <= float(rec.get("len_ratio") or 0.0) <= ratio[1]):
            dropped["len_ratio"] += 1
        elif f["min_num_f1"] is not None and rec.get("num_f1") is not None and rec["num_f1"] < f["min_num_f1"]:
            dropped["num_f1"] += 1
        else:
            kept.append(rec)
    return kept, dropped


@dataclass
class _LinkUnit:
    uid: str
    text: str
    kind: str


@dataclass
class Packed:
    src_lines: list[str]
    tgt_lines: list[str]
    orders: list[int]
    src_tokens: int
    cut: str

    @property
    def src_text(self) -> str:
        return JOINER.join(self.src_lines)

    @property
    def tgt_text(self) -> str:
        return JOINER.join(self.tgt_lines)


def _segments(links: Sequence[dict[str, Any]], break_at_gaps: bool) -> Iterator[list[dict[str, Any]]]:
    if not break_at_gaps:
        if links:
            yield list(links)
        return
    cur: list[dict[str, Any]] = []
    for rec in links:
        if cur and rec["order"] != cur[-1]["order"] + 1:
            yield cur
            cur = []
        cur.append(rec)
    if cur:
        yield cur


def pack_links(
    links: Sequence[dict[str, Any]],
    direction: str,
    count_tokens: Callable[[str], int],
    chunking_cfg: Mapping[str, Any],
    *,
    oversize: str = "drop",
    break_at_gaps: bool = False,
) -> tuple[list[Packed], Counter]:
    if direction not in SRC_TGT:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if oversize not in OVERSIZE_POLICIES:
        raise ValueError(f"oversize_links must be one of {OVERSIZE_POLICIES}, got {oversize!r}")
    src, tgt = SRC_TGT[direction]
    budget = int(chunking_cfg["budget_tokens"][direction])
    cap = min(int(chunking_cfg.get("max_unit_tokens") or budget), budget)
    stats: Counter = Counter()
    out: list[Packed] = []
    for seg in _segments(links, break_at_gaps):
        run: list[dict[str, Any]] = []

        def flush() -> None:
            if not run:
                return
            units = [_LinkUnit(uid=str(k), text=one_line(r[src]), kind=r.get(f"kind_{src}", "paragraph"))
                     for k, r in enumerate(run)]
            chunks = plan_chunks(units, count_tokens, budget,
                                 cut_before_heading=bool(chunking_cfg.get("cut_before_heading", True)),
                                 min_fill_before_heading_cut=float(chunking_cfg.get("min_fill_before_heading_cut", 0.5)),
                                 max_unit_tokens=cap, max_units_per_chunk=chunking_cfg.get("max_units_per_chunk"))
            for ch in chunks:
                if any("#" in u for u in ch.uids):
                    raise AssertionError("plan_chunks split a link; the oversize guard failed")
                recs = [run[int(u)] for u in ch.uids]
                out.append(Packed(src_lines=ch.lines, tgt_lines=[one_line(r[tgt]) for r in recs],
                                  orders=[int(r["order"]) for r in recs], src_tokens=ch.n_tokens, cut=ch.cut))
                stats["links_used"] += len(recs)
            run.clear()

        for rec in seg:
            n = count_tokens(one_line(rec[src]))
            if n > cap:
                stats["oversize_links"] += 1
                if oversize == "alone":
                    flush()
                    out.append(Packed(src_lines=[one_line(rec[src])], tgt_lines=[one_line(rec[tgt])],
                                      orders=[int(rec["order"])], src_tokens=n, cut="oversize"))
                    stats["links_used"] += 1
                continue
            run.append(rec)
        flush()
    for p in out:
        if len(p.src_lines) != len(p.tgt_lines):
            raise AssertionError("source and target line counts differ")
    return out, stats


def render_prompt(tokenizer: Any, family: str, src_text: str, tgt_lang: str, prompt_cfg: Mapping[str, Any],
                  context: str | None = None) -> str:
    template = prompt_cfg["context_template"] if context else prompt_cfg["template"]
    user = render_user_prompt(family, src_text, tgt_lang, context=context, template=template,
                              lang_names=prompt_cfg.get("lang_names", "en"))
    return build_prompt_text(tokenizer, family, user, mode=prompt_cfg.get("mode", "generation_prompt"))


def doc_rng(seed: int, sample_id: str, direction: str) -> random.Random:
    return random.Random(f"{seed}|{sample_id}|{direction}")


def make_examples(
    packed: Sequence[Packed],
    *,
    sample_id: str,
    category: str,
    direction: str,
    family: str,
    tokenizer: Any,
    prompt_cfg: Mapping[str, Any],
    context_p: float = 0.0,
    seed: int = 0,
) -> list[dict[str, Any]]:
    if context_p and prompt_cfg.get("context_template") not in CONTEXT_TEMPLATES:
        raise ValueError(f"context p > 0 needs prompt.context_template in {sorted(CONTEXT_TEMPLATES)}")
    src, tgt = SRC_TGT[direction]
    rng = doc_rng(seed, sample_id, direction)
    suffix = completion_suffix(tokenizer, family)
    out = []
    for k, p in enumerate(packed):
        draw = rng.random()
        context = packed[k - 1].src_text if (k > 0 and context_p > 0 and draw < context_p) else None
        prompt = render_prompt(tokenizer, family, p.src_text, tgt, prompt_cfg, context=context)
        out.append({
            "id": f"{direction}|{sample_id}|{k:04d}",
            "sample_id": sample_id,
            "category": category,
            "direction": direction,
            "src_lang": src,
            "tgt_lang": tgt,
            "prompt": prompt,
            "completion": p.tgt_text + suffix,
            "n_lines": len(p.src_lines),
            "orders": p.orders,
            "src_tokens": p.src_tokens,
            "context": context is not None,
            "cut": p.cut,
        })
    return out


def add_token_counts(examples: list[dict[str, Any]], tokenizer: Any, batch_size: int = 256) -> None:
    for key in ("prompt", "completion"):
        for k in range(0, len(examples), batch_size):
            part = examples[k:k + batch_size]
            if tokenizer is None:
                counts = [len(ex[key].split()) for ex in part]
            else:
                counts = [len(ids) for ids in encode_batch(tokenizer, [ex[key] for ex in part])]
            for ex, n in zip(part, counts):
                ex[f"{key}_tokens"] = n


def check_example(ex: Mapping[str, Any], family: str, suffix: str) -> None:
    body = ex["completion"][: -len(suffix)] if suffix else ex["completion"]
    if not ex["completion"].endswith(suffix):
        raise AssertionError(f"{ex['id']}: completion does not end with the completion suffix")
    if not (unicodedata.is_normalized("NFC", ex["prompt"]) and unicodedata.is_normalized("NFC", ex["completion"])):
        raise AssertionError(f"{ex['id']}: prompt or completion is not NFC (inference text always is)")
    if len(body.split(JOINER)) != ex["n_lines"] or any(not ln.strip() for ln in body.split(JOINER)):
        raise AssertionError(f"{ex['id']}: target line count differs from the source or a line is blank")
    if family == "hymt":
        if ex["prompt"].count(HY_BOS) != 1 or not ex["prompt"].startswith(HY_BOS):
            raise AssertionError(f"{ex['id']}: prompt must start with exactly one BOS")
        if HY_NO8 in ex["completion"]:
            raise AssertionError(f"{ex['id']}: completion contains the no_8 token")


@dataclass
class BuildSettings:
    family: str
    chunking: Mapping[str, Any]
    prompt: Mapping[str, Any]
    directions: Sequence[str] = DIRECTIONS
    filters: Mapping[str, Any] = field(default_factory=dict)
    packing: Mapping[str, Any] = field(default_factory=dict)
    context_p: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        bad = [d for d in self.directions if d not in SRC_TGT]
        if bad or not self.directions:
            raise ValueError(f"directions must be a non-empty subset of {DIRECTIONS}, got {list(self.directions)}")
        unknown = sorted(set(self.packing) - set(DEFAULT_PACKING))
        if unknown:
            raise ValueError(f"unknown packing keys {unknown}; known: {sorted(DEFAULT_PACKING)}")
        if not 0.0 <= float(self.context_p) <= 1.0:
            raise ValueError(f"context p must be in [0, 1], got {self.context_p}")

    @property
    def pack(self) -> dict[str, Any]:
        return {**DEFAULT_PACKING, **dict(self.packing)}


def build_examples(
    docs: Mapping[str, Sequence[dict[str, Any]]],
    settings: BuildSettings,
    tokenizer: Any = None,
    counter: TokenCounter | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counter = counter or TokenCounter(tokenizer)
    pack = settings.pack
    counts: dict[str, Any] = {"docs": len(docs), "links_in": 0, "links_dropped": Counter(),
                              "per_direction": {d: Counter() for d in settings.directions}}
    examples: list[dict[str, Any]] = []
    for sid in sorted(docs, key=natural_key):
        links = docs[sid]
        counts["links_in"] += len(links)
        kept, dropped = filter_links(links, settings.filters)
        counts["links_dropped"].update(dropped)
        if not kept:
            continue
        counter.prime(one_line(r[s]) for r in kept for s in ("en", "vi"))
        category = kept[0].get("category", "")
        for direction in settings.directions:
            packed, st = pack_links(kept, direction, counter, settings.chunking,
                                    oversize=pack["oversize_links"], break_at_gaps=bool(pack["break_at_gaps"]))
            counts["per_direction"][direction].update(st)
            counts["per_direction"][direction]["links_kept"] += len(kept)
            examples.extend(make_examples(packed, sample_id=sid, category=category, direction=direction,
                                          family=settings.family, tokenizer=tokenizer, prompt_cfg=settings.prompt,
                                          context_p=settings.context_p, seed=settings.seed))
    add_token_counts(examples, tokenizer)
    limit = pack["max_total_tokens"]
    if limit:
        before = Counter(ex["direction"] for ex in examples)
        examples = [ex for ex in examples if ex["prompt_tokens"] + ex["completion_tokens"] <= int(limit)]
        after = Counter(ex["direction"] for ex in examples)
        for d in settings.directions:
            counts["per_direction"][d]["over_max_total_tokens"] = before[d] - after[d]
    suffix = completion_suffix(tokenizer, settings.family)
    for ex in examples:
        check_example(ex, settings.family, suffix)
    counts["links_dropped"] = dict(counts["links_dropped"])
    counts["per_direction"] = {d: dict(c) for d, c in counts["per_direction"].items()}
    return examples, counts


def shuffle_examples(examples: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    out = sorted(examples, key=lambda ex: ex["id"])
    random.Random(seed).shuffle(out)
    return out


def cap_tokens(examples: Sequence[dict[str, Any]], max_tokens: int | None) -> list[dict[str, Any]]:
    if not max_tokens:
        return list(examples)
    out, total = [], 0
    for ex in examples:
        n = ex["prompt_tokens"] + ex["completion_tokens"]
        if total + n > max_tokens:
            break
        out.append(ex)
        total += n
    return out


def example_source(ex: Mapping[str, Any]) -> str:
    return ex.get("source") or IN_DOMAIN_SOURCE


def validate_extra(entries: Any) -> list[dict[str, Any]]:
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError(f"extra must be a list of sources, got {type(entries).__name__}")
    out, names = [], set()
    for e in entries:
        if not isinstance(e, Mapping):
            raise ValueError(f"each extra source must be a mapping, got {e!r}")
        unknown = sorted(set(e) - set(DEFAULT_EXTRA))
        if unknown:
            raise ValueError(f"unknown extra keys {unknown}; known: {sorted(DEFAULT_EXTRA)}")
        r = {**DEFAULT_EXTRA, **e}
        for key in ("name", "path", "seed"):
            if r[key] is None:
                raise ValueError(f"every extra source needs '{key}': {dict(e)}")
        name = str(r["name"])
        if name == IN_DOMAIN_SOURCE or "/" in name or "|" in name or name.startswith(EXTRA_PREFIX):
            raise ValueError(f"extra name {name!r} is reserved or holds '/' or '|' (they delimit example ids)")
        if name in names:
            raise ValueError(f"duplicate extra source name {name!r}")
        names.add(name)
        if int(r["max_lines"]) < 1 or int(r["block_pairs"]) < 1:
            raise ValueError(f"extra {name}: max_lines and block_pairs must be >= 1")
        if r["max_tokens"] is not None and int(r["max_tokens"]) < 1:
            raise ValueError(f"extra {name}: max_tokens must be >= 1 or null")
        out.append({"name": name, "path": str(r["path"]), "seed": int(r["seed"]),
                    "max_tokens": None if r["max_tokens"] is None else int(r["max_tokens"]),
                    "max_lines": int(r["max_lines"]), "block_pairs": int(r["block_pairs"])})
    return out


def read_pairs(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if not all(isinstance(rec.get(k), str) for k in ("id", "en", "vi")):
                raise ValueError(f"{path}:{n}: every pair needs string id, en and vi fields")
            if rec["id"] in seen:
                raise ValueError(f"{path}:{n}: duplicate pair id {rec['id']!r}")
            seen.add(rec["id"])
            out.append({"id": rec["id"], "en": rec["en"], "vi": rec["vi"]})
    return out


def shuffle_pairs(pairs: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(pairs, key=lambda p: hashlib.sha256(f"{seed}|{p['id']}".encode()).digest())


def build_extra_examples(
    pairs: Sequence[dict[str, Any]],
    settings: BuildSettings,
    *,
    name: str,
    seed: int,
    max_tokens: int | None = None,
    max_lines: int = DEFAULT_EXTRA["max_lines"],
    block_pairs: int = DEFAULT_EXTRA["block_pairs"],
    tokenizer: Any = None,
    counter: TokenCounter | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cap = settings.chunking.get("max_units_per_chunk")
    chunking = {**settings.chunking, "max_units_per_chunk": min(int(cap), max_lines) if cap else max_lines}
    extra_settings = replace(settings, chunking=chunking, filters={}, context_p=0.0)
    counter = counter or TokenCounter(tokenizer)
    stream = shuffle_pairs(pairs, seed)
    examples: list[dict[str, Any]] = []
    dropped: Counter = Counter()
    per_direction = {d: Counter() for d in settings.directions}
    total = blocks = taken = 0
    stopped_by = "pairs_exhausted"
    for b, start in enumerate(range(0, len(stream), block_pairs)):
        sid = f"{EXTRA_PREFIX}{name}/{b:05d}"
        links = [{"sample_id": sid, "order": start + k, "en": p["en"], "vi": p["vi"]}
                 for k, p in enumerate(stream[start:start + block_pairs])]
        exs, counts = build_examples({sid: links}, extra_settings, tokenizer, counter)
        n = sum(ex["prompt_tokens"] + ex["completion_tokens"] for ex in exs)
        if max_tokens is not None and total + n > max_tokens:
            stopped_by = "max_tokens"
            break
        for ex in exs:
            ex["source"] = name
        examples.extend(exs)
        total += n
        blocks += 1
        taken += len(links)
        dropped.update(counts["links_dropped"])
        for d, c in counts["per_direction"].items():
            per_direction[d].update(c)
    stats = {
        "pairs_in_file": len(pairs), "pairs_taken": taken, "blocks_taken": blocks, "stopped_by": stopped_by,
        "examples": len(examples), "total_tokens": total,
        "prompt_tokens": sum(ex["prompt_tokens"] for ex in examples),
        "completion_tokens": sum(ex["completion_tokens"] for ex in examples),
        "links_dropped": dict(dropped), "per_direction": {d: dict(c) for d, c in per_direction.items()},
        "max_units_per_chunk": chunking["max_units_per_chunk"],
    }
    return examples, stats


def percentiles(values: Sequence[float], qs: Sequence[int] = (50, 90, 99)) -> dict[str, float]:
    if not values:
        return {**{f"p{q}": 0 for q in qs}, "mean": 0.0, "max": 0}
    s = sorted(values)
    out: dict[str, float] = {f"p{q}": s[max(0, math.ceil(q / 100 * len(s)) - 1)] for q in qs}
    out["mean"] = round(sum(s) / len(s), 2)
    out["max"] = s[-1]
    return out


def summarize(examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def block(exs: Sequence[dict[str, Any]]) -> dict[str, Any]:
        prompt = [ex["prompt_tokens"] for ex in exs]
        comp = [ex["completion_tokens"] for ex in exs]
        total = [p + c for p, c in zip(prompt, comp)]
        return {
            "examples": len(exs),
            "docs": len({ex["sample_id"] for ex in exs}),
            "lines": sum(ex["n_lines"] for ex in exs),
            "prompt_tokens": sum(prompt),
            "completion_tokens": sum(comp),
            "total_tokens": sum(total),
            "contextual_examples": sum(1 for ex in exs if ex["context"]),
            "cuts": dict(Counter(ex["cut"] for ex in exs)),
            "src_tokens_pct": percentiles([ex["src_tokens"] for ex in exs]),
            "prompt_tokens_pct": percentiles(prompt),
            "completion_tokens_pct": percentiles(comp),
            "total_tokens_pct": percentiles(total),
            "lines_pct": percentiles([ex["n_lines"] for ex in exs]),
        }

    def with_directions(exs: Sequence[dict[str, Any]]) -> dict[str, Any]:
        out = block(exs)
        out["per_direction"] = {d: block([ex for ex in exs if ex["direction"] == d])
                                for d in sorted({ex["direction"] for ex in exs})}
        return out

    out = with_directions(examples)
    if any("source" in ex for ex in examples):
        groups: dict[str, list[dict[str, Any]]] = {}
        for ex in examples:
            groups.setdefault(example_source(ex), []).append(ex)
        out["per_source"] = {src: with_directions(g) for src, g in sorted(groups.items())}
    return out


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(p)
    return n
