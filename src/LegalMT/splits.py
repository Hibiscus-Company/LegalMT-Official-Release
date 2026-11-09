"""Train-carved dev split for LegalMT: which train documents are dev, dev_mini and train_fit."""
from __future__ import annotations

import hashlib
import json
import re
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import paths
from .data import load_split

ROLES = ("train_fit", "dev", "dev_mini", "public_test")
_ID_NUM = re.compile(r"^(.*?)(\d+)$")


@dataclass
class SplitSpec:
    version: str
    seed: int
    protocol: dict[str, Any]
    dev_ids: list[str]
    dev_mini_ids: list[str]
    train_fit_ids: list[str]
    excluded: dict[str, list[str]]
    doc_info: dict[str, dict[str, Any]]
    eligibility: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1, sort_keys=False) + "\n"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SplitSpec":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


def natural_key(sample_id: str) -> tuple[str, int, str]:
    m = _ID_NUM.match(sample_id)
    return (m.group(1), int(m.group(2)), "") if m else (sample_id, -1, sample_id)


def sort_ids(ids: Sequence[str]) -> list[str]:
    return sorted(ids, key=natural_key)


def spec_path(version: str = "v0") -> Path:
    return paths.CONFIG_DIR / f"dev_split_{version}.json"


def load_split_spec(version: str = "v0") -> SplitSpec:
    p = spec_path(version)
    if not p.is_file():
        raise FileNotFoundError(f"no split spec at {p}; build it with scripts/LegalMT/make_dev_split.py")
    return SplitSpec.from_dict(json.loads(p.read_text(encoding="utf-8")))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


_sha_cache: dict[tuple[str, int, int], str] = {}


def _cached_sha(path: Path) -> str:
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _sha_cache:
        _sha_cache[key] = file_sha256(path)
    return _sha_cache[key]


def docs_for(role: str, version: str = "v0", *, check_data: bool = True) -> pd.DataFrame:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    if role == "public_test":
        return load_split("public_test").df.reset_index(drop=True).copy()
    spec = load_split_spec(version)
    if check_data and spec.data.get("train_sha256"):
        actual = _cached_sha(paths.split_path("train"))
        if actual != spec.data["train_sha256"]:
            raise RuntimeError(
                f"train CSV sha256 {actual[:12]} differs from the one split {version} was built on "
                f"({spec.data['train_sha256'][:12]}); rebuild the split or pass check_data=False knowingly"
            )
    ids = {"train_fit": spec.train_fit_ids, "dev": spec.dev_ids, "dev_mini": spec.dev_mini_ids}[role]
    df = load_split("train").df
    pos = {sid: i for i, sid in enumerate(df["sample_id"])}
    missing = [s for s in ids if s not in pos]
    if missing:
        raise KeyError(f"{len(missing)} {role} ids are not in train, e.g. {missing[:3]}")
    return df.iloc[[pos[s] for s in ids]].reset_index(drop=True).copy()


def whitespace_words(text: str) -> int:
    return len(text.split())


def largest_remainder_quotas(counts: Mapping[str, int], total: int, minimum: int = 0) -> dict[str, int]:
    cats = sorted(counts)
    if minimum * len(cats) > total:
        raise ValueError(f"cannot give {minimum} to each of {len(cats)} categories within {total}")
    n = sum(counts.values())
    if n <= 0:
        raise ValueError("counts must sum to a positive number")
    exact = {c: total * counts[c] / n for c in cats}
    quotas = {c: max(minimum, int(np.floor(exact[c]))) for c in cats}
    left = total - sum(quotas.values())
    while left > 0:
        c = min(cats, key=lambda c: (-(exact[c] - quotas[c]), c))
        quotas[c] += 1
        left -= 1
    while left < 0:
        cands = [c for c in cats if quotas[c] > minimum]
        c = min(cands, key=lambda c: (exact[c] - quotas[c], c))
        quotas[c] -= 1
        left += 1
    return quotas


def category_rng(seed: int, category: str, stream: int = 0) -> np.random.Generator:
    return np.random.default_rng([int(seed), zlib.crc32(category.encode("utf-8")), int(stream)])


def length_stratified_draw(
    ids: Sequence[str], words: Mapping[str, int], quota: int, rng: np.random.Generator
) -> list[tuple[str, int]]:
    ordered = sorted(ids, key=lambda s: (words[s], natural_key(s)))
    if quota <= 0:
        return []
    if len(ordered) <= quota:
        return [(s, k) for k, s in enumerate(ordered)]
    strata = np.array_split(np.arange(len(ordered)), quota)
    out = []
    for k, st in enumerate(strata):
        pick = int(st[int(rng.integers(len(st)))])
        out.append((ordered[pick], k))
    return out


@dataclass
class SplitInputs:
    sample_ids: list[str]
    categories: list[str]
    en_words: list[int]
    vi_words: list[int]
    broken: dict[str, list[str]]
    identical_pairs: list[tuple[str, str]]
    containment: dict[str, tuple[np.ndarray, np.ndarray]]
    en_chars: list[int] | None = None
    vi_chars: list[int] | None = None

    @property
    def words(self) -> dict[str, int]:
        return {s: e + v for s, e, v in zip(self.sample_ids, self.en_words, self.vi_words)}


def _max_containment(inputs: SplitInputs) -> np.ndarray:
    mats = [np.maximum(en, vi) for en, vi in inputs.containment.values()]
    C = mats[0].copy()
    for m in mats[1:]:
        C = np.maximum(C, m)
    np.fill_diagonal(C, 0.0)
    return C


def _pick_dev_mini(
    dev: list[str], cats: Mapping[str, str], words: Mapping[str, int], train_counts: Mapping[str, int],
    n_mini: int, seed: int, notes: list[str],
) -> list[str]:
    median = float(np.median([words[s] for s in dev]))
    quotas = largest_remainder_quotas(train_counts, n_mini, minimum=1)
    pool = {c: sort_ids([s for s in dev if cats[s] == c and words[s] <= median]) for c in quotas}
    chosen: dict[str, list[str]] = {}
    shortfall = 0
    for c in sorted(quotas):
        cand = pool[c]
        if not cand:
            fallback = min((s for s in dev if cats[s] == c), key=lambda s: (words[s], natural_key(s)))
            notes.append(f"dev_mini: no {c} dev doc at or below the median; took the shortest ({fallback})")
            chosen[c] = [fallback]
            shortfall += quotas[c] - 1
            continue
        q = min(quotas[c], len(cand))
        shortfall += quotas[c] - q
        rng = category_rng(seed, c, stream=1)
        idx = sorted(rng.choice(len(cand), size=q, replace=False).tolist())
        chosen[c] = [cand[i] for i in idx]
    if shortfall:
        order = sorted(quotas, key=lambda c: (-train_counts[c], c))
        for c in order:
            spare = [s for s in pool[c] if s not in chosen[c]]
            while shortfall and spare:
                chosen[c].append(spare.pop(0))
                shortfall -= 1
        if shortfall:
            notes.append(f"dev_mini: {shortfall} slots left unfilled (not enough short dev docs)")
    return sort_ids([s for c in sorted(chosen) for s in chosen[c]])


def build_split_spec(
    inputs: SplitInputs, params: Mapping[str, Any], version: str, data: Mapping[str, Any] | None = None
) -> SplitSpec:
    seed = int(params["seed"])
    ids = list(inputs.sample_ids)
    cats = dict(zip(ids, inputs.categories))
    words = inputs.words
    en_w = dict(zip(ids, inputs.en_words))
    vi_w = dict(zip(ids, inputs.vi_words))
    pos = {s: i for i, s in enumerate(ids)}
    thr = float(params["near_duplicates"]["threshold"])
    rel_thr = float(params["near_duplicates"]["near_relative_threshold"])
    notes: list[str] = []

    broken = sort_ids({s for lst in inputs.broken.values() for s in lst})
    broken_set = set(broken)
    identical_members = sort_ids({s for p in inputs.identical_pairs for s in p})
    dup_dropped = sort_ids({p[1] for p in inputs.identical_pairs} - broken_set)
    dropped_set = broken_set | set(dup_dropped)
    C = _max_containment(inputs)
    near = [s for s in ids if C[pos[s]].max(initial=0.0) >= thr]

    ineligible = broken_set | set(identical_members) | set(near)
    eligible = [s for s in ids if s not in ineligible]

    train_counts: dict[str, int] = {}
    for s in ids:
        train_counts[cats[s]] = train_counts.get(cats[s], 0) + 1
    n_dev = int(params["dev_size"])
    lr = largest_remainder_quotas(train_counts, n_dev)
    quotas = {str(k): int(v) for k, v in params.get("dev_quotas", lr).items()}
    if quotas != lr:
        notes.append(f"dev_quotas from config {quotas} differ from largest remainder {lr}")
    if sum(quotas.values()) != n_dev:
        raise ValueError(f"dev_quotas sum to {sum(quotas.values())}, dev_size is {n_dev}")

    dev: list[str] = []
    stratum: dict[str, tuple[int, int]] = {}
    for c in sorted(quotas):
        pool = [s for s in eligible if cats[s] == c]
        if len(pool) < quotas[c]:
            notes.append(f"{c}: only {len(pool)} eligible docs for a quota of {quotas[c]}")
        for s, k in length_stratified_draw(pool, words, quotas[c], category_rng(seed, c, stream=0)):
            dev.append(s)
            stratum[s] = (k, quotas[c])
    dev = sort_ids(dev)

    dev_mini = _pick_dev_mini(dev, cats, words, train_counts, int(params["dev_mini_size"]), seed, notes)

    dev_set, mini_set = set(dev), set(dev_mini)
    train_fit = sort_ids([s for s in ids if s not in dev_set and s not in dropped_set])
    tf_idx = np.array([pos[s] for s in train_fit])

    doc_info: dict[str, dict[str, Any]] = {}
    for s in dev:
        i = pos[s]
        detail = {}
        for measure, (en, vi) in inputs.containment.items():
            detail[measure] = {"en": round(float(en[i, tf_idx].max(initial=0.0)), 4),
                               "vi": round(float(vi[i, tf_idx].max(initial=0.0)), 4)}
        row = C[i, tf_idx]
        j = int(row.argmax()) if len(row) else -1
        mx = float(row[j]) if j >= 0 else 0.0
        k, q = stratum[s]
        ratio = None
        if inputs.en_chars is not None and inputs.vi_chars is not None and inputs.en_chars[i]:
            ratio = round(inputs.vi_chars[i] / inputs.en_chars[i], 3)
        doc_info[s] = {
            "category": cats[s],
            "words": int(words[s]),
            "en_words": int(en_w[s]),
            "vi_words": int(vi_w[s]),
            "length_stratum": f"{k + 1}/{q}",
            "max_containment_into_train_fit": round(mx, 4),
            "nearest_train_fit_id": train_fit[j] if j >= 0 else None,
            "containment_detail": detail,
            "near_relative": bool(mx >= rel_thr),
            "in_dev_mini": s in mini_set,
            "vi_en_char_ratio": ratio,
        }

    summary = split_summary(dev, dev_mini, train_fit, cats, en_w, vi_w, doc_info, params)
    eligibility = {
        "n_train": len(ids),
        "n_eligible": len(eligible),
        "ineligible_broken": broken,
        "ineligible_identical_pair_member": identical_members,
        "ineligible_near_duplicate": sort_ids(set(near) - broken_set - set(identical_members)),
        "eligible_by_category": {c: sum(1 for s in eligible if cats[s] == c) for c in sorted(quotas)},
        "dev_quotas": {c: quotas[c] for c in sorted(quotas)},
        "largest_remainder_quotas": {c: lr[c] for c in sorted(lr)},
        "notes": notes,
    }
    return SplitSpec(
        version=version,
        seed=seed,
        protocol=json.loads(json.dumps(dict(params))),
        dev_ids=dev,
        dev_mini_ids=dev_mini,
        train_fit_ids=train_fit,
        excluded={"broken": broken, "duplicate_dropped": dup_dropped},
        doc_info=doc_info,
        eligibility=eligibility,
        data=dict(data or {}),
        summary=summary,
    )


def split_summary(
    dev: Sequence[str], dev_mini: Sequence[str], train_fit: Sequence[str], cats: Mapping[str, str],
    en_w: Mapping[str, int], vi_w: Mapping[str, int], doc_info: Mapping[str, Mapping[str, Any]],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    tpw = params.get("tokens_per_word_estimate", {"en": 1.0, "vi": 1.0})

    def tok(s: str) -> float:
        return en_w[s] * float(tpw["en"]) + vi_w[s] * float(tpw["vi"])

    per_cat: dict[str, dict[str, int]] = {}
    for role, lst in (("dev", dev), ("dev_mini", dev_mini), ("train_fit", train_fit)):
        for s in lst:
            d = per_cat.setdefault(cats[s], {})
            d[f"{role}_docs"] = d.get(f"{role}_docs", 0) + 1
            d[f"{role}_words"] = d.get(f"{role}_words", 0) + en_w[s] + vi_w[s]
    all_ids = set(dev) | set(train_fit)
    total_words = sum(en_w[s] + vi_w[s] for s in all_ids)
    dev_words = sum(en_w[s] + vi_w[s] for s in dev)
    dev_tok = sum(tok(s) for s in dev)
    total_tok = sum(tok(s) for s in all_ids)
    dev_doc_words = sorted((en_w[s] + vi_w[s] for s in dev), reverse=True)
    near = sort_ids([s for s in dev if doc_info[s]["near_relative"]])
    lo, hi = params.get("length_ratio_flag", [0.6, 1.3])
    ratio_flags = sort_ids([
        s for s in dev
        if doc_info[s].get("vi_en_char_ratio") is not None and not lo <= doc_info[s]["vi_en_char_ratio"] <= hi
    ])
    return {
        "n_dev": len(dev),
        "n_dev_mini": len(dev_mini),
        "n_train_fit": len(train_fit),
        "per_category": {c: per_cat[c] for c in sorted(per_cat)},
        "dev_words": dev_words,
        "dev_mini_words": sum(en_w[s] + vi_w[s] for s in dev_mini),
        "dev_share_of_dev_plus_train_fit_words": round(dev_words / total_words, 4) if total_words else 0.0,
        "dev_share_of_dev_plus_train_fit_tokens_ESTIMATE": round(dev_tok / total_tok, 4) if total_tok else 0.0,
        "dev_largest_doc_word_share": round(dev_doc_words[0] / dev_words, 4) if dev_words else 0.0,
        "dev_top3_doc_word_share": round(sum(dev_doc_words[:3]) / dev_words, 4) if dev_words else 0.0,
        "dev_median_words": float(np.median([en_w[s] + vi_w[s] for s in dev])) if dev else 0.0,
        "dev_mean_words": round(float(np.mean([en_w[s] + vi_w[s] for s in dev])), 2) if dev else 0.0,
        "train_fit_mean_words": round(float(np.mean([en_w[s] + vi_w[s] for s in train_fit])), 2) if train_fit else 0.0,
        "near_relative_dev_docs": near,
        "n_near_relative": len(near),
        "dev_length_ratio_flagged": ratio_flags,
    }
