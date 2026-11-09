#!/usr/bin/env python3
"""Build the train-carved dev split and re-check the data-quality lists it depends on.

    uv run python scripts/LegalMT/make_dev_split.py
    uv run python scripts/LegalMT/make_dev_split.py --dry-run
    uv run python scripts/LegalMT/make_dev_split.py --check
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from LegalMT import quality as Q
from LegalMT import splits as S
from LegalMT.data import load_split
from LegalMT.paths import load_config, split_path
from LegalMT.textnorm import normalize_text


def derive_quality(df, params: dict, *, verbose: bool = True) -> dict:
    nd = params["near_duplicates"]
    ids = df["sample_id"].tolist()
    en_raw, vi_raw = df["english_markdown"].tolist(), df["vietnamese_markdown"].tolist()
    checks = Q.language_checks(ids, en_raw, vi_raw, threshold=float(params["language_vi_share_threshold"]))
    en, vi = Q.corrected_columns(checks, en_raw, vi_raw)
    en_sets = Q.shingle_sets(en, n=int(nd["shingle_n"]), sample_k=int(nd.get("sample_k", 1)))
    vi_sets = Q.shingle_sets(vi, n=int(nd["shingle_n"]), sample_k=int(nd.get("sample_k", 1)))
    out = {
        "ids": ids,
        "checks": checks,
        "broken": Q.broken_pairs(checks),
        "identical_pairs": Q.identical_pairs(ids, en_sets, vi_sets),
        "exact_text_groups": Q.exact_duplicate_groups(ids, en, vi),
        "en_sets": en_sets,
        "vi_sets": vi_sets,
        "near": {},
    }
    for measure, max_df in nd["measures"].items():
        out["near"][measure] = Q.near_duplicates(ids, en_sets, vi_sets, max_df=max_df)
    if verbose:
        en_shares = np.array([c.en_col_share for c in checks if c.status == "ok"])
        vi_shares = np.array([c.vi_col_share for c in checks if c.status == "ok"])
        print(f"  VI word share, valid pairs: EN column P99 {np.percentile(en_shares, 99):.3f} max {en_shares.max():.3f}; "
              f"VI column min {vi_shares.min():.3f} P1 {np.percentile(vi_shares, 1):.3f}")
    return out


def compare_quality(dq: dict, derived: dict) -> list[str]:
    diffs = []
    for key in ("swapped", "both_vietnamese", "both_english"):
        missing, extra = Q.diff_lists(dq["broken_pairs"].get(key, []), derived["broken"][key])
        if missing:
            diffs.append(f"broken_pairs.{key}: in YAML but not re-derived: {missing}")
        if extra:
            diffs.append(f"broken_pairs.{key}: re-derived but not in YAML: {extra}")
    yaml_pairs = [tuple(p) for p in dq["identical_pairs"]]
    missing, extra = Q.diff_lists(yaml_pairs, derived["identical_pairs"])
    if missing:
        diffs.append(f"identical_pairs: in YAML but not re-derived: {missing}")
    if extra:
        diffs.append(f"identical_pairs: re-derived but not in YAML: {extra}")
    return diffs


def print_quality(dq: dict, derived: dict, diffs: list[str]) -> None:
    b = derived["broken"]
    print(f"  broken pairs re-derived: swapped {len(b['swapped'])}, both_vietnamese {len(b['both_vietnamese'])}, "
          f"both_english {len(b['both_english'])} (YAML: {sum(len(v) for v in dq['broken_pairs'].values())} total)")
    pairs = derived["identical_pairs"]
    cross = sum(1 for a, c in pairs if a.split("/")[0] != c.split("/")[0])
    print(f"  identical pairs (same word 5-gram set in both languages): {len(pairs)} ({cross} cross-category); "
          f"YAML {len(dq['identical_pairs'])}")
    print(f"  exact text duplicates after NFC + whitespace collapse: {len(derived['exact_text_groups'])} groups")
    if diffs:
        print("  QUALITY MISMATCH with data_quality YAML (the split uses the YAML lists):")
        for d in diffs:
            print(f"    - {d}")
    else:
        print("  data_quality YAML lists match the re-derivation exactly.")
    for measure, res in derived["near"].items():
        parts = []
        for thr in (0.3, 0.5, 0.8):
            g = Q.group_stats(res.groups(thr))
            parts.append(f"{thr}: {g['n_groups']} groups / {g['docs_in_multi_doc_groups']} docs in multi-doc groups / largest {g['largest_group']}")
        print(f"  near-duplicate groups, measure {measure} (max_df {res.max_df}): " + "; ".join(parts))
    print("  (exact measure: 0.8 -> 792 groups / 110 docs; 0.5 -> 701 / 249; 0.3 -> giant component of 290)")


def build_inputs(df, dq: dict, derived: dict) -> S.SplitInputs:
    ids = derived["ids"]
    return S.SplitInputs(
        sample_ids=ids,
        categories=df["category"].tolist(),
        en_words=[S.whitespace_words(t) for t in df["english_markdown"]],
        vi_words=[S.whitespace_words(t) for t in df["vietnamese_markdown"]],
        broken={k: list(v) for k, v in dq["broken_pairs"].items()},
        identical_pairs=[tuple(p) for p in dq["identical_pairs"]],
        containment={m: (r.containment["en"], r.containment["vi"]) for m, r in derived["near"].items()},
        en_chars=[len(normalize_text(t)) for t in df["english_markdown"]],
        vi_chars=[len(normalize_text(t)) for t in df["vietnamese_markdown"]],
    )


def print_split(spec: S.SplitSpec) -> None:
    sm = spec.summary
    el = spec.eligibility
    print(f"  train {el['n_train']} docs; dev-eligible {el['n_eligible']} "
          f"(ineligible: broken {len(el['ineligible_broken'])}, identical-pair members "
          f"{len(el['ineligible_identical_pair_member'])}, other near-duplicates {len(el['ineligible_near_duplicate'])})")
    print(f"  dev {sm['n_dev']}, dev_mini {sm['n_dev_mini']}, train_fit {sm['n_train_fit']}, "
          f"excluded broken {len(spec.excluded['broken'])} + duplicate_dropped {len(spec.excluded['duplicate_dropped'])}")
    print("\n  | category | eligible | dev docs | dev words | dev_mini docs | dev_mini words | train_fit docs | train_fit words |")
    print("  |---|---:|---:|---:|---:|---:|---:|---:|")
    for c, d in sm["per_category"].items():
        print(f"  | {c} | {el['eligible_by_category'].get(c, 0)} | {d.get('dev_docs', 0)} | {d.get('dev_words', 0):,} | "
              f"{d.get('dev_mini_docs', 0)} | {d.get('dev_mini_words', 0):,} | {d.get('train_fit_docs', 0)} | {d.get('train_fit_words', 0):,} |")
    print(f"\n  dev words {sm['dev_words']:,} (EN+VI whitespace words), dev_mini words {sm['dev_mini_words']:,}")
    print(f"  dev share of dev+train_fit: words {sm['dev_share_of_dev_plus_train_fit_words']:.2%}, "
          f"tokens {sm['dev_share_of_dev_plus_train_fit_tokens_ESTIMATE']:.2%} (ESTIMATE, HY tokens/word)")
    print(f"  dev mean words/doc {sm['dev_mean_words']:,} vs train_fit {sm['train_fit_mean_words']:,}; "
          f"dev median {sm['dev_median_words']:,}; largest dev doc holds {sm['dev_largest_doc_word_share']:.1%} "
          f"of dev words, top 3 hold {sm['dev_top3_doc_word_share']:.1%}")
    print(f"  near_relative dev docs (max containment into train_fit >= "
          f"{spec.protocol['near_duplicates']['near_relative_threshold']}): {sm['n_near_relative']} / {sm['n_dev']}")
    lo, hi = spec.protocol.get("length_ratio_flag", [0.6, 1.3])
    print(f"  dev docs with VI/EN char ratio outside [{lo}, {hi}] (flag only, reference may be partial): "
          f"{sm['dev_length_ratio_flagged']}")
    print("\n  | dev doc | category | words | stratum | max containment | near_relative | VI/EN chars | dev_mini |")
    print("  |---|---|---:|---:|---:|---|---:|---|")
    for s in spec.dev_ids:
        d = spec.doc_info[s]
        print(f"  | {s} | {d['category']} | {d['words']:,} | {d['length_stratum']} | "
              f"{d['max_containment_into_train_fit']:.3f} | {'yes' if d['near_relative'] else 'no'} | "
              f"{d['vi_en_char_ratio']} | {'yes' if d['in_dev_mini'] else ''} |")
    for n in el["notes"]:
        print(f"  NOTE: {n}")


def public_overlap(spec: S.SplitSpec, derived: dict, params: dict) -> dict[str, np.ndarray]:
    pt = load_split("public_test").df
    nd = params["near_duplicates"]
    pids = pt["sample_id"].tolist()
    checks = Q.language_checks(pids, pt["english_markdown"].tolist(), pt["vietnamese_markdown"].tolist(),
                               threshold=float(params["language_vi_share_threshold"]))
    en, vi = Q.corrected_columns(checks, pt["english_markdown"].tolist(), pt["vietnamese_markdown"].tolist())
    n, k = int(nd["shingle_n"]), int(nd.get("sample_k", 1))
    q_sets = {"en": Q.shingle_sets(en, n=n, sample_k=k), "vi": Q.shingle_sets(vi, n=n, sample_k=k)}
    pos = {s: i for i, s in enumerate(derived["ids"])}
    tf = np.array([pos[s] for s in spec.train_fit_ids])
    best = np.zeros(len(pids))
    out: dict[str, np.ndarray] = {}
    for measure, res in derived["near"].items():
        m_best = np.zeros(len(pids))
        for lang in ("en", "vi"):
            I, qsz = Q.query_intersections(res.indexes[lang], q_sets[lang], max_df=res.max_df)
            C = Q.containment(I, qsz, res.sizes[lang])[:, tf]
            m_best = np.maximum(m_best, C.max(axis=1))
        best = np.maximum(best, m_best)
        out[measure] = m_best
        print(f"  public test, measure {measure}: docs with max containment into train_fit >= 0.8: "
              f"{int((m_best >= 0.8).sum())}/{len(pids)}, >= 0.5: {int((m_best >= 0.5).sum())}/{len(pids)}")
    print(f"  public test, max over measures: >= 0.5: {int((best >= 0.5).sum())}/{len(pids)} "
          f"({(best >= 0.5).mean():.0%}); dev near_relative: {spec.summary['n_near_relative']}/{spec.summary['n_dev']} "
          f"({spec.summary['n_near_relative'] / max(1, spec.summary['n_dev']):.0%})")
    print(f"  public test status counts: {dict(sorted(Counter(c.status for c in checks).items()))}")
    out["max"] = best
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v0", help="protocol version block in configs/LegalMT/splits.yaml")
    ap.add_argument("--out", type=Path, default=None, help="output JSON (default configs/LegalMT/dev_split_<version>.json)")
    ap.add_argument("--dry-run", action="store_true", help="print everything, write nothing")
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if the output file would change")
    ap.add_argument("--strict", action="store_true", help="exit 1 when the data_quality YAML differs from the re-derivation")
    ap.add_argument("--no-public", action="store_true", help="skip the public-test overlap report")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing split JSON whose content would change (never needed for a first build)")
    args = ap.parse_args(argv)

    params = load_config("splits")[args.version]
    dq = load_config(params.get("data_quality_config", "data_quality_v0"))
    out_path = args.out if args.out is not None else S.spec_path(args.version)

    print(f"=== LegalMT dev split {args.version} (provisional protocol) ===")
    train_csv = split_path("train")
    df = load_split("train").df
    data = {"train_csv": str(train_csv.relative_to(REPO_ROOT)), "train_sha256": S.file_sha256(train_csv), "train_rows": len(df)}
    print(f"train rows {len(df)}, sha256 {data['train_sha256'][:16]}...")

    print("\n-- 1. data quality re-derivation --")
    derived = derive_quality(df, params)
    diffs = compare_quality(dq, derived)
    print_quality(dq, derived, diffs)

    print("\n-- 2. split --")
    spec = S.build_split_spec(build_inputs(df, dq, derived), params, args.version, data)
    print_split(spec)

    if not args.no_public:
        print("\n-- 3. public-test overlap (for comparison only; the public test is never used to build the split) --")
        public_overlap(spec, derived, params)

    text = spec.to_json()
    rc = 0
    if args.check:
        old = out_path.read_text(encoding="utf-8") if out_path.is_file() else None
        same = old == text
        print(f"\ncheck: {out_path} {'is up to date' if same else 'WOULD CHANGE'}")
        rc = 0 if same else 1
    elif args.dry_run:
        print("\ndry run: nothing written")
    else:
        old = out_path.read_text(encoding="utf-8") if out_path.is_file() else None
        if old is not None and old != text and not args.overwrite:
            print(f"\nREFUSED: {out_path} exists and this build differs from it. Runs trained or scored on the "
                  f"existing split would no longer match it. Add a new version block to splits.yaml, or pass "
                  f"--overwrite if nothing depends on the old file yet.")
            return 2
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"\nwrote {out_path} ({len(text):,} bytes){' (unchanged)' if old == text else ''}")
    if diffs and args.strict:
        rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
