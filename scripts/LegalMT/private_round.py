#!/usr/bin/env python3
"""Run the AIHub private round in one command: translate every document, write results.csv and the zip, validate both.

    scripts/LegalMT/gpu_run.sh scripts/LegalMT/private_round.py --input FILE --model MERGED_DIR --out OUT_DIR
    ... --mock-public OUT_DIR/mock_input.csv
    ... --score-public
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from LegalMT import runs as R
from LegalMT.langid import detect_lang

csv.field_size_limit(1 << 30)
PLATFORM_DIR = {"en": "en-vi", "vi": "vi-en"}
SRC_OF = {"en-vi": "en", "vi-en": "vi"}
TGT_OF = {"en-vi": "vi", "vi-en": "en"}
SIDE = {"en": "english_markdown", "vi": "vietnamese_markdown"}
ID_KEYS = ("sample_id", "id", "doc_id")
DIR_KEYS = ("direction", "dir", "lang_pair")
LANG_KEYS = ("src_lang", "source_lang", "source_language", "language", "lang")
TEXT_KEYS = ("source", "src", "text", "input", "document", "content", "source_text", "source_markdown")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def norm_direction(v: str) -> str | None:
    s = str(v).strip().lower().replace("_", "-").replace("2", "-").replace(" ", "")
    s = {"english-vietnamese": "en-vi", "vietnamese-english": "vi-en"}.get(s, s)
    return s if s in SRC_OF else None


def load_records(path: Path) -> list[dict[str, Any]]:
    suf = path.suffix.lower()
    if suf == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))
    if suf == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suf == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            lists = [v for v in obj.values() if isinstance(v, list)]
            if len(lists) != 1:
                raise SystemExit(f"{path}: a JSON object must hold exactly one list, keys {sorted(obj)}")
            obj = lists[0]
        return list(obj)
    raise SystemExit(f"{path}: unsupported input type {suf!r} (csv, json, jsonl)")


def read_input(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    recs = load_records(path)
    if not recs:
        raise SystemExit(f"{path}: no records")
    cols = list(recs[0].keys())
    low = {c.lower().strip(): c for c in cols}
    pick = lambda keys: next((low[k] for k in keys if k in low), None)
    cid, cdir, clang, ctext = pick(ID_KEYS), pick(DIR_KEYS), pick(LANG_KEYS), pick(TEXT_KEYS)
    sides = {lang: low.get(col) for lang, col in SIDE.items()}
    report = {"file": str(path), "sha256": sha256_file(path), "records": len(recs), "columns": cols,
              "id": cid, "direction": cdir, "src_lang": clang, "text": ctext, "sides": sides}
    if cid is None or (ctext is None and not any(sides.values())):
        raise SystemExit(f"cannot find the id and text columns in {cols}; report: {json.dumps(report, default=str)}")
    docs, problems = [], []
    for i, r in enumerate(recs):
        sid = str(r[cid]).strip()
        direction, how = None, None
        if cdir and str(r.get(cdir) or "").strip():
            direction, how = norm_direction(r[cdir]), "direction column"
            if direction is None:
                problems.append(f"row {i} ({sid}): direction value not understood")
                continue
        elif clang and str(r.get(clang) or "").strip():
            lang = str(r[clang]).strip().lower()[:2]
            direction, how = PLATFORM_DIR.get(lang), "source-language column"
        if ctext:
            text = r.get(ctext) or ""
        else:
            filled = {lang: (r.get(col) or "") for lang, col in sides.items() if col and str(r.get(col) or "").strip()}
            if direction:
                text = filled.get(SRC_OF[direction], "")
            elif len(filled) == 1:
                (lang, text), = filled.items()
                direction, how = PLATFORM_DIR[lang], "the only non-empty side"
            else:
                problems.append(f"row {i} ({sid}): {len(filled)} non-empty sides and no direction")
                continue
        if not str(text).strip():
            problems.append(f"row {i} ({sid}): empty source text")
            continue
        detected = detect_lang(text)
        if direction is None:
            direction, how = PLATFORM_DIR[detected], "langid"
        docs.append({"sample_id": sid, "direction": direction, "src_lang": SRC_OF[direction], "text": text,
                     "how": how, "langid_agrees": detected == SRC_OF[direction]})
    keys = [(d["sample_id"], d["direction"]) for d in docs]
    dups = sorted({k for k in keys if keys.count(k) > 1})
    if dups:
        problems.append(f"{len(dups)} duplicated (id, direction) pairs, e.g. {dups[:3]}")
    report.update({"documents": len(docs), "by_direction": {d: sum(1 for x in docs if x["direction"] == d) for d in SRC_OF},
                   "direction_from": {h: sum(1 for x in docs if x["how"] == h) for h in sorted({x["how"] for x in docs})},
                   "langid_disagrees": [x["sample_id"] for x in docs if not x["langid_agrees"]],
                   "source_words": sum(len(x["text"].split()) for x in docs), "problems": problems})
    if problems:
        raise SystemExit("input problems (nothing translated): " + json.dumps(report, ensure_ascii=False, default=str))
    return docs, report


def mock_public(out: Path) -> None:
    from LegalMT.splits import docs_for
    df = docs_for("public_test", "v0")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["sample_id", "direction", "source"])
        for i, r in enumerate(df.itertuples(index=False)):
            d = "en-vi" if i % 2 == 0 else "vi-en"
            w.writerow([r.sample_id, d, getattr(r, SIDE[SRC_OF[d]])])
    print(f"wrote {out}: {len(df)} documents, one direction each (alternating)")


def write_submission(docs: list[dict[str, Any]], trans: dict[tuple[str, str], str], out_dir: Path, stamp: str) -> dict[str, Any]:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    w.writerow(["sample_id", "direction", "translation"])
    for d in docs:
        w.writerow([d["sample_id"], d["direction"], trans[(d["sample_id"], d["direction"])]])
    data = buf.getvalue().encode("utf-8")
    back = list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))
    want = {(d["sample_id"], d["direction"]) for d in docs}
    got = [(r["sample_id"], r["direction"]) for r in back]
    if len(got) != len(want) or set(got) != want or any(not r["translation"].strip() for r in back):
        raise SystemExit("results.csv failed its round-trip check")
    if any(r["translation"] != trans[(r["sample_id"], r["direction"])] for r in back):
        raise SystemExit("results.csv changed a translation on its round trip")
    (out_dir / "results.csv").write_bytes(data)
    zpath = out_dir / f"submission_{stamp}.zip"
    with zipfile.ZipFile(zpath, "x", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.csv", data)
    with zipfile.ZipFile(zpath) as zf:
        if zf.namelist() != ["results.csv"] or zf.read("results.csv") != data:
            raise SystemExit(f"{zpath} failed its byte check")
    wrong = [k for k in want if detect_lang(trans[k]) != TGT_OF[k[1]]]
    return {"zip": str(zpath), "zip_sha256": sha256_file(zpath), "csv_bytes": len(data), "rows": len(back),
            "translation_not_in_target_language": sorted(f"{a} {b}" for a, b in wrong)}


def score_public(docs: list[dict[str, Any]], trans: dict[tuple[str, str], str]) -> dict[str, Any]:
    from sacrebleu.metrics import BLEU, CHRF
    from LegalMT.splits import docs_for
    rows = {r.sample_id: r for r in docs_for("public_test", "v0").itertuples(index=False)}
    res = {}
    for d in SRC_OF:
        ks = [(x["sample_id"], x["direction"]) for x in docs if x["direction"] == d]
        hs = [trans[k] for k in ks]
        rs = [getattr(rows[k[0]], SIDE[TGT_OF[d]]) for k in ks]
        res[d] = {"docs": len(ks), "bleu": BLEU().corpus_score(hs, [rs]).score, "chrf": CHRF().corpus_score(hs, [rs]).score}
    res["macro"] = {m: sum(res[d][m] for d in SRC_OF) / 2 for m in ("bleu", "chrf")}
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="the private test input file (csv, json, jsonl)")
    ap.add_argument("--model", help="merged model directory (sets model.local_path); omit for configs without weights")
    ap.add_argument("--config", default="hymt2_final_graphs")
    ap.add_argument("--set", action="append", default=[], metavar="KEY.PATH=VALUE")
    ap.add_argument("--out", help="output folder (created; reused to resume)")
    ap.add_argument("--docs-per-call", type=int, default=10, help="documents per translate call (a crash loses at most one group)")
    ap.add_argument("--check-only", action="store_true", help="read and validate the input, print the report, translate nothing")
    ap.add_argument("--mock-public", metavar="FILE", help="write a private-style input built from the public test, then exit")
    ap.add_argument("--score-public", action="store_true", help="rehearsal: score against the public test references")
    a = ap.parse_args(argv)
    if a.mock_public:
        mock_public(Path(a.mock_public))
        return 0
    if not a.input:
        ap.error("--input is required")
    docs, report = read_input(Path(a.input))
    print("input: " + json.dumps({k: v for k, v in report.items() if k not in ("langid_disagrees",)}, ensure_ascii=False, default=str))
    if report["langid_disagrees"]:
        print(f"WARNING: langid disagrees with the given direction for {len(report['langid_disagrees'])} documents "
              f"(the given direction is used): {report['langid_disagrees'][:10]}")
    if a.check_only:
        return 0
    if not a.out:
        ap.error("--out is required")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tpath = out / "translations.jsonl"
    done = {}
    if tpath.is_file():
        for line in tpath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                done[(t["sample_id"], t["direction"])] = t
        print(f"resuming: {len(done)} documents already translated in {tpath}")

    from LegalMT.profiling import GpuMemorySampler, Stopwatch, memory_report
    from LegalMT.translate import Translator, load_infer_config, parse_set_overrides
    sets = list(a.set) + ([f"model.local_path={a.model}"] if a.model else [])
    cfg, cfg_path = load_infer_config(a.config, parse_set_overrides(sets))
    sampler, sw = GpuMemorySampler(interval=0.2).start(), Stopwatch()
    todo = [d for d in docs if (d["sample_id"], d["direction"]) not in done]
    tr = None
    try:
        if todo:
            with sw.lap("load"):
                tr = Translator(cfg)
            print(f"loaded {tr.info().get('config_name')} ({tr.backend_name}) in {tr.load_seconds:.1f} s; {len(todo)} documents to translate")
            for src_lang in ("en", "vi"):
                group_all = [d for d in todo if d["src_lang"] == src_lang]
                for k in range(0, len(group_all), max(1, a.docs_per_call)):
                    group = group_all[k:k + a.docs_per_call]
                    with sw.lap("translate"):
                        results = tr.translate_documents([(d["sample_id"], d["text"], d["src_lang"]) for d in group])
                    with tpath.open("a", encoding="utf-8") as fh:
                        for d, res in zip(group, results):
                            rec = {"sample_id": d["sample_id"], "direction": d["direction"], "translation": res.text,
                                   "seconds": res.stats.get("seconds"), "fallback_units": res.stats.get("fallback_units"),
                                   "truncated_requests": res.stats.get("truncated_requests")}
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            done[(d["sample_id"], d["direction"])] = rec
                    print(f"[{PLATFORM_DIR[src_lang]}] {min(k + a.docs_per_call, len(group_all))}/{len(group_all)} documents, "
                          f"translate {sw.seconds.get('translate', 0.0):.1f} s", flush=True)
    finally:
        sampler.stop()
        info = tr.info() if tr is not None else None
        if tr is not None:
            tr.close()
    trans = {k: v["translation"] for k, v in done.items()}
    missing = [(d["sample_id"], d["direction"]) for d in docs if (d["sample_id"], d["direction"]) not in trans]
    if missing:
        raise SystemExit(f"{len(missing)} documents have no translation, e.g. {missing[:3]}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sub = write_submission(docs, trans, out, stamp)
    weights = Path(a.model) / "model.safetensors" if a.model else None
    manifest = {"created": R.now_iso(), "input": report, "model": a.model,
                "model_sha256": sha256_file(weights) if weights and weights.is_file() else None,
                "config": cfg, "config_path": str(cfg_path) if cfg_path else None, "translator": info,
                "code_fingerprint": R.code_fingerprint(), "timing": {k: round(v, 1) for k, v in sw.seconds.items()},
                "memory": memory_report(sampler), "submission": sub}
    if a.score_public:
        manifest["public_score"] = score_public(docs, trans)
        print("public score (rehearsal): " + json.dumps(manifest["public_score"]))
    (out / f"manifest_{stamp}.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"wrote {sub['zip']} ({sub['rows']} rows, sha256 {sub['zip_sha256']}); "
          f"translations not in the target language: {len(sub['translation_not_in_target_language'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
