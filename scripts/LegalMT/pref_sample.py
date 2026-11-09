#!/usr/bin/env python3
"""Preference training, step 1 of 2: sample translations of SFT chunks from a merged model, keep (best, worst) pairs.

    scripts/LegalMT/gpu_run.sh scripts/LegalMT/pref_sample.py --sft sft_v1 --model MERGED_DIR --out OUT_DIR \
        [--n-prompts 4000] [--seeds 1,2,3,4] [--set backend.gpu_memory_utilization=0.85]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from LegalMT import paths
from LegalMT import runs as R

SAMPLING = ["generation.do_sample=true", "generation.temperature=0.7", "generation.top_p=0.6",
            "generation.top_k=20", "generation.repetition_penalty=1.0"]
GROUP = 256


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def select_examples(sft: str, n: int, seed: int) -> list[dict[str, Any]]:
    ex = [e for e in read_jsonl(paths.artifacts("sft", sft, "train.jsonl")) if e.get("source", "in_domain") == "in_domain"]
    ex.sort(key=lambda e: e["id"])
    random.Random(seed).shuffle(ex)
    return ex[:n] if n else ex


def run_pass(name: str, sets: list[str], exs: list[dict[str, Any]], a, out: Path) -> None:
    from LegalMT.prompts import encode_batch
    from LegalMT.translate import GenRequest, Translator, load_infer_config, parse_set_overrides
    from LegalMT.translate.backends import max_new_tokens_for

    path = out / f"cand_{name}.jsonl"
    done = {r["id"] for r in read_jsonl(path)}
    todo = [e for e in exs if e["id"] not in done]
    if not todo:
        print(f"[{name}] all {len(exs)} done", flush=True)
        return
    cfg, _ = load_infer_config(a.config, parse_set_overrides(list(a.set) + sets + [f"model.local_path={a.model}"]))
    tr = Translator(cfg)
    try:
        t0 = time.monotonic()
        for k in range(0, len(todo), GROUP):
            part = todo[k:k + GROUP]
            ids = encode_batch(tr.tokenizer, [e["prompt"] for e in part])
            for e, p in zip(part, ids):
                if len(p) != e["prompt_tokens"]:
                    raise SystemExit(f"{e['id']}: prompt tokenizes to {len(p)} ids, the build says {e['prompt_tokens']}")
            reqs = [GenRequest(prompt_ids=p, src_text="", max_new_tokens=max_new_tokens_for(e["src_tokens"], e["direction"], cfg["generation"]))
                    for e, p in zip(part, ids)]
            outs = tr.backend.generate(reqs)
            with path.open("a", encoding="utf-8") as fh:
                for e, o in zip(part, outs):
                    fh.write(json.dumps({"id": e["id"], "text": o.text, "token_ids": o.token_ids, "finished": o.finished,
                                         "n_out": o.n_out}, ensure_ascii=False) + "\n")
            print(f"[{name}] {min(k + GROUP, len(todo))}/{len(todo)} prompts, {time.monotonic() - t0:.0f} s", flush=True)
    finally:
        tr.close()


def make_pairs(exs: list[dict[str, Any]], names: list[str], out: Path, min_gap: float, eos: str, eos_id: int) -> dict[str, Any]:
    from sacrebleu.metrics import BLEU, CHRF

    bleu, chrf = BLEU(effective_order=True), CHRF()
    cands = {n: {r["id"]: r for r in read_jsonl(out / f"cand_{n}.jsonl")} for n in names}
    stats = {"examples": len(exs), "pairs": 0, "skipped_gap": 0, "skipped_best_truncated": 0, "missing": 0,
             "chosen_from": {}, "rejected_from": {}, "mean_reward": {n: 0.0 for n in names}}
    gaps = []
    with (out / "pairs.jsonl").open("w", encoding="utf-8") as fh:
        for e in exs:
            if any(e["id"] not in cands[n] for n in names):
                stats["missing"] += 1
                continue
            ref = e["completion"][: -len(eos)] if e["completion"].endswith(eos) else e["completion"]
            rows = []
            for n in names:
                c = cands[n][e["id"]]
                r = (bleu.sentence_score(c["text"], [ref]).score + chrf.sentence_score(c["text"], [ref]).score) / 2
                stats["mean_reward"][n] += r
                rows.append((r, n, c))
            best = max(rows, key=lambda x: x[0])
            worst = min(rows, key=lambda x: x[0])
            if not best[2]["finished"]:
                stats["skipped_best_truncated"] += 1
                continue
            if best[0] - worst[0] < min_gap:
                stats["skipped_gap"] += 1
                continue
            gaps.append(best[0] - worst[0])
            stats["pairs"] += 1
            stats["chosen_from"][best[1]] = stats["chosen_from"].get(best[1], 0) + 1
            stats["rejected_from"][worst[1]] = stats["rejected_from"].get(worst[1], 0) + 1
            fh.write(json.dumps({"id": e["id"], "direction": e["direction"],
                                 "chosen_ids": list(best[2]["token_ids"]) + [eos_id],
                                 "rejected_ids": list(worst[2]["token_ids"]) + ([eos_id] if worst[2]["finished"] else []),
                                 "chosen_reward": round(best[0], 3), "rejected_reward": round(worst[0], 3),
                                 "chosen_from": best[1], "rejected_from": worst[1]}) + "\n")
    n_ok = len(exs) - stats["missing"]
    stats["mean_reward"] = {n: round(v / max(1, n_ok), 3) for n, v in stats["mean_reward"].items()}
    stats["mean_gap"] = round(sum(gaps) / len(gaps), 3) if gaps else None
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft", required=True, help="SFT build name under <ARTIFACTS>/sft/")
    ap.add_argument("--model", required=True, help="merged model directory (the policy that samples)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="hymt2_final_graphs")
    ap.add_argument("--set", action="append", default=[], metavar="KEY.PATH=VALUE")
    ap.add_argument("--n-prompts", type=int, default=4000, help="0 = every in-domain example")
    ap.add_argument("--select-seed", type=int, default=20261007)
    ap.add_argument("--seeds", default="1,2,3,4")
    ap.add_argument("--min-gap", type=float, default=1.0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exs = select_examples(a.sft, a.n_prompts, a.select_seed)
    print(f"{len(exs)} in-domain prompts from {a.sft} (directions: "
          f"{sorted({e['direction'] for e in exs})}, {sum(e['direction'] == 'en2vi' for e in exs)} en2vi)", flush=True)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    names = ["greedy"] + [f"s{s}" for s in seeds]
    t0 = time.monotonic()
    run_pass("greedy", [], exs, a, out)
    for s in seeds:
        run_pass(f"s{s}", SAMPLING + [f"generation.seed={s}"], exs, a, out)
    gen_s = round(time.monotonic() - t0, 1)
    from transformers import AutoTokenizer

    from LegalMT.prompts import completion_suffix
    tok = AutoTokenizer.from_pretrained(a.model)
    eos = completion_suffix(tok, "hymt")
    stats = make_pairs(exs, names, out, a.min_gap, eos, int(tok.convert_tokens_to_ids(eos)))
    man = {"created": R.now_iso(), "sft": a.sft, "model": a.model, "config": a.config, "set": a.set,
           "n_prompts": len(exs), "select_seed": a.select_seed, "seeds": seeds, "sampling": SAMPLING,
           "min_gap": a.min_gap, "reward": "(sentence BLEU + sentence chrF) / 2 vs the reference",
           "generation_seconds_this_session": gen_s, "stats": stats, "code_fingerprint": R.code_fingerprint()}
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    print(f"pairs: {json.dumps(stats)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
