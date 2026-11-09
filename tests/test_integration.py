"""Cross-module integration checks for LegalMT (the seams between modules)."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from LegalMT import paths

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts" / "LegalMT"
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_legalmt_integ_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


@pytest.mark.data
def test_echo_script_chain_on_dev_mini(tmp_path):
    from LegalMT.splits import load_split_spec

    env = {**os.environ, "LEGALMT_ARTIFACTS_ROOT": str(tmp_path), "CUDA_VISIBLE_DEVICES": ""}

    def run(script: str, *args: str) -> str:
        res = subprocess.run([sys.executable, str(SCRIPTS / f"{script}.py"), *args], capture_output=True,
                             text=True, env=env, timeout=900)
        assert res.returncode == 0, f"{script} exited {res.returncode}: {res.stderr[-3000:]}"
        return res.stdout

    ids = set(load_split_spec("v0").dev_mini_ids)
    run("translate", "--config", "echo_smoke", "--split", "dev_mini", "--tag", "integ_echo")
    (run_dir,) = (tmp_path / "runs").iterdir()
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert man["status"] == "done" and man["split"] == "dev_mini" and man["directions"] == ["en2vi", "vi2en"]
    assert {"load_seconds", "translate_seconds", "total_seconds"} <= set(man["timing"])
    assert {"peak_mib_nvidia_smi", "peak_mib_torch_allocated"} <= set(man["memory"])
    assert {"words_per_second", "out_tokens_per_second"} <= set(man["throughput"])

    hyps = _read_jsonl(run_dir / "hyps.jsonl")
    by_dir: dict[str, set[str]] = defaultdict(set)
    for h in hyps:
        assert h["direction"] == f"{h['src_lang']}2{h['tgt_lang']}" and h["fallback_units"] == 0
        by_dir[h["direction"]].add(h["sample_id"])
    assert by_dir == {"en2vi": ids, "vi2en": ids} and len(hyps) == 2 * len(ids)
    units = _read_jsonl(run_dir / "units.jsonl")
    assert units and all(u["hyp"] == u["src"] and u["direction"] in by_dir for u in units)
    assert sum(h["n_units"] for h in hyps) == len(units)

    run("evaluate", str(run_dir))
    res = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    for d in ("en2vi", "vi2en"):
        r = res["directions"][d]
        assert r["n_docs"] == r["n_hyps"] == r["n_wrong_language"] == len(ids) and r["n_missing"] == 0
        assert 0 < r["bleu"] < 20
    index = _read_jsonl(tmp_path / "results_index.jsonl")
    assert [x["run_name"] for x in index] == [run_dir.name] and set(index[0]["metrics"]) >= {"en2vi", "vi2en"}

    run("evaluate_segments", str(run_dir))
    segs = _read_jsonl(run_dir / "segments.jsonl")
    assert segs and all(s["mt"] == s["src"] for s in segs)
    rs = json.loads((run_dir / "results_segments.json").read_text(encoding="utf-8"))
    for d in ("en2vi", "vi2en"):
        r = rs["directions"][d]
        assert r["n_docs"] == len(ids) and not r["skipped_docs"] and not r["missing_hyps"]
        assert r["n_segments"] == sum(1 for s in segs if s["direction"] == d)

    table = run("make_results_table", "--split", "dev_mini")
    rows = [ln for ln in table.splitlines() if ln.startswith(f"| {run_dir.name} |")]
    assert len(rows) == 2 and all(f"wrong-lang {len(ids)}/{len(ids)}" in r for r in rows)


@pytest.mark.data
def test_real_sft_dev_prompts_are_the_translators_token_ids():
    pytest.importorskip("transformers")
    from LegalMT import sft_data as S
    from LegalMT.chunking import JOINER, one_line
    from LegalMT.translate import Translator, load_infer_config

    out = paths.artifacts("sft", "sft_v1")
    if not (out / "manifest.json").is_file():
        pytest.skip(f"no SFT build at {out}")
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    aligned_path = Path(man["inputs"]["aligned"])
    if not aligned_path.is_file():
        pytest.skip(f"aligned input {aligned_path} is gone")
    infer_name = man["infer_config"]["name"]
    infer, _ = load_infer_config(infer_name, man["config"].get("infer_overrides") or None)
    from LegalMT.translate.core import STRUCTURAL_DEFAULTS, deep_merge

    for block in ("prompt", "chunking", "layout"):
        built = man["infer_config"]["resolved"].get(block)
        if isinstance(built, dict) and isinstance(STRUCTURAL_DEFAULTS.get(block), dict):
            built = deep_merge(STRUCTURAL_DEFAULTS[block], built)
        assert infer.get(block) == built, f"{infer_name}.{block} changed after the build"
    try:
        tr = Translator({**infer, "backend": {"name": "echo", "tokenizer": "model"}})
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"tokenizer snapshot not cached: {exc}")

    train_lora = _load_script("train_lora")
    dev = _read_jsonl(out / "dev.jsonl")
    rows = train_lora.tokenize_examples(tr.tokenizer, infer["model"]["family"], dev)
    aligned = {s: {r["order"]: r for r in recs} for s, recs in S.read_aligned(aligned_path, {e["sample_id"] for e in dev}).items()}
    checked = 0
    for ex, row in zip(dev, rows):
        if ex["context"]:
            continue
        src = JOINER.join(one_line(aligned[ex["sample_id"]][o][ex["src_lang"]]) for o in ex["orders"])
        req = tr._request(src, ex["direction"], ex["tgt_lang"], None)
        assert req.prompt_ids == row["input_ids"][: row["n_prompt"]], ex["id"]
        checked += 1
    assert checked > 0


from test_train_tiny import _cfg, artifacts, tiny_model, tiny_sft

SYNTHETIC_DOC = ("Article 1. Scope\nThe keeper of the garden shall water every row each morning.\n\n"
                 "Article 2. Effect\nThis rule takes effect on the date of signing.\n")


def test_trained_adapter_reaches_the_translator_like_the_merged_model(tiny_model, tiny_sft, artifacts):
    import torch

    from LegalMT.translate import Translator, load_infer_config

    train_lora = _load_script("train_lora")
    merge_lora = _load_script("merge_lora")
    cfg, cfg_path = _cfg(train_lora, tiny_model, tiny_sft)
    summary = train_lora.run(cfg, cfg_path, argv=["test"])
    adapter = Path(summary["adapter_dir"])
    merged = artifacts / "merged" / "integ"
    merge_lora.merge(adapter, merged, save_dtype="float32")

    def translator(model_dir: Path, **over) -> Translator:
        base = {"model": {"local_path": str(model_dir), "repo_id": None, "revision": None},
                "backend": {"device": "cpu", "batch_size": 4}, "generation": {"max_new_tokens_cap": 8}}
        icfg, _ = load_infer_config("hymt15_hf_greedy", {**base, **over})
        return Translator(icfg)

    trs = {
        "adapter": translator(tiny_model, lora_adapter=str(adapter)),
        "adapter_unmerged": translator(tiny_model, lora_adapter=str(adapter),
                                       backend={"device": "cpu", "batch_size": 4, "lora_merge": False}),
        "merged": translator(merged),
        "base": translator(tiny_model),
    }
    req = trs["base"]._request("The keeper shall water the garden.", "en2vi", "vi", None)
    x = torch.tensor([req.prompt_ids])
    with torch.no_grad():
        logits = {k: tr.backend.model(input_ids=x).logits for k, tr in trs.items()}
    assert torch.allclose(logits["adapter"], logits["merged"], atol=1e-5, rtol=0)
    assert torch.allclose(logits["adapter_unmerged"], logits["merged"], atol=1e-5, rtol=0)
    assert (logits["adapter"] - logits["base"]).abs().max() > 1e-3
    texts = {k: tr.translate_text(SYNTHETIC_DOC, "en").text for k, tr in trs.items() if k != "base"}
    assert texts["adapter"] == texts["merged"] == texts["adapter_unmerged"]
