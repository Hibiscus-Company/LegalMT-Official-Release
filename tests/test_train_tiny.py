"""train_lora.py and merge_lora.py on a tiny random HunYuanDenseV1 model (GPU env, CUDA hidden)."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

import pytest

from LegalMT import paths
from LegalMT import sft_data as S
from LegalMT.prompts import HY_ASSISTANT, HY_IDS
from LegalMT.translate import Translator, load_infer_config

REPO_ROOT = Path(__file__).resolve().parents[1]
HY_REPO, HY_REV = "tencent/HY-MT1.5-1.8B", "dbad03788f49709801014c95d481a514c272ca52"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_legalmt_{name}", REPO_ROOT / "scripts" / "LegalMT" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory) -> Path:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    pytest.importorskip("datasets")
    os.environ.setdefault("HF_HOME", str(paths.load_config("paths")["hf_cache_default"]))
    hub = pytest.importorskip("huggingface_hub")
    try:
        snap = hub.snapshot_download(HY_REPO, revision=HY_REV, local_files_only=True)
    except Exception as exc:
        pytest.skip(f"HY-MT1.5 snapshot not cached: {exc}")
    config = transformers.AutoConfig.from_pretrained(snap)
    for k, v in {"num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
                 "num_key_value_heads": 2, "head_dim": 16, "attention_head_dim": 16}.items():
        setattr(config, k, v)
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config)
    out = tmp_path_factory.mktemp("tiny_hy")
    model.save_pretrained(out)
    transformers.AutoTokenizer.from_pretrained(snap).save_pretrained(out)
    (out / "generation_config.json").write_bytes((Path(snap) / "generation_config.json").read_bytes())
    return out


def _docs() -> dict[str, list[dict]]:
    docs = {}
    for d in range(1, 5):
        sid = f"Syn/{d}"
        docs[sid] = [{
            "pair_id": f"{sid}#{k:05d}", "sample_id": sid, "category": "Syn", "order": k,
            "en": f"The keeper of garden {d} shall water row {k} every morning and record it in the log.",
            "vi": f"Người giữ vườn {d} phải tưới hàng {k} mỗi sáng và ghi vào sổ.",
            "link": "1-1", "method": "zip", "len_ratio": 0.8, "num_f1": None,
            "kind_en": "heading" if k % 5 == 0 else "paragraph", "kind_vi": "heading" if k % 5 == 0 else "paragraph",
        } for k in range(12)]
    return docs


@pytest.fixture(scope="module")
def tiny_sft(tiny_model, tmp_path_factory) -> Path:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model)
    infer, _ = load_infer_config("hymt15_hf_greedy", {"chunking": {"budget_tokens": {"en2vi": 60, "vi2en": 80}}})
    settings = S.BuildSettings(family="hymt", chunking=infer["chunking"], prompt=infer["prompt"], seed=3)
    out = tmp_path_factory.mktemp("sft_tiny")
    docs = _docs()
    train, _ = S.build_examples({s: docs[s] for s in ("Syn/1", "Syn/2", "Syn/3")}, settings, tok)
    dev, _ = S.build_examples({"Syn/4": docs["Syn/4"]}, settings, tok)
    S.write_jsonl(out / "train.jsonl", S.shuffle_examples(train, 3))
    S.write_jsonl(out / "dev.jsonl", S.shuffle_examples(dev, 3))
    (out / "manifest.json").write_text(json.dumps({
        "name": "sft_tiny", "family": "hymt", "trainable": True,
        "tokenizer": {"mode": "model", **S.tokenizer_fingerprint(tiny_model)},
        "infer_config": {"name": "hymt15_hf_greedy", "resolved": infer}}), encoding="utf-8")
    return out


@pytest.fixture()
def artifacts(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(paths, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    return tmp_path / "artifacts"


def _cfg(train_lora, tiny_model: Path, tiny_sft: Path, **training):
    over = {
        "model": {"local_path": str(tiny_model), "repo_id": None, "revision": None},
        "data": {"dir": str(tiny_sft), "sft": None, "max_seq_len": 512, "eval_max_examples": 4},
        "training": {"max_steps": 3, "gradient_accumulation_steps": 2, "optim": "adamw_torch", "logging_steps": 1,
                     "eval_steps": 2, "save_steps": 2, "eval_on_start": False, "device": "cpu", **training},
    }
    return train_lora.load_train_config("smoke", over)


def _capture(store: list, gc_flags: list | None = None):
    def hook(trainer):
        from transformers import TrainerCallback

        orig = trainer.data_collator

        def wrapped(features):
            batch = orig(features)
            store.append({k: v.clone() for k, v in batch.items()})
            return batch

        trainer.data_collator = wrapped
        if gc_flags is not None:
            class Probe(TrainerCallback):
                def on_step_begin(self, args, state, control, **kwargs):
                    gc_flags.append(bool(trainer.model.get_base_model().is_gradient_checkpointing))

            trainer.add_callback(Probe())

    return hook


def test_train_three_steps_prompt_ids_labels_adapter_and_merge(tiny_model, tiny_sft, artifacts):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    train_lora = _load_script("train_lora")
    merge_lora = _load_script("merge_lora")
    cfg, cfg_path = _cfg(train_lora, tiny_model, tiny_sft)
    assert cfg["model"]["dtype"] == "bfloat16" and cfg["training"]["bf16"]
    captured: list[dict] = []
    gc_flags: list[bool] = []
    summary = train_lora.run(cfg, cfg_path, argv=["test"], trainer_hook=_capture(captured, gc_flags))
    assert summary["complete"] and summary["global_step"] == 3
    assert gc_flags == [True, True, True]

    prep = train_lora.prepare_data(cfg)
    assert len(prep.train_rows) == 3 * 2 and summary["tokens_seen"] == prep.plan.total_tokens
    assert captured, "the collator was never called"
    for batch in captured:
        ids, labels = batch["input_ids"][0].tolist(), batch["labels"][0].tolist()
        n_prompt = next(i for i, x in enumerate(labels) if x != -100)
        assert all(x != -100 for x in labels[n_prompt:])
        assert labels[n_prompt:] == ids[n_prompt:] and labels[-1] == HY_IDS["eos"]
        assert ids[0] == HY_IDS["bos"] and ids.count(HY_IDS["bos"]) == 1 and HY_IDS["no8"] not in ids
        assert ids[n_prompt - 1] == HY_IDS["assistant"]
    first = captured[0]
    row = prep.train_rows[0]
    assert first["input_ids"][0].tolist() == row["input_ids"] and first["labels"][0].tolist() == row["labels"]
    exs = {json.loads(ln)["id"]: json.loads(ln) for ln in (tiny_sft / "train.jsonl").read_text(encoding="utf-8").splitlines()}
    ex = exs[row["id"]]
    src_text = ex["prompt"].split("\n\n", 1)[1][: -len(HY_ASSISTANT)]
    infer, _ = load_infer_config("echo_hytok_smoke", {"model": {"local_path": str(tiny_model), "repo_id": None,
                                                                "revision": None}})
    req = Translator(infer)._request(src_text, ex["direction"], ex["tgt_lang"], None)
    assert req.prompt_ids == row["input_ids"][: row["n_prompt"]]

    out = Path(summary["checkpoint_dir"])
    state = json.loads((out / "trainer_state.json").read_text(encoding="utf-8"))
    train_logs = [h for h in state["log_history"] if "loss" in h]
    assert len(train_logs) == 3 and all(math.isfinite(h["loss"]) for h in train_logs)
    evals = [h["eval_loss"] for h in state["log_history"] if "eval_loss" in h]
    assert evals and all(math.isfinite(x) for x in evals)
    for h in train_logs:
        assert h["tokens_seen"] == prep.plan.tokens_at(h["step"]) == h["num_tokens"]
    assert any("tokens_per_second" in h for h in train_logs)
    assert (out / "metrics.jsonl").is_file() and (out / "plan.json").is_file() and (out / "train_config.json").is_file()
    assert {json.loads(ln)["session"] for ln in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()} == {1}
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert plan["data_sha256"] == train_lora.data_fingerprint(tiny_sft) and plan["script_sha256"] == train_lora.script_sha256()
    assert plan["sft_input_checks"]["split_check"].startswith("skipped")
    man = json.loads((artifacts / "runs" / summary["run_name"] / "manifest.json").read_text(encoding="utf-8"))
    assert man["status"] == "done" and man["config"]["training"]["seed"] == 1 and man["throughput"]["tokens_per_second"]

    adapter = Path(summary["adapter_dir"])
    assert (adapter / "adapter_config.json").is_file() and (adapter / "adapter_model.safetensors").is_file()
    assert (adapter / "tokenizer_config.json").is_file()
    acfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    assert acfg["r"] == 64 and acfg["lora_alpha"] == 128 and set(acfg["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    base = AutoModelForCausalLM.from_pretrained(tiny_model, dtype=torch.float32)
    peft_model = PeftModel.from_pretrained(base, str(adapter)).eval()
    lora_b = [p for n, p in peft_model.named_parameters() if "lora_B" in n]
    assert lora_b and any(p.abs().sum() > 0 for p in lora_b)

    merged_dir = artifacts / "merged" / "tiny"
    mm = merge_lora.merge(adapter, merged_dir, save_dtype="float32")
    assert mm["base"]["source"].endswith("train_config.json") and mm["adapter_sha256"]
    merged = AutoModelForCausalLM.from_pretrained(merged_dir, dtype=torch.float32).eval()
    assert not any("lora_" in n for n, _ in merged.named_parameters())
    x = torch.tensor([row["input_ids"]])
    with torch.no_grad():
        a = peft_model(input_ids=x).logits
        b = merged(input_ids=x).logits
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)
    assert (merged_dir / "tokenizer_config.json").is_file() and (merged_dir / "generation_config.json").is_file()
    assert merge_lora.resolve_adapter(summary["run_name"]) == adapter.resolve()

    from safetensors import safe_open
    from transformers import AutoTokenizer

    from LegalMT.prompts import build_prompt_text, encode

    bf16_dir = artifacts / "merged" / "tiny_bf16"
    merge_lora.merge(adapter, bf16_dir)
    cfg_json = json.loads((bf16_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg_json["dtype"] == "bfloat16" and cfg_json["use_cache"] is True
    with safe_open(str(bf16_dir / "model.safetensors"), "pt") as f, safe_open(str(tiny_model / "model.safetensors"), "pt") as g:
        assert set(f.keys()) == set(g.keys())
        assert {f.get_tensor(k).dtype for k in f.keys()} == {torch.bfloat16}
    assert next(AutoModelForCausalLM.from_pretrained(bf16_dir).parameters()).dtype == torch.bfloat16
    t_base, t_merged = AutoTokenizer.from_pretrained(tiny_model), AutoTokenizer.from_pretrained(bf16_dir)
    for mode in ("generation_prompt", "card_no8"):
        p_base = build_prompt_text(t_base, "hymt", "Dịch đoạn văn mẫu này.", mode=mode)
        p_merged = build_prompt_text(t_merged, "hymt", "Dịch đoạn văn mẫu này.", mode=mode)
        assert p_base == p_merged and encode(t_base, p_base) == encode(t_merged, p_merged)


def test_interrupted_run_resumes_to_the_same_adapter(tiny_model, tiny_sft, artifacts):
    import torch
    from safetensors.torch import load_file
    from transformers import TrainerCallback

    train_lora = _load_script("train_lora")
    sft = artifacts.parent / "sft_copy"
    shutil.copytree(tiny_sft, sft)
    cfg, cfg_path = _cfg(train_lora, tiny_model, sft, bf16=False, eval_on_start=True)
    cfg_ref, _ = _cfg(train_lora, tiny_model, sft, bf16=False, eval_on_start=True)
    ref = train_lora.run(cfg_ref, cfg_path, argv=["test"])

    class StopAt2(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 2:
                control.should_training_stop = True

    first = train_lora.run(cfg, cfg_path, argv=["test"], trainer_hook=lambda tr: tr.add_callback(StopAt2()))
    assert not first["complete"] and first["global_step"] == 2
    man = json.loads((artifacts / "runs" / first["run_name"] / "manifest.json").read_text(encoding="utf-8"))
    assert man["status"] == "failed" and man["stopped_early"]

    train_file = sft / "train.jsonl"
    original = train_file.read_bytes()
    rows = [json.loads(ln) for ln in original.decode("utf-8").splitlines()]
    S.write_jsonl(train_file, S.shuffle_examples(rows, 99))
    assert train_file.read_bytes() != original
    started = json.loads((Path(first["checkpoint_dir"]) / "plan.json").read_text(encoding="utf-8"))
    assert train_lora.prepare_data(cfg).plan.fingerprint() == started["fingerprint"]
    with pytest.raises(ValueError, match="SFT data files"):
        train_lora.run(cfg, cfg_path, resume=first["run_name"], argv=["test"])
    train_file.write_bytes(original)

    resumed = train_lora.run(cfg, cfg_path, resume=first["run_name"], argv=["test"])
    assert resumed["complete"] and resumed["global_step"] == 3 and resumed["run_name"] == first["run_name"]
    man = json.loads((artifacts / "runs" / first["run_name"] / "manifest.json").read_text(encoding="utf-8"))
    assert man["status"] == "done" and len(man["sessions"]) == 2 and man["sessions_started"] == 2
    metrics = [json.loads(ln) for ln in (Path(resumed["checkpoint_dir"]) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {m["session"] for m in metrics} == {1, 2}
    assert any("eval_loss" in m and m["session"] == 1 and m["step"] == 0 for m in metrics)
    assert not any("eval_loss" in m and m["session"] == 2 and m["step"] == 2 for m in metrics)
    a = load_file(str(Path(ref["adapter_dir"]) / "adapter_model.safetensors"))
    b = load_file(str(Path(resumed["adapter_dir"]) / "adapter_model.safetensors"))
    assert a.keys() == b.keys()
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6), k
    changed = json.loads(json.dumps(cfg))
    changed["training"]["learning_rate"] = 1e-3
    with pytest.raises(ValueError):
        train_lora.run(changed, cfg_path, resume=first["run_name"], argv=["test"])


def test_plan_is_exact_deterministic_and_budgeted():
    train_lora = _load_script("train_lora")
    lengths = [10 + (k * 7) % 13 for k in range(20)]
    p = train_lora.build_plan(lengths, 4, token_budget=100, max_steps=None, seed=5)
    assert p.total_tokens >= 100 and sum(p.step_tokens[:-1]) < 100 and len(p.order) == 4 * p.max_steps
    assert p.step_tokens == [sum(lengths[i] for i in p.order[k * 4:(k + 1) * 4]) for k in range(p.max_steps)]
    assert p.fingerprint() == train_lora.build_plan(lengths, 4, token_budget=100, max_steps=None, seed=5).fingerprint()
    assert p.fingerprint() != train_lora.build_plan(lengths, 4, token_budget=100, max_steps=None, seed=6).fingerprint()
    q = train_lora.build_plan(lengths, 4, token_budget=None, max_steps=10, seed=5)
    assert sorted(q.order[:20]) == list(range(20)) == sorted(q.order[20:40]) and q.order[:20] != q.order[20:40]
    r = train_lora.build_plan(lengths, 4, token_budget=None, max_steps=10, seed=5, reshuffle=False)
    assert r.order[:20] == r.order[20:40] and q.epochs == 2.0
    assert train_lora.build_plan(lengths, 3, token_budget=None, max_steps=None, seed=1).max_steps == 7
    with pytest.raises(ValueError):
        train_lora.build_plan([], 4, token_budget=10, max_steps=None, seed=1)


def test_config_validation():
    train_lora = _load_script("train_lora")
    for name in ("smoke", "pilot_3M", "night_30M"):
        cfg, _ = train_lora.load_train_config(name)
        assert cfg["model"]["repo_id"] and cfg["model"]["revision"]
        assert isinstance(cfg["training"]["seed"], int) and isinstance(cfg["data"]["order_seed"], int)
        assert cfg["lora"]["r"] == 64 and cfg["lora"]["alpha"] == 128
    assert train_lora.load_train_config("pilot_3M")[0]["data"]["token_budget"] == 3_000_000
    assert train_lora.load_train_config("night_30M")[0]["data"]["token_budget"] == 30_000_000
    smoke, _ = train_lora.load_train_config("smoke")
    assert smoke["training"]["max_steps"] == 20 and smoke["data"]["max_seq_len"] == 1024
    with pytest.raises(ValueError):
        train_lora.load_train_config("smoke", {"training": {"sed": 1}})
    with pytest.raises(ValueError):
        train_lora.load_train_config("smoke", {"training": {"seed": None}})
    with pytest.raises(ValueError):
        train_lora.load_train_config("smoke", {"model": {"revision": None}})
    with pytest.raises(ValueError):
        train_lora.load_train_config("_base_hymt_lora")


def test_trainer_args_precision_and_chunked_loss(tiny_model, tiny_sft, artifacts):
    import torch

    train_lora = _load_script("train_lora")
    cfg, _ = _cfg(train_lora, tiny_model, tiny_sft, max_grad_norm=0.0, eval_on_start=True)
    prep = train_lora.prepare_data(cfg)
    fresh = train_lora.build_trainer(prep, artifacts / "t_fresh")
    a = fresh.args
    assert a.max_grad_norm == 0.0 and a.eval_on_start and a.train_sampling_strategy == "sequential"
    assert a.lr_scheduler_type == "cosine_with_min_lr" and a.lr_scheduler_kwargs == {"min_lr_rate": 0.1}
    assert a.warmup_steps == 0.1 and a.get_warmup_steps(1032) == 104 and a.loss_type == "chunked_nll"
    assert not train_lora.build_trainer(prep, artifacts / "t_resumed", resuming=True).args.eval_on_start
    trainable = {n: p for n, p in fresh.model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n and p.dtype == torch.float32 for n, p in trainable.items())
    assert {p.dtype for p in fresh.model.parameters() if not p.requires_grad} == {torch.bfloat16}

    ev = fresh.evaluate()["eval_loss"]
    model = fresh.model.eval()
    losses = []
    with torch.no_grad(), model.disable_adapter():
        base = model.get_base_model()
        for r in prep.eval_rows:
            hidden = base.model(input_ids=torch.tensor([r["input_ids"]])).last_hidden_state
            logits = base.lm_head(hidden).float()[0, :-1]
            losses.append(torch.nn.functional.cross_entropy(logits, torch.tensor(r["labels"][1:]), ignore_index=-100).item())
    assert len(losses) == len(prep.eval_rows) > 1
    assert math.isclose(ev, sum(losses) / len(losses), rel_tol=1e-4)


def test_resume_guards_and_config_ranges(tmp_path):
    train_lora = _load_script("train_lora")
    check = train_lora.check_resume_plan
    old = {"fingerprint": "a", "data_sha256": "d", "sft_tokenizer_sha256": "t", "script_sha256": "s"}
    check(old, dict(old))
    for key in ("fingerprint", "data_sha256", "sft_tokenizer_sha256"):
        for allow in (False, True):
            with pytest.raises(ValueError, match=key):
                check(old, {**old, key: "x"}, allow_code_change=allow)
    with pytest.raises(ValueError, match="train_lora.py"):
        check(old, {**old, "script_sha256": "x"})
    check(old, {**old, "script_sha256": "x"}, allow_code_change=True)
    check({"fingerprint": "a"}, old)

    (tmp_path / "train.jsonl").write_text("a\n", encoding="utf-8")
    fp1 = train_lora.data_fingerprint(tmp_path)
    (tmp_path / "dev.jsonl").write_text("b\n", encoding="utf-8")
    fp2 = train_lora.data_fingerprint(tmp_path)
    (tmp_path / "train.jsonl").write_text("c\n", encoding="utf-8")
    assert len({fp1, fp2, train_lora.data_fingerprint(tmp_path)}) == 3

    with pytest.raises(ValueError, match="warmup_ratio"):
        train_lora.load_train_config("smoke", {"training": {"warmup_ratio": 1.0}})
    with pytest.raises(ValueError, match="min_lr_rate"):
        train_lora.load_train_config("smoke", {"training": {"min_lr_rate": None}})
    assert train_lora.load_train_config("smoke", {"training": {"lr_scheduler_type": "cosine", "min_lr_rate": None}})


def test_split_leak_guard_at_training_time(tmp_path):
    train_lora = _load_script("train_lora")
    check = train_lora.check_split_roles
    check({"A/1", "A/2"}, {"B/1"}, {"A/1", "A/2", "A/3"}, {"B/1", "B/2"})
    with pytest.raises(ValueError, match="dev documents"):
        check({"A/1", "B/1"}, {"B/2"}, {"A/1"}, {"B/1", "B/2"})
    with pytest.raises(ValueError, match="not train_fit"):
        check({"A/1", "C/9"}, {"B/1"}, {"A/1"}, {"B/1"})
    with pytest.raises(ValueError, match="dev-file documents are not dev"):
        check({"A/1"}, {"A/2"}, {"A/1", "A/2"}, {"B/1"})
    got = train_lora.sft_input_checks({"name": "x"}, [{"sample_id": "Syn/1"}], [])
    assert got["split_check"].startswith("skipped") and got["aligned_current"] is None
    with pytest.raises(ValueError, match="does not match the current split"):
        train_lora.sft_input_checks({"inputs": {"split_version": "v0"}}, [{"sample_id": "Syn/1"}], [])
    aligned = tmp_path / "train.jsonl"
    aligned.write_text("x\n", encoding="utf-8")
    from LegalMT.splits import file_sha256, load_split_spec

    spec = load_split_spec("v0")
    ok = train_lora.sft_input_checks(
        {"inputs": {"split_version": "v0", "aligned": str(aligned), "aligned_sha256": file_sha256(aligned)}},
        [{"sample_id": spec.train_fit_ids[0]}], [{"sample_id": spec.dev_ids[0]}])
    assert ok == {"split_check": "ok: split v0", "aligned_current": True}
    stale = train_lora.sft_input_checks(
        {"inputs": {"split_version": "v0", "aligned": str(aligned), "aligned_sha256": "0" * 64}},
        [{"sample_id": spec.train_fit_ids[0]}], [])
    assert stale["aligned_current"] is False


def test_init_from_starts_training_at_the_given_adapter(tiny_model, tiny_sft, artifacts):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    train_lora = _load_script("train_lora")
    cfg, cfg_path = _cfg(train_lora, tiny_model, tiny_sft)
    first = train_lora.run(cfg, cfg_path, argv=["test"])
    src = Path(first["adapter_dir"])
    assert "init_from" not in json.loads((Path(first["checkpoint_dir"]) / "plan.json").read_text(encoding="utf-8"))

    def cfg_from(adapter: Path, **lora):
        c, _ = _cfg(train_lora, tiny_model, tiny_sft)
        c = copy.deepcopy(c)
        c["lora"] = {**c["lora"], "init_from": str(adapter), **lora}
        c["run_tag"] = "warm"
        return train_lora.resolve_train_config(c)

    seen: dict = {}

    def hook(trainer):
        seen.update({k: v.detach().float().clone() for k, v in get_peft_model_state_dict(trainer.model).items()})

    second = train_lora.run(cfg_from(src), cfg_path, argv=["test"], trainer_hook=hook)
    want = load_file(str(src / "adapter_model.safetensors"))
    assert set(seen) == set(want) and want
    assert all(torch.equal(seen[k], want[k].float()) for k in want)
    plan = json.loads((Path(second["checkpoint_dir"]) / "plan.json").read_text(encoding="utf-8"))
    assert plan["init_from"]["path"] == str(src) and len(plan["init_from"]["adapter_sha256"]) == 64
    with pytest.raises(ValueError, match="LoRA shape"):
        train_lora.run(cfg_from(src, r=int(cfg["lora"]["r"]) * 2), cfg_path, argv=["test"])
    with pytest.raises(ValueError, match="no adapter_config.json"):
        cfg_from(src.parent)


@pytest.mark.data
@pytest.mark.parametrize("name", ["sft_v1", "sft_v1_final"])
def test_real_sft_build_passes_the_split_guard(name):
    train_lora = _load_script("train_lora")
    out = paths.artifacts("sft", name)
    if not (out / "manifest.json").is_file():
        pytest.skip(f"no SFT build at {out}")
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    tr = train_lora.read_examples(out / "train.jsonl", S.DIRECTIONS)
    dv = train_lora.read_examples(out / "dev.jsonl", S.DIRECTIONS)
    got = train_lora.sft_input_checks(man, tr, dv)["split_check"]
    if man.get("final"):
        assert got.startswith(f"FINAL build on split {man['inputs']['split_version']}")
    else:
        assert got == f"ok: split {man['inputs']['split_version']}"


@pytest.mark.parametrize("accum,batch", [(2, 1), (1, 2), (3, 1)])
def test_train_loss_and_grad_are_token_weighted_over_the_window(tiny_model, tiny_sft, artifacts, accum, batch):
    import torch

    train_lora = _load_script("train_lora")
    cfg, _ = _cfg(train_lora, tiny_model, tiny_sft, max_steps=1, gradient_accumulation_steps=accum,
                  per_device_train_batch_size=batch, bf16=False, eval_steps=None, save_steps=None,
                  max_grad_norm=1e9, warmup_ratio=0.0)
    cfg["model"]["dtype"] = "float32"
    cfg["lora"]["dropout"] = 0.0
    prep = train_lora.prepare_data(cfg)
    trainer = train_lora.build_trainer(prep, artifacts / f"w{accum}x{batch}")
    model = trainer.model.train()
    total, n = 0.0, 0
    for r in prep.train_rows[: accum * batch]:
        logits = model(input_ids=torch.tensor([r["input_ids"]])).logits.float()[0, :-1]
        labels = torch.tensor(r["labels"][1:])
        total = total + torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")
        n += int((labels != -100).sum())
    loss = total / n
    loss.backward()
    grad_norm = math.sqrt(sum(float((p.grad.float() ** 2).sum()) for p in model.parameters() if p.grad is not None))
    model.zero_grad(set_to_none=True)
    trainer.train()
    log = next(h for h in trainer.state.log_history if "loss" in h)
    assert math.isclose(log["loss"], loss.item(), rel_tol=1e-4)
    assert math.isclose(log["grad_norm"], grad_norm, rel_tol=1e-3)


def test_merge_output_names():
    merge_lora = _load_script("merge_lora")
    assert merge_lora.default_name(Path("/a/checkpoints/20260925-120000_x/adapter")) == "20260925-120000_x"
    assert merge_lora.default_name(Path("/a/checkpoints/20260925-120000_x/checkpoint-100")) == "20260925-120000_x-checkpoint-100"
    assert merge_lora.default_name(Path("/a/my_adapter")) == "my_adapter"


@pytest.mark.parametrize("ratio", [None, 4.0])
def test_loraplus_ratio_puts_lora_b_in_scaled_groups(tiny_model, tiny_sft, artifacts, ratio):
    train_lora = _load_script("train_lora")
    cfg, cfg_path = _cfg(train_lora, tiny_model, tiny_sft, max_steps=1, learning_rate=1e-4,
                         lr_scheduler_type="constant", warmup_ratio=0.0, loraplus_lr_ratio=ratio)
    seen: list = []

    def hook(trainer):
        from transformers import TrainerCallback

        class Probe(TrainerCallback):
            def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
                names = {id(p): n for n, p in trainer.model.named_parameters()}
                seen.append([(g["lr"], sorted({"B" if "lora_B" in names[id(p)] else "other" for p in g["params"]}))
                             for g in optimizer.param_groups if g["params"]])

        trainer.add_callback(Probe())

    train_lora.run(cfg, cfg_path, argv=["test"], trainer_hook=hook)
    assert seen, "optimizer never observed"
    groups = seen[0]
    if ratio is None:
        assert all(abs(lr - 1e-4) < 1e-12 for lr, _ in groups)
    else:
        b = [lr for lr, kinds in groups if kinds == ["B"]]
        other = [lr for lr, kinds in groups if kinds == ["other"]]
        assert b and other and not [k for _, k in groups if len(k) > 1]
        assert all(abs(lr - 4e-4) < 1e-12 for lr in b) and all(abs(lr - 1e-4) < 1e-12 for lr in other)
