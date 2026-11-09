"""Tests for LegalMT.prompts, with and without the real tokenizers."""
from __future__ import annotations

import os

import pytest

from LegalMT import paths
from LegalMT import prompts as P

SRC = "Line one of the source.\nLine two; with a clause."


def test_templates_are_the_card_text():
    assert P.TEMPLATES["default"] == ("Translate the following segment into {target_language}, without additional "
                                      "explanation.\n\n{source_text}")
    assert P.TEMPLATES["contextual"] == ("{context}\n参考上面的信息，把下面的文本翻译成{target_language}，"
                                         "注意不需要翻译上文，也不要额外解释：\n{source_text}\n")
    assert P.TEMPLATES["terminology"] == ("参考下面的翻译：\n{terms}\n\n将以下文本翻译为{target_language}，"
                                          "注意只需要输出翻译后的结果，不要额外解释：\n{source_text}")
    assert P.LANG_NAMES == {"en": "English", "vi": "Vietnamese"}


def test_render_default_both_directions():
    assert P.render_user_prompt("hymt", SRC, "vi") == (
        "Translate the following segment into Vietnamese, without additional explanation.\n\n" + SRC)
    assert P.render_user_prompt("qwen", "Xin chào.", "en").startswith(
        "Translate the following segment into English, without")


def test_render_contextual_and_terminology():
    ctx = "Previous chunk line."
    out = P.render_user_prompt("hymt", SRC, "vi", context=ctx, template="contextual")
    assert out.startswith(ctx + "\n参考上面的信息") and out.endswith(SRC + "\n") and "Vietnamese" in out
    terms = [("court", "tòa án"), ("decree", "nghị định")]
    out = P.render_user_prompt("hymt", SRC, "vi", terms=terms, template="terminology")
    assert "court 翻译成 tòa án\ndecree 翻译成 nghị định\n\n" in out and out.endswith(SRC)
    out = P.render_user_prompt("hymt", SRC, "vi", terms=terms, template="hymt2_terminology")
    assert "court translates to tòa án" in out
    out = P.render_user_prompt("hymt", SRC, "en", context=ctx, template="hymt2_contextual", lang_names="zh")
    assert "[Background Information]\n" + ctx in out and "英语" in out


def test_render_rejects_misuse():
    with pytest.raises(ValueError):
        P.render_user_prompt("hymt", SRC, "vi", template="contextual")
    with pytest.raises(ValueError):
        P.render_user_prompt("hymt", SRC, "vi", context="x")
    with pytest.raises(ValueError):
        P.render_user_prompt("hymt", SRC, "vi", template="terminology")
    with pytest.raises(ValueError):
        P.render_user_prompt("hymt", SRC, "fr")
    with pytest.raises(ValueError):
        P.render_user_prompt("llama", SRC, "vi")
    with pytest.raises(ValueError):
        P.render_user_prompt("hymt", SRC, "vi", template="nope")


def test_braces_and_placeholders_in_source_survive():
    src = "Clause {1} refers to {source_text} and {context} literally."
    out = P.render_user_prompt("hymt", src, "vi", context="ctx {source_text}", template="contextual")
    assert out.count(src) == 1 and out.startswith("ctx {source_text}\n")


def test_manual_hymt_render():
    user = P.render_user_prompt("hymt", SRC, "vi")
    gp = P.render_hymt_manual(user)
    assert gp == P.HY_BOS + P.HY_USER + user + P.HY_ASSISTANT
    assert P.render_hymt_manual(user, mode="card_no8").endswith(P.HY_NO8)
    ex = P.render_hymt_manual(user, mode="card_no8", answer="A.")
    assert ex.endswith("A." + P.HY_EOS + P.HY_NO8)
    assert P.build_prompt_text(None, "hymt", user) == gp
    prompt, completion = P.build_training_text(None, "hymt", user, "Dòng một.")
    assert prompt == gp and completion == "Dòng một." + P.HY_EOS and P.HY_NO8 not in completion
    with pytest.raises(ValueError):
        P.build_prompt_text(None, "hymt", user, mode="bad")


def test_bos_guard_and_qwen_needs_tokenizer():
    with pytest.raises(ValueError):
        P.build_prompt_text(None, "hymt", "text with a stray " + P.HY_BOS)
    with pytest.raises(ValueError):
        P.build_prompt_text(None, "qwen", "x")


class _FakeQwenTok:
    eos_token = "<|im_end|>"
    eos_token_id = 7

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kw):
        self.calls.append(kw)
        return "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, text, add_special_tokens=True):
        self.calls.append({"add_special_tokens": add_special_tokens})
        if isinstance(text, list):
            return {"input_ids": [[1] * len(t) for t in text]}
        return {"input_ids": ([0] if add_special_tokens else []) + [1] * len(text)}

    def convert_tokens_to_ids(self, tok):
        return 7


def test_qwen_family_and_encode_without_special_tokens():
    tok = _FakeQwenTok()
    text = P.build_prompt_text(tok, "qwen", "hi")
    assert tok.calls[-1] == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    assert P.encode(tok, text) == [1] * len(text)
    assert tok.calls[-1] == {"add_special_tokens": False}
    assert P.encode_batch(tok, ["ab", "c"]) == [[1, 1], [1]]
    assert P.completion_suffix(tok, "qwen") == "<|im_end|>"
    assert P.stop_token_ids(tok, "qwen") == [7]
    with pytest.raises(ValueError):
        P.build_prompt_text(tok, "qwen", "hi", mode="card_no8")


def test_completion_suffix_hymt_without_tokenizer():
    assert P.completion_suffix(None, "hymt") == P.HY_EOS
    assert P.stop_token_ids(None, "hymt") == [P.HY_IDS["eos"]] == [120020]


HY_SNAPSHOTS = [
    ("tencent/HY-MT1.5-1.8B", "dbad03788f49709801014c95d481a514c272ca52"),
    ("tencent/Hy-MT2-1.8B", "9a341cd1b679d3efd23b46e847b01745a71ed792"),
    ("tencent/HY-MT1.5-1.8B-GPTQ-Int4", "614b90aaac3987fbe4d6b3c976000b8c996cf5ca"),
]


def _hy_tokenizer(repo_id: str, revision: str):
    pytest.importorskip("transformers")
    os.environ.setdefault("HF_HOME", str(paths.load_config("paths")["hf_cache_default"]))
    hub = pytest.importorskip("huggingface_hub")
    try:
        path = hub.snapshot_download(repo_id, revision=revision, local_files_only=True)
    except Exception as exc:
        pytest.skip(f"{repo_id}@{revision[:8]} not cached: {exc}")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


@pytest.mark.parametrize("repo_id,revision", HY_SNAPSHOTS)
@pytest.mark.parametrize("template", ["default", "contextual", "hymt2_default"])
def test_real_hy_tokenizer_prompts(repo_id, revision, template):
    tok = _hy_tokenizer(repo_id, revision)
    ctx = "Điều 3. Previous context line." if template == "contextual" else None
    user = P.render_user_prompt("hymt", SRC, "vi", context=ctx, template=template)
    for mode, last_id in (("generation_prompt", 120007), ("card_no8", 120026)):
        text = P.build_prompt_text(tok, "hymt", user, mode=mode)
        assert text == P.render_hymt_manual(user, mode=mode)
        ids = P.encode(tok, text)
        assert ids[0] == 120000 and ids.count(120000) == 1
        assert ids[1] == 120006 and ids[-1] == last_id
        assert tok.decode(ids, skip_special_tokens=True).endswith(SRC + ("\n" if template == "contextual" else ""))


@pytest.mark.parametrize("repo_id,revision", HY_SNAPSHOTS)
def test_real_hy_eos_suffix_and_training_text(repo_id, revision):
    tok = _hy_tokenizer(repo_id, revision)
    suffix = P.completion_suffix(tok, "hymt")
    assert suffix == P.HY_EOS and tok.convert_tokens_to_ids(suffix) == 120020 == tok.eos_token_id
    assert P.stop_token_ids(tok, "hymt") == [120020]
    user = P.render_user_prompt("hymt", SRC, "vi")
    prompt, completion = P.build_training_text(tok, "hymt", user, "Dòng một.\nDòng hai.")
    full_ids = P.encode(tok, prompt + completion)
    assert full_ids[-1] == 120020 and 120026 not in full_ids and full_ids.count(120000) == 1
    jinja = tok.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": "A."}],
                                    tokenize=False, add_generation_prompt=False)
    assert jinja.endswith(P.HY_EOS + P.HY_NO8)


def test_gptq_tokenizer_ignores_its_baked_padding_and_truncation():
    gptq = _hy_tokenizer(*HY_SNAPSHOTS[2])
    ref = _hy_tokenizer(*HY_SNAPSHOTS[0])
    long = " ".join(f"word{i}" for i in range(3000))
    prompt = P.build_prompt_text(gptq, "hymt", P.render_user_prompt("hymt", SRC, "vi"))
    for text in ("Short synthetic line.", long, prompt):
        assert P.encode(gptq, text) == P.encode(ref, text)
    assert len(P.encode(gptq, long)) > 4096 and len(P.encode(gptq, "Short synthetic line.")) < 10
    assert P.encode_batch(gptq, ["a", long]) == P.encode_batch(ref, ["a", long])
