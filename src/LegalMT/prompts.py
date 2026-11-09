"""Prompt construction for inference and SFT; the only place that turns source text into model input."""
from __future__ import annotations

import re
from typing import Any, Sequence

LANG_NAMES = {"en": "English", "vi": "Vietnamese"}
LANG_NAMES_ZH = {"en": "英语", "vi": "越南语"}

FAMILIES = ("hymt", "qwen")
MODES = ("generation_prompt", "card_no8")

HY_BOS = "<｜hy_begin▁of▁sentence｜>"
HY_USER = "<｜hy_User｜>"
HY_ASSISTANT = "<｜hy_Assistant｜>"
HY_EOS = "<｜hy_place▁holder▁no▁2｜>"
HY_NO8 = "<｜hy_place▁holder▁no▁8｜>"
HY_IDS = {"bos": 120000, "user": 120006, "assistant": 120007, "eos": 120020, "no8": 120026, "pad": 120002}
QWEN_EOT = "<|im_end|>"


TEMPLATES: dict[str, str] = {
    "default": "Translate the following segment into {target_language}, without additional explanation.\n\n{source_text}",
    "contextual": ("{context}\n参考上面的信息，把下面的文本翻译成{target_language}，注意不需要翻译上文，也不要额外解释：\n"
                   "{source_text}\n"),
    "terminology": ("参考下面的翻译：\n{terms}\n\n将以下文本翻译为{target_language}，注意只需要输出翻译后的结果，不要额外解释：\n"
                    "{source_text}"),
    "hymt2_default": ("Translate the following text into {target_language}. Note that you should only output the "
                      "translated result without any additional explanation:\n\n{source_text}"),
    "hymt2_terminology": ("Reference the following translations:\n{terms}\n\nTranslate the following text into "
                          "{target_language}. Note that you must ONLY output the translated result without any "
                          "additional explanation:\n\n{source_text}"),
    "hymt2_contextual": ("[Background Information]\n{context}\n\nPlease translate the following text into "
                         "{target_language}, taking the provided background information into consideration.\n\n"
                         "[Source Text]\n{source_text}"),
}
TERM_JOINERS = {"terminology": "{src} 翻译成 {tgt}", "hymt2_terminology": "{src} translates to {tgt}"}
CONTEXT_TEMPLATES = frozenset({"contextual", "hymt2_contextual"})
_PLACEHOLDER = re.compile(r"\{(target_language|source_text|context|terms)\}")


def _lang_name(lang: str, lang_names: str) -> str:
    table = {"en": LANG_NAMES, "zh": LANG_NAMES_ZH}.get(lang_names)
    if table is None:
        raise ValueError(f"lang_names must be 'en' or 'zh', got {lang_names!r}")
    if lang not in table:
        raise ValueError(f"unknown language {lang!r}; expected one of {sorted(table)}")
    return table[lang]


def render_user_prompt(
    family: str,
    src_text: str,
    tgt_lang: str,
    *,
    context: str | None = None,
    terms: Sequence[tuple[str, str]] | None = None,
    template: str = "default",
    lang_names: str = "en",
) -> str:
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}; known: {sorted(TEMPLATES)}")
    fields: dict[str, str] = {"target_language": _lang_name(tgt_lang, lang_names), "source_text": src_text}
    if template in CONTEXT_TEMPLATES:
        if not context:
            raise ValueError(f"template {template!r} needs a non-empty context")
        fields["context"] = context
    elif context:
        raise ValueError(f"template {template!r} takes no context; use a contextual template")
    if template in TERM_JOINERS:
        if not terms:
            raise ValueError(f"template {template!r} needs at least one (source_term, target_term) pair")
        fields["terms"] = "\n".join(TERM_JOINERS[template].format(src=s, tgt=t) for s, t in terms)
    elif terms:
        raise ValueError(f"template {template!r} takes no terms; use a terminology template")
    return _PLACEHOLDER.sub(lambda m: fields[m.group(1)], TEMPLATES[template])


def render_hymt_manual(user_prompt: str, *, mode: str = "generation_prompt", answer: str | None = None) -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    text = HY_BOS + HY_USER + user_prompt
    if answer is not None:
        text += HY_ASSISTANT + answer + HY_EOS
    return text + (HY_ASSISTANT if mode == "generation_prompt" else HY_NO8)


def build_prompt_text(tokenizer: Any, family: str, user_prompt: str, *, mode: str = "generation_prompt") -> str:
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    messages = [{"role": "user", "content": user_prompt}]
    if family == "hymt":
        if tokenizer is None:
            text = render_hymt_manual(user_prompt, mode=mode)
        else:
            text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=(mode == "generation_prompt"))
        if text.count(HY_BOS) != 1 or not text.startswith(HY_BOS):
            raise ValueError("HY prompt must start with exactly one BOS; does the source text contain a BOS string?")
        return text
    if mode != "generation_prompt":
        raise ValueError("the qwen family has only mode 'generation_prompt'")
    if tokenizer is None:
        raise ValueError("the qwen family needs its tokenizer (the chat template differs across Qwen versions)")
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def completion_suffix(tokenizer: Any, family: str) -> str:
    if family == "hymt":
        eos = getattr(tokenizer, "eos_token", None) if tokenizer is not None else HY_EOS
        if eos != HY_EOS:
            raise ValueError(f"unexpected HY eos token {eos!r} (expected {HY_EOS!r})")
        return eos
    if family == "qwen":
        return QWEN_EOT
    raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")


def build_training_text(tokenizer: Any, family: str, user_prompt: str, answer: str,
                        *, mode: str = "generation_prompt") -> tuple[str, str]:
    return (build_prompt_text(tokenizer, family, user_prompt, mode=mode),
            answer + completion_suffix(tokenizer, family))


def encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def encode_batch(tokenizer: Any, texts: Sequence[str]) -> list[list[int]]:
    if not texts:
        return []
    return [list(ids) for ids in tokenizer(list(texts), add_special_tokens=False)["input_ids"]]


def stop_token_ids(tokenizer: Any, family: str) -> list[int]:
    if tokenizer is None:
        return [HY_IDS["eos"]] if family == "hymt" else []
    ids = [tokenizer.convert_tokens_to_ids(completion_suffix(tokenizer, family))]
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    return sorted({int(i) for i in ids if i is not None and i >= 0})
