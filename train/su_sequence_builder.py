"""SU (Speech Understanding) sequence construction helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch

from model.constants import DEFAULT_SPEECH_TOKEN


def build_su_messages(
    history: Sequence[Dict[str, Any]],
    *,
    system_prompt: str | None = "",
) -> Tuple[List[Dict[str, str]], List[torch.Tensor], List[int]]:
    """Build chat-template messages plus aligned speech waveforms."""
    messages: List[Dict[str, str]] = []
    waveforms: List[torch.Tensor] = []
    sampling_rates: List[int] = []

    # Always start with a system turn so the chat template is faithful.
    if history and history[0].get("role") == "system":
        sys_text = history[0].get("text") or ""
        messages.append({"role": "system", "content": str(sys_text)})
        turn_iter = history[1:]
    else:
        if system_prompt is None:
            system_prompt = ""
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string or None")
        # Always include a blank system turn when none is provided.
        messages.append({"role": "system", "content": system_prompt})
        turn_iter = history

    for turn in turn_iter:
        role = turn.get("role")
        if role == "user":
            # User turns are speech-only, aligned to waveforms.
            messages.append({"role": "user", "content": DEFAULT_SPEECH_TOKEN})
            audio = turn.get("audio")
            if not audio:
                raise ValueError("user turn is missing audio")
            waveforms.append(torch.as_tensor(audio["array"]))
            sampling_rates.append(int(audio["sampling_rate"]))
        elif role == "assistant":
            # Assistant turns are plain text context.
            text = turn.get("text")
            if text is None:
                raise ValueError("assistant turn is missing text")
            messages.append({"role": "assistant", "content": str(text)})
        else:
            raise ValueError(f"unexpected role in history: {role!r}")

    return messages, waveforms, sampling_rates


def build_su_token_ids(
    tokenizer,
    messages: Sequence[Dict[str, str]],
    *,
    assistant_text: str | None = None,
    enable_thinking: bool | None = None,
) -> Tuple[List[int], List[int] | None]:
    """Tokenize prompt ids and optional template-faithful reply ids."""
    template_kwargs: Dict[str, Any] = {}
    if enable_thinking is not None:
        # Let callers control Qwen-style thinking mode at prompt/render time.
        template_kwargs["enable_thinking"] = bool(enable_thinking)

    # Prompt ids end with a trailing assistant role stub.
    encoded = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        **template_kwargs,
    )
    prompt_token_ids = encoded["input_ids"]

    if assistant_text is None:
        return prompt_token_ids, None

    # Full ids include assistant text and chat-template end tokens.
    full_messages = list(messages) + [{"role": "assistant", "content": str(assistant_text)}]
    full_encoded = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        **template_kwargs,
    )
    full_token_ids = full_encoded["input_ids"]

    # Ensure the prompt is a strict prefix of the full template ids.
    if full_token_ids[: len(prompt_token_ids)] != prompt_token_ids:
        raise ValueError("chat template mismatch: prompt token ids are not a prefix of full token ids")

    reply_token_ids = full_token_ids[len(prompt_token_ids) :]
    return prompt_token_ids, reply_token_ids


__all__ = ["build_su_messages", "build_su_token_ids"]
