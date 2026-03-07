"""Tokenizer builders shared across pom modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from transformers import AutoTokenizer

from .constants import DEFAULT_SEP_TOKEN, DEFAULT_SPEECH_TOKEN


def build_pom_tokenizer(
    *,
    base_model_id: str = "Qwen/Qwen3-0.6B",
    cache_dir: Optional[str] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
) -> AutoTokenizer:
    """Build a tokenizer with pom-specific special tokens registered."""
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        cache_dir=cache_dir,
        **(tokenizer_kwargs or {}),
    )
    # Keep token registration centralized so all callers stay consistent.
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [DEFAULT_SPEECH_TOKEN, DEFAULT_SEP_TOKEN]}
    )
    return tokenizer


@dataclass(frozen=True)
class TokenIds:
    """Resolved special token ids used by pom runtime paths."""

    speech_id: int
    sep_id: int


def _resolve_required_token_id(tokenizer, token: str) -> int:
    """Resolve one required token id and fail fast if missing or unknown."""
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or not isinstance(token_id, int) or token_id < 0:
        raise ValueError(f"tokenizer is missing a valid token id for {token!r}")
    unk_id = tokenizer.unk_token_id
    if unk_id is not None and token_id == unk_id:
        raise ValueError(f"tokenizer resolved {token!r} to unk_token_id")
    return token_id


def resolve_token_ids(tokenizer) -> TokenIds:
    """Resolve and validate required special token ids once at startup."""
    return TokenIds(
        speech_id=_resolve_required_token_id(tokenizer, DEFAULT_SPEECH_TOKEN),
        sep_id=_resolve_required_token_id(tokenizer, DEFAULT_SEP_TOKEN),
    )


def ensure_tokenizer_contract(tokenizer, token_ids: Optional[TokenIds] = None) -> TokenIds:
    """Validate one tokenizer/token-id contract and return the effective token ids."""
    resolved = resolve_token_ids(tokenizer)
    if token_ids is None:
        return resolved

    # Keep startup strict: injected token ids must exactly match the tokenizer instance.
    if int(token_ids.speech_id) != int(resolved.speech_id):
        raise ValueError("tokenizer <speech> id does not match provided token_ids.speech_id")
    if int(token_ids.sep_id) != int(resolved.sep_id):
        raise ValueError("tokenizer <sep> id does not match provided token_ids.sep_id")
    return token_ids


__all__ = ["TokenIds", "build_pom_tokenizer", "ensure_tokenizer_contract", "resolve_token_ids"]
