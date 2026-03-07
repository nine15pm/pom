from __future__ import annotations

import pytest

from model.constants import DEFAULT_SEP_TOKEN, DEFAULT_SPEECH_TOKEN
from model.tokenizers import TokenIds, ensure_tokenizer_contract


class _FakeTokenizer:
    """Tiny tokenizer stub for contract tests without model downloads."""

    def __init__(self, mapping: dict[str, int], *, unk_token_id: int | None = None) -> None:
        # Store a direct token->id map to emulate convert_tokens_to_ids.
        self._mapping = dict(mapping)
        self.unk_token_id = unk_token_id

    def convert_tokens_to_ids(self, token: str) -> int | None:
        # Mirror HF behavior: return mapped id or unk id when configured.
        if token in self._mapping:
            return int(self._mapping[token])
        return self.unk_token_id


def test_ensure_tokenizer_contract_resolves_when_token_ids_not_provided():
    # Default path should resolve ids directly from the tokenizer instance.
    tokenizer = _FakeTokenizer({DEFAULT_SPEECH_TOKEN: 101, DEFAULT_SEP_TOKEN: 102}, unk_token_id=0)
    token_ids = ensure_tokenizer_contract(tokenizer)
    assert token_ids == TokenIds(speech_id=101, sep_id=102)


def test_ensure_tokenizer_contract_accepts_matching_explicit_token_ids():
    # Injected token ids are accepted when they match tokenizer-resolved ids.
    tokenizer = _FakeTokenizer({DEFAULT_SPEECH_TOKEN: 201, DEFAULT_SEP_TOKEN: 202}, unk_token_id=0)
    explicit = TokenIds(speech_id=201, sep_id=202)
    token_ids = ensure_tokenizer_contract(tokenizer, explicit)
    assert token_ids is explicit


def test_ensure_tokenizer_contract_rejects_mismatched_speech_id():
    # Mismatch on <speech> id must fail fast to prevent silent contract drift.
    tokenizer = _FakeTokenizer({DEFAULT_SPEECH_TOKEN: 301, DEFAULT_SEP_TOKEN: 302}, unk_token_id=0)
    with pytest.raises(ValueError, match="<speech>"):
        _ = ensure_tokenizer_contract(tokenizer, TokenIds(speech_id=999, sep_id=302))


def test_ensure_tokenizer_contract_rejects_mismatched_sep_id():
    # Mismatch on <sep> id must fail fast to prevent silent contract drift.
    tokenizer = _FakeTokenizer({DEFAULT_SPEECH_TOKEN: 401, DEFAULT_SEP_TOKEN: 402}, unk_token_id=0)
    with pytest.raises(ValueError, match="<sep>"):
        _ = ensure_tokenizer_contract(tokenizer, TokenIds(speech_id=401, sep_id=999))


def test_ensure_tokenizer_contract_rejects_unk_resolution():
    # Required tokens resolving to unk are invalid and should raise.
    tokenizer = _FakeTokenizer({DEFAULT_SPEECH_TOKEN: 0, DEFAULT_SEP_TOKEN: 2}, unk_token_id=0)
    with pytest.raises(ValueError, match="unk_token_id"):
        _ = ensure_tokenizer_contract(tokenizer)
