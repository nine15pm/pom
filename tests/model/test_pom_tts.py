from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM

from model.constants import DEFAULT_SEP_TOKEN, DEFAULT_SPEECH_TOKEN, SPEECH_VOCAB_SIZE
from model.pom_tts import PomTTS, build_pom_tts
from model.tokenizers import TokenIds, build_pom_tokenizer, resolve_token_ids


def test_pom_tts_builder_aligns_vocab_sizes(base_model_id):
    tts, tokenizer, token_ids = build_pom_tts(base_model_id=base_model_id)
    assert tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN) != tokenizer.unk_token_id
    assert tokenizer.convert_tokens_to_ids(DEFAULT_SEP_TOKEN) != tokenizer.unk_token_id
    assert token_ids.sep_id == tokenizer.convert_tokens_to_ids(DEFAULT_SEP_TOKEN)
    assert tts.speech_token_offset == len(tokenizer)
    assert tts.total_vocab_size == len(tokenizer) + SPEECH_VOCAB_SIZE


def test_pom_tts_builder_accepts_injected_tokenizer_contract(base_model_id):
    shared_tokenizer = build_pom_tokenizer(base_model_id=base_model_id)
    shared_token_ids = resolve_token_ids(shared_tokenizer)

    tts, tokenizer, token_ids = build_pom_tts(
        base_model_id=base_model_id,
        tokenizer=shared_tokenizer,
        token_ids=shared_token_ids,
    )

    assert tokenizer is shared_tokenizer
    assert token_ids == shared_token_ids
    assert tts.speech_token_offset == len(shared_tokenizer)


def test_pom_tts_builder_rejects_mismatched_injected_token_ids(base_model_id):
    shared_tokenizer = build_pom_tokenizer(base_model_id=base_model_id)
    shared_token_ids = resolve_token_ids(shared_tokenizer)

    with pytest.raises(ValueError, match="<sep>"):
        _ = build_pom_tts(
            base_model_id=base_model_id,
            tokenizer=shared_tokenizer,
            token_ids=TokenIds(
                speech_id=int(shared_token_ids.speech_id),
                sep_id=int(shared_token_ids.sep_id) + 1,
            ),
        )


def test_pom_tts_map_unit_ids_enforces_stage2_target_contract(base_model_id):
    # Build one real model/tokenizer pair so mapping behavior matches Stage-2 training.
    tts, _, _ = build_pom_tts(base_model_id=base_model_id)
    if tts.speech_token_offset is None:
        raise ValueError("speech_token_offset must be set for Stage-2 unit mapping")

    # Valid unit ids should map by a fixed offset into LM vocabulary ids.
    units = torch.tensor([0, 7, SPEECH_VOCAB_SIZE - 1], device="cuda", dtype=torch.long)
    mapped = tts.map_unit_ids(units)
    assert torch.equal(mapped, units + int(tts.speech_token_offset))

    # Stage-2 correctness depends on rejecting invalid unit id ranges early.
    with pytest.raises(ValueError, match="must be in \\[0, speech_vocab_size-1\\]"):
        _ = tts.map_unit_ids(torch.tensor([-1], device="cuda", dtype=torch.long))
    with pytest.raises(ValueError, match="must be in \\[0, speech_vocab_size-1\\]"):
        _ = tts.map_unit_ids(torch.tensor([SPEECH_VOCAB_SIZE], device="cuda", dtype=torch.long))


def test_pom_tts_builder_keeps_stage2_vocab_metadata_consistent(base_model_id):
    # Build one real model so we can validate metadata used by Stage-2 batching logic.
    tts, tokenizer, _ = build_pom_tts(base_model_id=base_model_id)

    # Stage-2 assumes speech ids are an offset extension after text tokenizer ids.
    assert tts.text_vocab_size == len(tokenizer)
    assert tts.speech_vocab_size == SPEECH_VOCAB_SIZE
    assert tts.speech_token_offset == len(tokenizer)

    # Stage-2 token lookup correctness depends on config + embedding size agreement.
    assert tts.total_vocab_size == int(tts.config.vocab_size)
    assert int(tts.get_input_embeddings().num_embeddings) == tts.total_vocab_size


def test_pom_tts_packaging_roundtrip_preserves_weights_and_metadata(base_model_id, tmp_path):
    # Build one real PomTTS and run one deterministic forward for roundtrip comparison.
    tts, _, _ = build_pom_tts(base_model_id=base_model_id)
    tts.eval()
    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        logits_before = tts(input_ids=input_ids, use_cache=False, return_dict=True).logits

    # Save as a HF package and reload through the model class API.
    save_dir = tmp_path / "pom_tts_roundtrip"
    tts.save_pretrained(save_dir, safe_serialization=False)
    reloaded = PomTTS.from_pretrained(save_dir)
    reloaded.eval()
    with torch.no_grad():
        logits_after = reloaded(input_ids=input_ids, use_cache=False, return_dict=True).logits

    # Packaging correctness requires exact metadata persistence after reload.
    assert reloaded.speech_vocab_size == tts.speech_vocab_size
    assert reloaded.text_vocab_size == tts.text_vocab_size
    assert reloaded.speech_token_offset == tts.speech_token_offset
    assert reloaded.total_vocab_size == tts.total_vocab_size

    # Roundtrip correctness requires equivalent model outputs on the same input.
    assert torch.allclose(logits_after, logits_before)

    # AutoModel loading should also resolve back to PomTTS via registered config.
    auto_loaded = AutoModelForCausalLM.from_pretrained(save_dir)
    assert isinstance(auto_loaded, PomTTS)


def test_pom_tts_safetensors_reload_preserves_stage2_mapping_contract(base_model_id, tmp_path):
    # Build and save a clean safetensors package to mirror artifact handoff usage.
    tts, _, _ = build_pom_tts(base_model_id=base_model_id)
    save_dir = tmp_path / "pom_tts_safetensors"
    tts.save_pretrained(save_dir, safe_serialization=True)
    assert (save_dir / "model.safetensors").exists()

    # Reload via the class API and verify Stage-2 mapping assumptions still hold.
    reloaded = PomTTS.from_pretrained(save_dir)
    if reloaded.speech_token_offset is None:
        raise ValueError("speech_token_offset must exist after packaged reload")
    units = torch.tensor([0, 1, SPEECH_VOCAB_SIZE - 1], dtype=torch.long)
    mapped = reloaded.map_unit_ids(units)

    # Stage-2 target building depends on fixed-offset mapping and vocab-size agreement.
    assert torch.equal(mapped, units + int(reloaded.speech_token_offset))
    assert int(reloaded.get_input_embeddings().num_embeddings) == reloaded.total_vocab_size
