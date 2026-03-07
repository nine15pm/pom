from __future__ import annotations

import pytest
import torch
import torchaudio

from transformers import AutoTokenizer, Qwen3ForCausalLM

from model.constants import DEFAULT_SEP_TOKEN, DEFAULT_SPEECH_TOKEN, IGNORE_INDEX
from model.pom_thinker import PomThinker, PomThinkerConfig, build_thinker
from model.tokenizers import TokenIds, build_pom_tokenizer, resolve_token_ids


DEVICE = torch.device("cuda")
DTYPE = torch.float32


def _simple_text_ids(tokenizer, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("Tokenizer returned empty ids for test text")
    return ids


def _speech_spec(whisper_tiny_id: str) -> dict:
    return {
        "encoder_id": whisper_tiny_id,
        "frame_stack": 5,
        "projector_hidden_dim": 2048,
    }


def test_builder_adds_speech_token_and_modules(base_model_id, whisper_tiny_id):
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_vocab = len(base_tokenizer)

    model, tokenizer, _ = build_thinker(
        base_model_id=base_model_id,
        speech=_speech_spec(whisper_tiny_id),
    )

    assert len(tokenizer) == base_vocab + 2
    speech_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
    sep_id = tokenizer.convert_tokens_to_ids(DEFAULT_SEP_TOKEN)
    assert speech_id == model.config.speech_token_id
    assert sep_id != tokenizer.unk_token_id
    assert model.get_speech_encoder() is not None
    assert model.get_speech_projector() is not None


def test_builder_without_speech_keeps_modules_none(base_model_id):
    model, tokenizer, _ = build_thinker(
        base_model_id=base_model_id,
        speech=None,
    )
    assert model.get_speech_encoder() is None
    assert model.get_speech_projector() is None
    assert tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN) == model.config.speech_token_id
    assert tokenizer.convert_tokens_to_ids(DEFAULT_SEP_TOKEN) != tokenizer.unk_token_id


def test_builder_bootstrap_uses_pom_thinker_config(base_model_id):
    model, _, _ = build_thinker(
        base_model_id=base_model_id,
        speech=None,
    )

    # Regression guard: bootstrap must end with PomThinker config type/model_type.
    assert isinstance(model.config, PomThinkerConfig)
    assert model.config.model_type == "pom_thinker_qwen3"


def test_fusion_with_provided_features_replaces_speech_tokens(base_model_id):
    model, tokenizer, _ = build_thinker(
        base_model_id=base_model_id,
        speech=None,
    )
    model.to(device=DEVICE)
    model.eval()
    embed_dim = model.get_input_embeddings().embedding_dim
    speech_id = model.config.speech_token_id

    text_a = _simple_text_ids(tokenizer, "hi")[0]
    text_b = _simple_text_ids(tokenizer, "bye")[0]

    input_ids = torch.tensor([[speech_id, text_a, speech_id, text_b]], device=DEVICE)
    attention_mask = torch.ones_like(input_ids, device=DEVICE)
    labels = input_ids.clone()

    speech_feat_1 = torch.randn(3, embed_dim, device=DEVICE)
    speech_mask_1 = torch.tensor([1, 1, 0], dtype=torch.bool, device=DEVICE)  # trims to 2 frames
    speech_feat_2 = torch.randn(1, embed_dim, device=DEVICE)

    out = model.prepare_inputs_labels_for_speech_and_text(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=labels,
        speech_features=[(speech_feat_1, speech_mask_1), speech_feat_2],
    )
    _, pos_ids, attn, _, embeds, out_labels = out

    # Expected order: speech1(2), text_a(1), speech2(1), text_b(1) -> len 5
    assert embeds.shape[1] == 5
    assert torch.all(attn[0, :5])
    assert torch.equal(pos_ids[0, :5], torch.arange(5, device=DEVICE))

    # Speech spans masked from loss; text tokens keep labels
    assert torch.all(out_labels[0, 0:2] == IGNORE_INDEX)
    assert out_labels[0, 2].item() == labels[0, 1].item()
    assert out_labels[0, 3] == IGNORE_INDEX
    assert out_labels[0, 4].item() == labels[0, 3].item()


def test_no_speech_path_matches_base_logits(base_model_id):
    base_model = Qwen3ForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=DTYPE,
    )
    base_vocab = base_model.get_input_embeddings().num_embeddings
    pom_model, tokenizer, _ = build_thinker(
        base_model_id=base_model_id,
        speech=None,
        torch_dtype=DTYPE,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.to(device=DEVICE)
    pom_model.to(device=DEVICE)
    base_model.eval()
    pom_model.eval()

    ids = tokenizer("just text", add_special_tokens=False, return_tensors="pt")["input_ids"].to(device=DEVICE)
    attention_mask = torch.ones_like(ids, device=DEVICE)

    with torch.no_grad():
        base_logits = base_model(input_ids=ids, attention_mask=attention_mask).logits
        pom_logits = pom_model(input_ids=ids, attention_mask=attention_mask).logits

    assert torch.allclose(
        base_logits[:, :, :base_vocab],
        pom_logits[:, :, :base_vocab],
        atol=1e-4,
        rtol=1e-4,
    )


def test_end_to_end_with_waveforms_runs_and_freezes_encoder(
    base_model_id,
    whisper_tiny_id,
    fixture_audio_paths,
    load_fixture_audio,
):
    model, tokenizer, _ = build_thinker(
        base_model_id=base_model_id,
        speech=_speech_spec(whisper_tiny_id),
    )
    model.to(device=DEVICE)
    model.eval()

    speech_id = model.config.speech_token_id
    text_token = _simple_text_ids(tokenizer, "hi")[0]
    input_ids = torch.tensor([[speech_id, text_token]], device=DEVICE)
    attention_mask = torch.ones_like(input_ids, device=DEVICE)
    labels = input_ids.clone()

    waveform, sampling_rate = load_fixture_audio(fixture_audio_paths[0])
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    waveform = waveform[: sampling_rate]  # keep ~1s for speed
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            speech_waveforms=[waveform],
            speech_sampling_rate=sampling_rate,
        )

    assert outputs.logits.shape[1] >= input_ids.shape[1]
    assert not torch.isnan(outputs.loss)

    encoder = model.get_speech_encoder()
    assert encoder is not None
    assert all(not p.requires_grad for p in encoder.parameters())


def test_hf_roundtrip_with_speech_waveform_forward(
    base_model_id,
    whisper_tiny_id,
    fixture_audio_paths,
    load_fixture_audio,
    tmp_path,
):
    """HF save/load should rebuild speech modules and run one real waveform forward."""
    model, tokenizer, _ = build_thinker(
        base_model_id=base_model_id,
        speech=_speech_spec(whisper_tiny_id),
    )
    save_dir = tmp_path / "pom_thinker_hf_roundtrip"
    model.save_pretrained(save_dir, safe_serialization=False)
    tokenizer.save_pretrained(save_dir)

    reloaded = PomThinker.from_pretrained(save_dir)
    reloaded.to(device=DEVICE)
    reloaded.eval()

    # Roundtrip must preserve speech module availability for inference/training paths.
    assert reloaded.get_speech_encoder() is not None
    assert reloaded.get_speech_projector() is not None

    speech_id = int(reloaded.config.speech_token_id)
    text_token = _simple_text_ids(tokenizer, "hi")[0]
    input_ids = torch.tensor([[speech_id, text_token]], device=DEVICE)
    attention_mask = torch.ones_like(input_ids, device=DEVICE)
    labels = input_ids.clone()

    # Use one real fixture waveform to validate the full speech feature path.
    waveform, sampling_rate = load_fixture_audio(fixture_audio_paths[0])
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    waveform = waveform[: sampling_rate]

    with torch.no_grad():
        outputs = reloaded(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            speech_waveforms=[waveform],
            speech_sampling_rate=sampling_rate,
        )

    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[1] >= input_ids.shape[1]
    assert outputs.logits.shape[2] == reloaded.get_output_embeddings().weight.shape[0]
    assert torch.isfinite(outputs.logits).all()


def test_variable_batch_speech_lengths_forward(
    base_model_id,
    whisper_tiny_id,
    fixture_audio_paths,
    load_fixture_audio,
):
    model, tokenizer, _ = build_thinker(
        base_model_id=base_model_id,
        speech=_speech_spec(whisper_tiny_id),
    )
    model.to(device=DEVICE)
    model.eval()

    speech_id = model.config.speech_token_id
    text_token = _simple_text_ids(tokenizer, "ok")[0]

    input_ids = torch.tensor(
        [
            [speech_id, text_token, speech_id],
            [text_token, speech_id, text_token],
        ],
        device=DEVICE,
    )
    attention_mask = torch.ones_like(input_ids, device=DEVICE)
    labels = input_ids.clone()

    waveform_a, sr_a = load_fixture_audio(fixture_audio_paths[0])
    if waveform_a.ndim == 2:
        waveform_a = waveform_a.mean(dim=0)
    waveform_a = waveform_a[: sr_a]

    waveform_b, sr_b = load_fixture_audio(fixture_audio_paths[1])
    if waveform_b.ndim == 2:
        waveform_b = waveform_b.mean(dim=0)
    if sr_b != sr_a:
        waveform_b = torchaudio.functional.resample(waveform_b, orig_freq=sr_b, new_freq=sr_a)
        sr_b = sr_a
    waveform_b = waveform_b[: sr_b]

    waveforms = [waveform_a, waveform_b, waveform_a[: sr_a // 2]]

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            speech_waveforms=waveforms,
            speech_sampling_rate=sr_a,
        )

    assert outputs.logits.shape[0] == input_ids.shape[0]
    assert not torch.isnan(outputs.loss)


def test_su_trainable_params(base_model_id, whisper_tiny_id):
    model, _, _ = build_thinker(
        base_model_id=base_model_id,
        speech=_speech_spec(whisper_tiny_id),
    )

    encoder = model.get_speech_encoder()
    projector = model.get_speech_projector()

    assert encoder is not None
    assert projector is not None
    assert all(not p.requires_grad for p in encoder.parameters())
    assert all(p.requires_grad for p in projector.parameters())
    assert all(p.requires_grad for p in model.model.parameters())


def test_builder_accepts_injected_tokenizer_contract(base_model_id):
    shared_tokenizer = build_pom_tokenizer(base_model_id=base_model_id)
    shared_token_ids = resolve_token_ids(shared_tokenizer)

    model, tokenizer, token_ids = build_thinker(
        base_model_id=base_model_id,
        speech=None,
        tokenizer=shared_tokenizer,
        token_ids=shared_token_ids,
    )

    assert tokenizer is shared_tokenizer
    assert token_ids == shared_token_ids
    assert model.config.speech_token_id == shared_token_ids.speech_id


def test_builder_rejects_mismatched_injected_token_ids(base_model_id):
    shared_tokenizer = build_pom_tokenizer(base_model_id=base_model_id)
    shared_token_ids = resolve_token_ids(shared_tokenizer)

    with pytest.raises(ValueError, match="<speech>"):
        _ = build_thinker(
            base_model_id=base_model_id,
            speech=None,
            tokenizer=shared_tokenizer,
            token_ids=TokenIds(
                speech_id=int(shared_token_ids.speech_id) + 1,
                sep_id=int(shared_token_ids.sep_id),
            ),
        )
