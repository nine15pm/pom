from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM

from model.pom_talker import PomTalker, build_talker


DEVICE = torch.device("cuda")


@pytest.fixture(scope="module")
def shared_stage2_talker(base_model_id):
    # Build one Stage-2 PomTalker instance for this module to keep tests fast.
    talker, _, _ = build_talker(base_model_id=base_model_id)
    return talker.to(DEVICE)


def test_pom_talker_builder_wires_speech_lm_and_gate(shared_stage2_talker):
    # Validate core Stage-2 ownership: PomTalker owns speech LM weights + gate fusion.
    talker = shared_stage2_talker
    assert isinstance(talker, PomTalker)

    hidden_dim = int(getattr(talker.config, "hidden_size"))
    embed_dim = int(talker.get_input_embeddings().embedding_dim)
    assert int(talker.config.llm_hidden_dim) == hidden_dim
    assert int(talker.gate_fusion.llm_hidden_dim) == hidden_dim
    assert int(talker.gate_fusion.speech_embed_dim) == embed_dim


def test_pom_talker_fuse_returns_expected_shape_and_gradients(shared_stage2_talker):
    # Validate fused outputs have correct shape and backprop reaches gate + Talker embeddings.
    talker = shared_stage2_talker
    talker.train()
    talker.zero_grad(set_to_none=True)

    hidden_dim = int(talker.config.llm_hidden_dim)
    embed_tokens = talker.get_input_embeddings()
    embed_weight = embed_tokens.weight
    embed_dim = int(embed_weight.shape[-1])

    hidden_row = torch.randn((4, hidden_dim), device=DEVICE, dtype=embed_weight.dtype)
    content_ids = torch.tensor([0, 1, 2, 3], device=DEVICE, dtype=torch.long)

    fused_rows = talker.fuse(hidden_rows=[hidden_row], content_ids=[content_ids])
    assert len(fused_rows) == 1
    assert fused_rows[0].shape == (4, embed_dim)
    assert fused_rows[0].device == embed_weight.device

    # Use a tiny scalar objective so we can assert gradients on key trainable paths.
    fused_rows[0].sum().backward()
    gate_has_grad = any(
        param.grad is not None and torch.any(param.grad != 0)
        for param in talker.gate_fusion.parameters()
    )
    assert gate_has_grad

    embed_grad = talker.get_input_embeddings().weight.grad
    assert embed_grad is not None
    assert torch.any(embed_grad[content_ids] != 0)


def test_pom_talker_fuse_rejects_hidden_content_token_count_mismatch(shared_stage2_talker):
    # Validate fail-fast behavior when hidden rows and content ids are not token-aligned.
    talker = shared_stage2_talker
    hidden_dim = int(talker.config.llm_hidden_dim)
    embed_weight = talker.get_input_embeddings().weight
    hidden_row = torch.zeros((3, hidden_dim), device=DEVICE, dtype=embed_weight.dtype)
    content_ids = torch.tensor([0, 1], device=DEVICE, dtype=torch.long)

    with pytest.raises(ValueError, match="hidden/content token count mismatch"):
        _ = talker.fuse(hidden_rows=[hidden_row], content_ids=[content_ids])


def test_pom_talker_forward_matches_direct_qwen_forward(shared_stage2_talker):
    # Validate forward(...) uses the model's own LM path with no wrapper indirection.
    talker = shared_stage2_talker
    talker.eval()

    input_ids = torch.tensor([[0, 1, 2]], device=DEVICE, dtype=torch.long)
    with torch.no_grad():
        outputs_talker = talker(input_ids=input_ids, use_cache=False, return_dict=True)
        outputs_direct = super(PomTalker, talker).forward(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        )

    assert torch.equal(outputs_talker.logits, outputs_direct.logits)


def test_pom_talker_packaging_roundtrip_preserves_weights_and_metadata(base_model_id, tmp_path):
    # Validate save/load roundtrip so Talker can be a clean Stage-2 package artifact.
    talker, _, _ = build_talker(base_model_id=base_model_id)
    talker.eval()
    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        logits_before = talker(input_ids=input_ids, use_cache=False, return_dict=True).logits

    save_dir = tmp_path / "pom_talker_roundtrip"
    talker.save_pretrained(save_dir, safe_serialization=False)
    reloaded = PomTalker.from_pretrained(save_dir)
    reloaded.eval()
    with torch.no_grad():
        logits_after = reloaded(input_ids=input_ids, use_cache=False, return_dict=True).logits

    assert reloaded.speech_vocab_size == talker.speech_vocab_size
    assert reloaded.text_vocab_size == talker.text_vocab_size
    assert reloaded.speech_token_offset == talker.speech_token_offset
    assert reloaded.total_vocab_size == talker.total_vocab_size
    assert int(reloaded.config.llm_hidden_dim) == int(talker.config.llm_hidden_dim)
    assert torch.allclose(logits_after, logits_before)

    auto_loaded = AutoModelForCausalLM.from_pretrained(save_dir)
    assert isinstance(auto_loaded, PomTalker)
