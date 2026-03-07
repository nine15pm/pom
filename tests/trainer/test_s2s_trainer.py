import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
import yaml

from model.pom_talker import build_talker
from model.pom_thinker import build_thinker
from train.s2s_trainer import _run_frozen_thinker


def _repo_root() -> Path:
    """Return the repository root for subprocess trainer runs."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repo root (pyproject.toml not found)")


def _write_json(path: Path, payload) -> None:
    """Write one JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write JSONL records to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_config(path: Path, data: dict) -> None:
    """Write one YAML trainer config."""
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _copy_fixture_audio(src: Path, dst: Path) -> None:
    """Copy one WAV fixture into the temporary audio tree."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def _run_trainer(
    module_name: str,
    config_path: Path,
    *,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Run one trainer module with a config path and optional CLI args."""
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    cmd = [sys.executable, "-m", module_name, "--config", str(config_path)]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(
        cmd,
        cwd=_repo_root(),
        env=env,
        capture_output=True,
        text=True,
    )


def _load_model_state(path: Path) -> dict:
    """Load one torch checkpoint state dict from disk."""
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise AssertionError(f"expected state_dict mapping in {path}")
    return state


def _normalize_state_keys(state: dict) -> dict:
    """Normalize checkpoint keys by stripping one optional DDP prefix."""
    normalized = {}
    for key, value in state.items():
        if key.startswith("module."):
            normalized[key[len("module."):]] = value
        else:
            normalized[key] = value
    return normalized


def _make_su_config(
    *,
    model_id: str,
    whisper_id: str,
    json_path: Path,
    audio_root: Path,
    output_dir: Path,
) -> dict:
    """Build a tiny valid SU config for checkpoint bootstrap tests."""
    return {
        "model": {
            "id": model_id,
            "speech_encoder_id": whisper_id,
            "frame_stack": 5,
            "adapter_hidden_dim": 2048,
        },
        "data": {
            "json_path": str(json_path),
            "audio_root": str(audio_root),
        },
        "training": {
            "batch_size": 1,
            "grad_accum": 1,
            "precision": "no",
            "num_workers": 0,
            "seed": 123,
            "lr": 5.0e-5,
            "steps": 1,
            "profile": False,
            "output_dir": str(output_dir),
        },
        "wandb": {"enabled": False},
    }


def _make_tts_config(
    *,
    model_id: str,
    json_path: Path,
    output_dir: Path,
) -> dict:
    """Build a tiny valid TTS config for checkpoint bootstrap tests."""
    return {
        "model": {
            "id": model_id,
        },
        "data": {
            "json_path": str(json_path),
        },
        "training": {
            "batch_size": 1,
            "grad_accum": 1,
            "precision": "no",
            "num_workers": 0,
            "seed": 123,
            "lr": 5.0e-4,
            "steps": 1,
            "read_length": 3,
            "write_length": 10,
            "profile": False,
            "output_dir": str(output_dir),
        },
        "wandb": {"enabled": False},
    }


def _make_s2s_config(
    *,
    model_id: str,
    whisper_id: str,
    json_path: Path,
    audio_root: Path,
    output_dir: Path,
    thinker_checkpoint: Path,
    tts_checkpoint: Path,
) -> dict:
    """Build a tiny valid S2S config with fresh Stage-2 bootstrap checkpoints."""
    return {
        "model": {
            "id": model_id,
            "speech_encoder_id": whisper_id,
            "frame_stack": 5,
            "adapter_hidden_dim": 2048,
        },
        "data": {
            "json_path": str(json_path),
            "audio_root": str(audio_root),
        },
        "training": {
            "batch_size": 1,
            "grad_accum": 1,
            "precision": "no",
            "num_workers": 0,
            "seed": 123,
            "lr": 1.0e-3,
            "steps": 1,
            "read_length": 3,
            "write_length": 10,
            "thinker_checkpoint": str(thinker_checkpoint),
            "tts_checkpoint": str(tts_checkpoint),
            "profile": False,
            "output_dir": str(output_dir),
        },
        "wandb": {"enabled": False},
    }


def _write_minimal_s2s_shard(shard_dir: Path, *, sample_count: int = 1) -> None:
    """Write one minimal valid Stage-2 shard directory with schema manifest."""
    _write_json(shard_dir / "manifest.json", {"schema": "s2s_pairs_v1"})
    _write_jsonl(
        shard_dir / "shard-00000.jsonl",
        [
            {
                "id": f"s2s-{idx}",
                "source_id": "conv-0",
                "turn_index": idx,
                "history": [{"role": "user", "text": None, "audio": {"path": "audio/a.wav"}}],
                "assistant_text": "ok",
                "unit_ids": list(range(10)),
            }
            for idx in range(sample_count)
        ],
    )


def _try_strict_load_thinker(state: dict, *, model_id: str, whisper_id: str) -> bool:
    """Return whether a state dict strictly loads into a public Thinker builder model."""
    thinker, _, _ = build_thinker(
        base_model_id=model_id,
        speech={
            "encoder_id": whisper_id,
            "frame_stack": 5,
            "projector_hidden_dim": 2048,
        },
    )
    try:
        thinker.load_state_dict(state, strict=True)
    except RuntimeError:
        return False
    return True


def _try_strict_load_talker(state: dict, *, model_id: str) -> bool:
    """Return whether a state dict strictly loads into a public Talker builder model."""
    talker, _, _ = build_talker(base_model_id=model_id)
    try:
        talker.load_state_dict(state, strict=True)
    except RuntimeError:
        return False
    return True


def _split_s2s_checkpoint_states(
    checkpoint_dir: Path,
    *,
    model_id: str,
    whisper_id: str,
) -> tuple[dict, dict]:
    """Classify S2S model files by strict-load behavior into Thinker and Talker states."""
    model_paths = sorted(checkpoint_dir.glob("pytorch_model*.bin"))
    if not model_paths:
        raise AssertionError(f"no model checkpoint files found under {checkpoint_dir}")

    thinker_state = None
    talker_state = None
    for model_path in model_paths:
        state = _normalize_state_keys(_load_model_state(model_path))
        can_load_thinker = _try_strict_load_thinker(
            state,
            model_id=model_id,
            whisper_id=whisper_id,
        )
        can_load_talker = _try_strict_load_talker(state, model_id=model_id)
        if can_load_thinker and not can_load_talker:
            thinker_state = state
            continue
        if can_load_talker and not can_load_thinker:
            talker_state = state
            continue
        raise AssertionError(
            f"unable to classify checkpoint file {model_path.name}: thinker={can_load_thinker} talker={can_load_talker}"
        )

    if thinker_state is None:
        raise AssertionError(f"could not locate thinker state in {checkpoint_dir}")
    if talker_state is None:
        raise AssertionError(f"could not locate talker state in {checkpoint_dir}")
    return thinker_state, talker_state


def _assert_no_step_checkpoints(output_dir: Path) -> None:
    """Assert a run produced no step checkpoints (fail-fast behavior)."""
    checkpoint_root = output_dir / "checkpoints"
    if not checkpoint_root.exists():
        return
    assert not list(checkpoint_root.glob("step-*"))


def test_run_frozen_thinker_uses_hf_base_model_contract():
    """Ensure Stage-2 frozen forward uses HF base_model (not legacy get_model)."""

    class _DummyBackbone(torch.nn.Module):
        """Return deterministic hidden states so the trainer path is easy to verify."""

        def forward(
            self,
            *,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            inputs_embeds=None,
            use_cache=False,
            return_dict=True,
        ):
            del input_ids, attention_mask, position_ids, use_cache, return_dict
            return SimpleNamespace(last_hidden_state=inputs_embeds + 1.0)

    class _DummyThinker(torch.nn.Module):
        """Expose the HF contract and fail loudly if legacy access is used."""

        def __init__(self) -> None:
            super().__init__()
            self.base_model = _DummyBackbone()

        def get_model(self):
            # This test guards against regressions back to old LLaVA-style wrappers.
            raise AssertionError("legacy get_model() should not be used in Stage-2")

    thinker = _DummyThinker()
    inputs_embeds = torch.randn(2, 3, 4)
    position_ids = torch.arange(3, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    attention_mask = torch.ones((2, 3), dtype=torch.bool)

    last_hidden = _run_frozen_thinker(
        thinker=thinker,
        position_ids=position_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
    )

    assert torch.allclose(last_hidden, inputs_embeds + 1.0)


@pytest.fixture(scope="module")
def s2s_bootstrap_artifacts(
    tmp_path_factory,
    fixture_audio_paths,
    local_base_model_id,
    local_whisper_tiny_id,
):
    """Create shared SU/TTS bootstrap checkpoints plus one minimal S2S shard."""
    root = tmp_path_factory.mktemp("s2s_trainer_bootstrap")
    audio_root = root / "audio_root"
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")

    su_json = root / "su_train.json"
    _write_json(
        su_json,
        [
            {
                "id": "su-conv-0",
                "conversations": [
                    {"from": "user", "value": "u0", "audio": "audio/a.wav"},
                    {"from": "assistant", "value": "ok"},
                ],
            }
        ],
    )
    su_output = root / "out_su"
    su_cfg = root / "su.yaml"
    _write_config(
        su_cfg,
        _make_su_config(
            model_id=local_base_model_id,
            whisper_id=local_whisper_tiny_id,
            json_path=su_json,
            audio_root=audio_root,
            output_dir=su_output,
        ),
    )
    su_result = _run_trainer("train.su_trainer", su_cfg)
    su_log = (su_result.stdout or "") + (su_result.stderr or "")
    assert su_result.returncode == 0, su_log

    tts_jsonl = root / "tts_train.jsonl"
    _write_jsonl(
        tts_jsonl,
        [
            {
                "id": "tts-0",
                "source_id": "conv-0",
                "turn_index": 1,
                "assistant_text": "ok",
                "unit_ids": list(range(10)),
            }
        ],
    )
    tts_output = root / "out_tts"
    tts_cfg = root / "tts.yaml"
    _write_config(
        tts_cfg,
        _make_tts_config(
            model_id=local_base_model_id,
            json_path=tts_jsonl,
            output_dir=tts_output,
        ),
    )
    tts_result = _run_trainer("train.tts_trainer", tts_cfg)
    tts_log = (tts_result.stdout or "") + (tts_result.stderr or "")
    assert tts_result.returncode == 0, tts_log

    s2s_shards = root / "s2s_shards"
    _write_minimal_s2s_shard(s2s_shards, sample_count=1)

    return {
        "root": root,
        "model_id": local_base_model_id,
        "whisper_id": local_whisper_tiny_id,
        "audio_root": audio_root,
        "s2s_shards": s2s_shards,
        "thinker_ckpt": su_output / "checkpoints" / "step-00000001",
        "tts_ckpt": tts_output / "checkpoints" / "step-00000001",
    }


def test_s2s_trainer_fresh_bootstrap_smoke(tmp_path, s2s_bootstrap_artifacts):
    """Run Stage-2 from shared bootstrap checkpoints and verify checkpoint output."""
    output_dir = tmp_path / "out_s2s"
    s2s_cfg = tmp_path / "s2s.yaml"
    _write_config(
        s2s_cfg,
        _make_s2s_config(
            model_id=s2s_bootstrap_artifacts["model_id"],
            whisper_id=s2s_bootstrap_artifacts["whisper_id"],
            json_path=s2s_bootstrap_artifacts["s2s_shards"],
            audio_root=s2s_bootstrap_artifacts["audio_root"],
            output_dir=output_dir,
            thinker_checkpoint=s2s_bootstrap_artifacts["thinker_ckpt"],
            tts_checkpoint=s2s_bootstrap_artifacts["tts_ckpt"],
        ),
    )

    result = _run_trainer("train.s2s_trainer", s2s_cfg)
    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, output

    final_checkpoint = output_dir / "checkpoints" / "step-00000001"
    assert final_checkpoint.exists()
    state_path = final_checkpoint / "trainer_state.json"
    assert state_path.exists()
    trainer_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert trainer_state["global_step"] == 1


def test_s2s_trainer_keeps_thinker_frozen_and_updates_talker(tmp_path, s2s_bootstrap_artifacts):
    """Verify Stage-2 leaves Thinker unchanged while updating Talker-owned LM weights."""
    output_dir = tmp_path / "out_s2s"
    s2s_cfg_data = _make_s2s_config(
        model_id=s2s_bootstrap_artifacts["model_id"],
        whisper_id=s2s_bootstrap_artifacts["whisper_id"],
        json_path=s2s_bootstrap_artifacts["s2s_shards"],
        audio_root=s2s_bootstrap_artifacts["audio_root"],
        output_dir=output_dir,
        thinker_checkpoint=s2s_bootstrap_artifacts["thinker_ckpt"],
        tts_checkpoint=s2s_bootstrap_artifacts["tts_ckpt"],
    )
    # Use two steps so we observe at least one optimizer update.
    s2s_cfg_data["training"]["steps"] = 2
    s2s_cfg = tmp_path / "s2s.yaml"
    _write_config(s2s_cfg, s2s_cfg_data)

    s2s_result = _run_trainer("train.s2s_trainer", s2s_cfg)
    s2s_log = (s2s_result.stdout or "") + (s2s_result.stderr or "")
    assert s2s_result.returncode == 0, s2s_log

    su_state_before = _normalize_state_keys(_load_model_state(s2s_bootstrap_artifacts["thinker_ckpt"] / "pytorch_model.bin"))
    tts_state_before = _normalize_state_keys(_load_model_state(s2s_bootstrap_artifacts["tts_ckpt"] / "pytorch_model.bin"))
    thinker_state_after, talker_state_after = _split_s2s_checkpoint_states(
        output_dir / "checkpoints" / "step-00000002",
        model_id=s2s_bootstrap_artifacts["model_id"],
        whisper_id=s2s_bootstrap_artifacts["whisper_id"],
    )

    # Thinker checkpoint must remain exactly equal to the frozen SU bootstrap checkpoint.
    assert set(thinker_state_after.keys()) == set(su_state_before.keys())
    for key, before in su_state_before.items():
        assert torch.equal(before, thinker_state_after[key]), f"thinker changed at {key}"

    # Compare Talker LM weights via public modules rather than key-name assumptions.
    talker_before, _, _ = build_talker(base_model_id=s2s_bootstrap_artifacts["model_id"])
    talker_before.load_state_dict(
        {**tts_state_before, **{k: v for k, v in talker_before.state_dict().items() if k.startswith("gate_fusion.")}},
        strict=True,
    )
    talker_after, _, _ = build_talker(base_model_id=s2s_bootstrap_artifacts["model_id"])
    talker_after.load_state_dict(talker_state_after, strict=True)

    any_lm_weight_changed = False
    for key, before in talker_before.state_dict().items():
        if key.startswith("gate_fusion."):
            continue
        after = talker_after.state_dict()[key]
        if not torch.equal(before, after):
            any_lm_weight_changed = True
            break
    assert any_lm_weight_changed, "no talker LM weights changed during Stage-2 training"


def test_s2s_trainer_rejects_wrong_tts_bootstrap_checkpoint(tmp_path, s2s_bootstrap_artifacts):
    """Fail fast when Stage-2 tts_checkpoint points to a non-TTS checkpoint."""
    output_dir = tmp_path / "out_s2s"
    s2s_cfg = tmp_path / "s2s.yaml"
    _write_config(
        s2s_cfg,
        _make_s2s_config(
            model_id=s2s_bootstrap_artifacts["model_id"],
            whisper_id=s2s_bootstrap_artifacts["whisper_id"],
            json_path=s2s_bootstrap_artifacts["s2s_shards"],
            audio_root=s2s_bootstrap_artifacts["audio_root"],
            output_dir=output_dir,
            thinker_checkpoint=s2s_bootstrap_artifacts["thinker_ckpt"],
            # Intentional mismatch: this path is a Thinker checkpoint, not a TTS checkpoint.
            tts_checkpoint=s2s_bootstrap_artifacts["thinker_ckpt"],
        ),
    )

    result = _run_trainer("train.s2s_trainer", s2s_cfg)
    # Assert the failure is the expected strict-load mismatch, not an unrelated crash.
    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0
    assert "tts checkpoint key mismatch" in output
    _assert_no_step_checkpoints(output_dir)


def test_s2s_trainer_normalizes_tts_shape_mismatch_error(tmp_path, s2s_bootstrap_artifacts):
    """Fail with normalized tts checkpoint mismatch when one tensor shape is incompatible."""
    output_dir = tmp_path / "out_s2s_tts_shape_mismatch"
    s2s_cfg = tmp_path / "s2s_tts_shape_mismatch.yaml"

    # Copy a real TTS checkpoint and corrupt one tensor shape to force RuntimeError on load.
    bad_tts_ckpt = tmp_path / "bad_tts_ckpt"
    bad_tts_ckpt.mkdir(parents=True, exist_ok=False)
    shutil.copy(
        s2s_bootstrap_artifacts["tts_ckpt"] / "pytorch_model.bin",
        bad_tts_ckpt / "pytorch_model.bin",
    )
    state = _load_model_state(bad_tts_ckpt / "pytorch_model.bin")
    # Pick a deterministic small non-scalar tensor so corruption stays lightweight and stable.
    tensor_candidates = sorted(
        (
            (key, value)
            for key, value in state.items()
            if isinstance(value, torch.Tensor) and value.ndim > 0
        ),
        key=lambda item: (int(item[1].numel()), item[0]),
    )
    if not tensor_candidates:
        raise AssertionError("checkpoint must contain at least one non-scalar tensor")
    probe_key, probe = tensor_candidates[0]
    bad_shape = list(probe.shape)
    bad_shape[0] = bad_shape[0] + 1
    state[probe_key] = torch.zeros(
        bad_shape,
        dtype=probe.dtype,
    )
    torch.save(state, bad_tts_ckpt / "pytorch_model.bin")
    (bad_tts_ckpt / "trainer_state.json").write_text(
        json.dumps({"global_step": 0}, indent=2),
        encoding="utf-8",
    )

    _write_config(
        s2s_cfg,
        _make_s2s_config(
            model_id=s2s_bootstrap_artifacts["model_id"],
            whisper_id=s2s_bootstrap_artifacts["whisper_id"],
            json_path=s2s_bootstrap_artifacts["s2s_shards"],
            audio_root=s2s_bootstrap_artifacts["audio_root"],
            output_dir=output_dir,
            thinker_checkpoint=s2s_bootstrap_artifacts["thinker_ckpt"],
            tts_checkpoint=bad_tts_ckpt,
        ),
    )

    result = _run_trainer("train.s2s_trainer", s2s_cfg)
    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0
    assert f"tts checkpoint key mismatch from {bad_tts_ckpt}" in output
    _assert_no_step_checkpoints(output_dir)


def test_s2s_trainer_requires_explicit_tts_checkpoint(tmp_path, s2s_bootstrap_artifacts):
    """Fail fast when training.tts_checkpoint is missing in Stage-2 config."""
    output_dir = tmp_path / "out_s2s_missing_tts_checkpoint"
    s2s_cfg = tmp_path / "s2s_missing_tts_checkpoint.yaml"
    cfg = _make_s2s_config(
        model_id=s2s_bootstrap_artifacts["model_id"],
        whisper_id=s2s_bootstrap_artifacts["whisper_id"],
        json_path=s2s_bootstrap_artifacts["s2s_shards"],
        audio_root=s2s_bootstrap_artifacts["audio_root"],
        output_dir=output_dir,
        thinker_checkpoint=s2s_bootstrap_artifacts["thinker_ckpt"],
        tts_checkpoint=s2s_bootstrap_artifacts["tts_ckpt"],
    )
    # Remove canonical TTS checkpoint to enforce the strict config contract.
    cfg["training"].pop("tts_checkpoint")
    # Legacy alias should not be accepted by the Stage-2 loader contract.
    cfg["training"]["talker_checkpoint"] = str(s2s_bootstrap_artifacts["tts_ckpt"])
    _write_config(s2s_cfg, cfg)

    result = _run_trainer("train.s2s_trainer", s2s_cfg)
    # Check the exact fail-fast contract so this test is robust against unrelated failures.
    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0
    assert "training.tts_checkpoint must be set for Stage-2 runs" in output
    _assert_no_step_checkpoints(output_dir)


@pytest.mark.parametrize(
    ("missing_source", "expected_error"),
    [
        ("thinker", "pytorch_model.bin not found in checkpoint"),
        ("tts", "pytorch_model.bin not found in checkpoint"),
    ],
)
def test_s2s_trainer_bootstrap_requires_pytorch_model_bin_for_both_sources(
    tmp_path,
    s2s_bootstrap_artifacts,
    missing_source: str,
    expected_error: str,
):
    """Fail fast unless both bootstrap checkpoint dirs contain pytorch_model.bin."""
    output_dir = tmp_path / f"out_s2s_missing_bin_{missing_source}"
    s2s_cfg = tmp_path / f"s2s_missing_bin_{missing_source}.yaml"

    # Create one empty checkpoint-like directory to simulate a missing .bin payload.
    missing_ckpt_dir = tmp_path / f"missing_bin_{missing_source}"
    missing_ckpt_dir.mkdir(parents=True, exist_ok=False)
    # Add trainer_state to keep this directory shape realistic for training artifacts.
    (missing_ckpt_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": 0}, indent=2),
        encoding="utf-8",
    )

    thinker_checkpoint = s2s_bootstrap_artifacts["thinker_ckpt"]
    tts_checkpoint = s2s_bootstrap_artifacts["tts_ckpt"]
    if missing_source == "thinker":
        thinker_checkpoint = missing_ckpt_dir
    else:
        tts_checkpoint = missing_ckpt_dir

    _write_config(
        s2s_cfg,
        _make_s2s_config(
            model_id=s2s_bootstrap_artifacts["model_id"],
            whisper_id=s2s_bootstrap_artifacts["whisper_id"],
            json_path=s2s_bootstrap_artifacts["s2s_shards"],
            audio_root=s2s_bootstrap_artifacts["audio_root"],
            output_dir=output_dir,
            thinker_checkpoint=thinker_checkpoint,
            tts_checkpoint=tts_checkpoint,
        ),
    )

    result = _run_trainer("train.s2s_trainer", s2s_cfg)
    # Assert the shared strict-loader contract: both bootstrap paths require .bin weights.
    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0
    assert expected_error in output
    _assert_no_step_checkpoints(output_dir)


def test_s2s_trainer_resume_rejects_compat_mismatch(tmp_path, s2s_bootstrap_artifacts):
    """Fail fast on resume when compat-defining Stage-2 config fields change."""
    base_output = tmp_path / "out_s2s_base"
    base_cfg = tmp_path / "s2s_base.yaml"
    _write_config(
        base_cfg,
        _make_s2s_config(
            model_id=s2s_bootstrap_artifacts["model_id"],
            whisper_id=s2s_bootstrap_artifacts["whisper_id"],
            json_path=s2s_bootstrap_artifacts["s2s_shards"],
            audio_root=s2s_bootstrap_artifacts["audio_root"],
            output_dir=base_output,
            thinker_checkpoint=s2s_bootstrap_artifacts["thinker_ckpt"],
            tts_checkpoint=s2s_bootstrap_artifacts["tts_ckpt"],
        ),
    )
    base_result = _run_trainer("train.s2s_trainer", base_cfg)
    base_log = (base_result.stdout or "") + (base_result.stderr or "")
    assert base_result.returncode == 0, base_log

    mismatch_output = tmp_path / "out_s2s_resume_mismatch"
    mismatch_cfg_data = _make_s2s_config(
        model_id=s2s_bootstrap_artifacts["model_id"],
        whisper_id=s2s_bootstrap_artifacts["whisper_id"],
        json_path=s2s_bootstrap_artifacts["s2s_shards"],
        audio_root=s2s_bootstrap_artifacts["audio_root"],
        output_dir=mismatch_output,
        thinker_checkpoint=s2s_bootstrap_artifacts["thinker_ckpt"],
        tts_checkpoint=s2s_bootstrap_artifacts["tts_ckpt"],
    )
    # Change one compat-defining field to trigger resume mismatch.
    mismatch_cfg_data["training"]["read_length"] = 2
    mismatch_cfg = tmp_path / "s2s_resume_mismatch.yaml"
    _write_config(mismatch_cfg, mismatch_cfg_data)

    mismatch_result = _run_trainer(
        "train.s2s_trainer",
        mismatch_cfg,
        extra_args=["--resume", str(base_output / "checkpoints" / "step-00000001")],
    )
    assert mismatch_result.returncode != 0
    _assert_no_step_checkpoints(mismatch_output)
