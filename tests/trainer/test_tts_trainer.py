import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import yaml


def _repo_root() -> Path:
    """Return the project root for subprocess execution."""
    # Walk up until we find the repo marker file.
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repo root (pyproject.toml not found)")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write JSON records to one JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_config(path: Path, data: dict) -> None:
    """Serialize a YAML config for the trainer."""
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _read_json(path: Path) -> dict:
    """Load one JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _run_tts_trainer(
    config_path: Path,
    *,
    extra_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run train.tts_trainer and return the subprocess result."""
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, "-m", "train.tts_trainer", "--config", str(config_path)]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(
        cmd,
        cwd=_repo_root(),
        env=env,
        capture_output=True,
        text=True,
    )


def _build_trainer_cfg(
    *,
    model_id: str,
    json_path: Path,
    output_dir: Path,
    steps: int,
    batch_size: int = 1,
    grad_accum: int = 1,
    precision: str = "no",
    num_workers: int = 0,
    seed: int = 123,
    lr: float = 5.0e-4,
    read_length: int = 64,
    write_length: int = 1,
    save_every: int | None = None,
    keep_last_n_checkpoints: int | None = None,
) -> dict:
    """Build a minimal TTS trainer config for integration tests."""
    training_cfg = {
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "precision": precision,
        "num_workers": num_workers,
        "seed": seed,
        "lr": lr,
        "steps": steps,
        "read_length": read_length,
        "write_length": write_length,
        "output_dir": str(output_dir),
    }
    if save_every is not None:
        training_cfg["save_every"] = save_every
    if keep_last_n_checkpoints is not None:
        training_cfg["keep_last_n_checkpoints"] = keep_last_n_checkpoints

    return {
        "model": {
            "id": model_id,
        },
        "data": {
            "json_path": str(json_path),
        },
        "training": training_cfg,
        "wandb": {"enabled": False},
    }


def _write_minimal_tts_two_shards(shard_dir: Path) -> None:
    """Write a tiny valid two-shard TTS dataset."""
    # Keep one valid sample per shard to make cursor progression deterministic.
    _write_jsonl(
        shard_dir / "shard-00000.jsonl",
        [
            {
                "id": "s0",
                "source_id": "conv-0",
                "turn_index": 0,
                "assistant_text": "a",
                "unit_ids": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        ],
    )
    _write_jsonl(
        shard_dir / "shard-00001.jsonl",
        [
            {
                "id": "s1",
                "source_id": "conv-1",
                "turn_index": 0,
                "assistant_text": "a",
                "unit_ids": [9, 10, 11, 12, 13, 14, 15, 16],
            }
        ],
    )


@pytest.fixture(scope="module")
def tts_sharded_base_run(tmp_path_factory, local_base_model_id):
    """Run one shared sharded training job for TTS resume tests."""
    root = tmp_path_factory.mktemp("tts_trainer_shared")
    shard_dir = root / "tts_shards"
    _write_minimal_tts_two_shards(shard_dir)

    output_dir = root / "out_base"
    cfg = _build_trainer_cfg(
        model_id=local_base_model_id,
        json_path=shard_dir,
        output_dir=output_dir,
        steps=3,
        save_every=2,
        keep_last_n_checkpoints=2,
    )
    cfg["data"]["shuffle_shards"] = True
    cfg["data"]["shuffle_seed"] = 123
    cfg_path = root / "config_base.yaml"
    _write_config(cfg_path, cfg)

    # Base run powers multiple tests so we only pay setup cost once.
    result = _run_tts_trainer(cfg_path)
    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, output

    return {
        "root": root,
        "shard_dir": shard_dir,
        "output_dir": output_dir,
        "step2_dir": output_dir / "checkpoints" / "step-00000002",
    }


def test_tts_trainer_sharded_resume_cursor_is_conservative(
    tts_sharded_base_run,
    local_base_model_id,
):
    """Resuming from a sharded checkpoint should keep a conservative cursor."""
    step2_state = _read_json(tts_sharded_base_run["step2_dir"] / "trainer_state.json")
    assert step2_state["next_shard_cursor"] == 1
    assert step2_state["shuffle_epoch"] == 0

    # Resume in a fresh output dir so this test does not mutate shared artifacts.
    resumed_output_dir = tts_sharded_base_run["root"] / "out_resume_cursor"
    resumed_cfg = _build_trainer_cfg(
        model_id=local_base_model_id,
        json_path=tts_sharded_base_run["shard_dir"],
        output_dir=resumed_output_dir,
        steps=3,
        keep_last_n_checkpoints=1,
    )
    resumed_cfg["data"]["shuffle_shards"] = True
    resumed_cfg["data"]["shuffle_seed"] = 123
    resumed_cfg_path = tts_sharded_base_run["root"] / "config_resume_cursor.yaml"
    _write_config(resumed_cfg_path, resumed_cfg)

    # Resume from step 2 and verify conservative cursor behavior is preserved.
    resumed = _run_tts_trainer(
        resumed_cfg_path,
        extra_args=["--resume", str(tts_sharded_base_run["step2_dir"])],
    )
    resumed_output = (resumed.stdout or "") + (resumed.stderr or "")
    assert resumed.returncode == 0, resumed_output
    step3_state = _read_json(
        resumed_output_dir / "checkpoints" / "step-00000003" / "trainer_state.json"
    )
    assert step3_state["global_step"] == 3
    assert step3_state["next_shard_cursor"] == 1
    assert step3_state["shuffle_epoch"] == 0


def test_tts_trainer_epoch_increments_on_exhaustion(tts_sharded_base_run):
    """A sharded run should bump shuffle_epoch after exhausting one pass."""
    # The shared base run uses 2 shards and 3 steps, forcing one exhaustion rollover.
    state = _read_json(
        tts_sharded_base_run["output_dir"]
        / "checkpoints"
        / "step-00000003"
        / "trainer_state.json"
    )
    assert state["shuffle_epoch"] == 1


def test_tts_trainer_resume_rejects_incompatible_config(
    tts_sharded_base_run,
    local_base_model_id,
):
    """Resume should fail fast when compat-defining config fields change."""
    # Reuse the same data and checkpoint, but intentionally change one compat key.
    mismatch_output_dir = tts_sharded_base_run["root"] / "out_resume_mismatch"
    mismatch_cfg = _build_trainer_cfg(
        model_id=local_base_model_id,
        json_path=tts_sharded_base_run["shard_dir"],
        output_dir=mismatch_output_dir,
        steps=3,
        read_length=2,
    )
    mismatch_cfg["data"]["shuffle_shards"] = True
    mismatch_cfg["data"]["shuffle_seed"] = 123
    mismatch_cfg_path = tts_sharded_base_run["root"] / "config_resume_mismatch.yaml"
    _write_config(mismatch_cfg_path, mismatch_cfg)

    # Resume must reject because checkpoint compat saved read_length=64.
    result = _run_tts_trainer(
        mismatch_cfg_path,
        extra_args=["--resume", str(tts_sharded_base_run["step2_dir"])],
    )
    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0
    assert "resume mismatch for read_length" in output


def test_tts_trainer_resume_wraps_stale_shard_cursor(
    tts_sharded_base_run,
    local_base_model_id,
):
    """Resume should wrap out-of-range shard cursor and advance shuffle epoch."""
    # Copy checkpoint so this test can mutate trainer_state without affecting others.
    stale_resume_dir = (
        tts_sharded_base_run["root"] / f"step-00000002-stale-{uuid.uuid4().hex}"
    )
    shutil.copytree(tts_sharded_base_run["step2_dir"], stale_resume_dir)
    stale_state_path = stale_resume_dir / "trainer_state.json"
    stale_state = _read_json(stale_state_path)
    stale_state["next_shard_cursor"] = 99
    stale_state_path.write_text(json.dumps(stale_state, indent=2), encoding="utf-8")

    wrap_output_dir = tts_sharded_base_run["root"] / "out_resume_wrap_cursor"
    wrap_cfg = _build_trainer_cfg(
        model_id=local_base_model_id,
        json_path=tts_sharded_base_run["shard_dir"],
        output_dir=wrap_output_dir,
        steps=3,
        keep_last_n_checkpoints=1,
    )
    wrap_cfg["data"]["shuffle_shards"] = True
    wrap_cfg["data"]["shuffle_seed"] = 123
    wrap_cfg_path = tts_sharded_base_run["root"] / "config_resume_wrap_cursor.yaml"
    _write_config(wrap_cfg_path, wrap_cfg)

    # Resume should log wrap-to-zero and still complete the remaining step.
    result = _run_tts_trainer(
        wrap_cfg_path,
        extra_args=["--resume", str(stale_resume_dir)],
    )
    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, output
    assert "wrapping to shard 0" in output

    final_state = _read_json(
        wrap_output_dir / "checkpoints" / "step-00000003" / "trainer_state.json"
    )
    assert final_state["global_step"] == 3
    assert final_state["next_shard_cursor"] == 0
    assert final_state["shuffle_epoch"] == 1
