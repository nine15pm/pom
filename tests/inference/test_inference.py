from __future__ import annotations

import base64
from copy import deepcopy
from pathlib import Path
import shutil
import time

import pytest
import torch
import torchaudio
import yaml
from transformers import AutoTokenizer

from inference.loader import load_runtime
from inference.offline import InferenceRequest, run_turn
from inference.streaming import StreamingConversationSession, StreamingInferenceRequest
from model.pom_talker import build_talker
from model.pom_thinker import build_thinker


def _base_cfg(
    *,
    cache_dir: str,
    tokenizer_source: str | None,
    speech_vocab_size: int = 6561,
) -> dict:
    """Build the smallest valid Phase A config for runtime + run_turn tests."""
    return {
        "models": {
            "artifact_mode": "hf",
            "cache_dir": cache_dir,
            "base_cache_dir": None,
            "speech_encoder_cache_dir": None,
            "speech_vocab_size": speech_vocab_size,
            "checkpoint": None,
        },
        "tokenizer": {
            "source": tokenizer_source,
            "enable_thinking": False,
            "assistant_stop_token": None,
        },
        "generation": {
            # Keep sampling close to real inference defaults, with small token caps for test speed.
            "text": {"max_new_tokens": 12, "temperature": 0.7, "top_p": 0.9},
            "speech": {
                "max_new_tokens": 16,
                "temperature": 0.8,
                "top_p": 0.95,
                "repetition_penalty": 1.1,
                "max_repeat_run": 30,
                "read_length": 3,
                "write_length": 10,
            },
        },
        "runtime": {"device": "cuda", "dtype": "bf16", "seed": 42},
        "output": {"output_wav_path": None},
    }


def _reset_decode_seed(cfg: dict) -> None:
    """Reset torch RNG so decode-driven tests are reproducible between calls."""
    # Streaming tests also use this to keep sampling behavior stable across runs.
    seed = int(cfg["runtime"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cache_dir_with_decoder(
    *,
    base_cache_dir: str,
    decoder_dir: str,
    output_root: Path,
) -> str:
    """Clone one cache dir and attach decoder assets at <cache_dir>/decoder."""
    target_cache_dir = output_root / "cache_with_decoder"
    shutil.copytree(Path(base_cache_dir), target_cache_dir)
    decoder_link = target_cache_dir / "decoder"
    # Replace any copied decoder path so this helper is stable across cache layouts.
    if decoder_link.is_symlink() or decoder_link.is_file():
        decoder_link.unlink()
    elif decoder_link.is_dir():
        shutil.rmtree(decoder_link)
    decoder_link.symlink_to(Path(decoder_dir), target_is_directory=True)
    return target_cache_dir.as_posix()


def _stream_one_turn(
    *,
    runtime,
    cfg: dict,
    request: StreamingInferenceRequest,
):
    """Collect one streaming turn using the current session-based public API."""
    # Use a fresh session so this helper matches one-turn behavior.
    session = StreamingConversationSession(runtime=runtime, cfg=cfg, max_history_turns=10)
    return list(session.stream_turn(request))


def _build_streaming_test_app(*, cfg: dict, tmp_path: Path, filename: str):
    """Create one real streaming server app bound to a temp inference config file."""
    from inference.server import StreamingServerConfig, create_app

    # Persist config so FastAPI startup loads runtime the same way as production.
    inference_cfg_path = tmp_path / filename
    inference_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    server_cfg = StreamingServerConfig(
        inference_config_path=inference_cfg_path.as_posix(),
        max_history_turns=10,
        host="127.0.0.1",
        port=8000,
        websocket_path="/ws",
        ws_max_frame_bytes=30_000_000,
        max_audio_bytes=20_000_000,
    )
    return create_app(cfg=server_cfg)


def _receive_turn_messages(websocket) -> list[dict]:
    """Collect all messages for one turn until terminal done/error."""
    messages: list[dict] = []
    while True:
        message = websocket.receive_json()
        messages.append(message)
        if message.get("type") in {"turn.done", "error"}:
            return messages


@pytest.fixture(scope="module")
def hf_runtime_and_cfg(tmp_path_factory, base_model_id: str, whisper_tiny_id: str):
    """Create a real HF runtime bundle from saved Thinker/Talker artifacts."""
    # Build real modules once with one shared tokenizer contract.
    thinker, tokenizer, token_ids = build_thinker(
        base_model_id=base_model_id,
        speech={
            "encoder_id": whisper_tiny_id,
            "frame_stack": 5,
            "projector_hidden_dim": 2048,
        },
    )
    talker, _, _ = build_talker(
        llm_hidden_dim=int(thinker.config.hidden_size),
        base_model_id=base_model_id,
        speech_vocab_size=6561,
        tokenizer=tokenizer,
        token_ids=token_ids,
    )

    # Save artifacts in HF layout so we exercise the default inference load path.
    root = tmp_path_factory.mktemp("inference_pipeline_hf")
    thinker_dir = root / "thinker"
    talker_dir = root / "talker"
    tokenizer_dir = root / "tokenizer"
    thinker.save_pretrained(thinker_dir, safe_serialization=False)
    talker.save_pretrained(talker_dir, safe_serialization=False)
    tokenizer.save_pretrained(tokenizer_dir)

    cfg = _base_cfg(
        cache_dir=str(root),
        tokenizer_source=None,
    )
    runtime = load_runtime(cfg)
    return runtime, cfg


def test_hf_e2e_turn_contract(hf_runtime_and_cfg, fixture_audio_paths):
    """Run one real turn and verify the public response contract shape/ranges."""
    runtime, cfg = hf_runtime_and_cfg
    request = InferenceRequest(
        audio_path=str(fixture_audio_paths[0]),
        decode_audio=False,
    )

    response = run_turn(runtime=runtime, cfg=cfg, request=request)

    # Assistant side must always return at least one content token.
    assert isinstance(response.assistant_text, str)
    assert len(response.assistant_text_token_ids) > 0
    assert all(isinstance(token_id, int) for token_id in response.assistant_text_token_ids)

    # Speech units are raw ids, so each id must be inside the configured unit vocabulary.
    speech_vocab_size = int(cfg["models"]["speech_vocab_size"])
    assert all(isinstance(unit_id, int) for unit_id in response.speech_token_ids)
    assert all(0 <= unit_id < speech_vocab_size for unit_id in response.speech_token_ids)


def test_streaming_outputs_are_valid_with_same_overrides(
    trained_runtime_and_cfg,
    fixture_audio_paths,
):
    """Validate one streaming turn under realistic decode settings."""
    runtime, cfg = trained_runtime_and_cfg
    audio_path = str(fixture_audio_paths[0])

    # Use one shared decode setting block close to production inference defaults.
    text_overrides = {"max_new_tokens": 12, "temperature": 0.7, "top_p": 0.9}
    speech_overrides = {
        "max_new_tokens": 16,
        "temperature": 0.8,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "max_repeat_run": 30,
        "read_length": 3,
        "write_length": 10,
    }

    # Run streaming under the same request overrides and collect terminal metadata.
    stream_events = _stream_one_turn(
        runtime=runtime,
        cfg=cfg,
        request=StreamingInferenceRequest(
            audio_path=audio_path,
            decode_audio=False,
            text_generation_overrides=text_overrides,
            speech_generation_overrides=speech_overrides,
        ),
    )

    # Streaming contract requires one terminal done event with final metadata.
    assert len(stream_events) > 0
    assert stream_events[-1].event == "done"
    assert stream_events[-1].result is not None
    final = stream_events[-1].result

    # Streaming should produce non-empty outputs with ids inside configured speech vocab bounds.
    speech_vocab_size = int(cfg["models"]["speech_vocab_size"])
    assert len(final.assistant_text_token_ids) > 0
    assert len(final.speech_token_ids) > 0
    assert len(final.speech_token_ids) <= int(speech_overrides["max_new_tokens"])
    assert all(0 <= int(unit_id) < speech_vocab_size for unit_id in final.speech_token_ids)

    # Streaming result fields should still satisfy the public completion contract.
    assert isinstance(final.assistant_text, str)
    assert isinstance(final.thinker_stop_seen, bool)
    assert isinstance(final.talker_eos_seen, bool)
    assert isinstance(final.queued_conditioning_consumed, bool)


def test_streaming_event_contract_and_order_decode_audio_false(
    trained_runtime_and_cfg,
    fixture_audio_paths,
):
    """Enforce minimal event ordering guarantees needed by a simple streaming client."""
    runtime, cfg = trained_runtime_and_cfg
    audio_path = str(fixture_audio_paths[0])

    # Keep sampling stable so this contract check is repeatable in real runtime tests.
    _reset_decode_seed(cfg)
    events = _stream_one_turn(
        runtime=runtime,
        cfg=cfg,
        request=StreamingInferenceRequest(audio_path=audio_path, decode_audio=False),
    )

    # Streaming must emit at least one event and always finish with done.
    assert len(events) > 0
    assert events[-1].event == "done"
    assert events[-1].result is not None

    # decode_audio=False should not emit audio chunks in the stream.
    assert all(event.event != "audio" for event in events)

    # Keep event vocabulary strict so server-client protocol stays explicit.
    allowed_events = {"text", "units", "done"}
    assert all(event.event in allowed_events for event in events)

    # done must appear exactly once and only as the terminal event.
    done_count = sum(1 for event in events if event.event == "done")
    assert done_count == 1

    # Every units event should carry at least one generated unit id.
    unit_events = [event for event in events if event.event == "units"]
    assert all(event.unit_ids is not None and len(event.unit_ids) > 0 for event in unit_events)

    # Concatenated streamed units must equal the final canonical speech output.
    streamed_units = [unit_id for event in unit_events for unit_id in (event.unit_ids or [])]
    final = events[-1].result
    assert streamed_units == final.speech_token_ids

    # Latest text snapshot should match final text shown at completion.
    text_events = [event for event in events if event.event == "text"]
    if text_events:
        assert text_events[-1].text_so_far == final.assistant_text


def test_streaming_interleaves_units_before_text_finishes(
    trained_runtime_and_cfg,
    fixture_audio_paths,
):
    """Ensure the V2 scheduler starts unit streaming before all text tokens are finished."""
    runtime, cfg = trained_runtime_and_cfg
    audio_path = str(fixture_audio_paths[0])

    # Keep decode deterministic and nudge scheduling toward early interleaving.
    _reset_decode_seed(cfg)
    events = _stream_one_turn(
        runtime=runtime,
        cfg=cfg,
        request=StreamingInferenceRequest(
            audio_path=audio_path,
            decode_audio=False,
            text_generation_overrides={"max_new_tokens": 32},
            speech_generation_overrides={"read_length": 1, "write_length": 1},
        ),
    )

    # Require a normal completed turn before checking cross-stream ordering.
    assert len(events) > 0
    assert events[-1].event == "done"
    assert events[-1].result is not None

    # Find first units event and last text event using only public stream events.
    first_units_idx = next((idx for idx, event in enumerate(events) if event.event == "units"), None)
    last_text_idx = next(
        (idx for idx in range(len(events) - 1, -1, -1) if events[idx].event == "text"),
        None,
    )

    assert first_units_idx is not None
    assert last_text_idx is not None
    # Interleaving means unit streaming begins before text generation fully finishes.
    assert int(first_units_idx) < int(last_text_idx)


def test_streaming_flushes_tail_conditioning_when_read_length_is_large(
    trained_runtime_and_cfg,
    fixture_audio_paths,
):
    """Ensure Talker still starts after Thinker stops even when read_length is never reached."""
    runtime, cfg = trained_runtime_and_cfg
    audio_path = str(fixture_audio_paths[0])

    # Force the scheduler into the tail-flush path with an intentionally large read_length.
    _reset_decode_seed(cfg)
    events = _stream_one_turn(
        runtime=runtime,
        cfg=cfg,
        request=StreamingInferenceRequest(
            audio_path=audio_path,
            decode_audio=False,
            text_generation_overrides={"max_new_tokens": 8},
            speech_generation_overrides={"read_length": 64, "write_length": 1},
        ),
    )

    # The stream should complete normally and still produce unit chunks.
    assert len(events) > 0
    assert events[-1].event == "done"
    assert events[-1].result is not None
    unit_events = [event for event in events if event.event == "units"]
    assert len(unit_events) > 0

    # Final output should confirm the turn produced speech and consumed conditioning.
    final = events[-1].result
    assert len(final.speech_token_ids) > 0
    assert final.queued_conditioning_consumed is True


def test_streaming_talker_early_stop_does_not_truncate_final_text(
    trained_runtime_and_cfg,
    fixture_audio_paths,
):
    """Ensure early Talker stop still yields a complete streaming text result."""
    runtime, cfg = trained_runtime_and_cfg
    audio_path = str(fixture_audio_paths[0])
    # Keep decode stable while forcing Talker to stop almost immediately.
    text_overrides = {"max_new_tokens": 48, "temperature": 1e-5, "top_p": 1.0}
    speech_overrides = {"max_new_tokens": 1, "read_length": 1, "write_length": 1}

    # Streaming should keep generating Thinker text even when speech side stops almost immediately.
    _reset_decode_seed(cfg)
    events = _stream_one_turn(
        runtime=runtime,
        cfg=cfg,
        request=StreamingInferenceRequest(
            audio_path=audio_path,
            decode_audio=False,
            text_generation_overrides=text_overrides,
            speech_generation_overrides=speech_overrides,
        ),
    )

    assert len(events) > 0
    assert events[-1].event == "done"
    assert events[-1].result is not None
    final = events[-1].result

    # Concatenated streamed units should match the final canonical unit sequence.
    unit_events = [event for event in events if event.event == "units"]
    streamed_units = [unit_id for event in unit_events for unit_id in (event.unit_ids or [])]
    assert streamed_units == final.speech_token_ids

    # Last streamed text snapshot should match the final text contract.
    text_events = [event for event in events if event.event == "text"]
    if text_events:
        assert text_events[-1].text_so_far == final.assistant_text

    # Early Talker stop should still produce completed text plus bounded speech units.
    assert len(final.assistant_text_token_ids) > 0
    assert len(final.speech_token_ids) <= 1


def test_streaming_first_units_arrive_early_in_turn_timeline(
    trained_runtime_and_cfg,
    fixture_audio_paths,
):
    """Guard against regressions where the first unit chunk is delayed until near turn end."""
    runtime, cfg = trained_runtime_and_cfg
    audio_path = str(fixture_audio_paths[0])

    # Bias settings toward a longer turn so early/late timing shape is observable.
    _reset_decode_seed(cfg)
    session = StreamingConversationSession(runtime=runtime, cfg=cfg, max_history_turns=10)
    stream = session.stream_turn(
        StreamingInferenceRequest(
            audio_path=audio_path,
            decode_audio=False,
            text_generation_overrides={"max_new_tokens": 64},
            speech_generation_overrides={"max_new_tokens": 48, "read_length": 1, "write_length": 1},
        )
    )

    t0 = time.perf_counter()
    first_units_ts: float | None = None
    done_ts: float | None = None
    saw_units = False

    # Consume real stream events while recording when units first appear and when the turn ends.
    for event in stream:
        now = time.perf_counter()
        if event.event == "units" and not saw_units:
            saw_units = True
            first_units_ts = now
        if event.event == "done":
            done_ts = now

    assert first_units_ts is not None
    assert done_ts is not None
    total = float(done_ts - t0)
    first_units_offset = float(first_units_ts - t0)
    assert total > 0.0
    # Ratio guard keeps this stable across machines while still catching thinker-first cliffs.
    assert first_units_offset / total < 0.8


def test_streaming_interleaved_scheduler_handles_truncated_history_multi_turn(
    trained_runtime_and_cfg,
    fixture_audio_paths,
):
    """Ensure repeated interleaved turns stay healthy when session history is aggressively truncated."""
    runtime, cfg = trained_runtime_and_cfg
    session = StreamingConversationSession(runtime=runtime, cfg=cfg, max_history_turns=1)
    audio_paths = [str(fixture_audio_paths[0]), str(fixture_audio_paths[1]), str(fixture_audio_paths[0])]

    # Run three real turns to exercise scheduling across repeated history truncation boundaries.
    for audio_path in audio_paths:
        _reset_decode_seed(cfg)
        events = list(
            session.stream_turn(
                StreamingInferenceRequest(
                    audio_path=audio_path,
                    decode_audio=False,
                    text_generation_overrides={"max_new_tokens": 24},
                    speech_generation_overrides={"read_length": 1, "write_length": 1},
                )
            )
        )

        assert len(events) > 0
        assert events[-1].event == "done"
        assert events[-1].result is not None
        # Each turn should stream at least one unit chunk, not only a terminal summary.
        assert any(event.event == "units" for event in events)
        final = events[-1].result
        assert len(final.assistant_text_token_ids) > 0
        assert len(final.speech_token_ids) > 0


def test_streaming_units_chunk_size_respects_write_length(
    trained_runtime_and_cfg,
    fixture_audio_paths,
):
    """Ensure streamed unit chunk sizes obey the configured write_length pacing bound."""
    runtime, cfg = trained_runtime_and_cfg
    audio_path = str(fixture_audio_paths[0])
    write_length = 2

    # Use a tiny write_length so chunk-size violations are easy to detect.
    _reset_decode_seed(cfg)
    events = _stream_one_turn(
        runtime=runtime,
        cfg=cfg,
        request=StreamingInferenceRequest(
            audio_path=audio_path,
            decode_audio=False,
            text_generation_overrides={"max_new_tokens": 32},
            speech_generation_overrides={"read_length": 1, "write_length": write_length},
        ),
    )

    assert len(events) > 0
    assert events[-1].event == "done"
    unit_chunks = [event.unit_ids for event in events if event.event == "units"]
    assert len(unit_chunks) > 0
    # Black-box pacing contract: each emitted chunk length must be bounded by write_length.
    assert all(chunk is not None and 1 <= len(chunk) <= write_length for chunk in unit_chunks)


def test_streaming_respects_speech_max_new_tokens_override(trained_runtime_and_cfg, fixture_audio_paths):
    """Bound streaming speech output length when request overrides max_new_tokens."""
    runtime, cfg = trained_runtime_and_cfg
    audio_path = str(fixture_audio_paths[0])

    # Keep sampling stable so this behavior check is deterministic enough.
    _reset_decode_seed(cfg)
    events = _stream_one_turn(
        runtime=runtime,
        cfg=cfg,
        request=StreamingInferenceRequest(
            audio_path=audio_path,
            decode_audio=False,
            speech_generation_overrides={"max_new_tokens": 1},
        ),
    )

    # Streaming should terminate normally with one done event.
    assert len(events) > 0
    assert events[-1].event == "done"
    assert events[-1].result is not None
    done_count = sum(1 for event in events if event.event == "done")
    assert done_count == 1

    # Final speech output must honor the explicit request-bound generation cap.
    final = events[-1].result
    assert len(final.speech_token_ids) <= 1


def test_streaming_decode_audio_requires_decoder_assets(
    trained_runtime_no_decoder_and_cfg,
    fixture_audio_paths,
):
    """Reject streaming waveform decode when runtime has no decoder assets loaded."""
    runtime, cfg = trained_runtime_no_decoder_and_cfg
    audio_path = str(fixture_audio_paths[0])

    # Streaming should fail before generation when decode is requested without decoder.
    with pytest.raises(ValueError, match="requires decoder assets"):
        _ = _stream_one_turn(
            runtime=runtime,
            cfg=cfg,
            request=StreamingInferenceRequest(audio_path=audio_path, decode_audio=True),
        )


def test_streaming_audio_chunks_with_real_decoder(
    trained_runtime_with_decoder_and_cfg,
    fixture_audio_paths,
):
    """Emit real audio chunks during streaming when decoder assets are configured."""
    runtime_with_decoder, cfg_with_decoder = trained_runtime_with_decoder_and_cfg
    audio_path = str(fixture_audio_paths[0])

    # Keep sampling stable so this streaming contract check is repeatable.
    _reset_decode_seed(cfg_with_decoder)
    events = _stream_one_turn(
        runtime=runtime_with_decoder,
        cfg=cfg_with_decoder,
        request=StreamingInferenceRequest(audio_path=audio_path, decode_audio=True),
    )

    # Streaming must terminate with one final done event.
    assert len(events) > 0
    assert events[-1].event == "done"
    assert events[-1].result is not None
    done_count = sum(1 for event in events if event.event == "done")
    assert done_count == 1

    # Real streaming decode should emit at least one non-empty audio chunk.
    audio_events = [event for event in events if event.event == "audio"]
    assert len(audio_events) > 0
    assert all(event.wav is not None for event in audio_events)
    assert all(event.wav.ndim == 1 and int(event.wav.numel()) > 0 for event in audio_events)
    assert all(event.sample_rate is not None and int(event.sample_rate) > 0 for event in audio_events)

    # Sample rate should stay consistent across all emitted chunks.
    sample_rates = {int(event.sample_rate) for event in audio_events if event.sample_rate is not None}
    assert len(sample_rates) == 1


def test_streaming_session_supports_multi_turn_and_clear(trained_runtime_and_cfg, fixture_audio_paths):
    """Run multi-turn streaming on one session and verify clear() resets context."""
    runtime, cfg = trained_runtime_and_cfg
    audio_turn_1 = str(fixture_audio_paths[0])
    audio_turn_2 = str(fixture_audio_paths[1])
    session = StreamingConversationSession(runtime=runtime, cfg=cfg, max_history_turns=10)

    # Turn 1: run one real request and validate terminal event shape.
    turn1_events = list(
        session.stream_turn(
            StreamingInferenceRequest(audio_path=audio_turn_1, decode_audio=False),
        )
    )
    assert len(turn1_events) > 0
    assert turn1_events[-1].event == "done"
    assert turn1_events[-1].result is not None

    # Turn 2: same websocket-like session should handle the next request without reset.
    turn2_events = list(
        session.stream_turn(
            StreamingInferenceRequest(audio_path=audio_turn_2, decode_audio=False),
        )
    )
    assert len(turn2_events) > 0
    assert turn2_events[-1].event == "done"
    assert turn2_events[-1].result is not None

    # After clear(), this session should behave like a no-history session at the same turn index.
    session.clear()
    cleared_events = list(
        session.stream_turn(
            StreamingInferenceRequest(audio_path=audio_turn_1, decode_audio=False),
        )
    )
    assert len(cleared_events) > 0
    assert cleared_events[-1].event == "done"
    assert cleared_events[-1].result is not None
    cleared_final = cleared_events[-1].result

    # Build a reference session with the same turn index and cleared history before the final turn.
    ref_session = StreamingConversationSession(runtime=runtime, cfg=cfg, max_history_turns=10)
    for burn_audio in (audio_turn_1, audio_turn_2):
        burn_events = list(
            ref_session.stream_turn(
                StreamingInferenceRequest(audio_path=burn_audio, decode_audio=False),
            )
        )
        assert len(burn_events) > 0
        assert burn_events[-1].event == "done"
        assert burn_events[-1].result is not None
        ref_session.clear()

    ref_events = list(
        ref_session.stream_turn(
            StreamingInferenceRequest(audio_path=audio_turn_1, decode_audio=False),
        )
    )
    assert len(ref_events) > 0
    assert ref_events[-1].event == "done"
    assert ref_events[-1].result is not None
    ref_final = ref_events[-1].result

    assert cleared_final.assistant_text == ref_final.assistant_text
    assert cleared_final.assistant_text_token_ids == ref_final.assistant_text_token_ids
    assert cleared_final.speech_token_ids == ref_final.speech_token_ids
    assert cleared_final.thinker_stop_seen == ref_final.thinker_stop_seen
    assert cleared_final.talker_eos_seen == ref_final.talker_eos_seen
    assert cleared_final.queued_conditioning_consumed == ref_final.queued_conditioning_consumed


def test_streaming_websocket_turn_contract_real_app(trained_runtime_and_cfg, fixture_audio_paths, tmp_path):
    """Exercise one real WebSocket turn and verify protocol event ordering."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    _, cfg = trained_runtime_and_cfg
    audio_bytes = Path(str(fixture_audio_paths[0])).read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    app = _build_streaming_test_app(
        cfg=cfg,
        tmp_path=tmp_path,
        filename="inference_for_server.yaml",
    )

    # Use one websocket connection like the browser push-to-talk flow.
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {
                    "type": "turn.start",
                    "turn_id": "turn-1",
                    "audio_b64": audio_b64,
                    "decode_audio": False,
                }
            )
            messages = _receive_turn_messages(websocket)

    # A valid request must end with one terminal turn.done and no error.
    assert len(messages) > 0
    assert messages[-1]["type"] == "turn.done"
    assert all(message.get("type") != "error" for message in messages)
    assert sum(1 for message in messages if message.get("type") == "turn.done") == 1

    # Event sequence and turn IDs must stay consistent for the whole streamed turn.
    seqs = [int(message["seq"]) for message in messages]
    assert seqs == list(range(len(seqs)))
    assert all(message.get("turn_id") == "turn-1" for message in messages)

    # Server should emit only the public protocol events for decode_audio=false.
    allowed_types = {"text.delta", "units.chunk", "turn.done"}
    assert all(message["type"] in allowed_types for message in messages)


def test_streaming_websocket_supports_two_turns_same_socket(
    trained_runtime_and_cfg,
    fixture_audio_paths,
    tmp_path,
):
    """Run two valid turns on one socket to verify conversation continuity."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    _, cfg = trained_runtime_and_cfg
    audio_a = base64.b64encode(Path(str(fixture_audio_paths[0])).read_bytes()).decode("ascii")
    audio_b = base64.b64encode(Path(str(fixture_audio_paths[1])).read_bytes()).decode("ascii")
    app = _build_streaming_test_app(
        cfg=cfg,
        tmp_path=tmp_path,
        filename="inference_for_server_two_turns.yaml",
    )

    # One socket should support multiple sequential turn.start requests.
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {
                    "type": "turn.start",
                    "turn_id": "turn-1",
                    "audio_b64": audio_a,
                    "decode_audio": False,
                }
            )
            turn1_messages = _receive_turn_messages(websocket)

            websocket.send_json(
                {
                    "type": "turn.start",
                    "turn_id": "turn-2",
                    "audio_b64": audio_b,
                    "decode_audio": False,
                }
            )
            turn2_messages = _receive_turn_messages(websocket)

    # Both turns must complete successfully on the same socket.
    assert len(turn1_messages) > 0
    assert len(turn2_messages) > 0
    assert turn1_messages[-1]["type"] == "turn.done"
    assert turn2_messages[-1]["type"] == "turn.done"
    assert all(message.get("type") != "error" for message in turn1_messages)
    assert all(message.get("type") != "error" for message in turn2_messages)
    assert all(message.get("turn_id") == "turn-1" for message in turn1_messages)
    assert all(message.get("turn_id") == "turn-2" for message in turn2_messages)

    # Sequence numbers should restart per turn because seq is turn-local.
    turn1_seqs = [int(message["seq"]) for message in turn1_messages]
    turn2_seqs = [int(message["seq"]) for message in turn2_messages]
    assert turn1_seqs == list(range(len(turn1_seqs)))
    assert turn2_seqs == list(range(len(turn2_seqs)))


def test_streaming_websocket_recovers_after_invalid_request(
    trained_runtime_and_cfg,
    fixture_audio_paths,
    tmp_path,
):
    """Keep socket usable after a bad request by handling error then next valid turn."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    _, cfg = trained_runtime_and_cfg
    valid_audio = base64.b64encode(Path(str(fixture_audio_paths[0])).read_bytes()).decode("ascii")
    app = _build_streaming_test_app(
        cfg=cfg,
        tmp_path=tmp_path,
        filename="inference_for_server_recovery.yaml",
    )

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            # First send malformed base64 and expect one terminal error for that request.
            websocket.send_json(
                {
                    "type": "turn.start",
                    "turn_id": "bad-turn",
                    "audio_b64": "%%%not-base64%%%",
                    "decode_audio": False,
                }
            )
            error_message = websocket.receive_json()
            assert error_message["type"] == "error"
            assert error_message["turn_id"] == "bad-turn"

            # Then send a valid turn on the same socket and expect normal completion.
            websocket.send_json(
                {
                    "type": "turn.start",
                    "turn_id": "good-turn",
                    "audio_b64": valid_audio,
                    "decode_audio": False,
                }
            )
            good_turn_messages = _receive_turn_messages(websocket)

    # Good turn should complete after the previous error without reconnecting.
    assert len(good_turn_messages) > 0
    assert good_turn_messages[-1]["type"] == "turn.done"
    assert all(message.get("type") != "error" for message in good_turn_messages)
    assert all(message.get("turn_id") == "good-turn" for message in good_turn_messages)
    seqs = [int(message["seq"]) for message in good_turn_messages]
    assert seqs == list(range(len(seqs)))


def test_streaming_websocket_disconnect_then_reconnect_still_works(
    trained_runtime_and_cfg,
    fixture_audio_paths,
    tmp_path,
):
    """Handle client disconnect during a turn and keep server usable for new sockets."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    _, cfg = trained_runtime_and_cfg
    audio_b64 = base64.b64encode(Path(str(fixture_audio_paths[0])).read_bytes()).decode("ascii")
    app = _build_streaming_test_app(
        cfg=cfg,
        tmp_path=tmp_path,
        filename="inference_for_server_disconnect.yaml",
    )

    with TestClient(app) as client:
        # First connection: start a turn then drop the socket immediately.
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {
                    "type": "turn.start",
                    "turn_id": "drop-turn",
                    "audio_b64": audio_b64,
                    "decode_audio": False,
                }
            )

        # Second connection: run a normal turn to prove server health after disconnect.
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {
                    "type": "turn.start",
                    "turn_id": "reconnect-turn",
                    "audio_b64": audio_b64,
                    "decode_audio": False,
                }
            )
            messages = _receive_turn_messages(websocket)

    # Reconnected socket must complete successfully with standard event sequencing.
    assert len(messages) > 0
    assert messages[-1]["type"] == "turn.done"
    assert all(message.get("type") != "error" for message in messages)
    assert all(message.get("turn_id") == "reconnect-turn" for message in messages)
    seqs = [int(message["seq"]) for message in messages]
    assert seqs == list(range(len(seqs)))


@pytest.fixture(scope="module")
def checkpoint_runtime_and_cfg(
    tmp_path_factory,
    hf_runtime_and_cfg,
    base_model_id: str,
    whisper_tiny_id: str,
):
    """Create a real checkpoint-mode runtime bundle from strict .bin artifacts."""
    runtime, _ = hf_runtime_and_cfg

    # Save strict checkpoint files so we exercise the explicit fallback load path.
    root = tmp_path_factory.mktemp("inference_pipeline_ckpt")
    thinker_ckpt_dir = root / "thinker"
    talker_ckpt_dir = root / "talker"
    thinker_ckpt_dir.mkdir(parents=True, exist_ok=False)
    talker_ckpt_dir.mkdir(parents=True, exist_ok=False)
    torch.save(runtime.thinker.state_dict(), thinker_ckpt_dir / "pytorch_model.bin")
    torch.save(runtime.talker.state_dict(), talker_ckpt_dir / "pytorch_model.bin")

    cfg = _base_cfg(
        cache_dir=str(root),
        tokenizer_source=base_model_id,
    )
    cfg["models"]["artifact_mode"] = "checkpoint"
    cfg["models"]["checkpoint"] = {
        "base_model_id": base_model_id,
        "speech_encoder_id": whisper_tiny_id,
        "frame_stack": 5,
        "projector_hidden_dim": 2048,
    }
    runtime = load_runtime(cfg)
    return runtime, cfg


def test_checkpoint_e2e_turn_contract(checkpoint_runtime_and_cfg, fixture_audio_paths):
    """Run one real checkpoint-mode turn and verify the same response contract."""
    runtime, cfg = checkpoint_runtime_and_cfg
    request = InferenceRequest(
        audio_path=str(fixture_audio_paths[0]),
        decode_audio=False,
    )

    response = run_turn(runtime=runtime, cfg=cfg, request=request)

    # Checkpoint fallback must satisfy the same public output guarantees as HF mode.
    assert isinstance(response.assistant_text, str)
    assert len(response.assistant_text_token_ids) > 0
    assert all(isinstance(token_id, int) for token_id in response.assistant_text_token_ids)
    speech_vocab_size = int(cfg["models"]["speech_vocab_size"])
    assert all(isinstance(unit_id, int) for unit_id in response.speech_token_ids)
    assert all(0 <= unit_id < speech_vocab_size for unit_id in response.speech_token_ids)


def test_decode_audio_requires_decoder_assets(hf_runtime_and_cfg, fixture_audio_paths):
    """Fail fast when waveform decode is requested without decoder assets."""
    runtime, cfg = hf_runtime_and_cfg
    request = InferenceRequest(
        audio_path=str(fixture_audio_paths[0]),
        decode_audio=True,
    )

    with pytest.raises(ValueError, match="requires decoder assets"):
        _ = run_turn(runtime=runtime, cfg=cfg, request=request)


def test_decode_audio_default_requires_decoder_assets(hf_runtime_and_cfg, fixture_audio_paths):
    """Fail fast by default when decode_audio is omitted and decoder assets are missing."""
    runtime, cfg = hf_runtime_and_cfg
    # Omit decode_audio to validate the request default contract.
    request = InferenceRequest(audio_path=str(fixture_audio_paths[0]))

    with pytest.raises(ValueError, match="requires decoder assets"):
        _ = run_turn(runtime=runtime, cfg=cfg, request=request)


def test_decode_audio_opt_out_runs_without_decoder(hf_runtime_and_cfg, fixture_audio_paths):
    """Allow text + unit decode when request explicitly opts out of waveform decode."""
    runtime, cfg = hf_runtime_and_cfg
    request = InferenceRequest(
        audio_path=str(fixture_audio_paths[0]),
        decode_audio=False,
    )

    response = run_turn(runtime=runtime, cfg=cfg, request=request)
    assert isinstance(response.assistant_text, str)
    assert len(response.assistant_text_token_ids) > 0
    assert all(isinstance(unit_id, int) for unit_id in response.speech_token_ids)
    assert response.wav is None
    assert response.sample_rate is None
    assert response.wav_path is None


def test_decode_audio_opt_out_does_not_write_wav(hf_runtime_and_cfg, fixture_audio_paths, tmp_path):
    """Do not write wav output when request opts out of decode."""
    runtime, cfg = hf_runtime_and_cfg
    # Configure an output path and verify decode opt-out still avoids file writes.
    cfg_no_decode = deepcopy(cfg)
    output_wav_path = tmp_path / "should_not_exist.wav"
    cfg_no_decode["output"]["output_wav_path"] = output_wav_path.as_posix()
    request = InferenceRequest(
        audio_path=str(fixture_audio_paths[0]),
        decode_audio=False,
    )

    response = run_turn(runtime=runtime, cfg=cfg_no_decode, request=request)
    assert output_wav_path.exists() is False
    assert response.wav is None
    assert response.sample_rate is None
    assert response.wav_path is None


def test_decode_audio_real_decoder_returns_waveform(
    hf_runtime_and_cfg,
    fixture_audio_paths,
    decoder_dir: str,
    tmp_path,
):
    """Decode real unit output to waveform when decoder assets are configured."""
    _, cfg = hf_runtime_and_cfg
    # Rebuild runtime with decoder assets to exercise true unit->wav integration.
    cfg_with_decoder = deepcopy(cfg)
    cfg_with_decoder["models"]["cache_dir"] = _cache_dir_with_decoder(
        base_cache_dir=str(cfg["models"]["cache_dir"]),
        decoder_dir=decoder_dir,
        output_root=tmp_path,
    )
    runtime_with_decoder = load_runtime(cfg_with_decoder)

    # Omit decode_audio to verify the default decode-on behavior.
    request = InferenceRequest(audio_path=str(fixture_audio_paths[0]))
    response = run_turn(runtime=runtime_with_decoder, cfg=cfg_with_decoder, request=request)

    assert len(response.speech_token_ids) > 0
    assert response.wav is not None
    assert response.wav.ndim == 1
    assert int(response.wav.numel()) > 0
    assert isinstance(response.sample_rate, int) and response.sample_rate > 0
    assert response.wav_path is None


def test_decode_audio_writes_wav_when_output_path_is_set(
    hf_runtime_and_cfg,
    fixture_audio_paths,
    decoder_dir: str,
    tmp_path,
):
    """Write decoded waveform to output_wav_path when configured."""
    _, cfg = hf_runtime_and_cfg
    # Configure decoder assets and one explicit output wav path.
    cfg_with_decoder = deepcopy(cfg)
    cfg_with_decoder["models"]["cache_dir"] = _cache_dir_with_decoder(
        base_cache_dir=str(cfg["models"]["cache_dir"]),
        decoder_dir=decoder_dir,
        output_root=tmp_path,
    )
    output_wav_path = tmp_path / "decoded_turn.wav"
    cfg_with_decoder["output"]["output_wav_path"] = output_wav_path.as_posix()
    runtime_with_decoder = load_runtime(cfg_with_decoder)

    # Omit decode_audio to validate default decode+write behavior.
    request = InferenceRequest(audio_path=str(fixture_audio_paths[0]))
    response = run_turn(runtime=runtime_with_decoder, cfg=cfg_with_decoder, request=request)

    assert response.wav_path == output_wav_path.as_posix()
    assert output_wav_path.is_file()
    saved_wav, saved_sr = torchaudio.load(output_wav_path.as_posix())
    assert saved_wav.ndim == 2 and int(saved_wav.shape[1]) > 0
    assert isinstance(response.sample_rate, int) and response.sample_rate > 0
    assert int(saved_sr) == int(response.sample_rate)


def test_runtime_contract_fail_fast(hf_runtime_and_cfg, base_model_id: str):
    """Fail fast on the most critical runtime contract violations."""
    _, cfg = hf_runtime_and_cfg

    # Phase A is CUDA-only, so CPU should be rejected immediately.
    bad_device_cfg = deepcopy(cfg)
    bad_device_cfg["runtime"]["device"] = "cpu"
    with pytest.raises(ValueError, match="runtime.device must be 'cuda'"):
        _ = load_runtime(bad_device_cfg)

    # Tokenizer without Pom special tokens must be rejected before serving.
    bad_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    bad_tokenizer_cfg = deepcopy(cfg)
    bad_tokenizer_cache = Path(str(cfg["models"]["cache_dir"])).parent / "cache_bad_tokenizer_contract"
    shutil.copytree(Path(str(cfg["models"]["cache_dir"])), bad_tokenizer_cache)
    bad_tokenizer.save_pretrained(bad_tokenizer_cache / "tokenizer")
    bad_tokenizer_cfg["models"]["cache_dir"] = bad_tokenizer_cache.as_posix()
    with pytest.raises(ValueError, match="<speech>"):
        _ = load_runtime(bad_tokenizer_cfg)

    # Configured speech vocab must match the loaded Talker artifact contract.
    bad_vocab_cfg = deepcopy(cfg)
    bad_vocab_cfg["models"]["speech_vocab_size"] = 6500
    with pytest.raises(ValueError, match="speech_vocab_size"):
        _ = load_runtime(bad_vocab_cfg)


def test_generation_overrides_contract(hf_runtime_and_cfg, fixture_audio_paths):
    """Reject unknown overrides and honor valid override limits."""
    runtime, cfg = hf_runtime_and_cfg
    audio_path = str(fixture_audio_paths[0])

    # Unknown request override keys must fail fast to avoid silent behavior drift.
    bad_request = InferenceRequest(
        audio_path=audio_path,
        decode_audio=False,
        speech_generation_overrides={"unknown_key": 1},
    )
    with pytest.raises(ValueError, match="unknown generation.speech override keys"):
        _ = run_turn(runtime=runtime, cfg=cfg, request=bad_request)

    # A valid override should constrain output behavior in the expected direction.
    good_request = InferenceRequest(
        audio_path=audio_path,
        decode_audio=False,
        speech_generation_overrides={"max_new_tokens": 1},
    )
    response = run_turn(runtime=runtime, cfg=cfg, request=good_request)
    assert len(response.speech_token_ids) <= 1


def test_alignment_guard_fails_on_interleaved_control_tokens(
    hf_runtime_and_cfg,
    fixture_audio_paths,
    tmp_path,
):
    """Force a real interleaved-control case and verify alignment fails."""
    runtime, cfg = hf_runtime_and_cfg
    audio_path = str(fixture_audio_paths[0])

    # Run one baseline turn so we can pick a real content token from model output.
    baseline = run_turn(
        runtime=runtime,
        cfg=cfg,
        request=InferenceRequest(audio_path=audio_path, decode_audio=False),
    )
    assert len(baseline.assistant_text_token_ids) >= 3
    middle_idx = len(baseline.assistant_text_token_ids) // 2
    middle_token_id = int(baseline.assistant_text_token_ids[middle_idx])
    assert middle_token_id not in set(int(t) for t in runtime.tokenizer.all_special_ids)

    # Mark one interior content token as special in tokenizer-only metadata.
    token_text = runtime.tokenizer.convert_ids_to_tokens(middle_token_id)
    assert isinstance(token_text, str) and token_text != ""
    mutated_tokenizer = AutoTokenizer.from_pretrained(
        str(Path(str(cfg["models"]["cache_dir"])) / "tokenizer")
    )
    mutated_tokenizer.add_special_tokens({"additional_special_tokens": [token_text]})
    # Ensure we marked the same existing token id as special, not a new added token.
    assert int(mutated_tokenizer.convert_tokens_to_ids(token_text)) == middle_token_id
    mutated_tokenizer_dir = tmp_path / "mutated_alignment_tokenizer"
    mutated_tokenizer.save_pretrained(mutated_tokenizer_dir)

    # Reload runtime against the mutated tokenizer source and expect alignment failure.
    mutated_cfg = deepcopy(cfg)
    mutated_cache_dir = tmp_path / "cache_mutated_alignment_tokenizer"
    shutil.copytree(Path(str(cfg["models"]["cache_dir"])), mutated_cache_dir)
    shutil.rmtree(mutated_cache_dir / "tokenizer")
    shutil.copytree(mutated_tokenizer_dir, mutated_cache_dir / "tokenizer")
    mutated_cfg["models"]["cache_dir"] = str(mutated_cache_dir)
    mutated_runtime = load_runtime(mutated_cfg)
    with pytest.raises(ValueError, match="no match"):
        _ = run_turn(
            runtime=mutated_runtime,
            cfg=mutated_cfg,
            request=InferenceRequest(audio_path=audio_path, decode_audio=False),
        )
