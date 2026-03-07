"""Decode process with in-file decode logic and process loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cosyvoice2.speech_decoder import SpeechDecoder
from inference.stream_protocol import (
    DecodeChunk,
    DecodeTurnChunk,
    DecodeTurnEnd,
    DecodeTurnStart,
    Shutdown,
    WorkerError,
    WorkerReady,
    split_new_units,
)


@dataclass
class _TurnState:
    """Track one decode turn state for cumulative-unit delta decode."""

    enabled: bool
    decoder_state: Any
    consumed_units: int = 0


def _build_decoder(*, cfg: dict[str, Any]) -> SpeechDecoder:
    """Load only decoder assets for the decode process."""
    cache_dir = Path(str(cfg["models"]["cache_dir"]))
    decoder_dir = cache_dir / "decoder"
    if not decoder_dir.is_dir():
        raise FileNotFoundError(f"decoder assets not found: {decoder_dir}")
    return SpeechDecoder(decoder_dir, device="cuda")


def run_decode_process(
    *,
    cfg: dict[str, Any],
    request_queue: Any,
    event_queue: Any,
) -> None:
    """Run a blocking decode worker loop on one dedicated process."""
    # Fail fast on startup errors so parent runtime does not hang waiting forever.
    try:
        decoder = _build_decoder(cfg=cfg)
    except Exception as exc:
        event_queue.put(WorkerError(stage="decode", message=f"startup failed: {exc}"))
        return
    event_queue.put(WorkerReady(stage="decode"))
    turns: dict[str, _TurnState] = {}
    while True:
        request = request_queue.get()
        if isinstance(request, Shutdown):
            return
        if isinstance(request, DecodeTurnStart):
            turns[request.turn_id] = _TurnState(
                enabled=bool(request.enabled),
                decoder_state=decoder.create_stream_state() if bool(request.enabled) else None,
                consumed_units=0,
            )
            continue
        if isinstance(request, DecodeTurnEnd):
            turns.pop(request.turn_id, None)
            continue
        if not isinstance(request, DecodeTurnChunk):
            event_queue.put(WorkerError(stage="decode", message=f"unsupported request type: {type(request).__name__}"))
            continue

        turn_id = str(request.chunk.turn_id)
        state = turns.get(turn_id)
        if state is None:
            event_queue.put(WorkerError(stage="decode", message="decode chunk for unknown turn", turn_id=turn_id))
            continue
        if not state.enabled:
            if bool(request.chunk.finalize):
                turns.pop(turn_id, None)
                event_queue.put(DecodeTurnEnd(turn_id=turn_id))
            continue

        try:
            # Decode only unseen units from cumulative generation state.
            new_units, next_consumed = split_new_units(
                cumulative=request.chunk.unit_ids,
                consumed=int(state.consumed_units),
            )
            if not new_units and not bool(request.chunk.finalize):
                continue
            audio_chunk, next_decoder_state = decoder.decode_unit_chunk(
                state=state.decoder_state,
                new_unit_ids=new_units,
                finalize=bool(request.chunk.finalize),
            )
            state.decoder_state = next_decoder_state
            state.consumed_units = int(next_consumed)
            if int(audio_chunk.numel()) > 0:
                event_queue.put(
                    DecodeChunk(
                        turn_id=turn_id,
                        wav=audio_chunk,
                        sample_rate=int(decoder.sample_rate),
                    )
                )
            if bool(request.chunk.finalize):
                turns.pop(turn_id, None)
                event_queue.put(DecodeTurnEnd(turn_id=turn_id))
        except Exception as exc:
            turns.pop(turn_id, None)
            event_queue.put(WorkerError(stage="decode", message=str(exc), turn_id=turn_id))


__all__ = ["run_decode_process"]
