"""Streaming process contracts shared by generation, decode, and server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class GenerationChunk:
    """Carry one generation update with cumulative text/units and finalize."""

    turn_id: str
    text: str
    unit_ids: list[int]
    finalize: bool


@dataclass(frozen=True)
class GenerationFinalState:
    """Carry generation stop metadata for the final turn result."""

    assistant_text_token_ids: list[int]
    speech_token_ids: list[int]
    thinker_stop_seen: bool
    talker_eos_seen: bool
    queued_conditioning_consumed: bool


@dataclass(frozen=True)
class DecodeChunk:
    """Carry one decoded audio chunk produced from unit deltas."""

    turn_id: str
    wav: torch.Tensor
    sample_rate: int


@dataclass(frozen=True)
class TurnStart:
    """Start one turn in the generation process."""

    session_id: str
    turn_id: str
    audio_bytes: bytes
    decode_audio: bool
    text_generation_overrides: dict[str, object] | None = None
    speech_generation_overrides: dict[str, object] | None = None
    clear_conversation: bool = False


@dataclass(frozen=True)
class SessionClear:
    """Clear one generation-session conversation history."""

    session_id: str


@dataclass(frozen=True)
class TurnCancel:
    """Cancel one in-flight generation turn."""

    session_id: str
    turn_id: str


@dataclass(frozen=True)
class DecodeTurnStart:
    """Start one turn in the decode process."""

    turn_id: str
    enabled: bool


@dataclass(frozen=True)
class DecodeTurnChunk:
    """Forward one generation chunk to the decode process."""

    chunk: GenerationChunk


@dataclass(frozen=True)
class DecodeTurnEnd:
    """Mark one decode turn as complete."""

    turn_id: str


@dataclass(frozen=True)
class TurnDone:
    """Mark one turn as complete with final generation metadata."""

    turn_id: str
    final_state: GenerationFinalState


@dataclass(frozen=True)
class WorkerError:
    """Carry one structured worker-side failure."""

    stage: str
    message: str
    turn_id: str | None = None


@dataclass(frozen=True)
class WorkerReady:
    """Signal that one worker completed startup and is ready."""

    stage: str


@dataclass(frozen=True)
class Shutdown:
    """Stop one worker process loop."""


GenerationProcessRequest = TurnStart | SessionClear | Shutdown
GenerationProcessEvent = GenerationChunk | TurnDone | WorkerError | WorkerReady

DecodeProcessRequest = DecodeTurnStart | DecodeTurnChunk | DecodeTurnEnd | Shutdown
DecodeProcessEvent = DecodeChunk | DecodeTurnEnd | WorkerError | WorkerReady


def split_new_units(*, cumulative: Sequence[int], consumed: int) -> tuple[list[int], int]:
    """Return only unseen unit deltas from a cumulative unit sequence."""
    if int(consumed) < 0:
        raise ValueError("consumed must be >= 0")
    total = int(len(cumulative))
    if int(consumed) > total:
        raise ValueError("cumulative unit sequence cannot shrink")
    new_units = [int(unit_id) for unit_id in cumulative[int(consumed) :]]
    return new_units, total


__all__ = [
    "DecodeProcessEvent",
    "DecodeProcessRequest",
    "DecodeTurnChunk",
    "DecodeTurnEnd",
    "DecodeTurnStart",
    "DecodeChunk",
    "GenerationProcessEvent",
    "GenerationProcessRequest",
    "GenerationChunk",
    "GenerationFinalState",
    "SessionClear",
    "Shutdown",
    "TurnDone",
    "TurnCancel",
    "TurnStart",
    "WorkerError",
    "WorkerReady",
    "split_new_units",
]
