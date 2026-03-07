"""Tiny FastAPI WebSocket server for Phase B push-to-talk streaming."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import torch
import uvicorn
import yaml

import app as app_pkg
from inference.config import load_inference_config
from inference.inference import resolve_speech_decode_config
from inference.stream_protocol import DecodeChunk, GenerationChunk, GenerationFinalState, TurnDone, split_new_units
from inference.stream_runtime import StreamRuntime


@dataclass(frozen=True)
class StreamingServerConfig:
    """Hold validated server/runtime settings for WebSocket serving."""

    inference_config_path: str
    max_history_turns: int
    host: str
    port: int
    websocket_path: str
    ws_max_frame_bytes: int
    max_audio_bytes: int


@dataclass(frozen=True)
class TurnStartRequest:
    """Hold one parsed turn.start request from a WebSocket client."""

    turn_id: str
    audio_b64: str
    decode_audio: bool
    clear_conversation: bool
    text_generation_overrides: dict[str, object] | None
    speech_generation_overrides: dict[str, object] | None


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    """Validate one config/request object is a JSON/YAML mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _reject_unknown_keys(section: dict[str, Any], *, allowed: set[str], path: str) -> None:
    """Reject unknown keys so config and protocol stay explicit."""
    unknown = sorted(key for key in section.keys() if key not in allowed)
    if unknown:
        raise ValueError(f"unknown keys in {path}: {unknown}")


def _require_str(section: dict[str, Any], *, key: str, path: str) -> str:
    """Validate one required non-empty string field."""
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}.{key} must be a non-empty string")
    return value.strip()


def _require_positive_int(section: dict[str, Any], *, key: str, path: str) -> int:
    """Validate one required integer field that must be > 0."""
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{path}.{key} must be an integer > 0")
    return int(value)


def load_streaming_server_config(path: str | Path = "configs/streaming.yaml") -> StreamingServerConfig:
    """Load and validate the tiny Phase B WebSocket server config."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = _as_mapping(raw, name="config")
    _reject_unknown_keys(cfg, allowed={"inference_config_path", "streaming", "server"}, path="config")

    inference_config_path = _require_str(cfg, key="inference_config_path", path="config")
    streaming = _as_mapping(cfg.get("streaming"), name="streaming")
    _reject_unknown_keys(streaming, allowed={"max_history_turns"}, path="streaming")
    max_history_turns = _require_positive_int(streaming, key="max_history_turns", path="streaming")
    server = _as_mapping(cfg.get("server"), name="server")
    _reject_unknown_keys(
        server,
        allowed={"host", "port", "websocket_path", "ws_max_frame_bytes", "max_audio_bytes"},
        path="server",
    )

    host = _require_str(server, key="host", path="server")
    port = _require_positive_int(server, key="port", path="server")
    websocket_path = _require_str(server, key="websocket_path", path="server")
    ws_max_frame_bytes = _require_positive_int(server, key="ws_max_frame_bytes", path="server")
    max_audio_bytes = _require_positive_int(server, key="max_audio_bytes", path="server")
    if not websocket_path.startswith("/"):
        raise ValueError("server.websocket_path must start with '/'")
    # Ensure app-level audio validation can run before transport drops oversized frames.
    required_frame_budget = ((int(max_audio_bytes) + 2) // 3) * 4 + 4096
    if int(ws_max_frame_bytes) < int(required_frame_budget):
        raise ValueError(
            "server.ws_max_frame_bytes is too small for server.max_audio_bytes after base64/json overhead"
        )

    return StreamingServerConfig(
        inference_config_path=inference_config_path,
        max_history_turns=max_history_turns,
        host=host,
        port=port,
        websocket_path=websocket_path,
        ws_max_frame_bytes=ws_max_frame_bytes,
        max_audio_bytes=max_audio_bytes,
    )


def _parse_turn_start(payload: dict[str, Any]) -> TurnStartRequest:
    """Validate and parse one turn.start client request payload."""
    _reject_unknown_keys(
        payload,
        allowed={
            "type",
            "turn_id",
            "audio_b64",
            "decode_audio",
            "clear_conversation",
            "text_generation_overrides",
            "speech_generation_overrides",
        },
        path="request",
    )
    message_type = _require_str(payload, key="type", path="request")
    if message_type != "turn.start":
        raise ValueError("request.type must be 'turn.start'")

    turn_id = _require_str(payload, key="turn_id", path="request")
    audio_b64 = _require_str(payload, key="audio_b64", path="request")

    decode_audio = payload.get("decode_audio", True)
    if not isinstance(decode_audio, bool):
        raise ValueError("request.decode_audio must be a boolean when set")
    clear_conversation = payload.get("clear_conversation", False)
    if not isinstance(clear_conversation, bool):
        raise ValueError("request.clear_conversation must be a boolean when set")

    text_overrides = payload.get("text_generation_overrides")
    if text_overrides is not None:
        text_overrides = _as_mapping(text_overrides, name="request.text_generation_overrides")

    speech_overrides = payload.get("speech_generation_overrides")
    if speech_overrides is not None:
        speech_overrides = _as_mapping(speech_overrides, name="request.speech_generation_overrides")

    return TurnStartRequest(
        turn_id=turn_id,
        audio_b64=audio_b64,
        decode_audio=decode_audio,
        clear_conversation=clear_conversation,
        text_generation_overrides=text_overrides,
        speech_generation_overrides=speech_overrides,
    )


def _parse_message_type(payload: dict[str, Any]) -> str:
    """Read and validate one request.type value."""
    message_type = _require_str(payload, key="type", path="request")
    allowed = {"turn.start", "session.clear"}
    if message_type not in allowed:
        raise ValueError(f"request.type must be one of: {sorted(allowed)}")
    return message_type


def _decode_audio_bytes(*, audio_b64: str, max_bytes: int) -> bytes:
    """Decode base64 audio payload and enforce a max raw byte size."""
    try:
        raw = base64.b64decode(audio_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("request.audio_b64 must be valid base64") from exc
    if len(raw) <= 0:
        raise ValueError("request.audio_b64 decoded to empty audio")
    if len(raw) > int(max_bytes):
        raise ValueError("request.audio payload exceeds server.max_audio_bytes")
    return raw


def _wav_to_pcm16_b64(wav: torch.Tensor) -> str:
    """Convert one float waveform chunk to base64-encoded PCM16 bytes."""
    mono = wav.detach().to(device="cpu", dtype=torch.float32).flatten()
    if int(mono.numel()) <= 0:
        return ""
    pcm16 = mono.clamp(min=-1.0, max=1.0).mul(32767.0).round().to(dtype=torch.int16)
    return base64.b64encode(pcm16.numpy().tobytes()).decode("ascii")


def _resolve_stop_reason(*, final_state: GenerationFinalState, max_speech_tokens: int) -> str:
    """Map final streaming result flags to one explicit stop reason."""
    if bool(final_state.talker_eos_seen):
        return "eos"
    if len(final_state.speech_token_ids) >= int(max_speech_tokens):
        return "max_tokens"
    return "repeat_guard"


async def _send_json_or_stop(websocket: WebSocket, payload: dict[str, Any]) -> bool:
    """Send one JSON message and return False when the client has disconnected."""
    try:
        await websocket.send_json(payload)
    except WebSocketDisconnect:
        return False
    return True


def _error_payload(*, turn_id: str | None, seq: int, message: str) -> dict[str, Any]:
    """Build one consistent terminal error payload for protocol failures."""
    payload: dict[str, Any] = {
        "type": "error",
        "seq": int(seq),
        "message": str(message),
    }
    if turn_id is not None:
        payload["turn_id"] = str(turn_id)
    return payload


async def _run_turn(
    *,
    websocket: WebSocket,
    runtime: StreamRuntime,
    inference_cfg: dict[str, Any],
    session_id: str,
    request: TurnStartRequest,
    max_audio_bytes: int,
) -> bool:
    """Run one turn request end-to-end and stream ordered server events."""
    # Keep seq local so all turn events (including terminal errors) stay monotonic.
    seq = 0
    turn_complete = False
    stream = None
    try:
        raw_audio = _decode_audio_bytes(audio_b64=request.audio_b64, max_bytes=max_audio_bytes)
        speech_cfg = resolve_speech_decode_config(inference_cfg, request.speech_generation_overrides)
        max_speech_tokens = int(speech_cfg.max_new_tokens)
        stream = runtime.stream_turn(
            session_id=str(session_id),
            turn_id=str(request.turn_id),
            audio_bytes=raw_audio,
            decode_audio=bool(request.decode_audio),
            text_generation_overrides=request.text_generation_overrides,
            speech_generation_overrides=request.speech_generation_overrides,
            clear_conversation=bool(request.clear_conversation),
        )
        emitted_units = 0
        last_text = ""
        for event in stream:
            if isinstance(event, GenerationChunk):
                # Emit latest assistant text snapshot as a simple running string.
                if event.text != last_text:
                    last_text = str(event.text)
                    if not await _send_json_or_stop(
                        websocket,
                        {
                            "type": "text.delta",
                            "turn_id": request.turn_id,
                            "seq": int(seq),
                            "text": last_text,
                            "token_ids": [],
                        },
                    ):
                        return False
                    seq += 1
                # Emit only new unit deltas while generation stays cumulative.
                new_units, next_emitted = split_new_units(cumulative=event.unit_ids, consumed=emitted_units)
                emitted_units = int(next_emitted)
                if new_units:
                    if not await _send_json_or_stop(
                        websocket,
                        {
                            "type": "units.chunk",
                            "turn_id": request.turn_id,
                            "seq": int(seq),
                            "unit_ids": [int(unit_id) for unit_id in new_units],
                        },
                    ):
                        return False
                    seq += 1
                continue

            if isinstance(event, DecodeChunk):
                audio_b64 = _wav_to_pcm16_b64(event.wav)
                if not audio_b64:
                    continue
                if not await _send_json_or_stop(
                    websocket,
                    {
                        "type": "audio.chunk",
                        "turn_id": request.turn_id,
                        "seq": int(seq),
                        "audio_b64": audio_b64,
                        "sample_rate": int(event.sample_rate),
                        "encoding": "pcm_s16le",
                    },
                ):
                    return False
                seq += 1
                continue

            if isinstance(event, TurnDone):
                final_state = event.final_state
                if not await _send_json_or_stop(
                    websocket,
                    {
                        "type": "turn.done",
                        "turn_id": request.turn_id,
                        "seq": int(seq),
                        "assistant_text": str(last_text),
                        "assistant_text_token_ids": [
                            int(token_id) for token_id in final_state.assistant_text_token_ids
                        ],
                        "speech_token_ids": [int(unit_id) for unit_id in final_state.speech_token_ids],
                        "thinker_stop_seen": bool(final_state.thinker_stop_seen),
                        "talker_eos_seen": bool(final_state.talker_eos_seen),
                        "queued_conditioning_consumed": bool(final_state.queued_conditioning_consumed),
                        "stop_reason": _resolve_stop_reason(final_state=final_state, max_speech_tokens=max_speech_tokens),
                    },
                ):
                    return False
                turn_complete = True
                seq += 1
                continue
    except ValueError as exc:
        return await _send_json_or_stop(
            websocket,
            _error_payload(turn_id=request.turn_id, seq=seq, message=str(exc)),
        )
    except Exception as exc:
        return await _send_json_or_stop(
            websocket,
            _error_payload(turn_id=request.turn_id, seq=seq, message=f"inference failed: {exc}"),
        )
    finally:
        # Cancel unfinished turns so disconnected clients do not leave workers blocked.
        if stream is not None and not bool(turn_complete):
            runtime.cancel_turn(session_id=str(session_id), turn_id=str(request.turn_id))
        if stream is not None:
            stream.close()
    return True


def create_app(*, cfg: StreamingServerConfig) -> FastAPI:
    """Create a FastAPI app bound to one validated streaming server config."""
    app = FastAPI(title="pom-streaming-server")
    app_dir = Path(app_pkg.__file__).resolve().parent
    if not app_dir.is_dir():
        raise FileNotFoundError(f"app directory not found: {app_dir}")
    app.state.server_cfg = cfg
    app.state.runtime_lock = asyncio.Lock()
    app.state.inference_cfg = None
    app.state.stream_runtime = None

    # Serve static app assets from a dedicated folder so UI stays decoupled from backend logic.
    app.mount("/app", StaticFiles(directory=app_dir), name="app")

    @app.get("/")
    async def root_page() -> FileResponse:
        """Serve the tiny browser app entrypoint."""
        return FileResponse(app_dir / "index.html")

    @app.get("/app/config")
    async def app_config() -> dict[str, str]:
        """Expose tiny frontend runtime config so WS path stays backend-driven."""
        return {"websocket_path": str(cfg.websocket_path)}

    @app.on_event("startup")
    async def _startup() -> None:
        # Build process runtime once so all sockets share warm workers.
        inference_cfg = load_inference_config(path=cfg.inference_config_path)
        runtime = StreamRuntime(
            cfg=inference_cfg,
            max_history_turns=int(cfg.max_history_turns),
        )
        runtime.start()
        app.state.inference_cfg = inference_cfg
        app.state.stream_runtime = runtime

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        # Close workers on shutdown so GPU resources are released.
        runtime = app.state.stream_runtime
        if runtime is not None:
            runtime.close()
        app.state.stream_runtime = None

    @app.websocket(cfg.websocket_path)
    async def websocket_turns(websocket: WebSocket) -> None:
        # Keep one accepted socket open for multiple sequential turn.start requests.
        await websocket.accept()
        # Scope one generation-session id to one websocket connection.
        session_id = str(uuid.uuid4())
        while True:
            seq = 0
            turn_id: str | None = None
            try:
                message_text = await websocket.receive_text()
            except WebSocketDisconnect:
                return

            try:
                payload = _as_mapping(json.loads(message_text), name="request")
                message_type = _parse_message_type(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                ok = await _send_json_or_stop(
                    websocket,
                    _error_payload(turn_id=turn_id, seq=seq, message=str(exc)),
                )
                if not ok:
                    return
                continue

            if message_type == "session.clear":
                _reject_unknown_keys(payload, allowed={"type"}, path="request")
                app.state.stream_runtime.clear_session(session_id=session_id)
                ok = await _send_json_or_stop(websocket, {"type": "session.cleared", "seq": 0})
                if not ok:
                    return
                continue

            request = _parse_turn_start(payload)
            turn_id = request.turn_id
            # Serialize turns so one GPU runtime is never driven concurrently.
            async with app.state.runtime_lock:
                connected = await _run_turn(
                    websocket=websocket,
                    runtime=app.state.stream_runtime,
                    inference_cfg=app.state.inference_cfg,
                    session_id=session_id,
                    request=request,
                    max_audio_bytes=int(cfg.max_audio_bytes),
                )
            if not connected:
                return

    return app


def _parse_args() -> argparse.Namespace:
    """Parse minimal CLI flags for launching the WebSocket server."""
    parser = argparse.ArgumentParser(description="Pom Phase B WebSocket server")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/streaming.yaml",
        help="Path to streaming server YAML config",
    )
    return parser.parse_args()


def main() -> None:
    """Load config, build app once, and run uvicorn server."""
    args = _parse_args()
    cfg = load_streaming_server_config(path=args.config)
    app = create_app(cfg=cfg)
    uvicorn.run(
        app,
        host=cfg.host,
        port=int(cfg.port),
        ws_max_size=int(cfg.ws_max_frame_bytes),
    )


if __name__ == "__main__":
    main()
