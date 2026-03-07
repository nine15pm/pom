"""Tiny streaming runtime that orchestrates generation/decode processes."""

from __future__ import annotations

import multiprocessing as mp
from queue import Empty
import time
from typing import Any, Generator

from inference.decode_process import run_decode_process
from inference.generation_process import run_generation_process
from inference.stream_protocol import (
    DecodeChunk,
    DecodeProcessEvent,
    DecodeTurnChunk,
    DecodeTurnEnd,
    DecodeTurnStart,
    GenerationChunk,
    GenerationProcessEvent,
    SessionClear,
    Shutdown,
    TurnDone,
    TurnCancel,
    TurnStart,
    WorkerError,
    WorkerReady,
)


class StreamRuntime:
    """Own generation/decode worker processes and route turn events."""

    def __init__(self, *, cfg: dict[str, Any], max_history_turns: int) -> None:
        """Store config and initialize process handles."""
        self.cfg = cfg
        self.max_history_turns = int(max_history_turns)
        self._ctx = mp.get_context("spawn")
        self._gen_req: Any = None
        self._gen_ctl: Any = None
        self._gen_evt: Any = None
        self._dec_req: Any = None
        self._dec_evt: Any = None
        self._gen_proc: mp.Process | None = None
        self._dec_proc: mp.Process | None = None

    def start(self) -> None:
        """Start worker processes exactly once."""
        if self._gen_proc is not None or self._dec_proc is not None:
            raise RuntimeError("stream runtime already started")
        self._gen_req = self._ctx.Queue()
        self._gen_ctl = self._ctx.Queue()
        self._gen_evt = self._ctx.Queue()
        self._dec_req = self._ctx.Queue()
        self._dec_evt = self._ctx.Queue()
        self._gen_proc = self._ctx.Process(
            target=run_generation_process,
            kwargs={
                "cfg": self.cfg,
                "max_history_turns": int(self.max_history_turns),
                "request_queue": self._gen_req,
                "event_queue": self._gen_evt,
                "control_queue": self._gen_ctl,
            },
            daemon=True,
        )
        self._dec_proc = self._ctx.Process(
            target=run_decode_process,
            kwargs={
                "cfg": self.cfg,
                "request_queue": self._dec_req,
                "event_queue": self._dec_evt,
            },
            daemon=True,
        )
        self._gen_proc.start()
        self._dec_proc.start()
        # Block startup until workers report ready (or fail fast with clear errors).
        self._wait_for_worker_ready(stage="generation")
        self._wait_for_worker_ready(stage="decode")

    def _wait_for_worker_ready(self, *, stage: str, timeout_s: float = 30.0) -> None:
        """Wait for one worker to report startup success or startup failure."""
        if stage == "generation":
            event_queue = self._gen_evt
            proc = self._gen_proc
        elif stage == "decode":
            event_queue = self._dec_evt
            proc = self._dec_proc
        else:
            raise ValueError(f"unsupported worker stage: {stage}")
        if event_queue is None or proc is None:
            raise RuntimeError("stream runtime is not started")

        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if proc.exitcode is not None:
                raise RuntimeError(f"{stage} worker exited during startup (exit_code={proc.exitcode})")
            timeout = min(0.25, max(0.0, deadline - time.monotonic()))
            try:
                event = event_queue.get(timeout=timeout)
            except Empty:
                continue
            if isinstance(event, WorkerError):
                raise RuntimeError(f"{stage} worker failed during startup: {event.message}")
            if isinstance(event, WorkerReady) and event.stage == stage:
                return
        raise RuntimeError(f"{stage} worker did not report ready within {timeout_s:.1f}s")

    def _raise_if_worker_dead(self) -> None:
        """Abort streaming immediately if either worker process has died."""
        if self._gen_proc is None or self._dec_proc is None:
            raise RuntimeError("stream runtime is not started")
        if self._gen_proc.exitcode is not None:
            raise RuntimeError(f"generation worker exited unexpectedly (exit_code={self._gen_proc.exitcode})")
        if self._dec_proc.exitcode is not None:
            raise RuntimeError(f"decode worker exited unexpectedly (exit_code={self._dec_proc.exitcode})")

    def close(self) -> None:
        """Stop workers and clean up process resources."""
        if self._gen_req is not None:
            self._gen_req.put(Shutdown())
        if self._dec_req is not None:
            self._dec_req.put(Shutdown())
        for proc in (self._gen_proc, self._dec_proc):
            if proc is None:
                continue
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        self._gen_proc = None
        self._dec_proc = None
        self._gen_ctl = None

    def clear_session(self, *, session_id: str) -> None:
        """Clear one generation-session history."""
        if self._gen_req is None:
            raise RuntimeError("stream runtime is not started")
        self._gen_req.put(SessionClear(session_id=str(session_id)))

    def cancel_turn(self, *, session_id: str, turn_id: str) -> None:
        """Cancel one in-flight turn and force decode-side cleanup."""
        if self._gen_ctl is not None:
            self._gen_ctl.put(TurnCancel(session_id=str(session_id), turn_id=str(turn_id)))
        if self._dec_req is not None:
            self._dec_req.put(DecodeTurnEnd(turn_id=str(turn_id)))

    def _drain_decode_events(self, *, turn_id: str) -> tuple[list[DecodeChunk], bool]:
        """Drain decode events for one turn without blocking."""
        chunks: list[DecodeChunk] = []
        decode_done = False
        if self._dec_evt is None:
            return chunks, decode_done
        while True:
            try:
                event = self._dec_evt.get_nowait()
            except Empty:
                break
            if isinstance(event, WorkerError):
                if event.turn_id is None or str(event.turn_id) == str(turn_id):
                    raise RuntimeError(f"decode worker failed: {event.message}")
                continue
            if isinstance(event, WorkerReady):
                continue
            if isinstance(event, DecodeTurnEnd):
                if str(event.turn_id) == str(turn_id):
                    decode_done = True
                continue
            if isinstance(event, DecodeChunk):
                if str(event.turn_id) == str(turn_id):
                    chunks.append(event)
                continue
        return chunks, decode_done

    def stream_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        audio_bytes: bytes,
        decode_audio: bool,
        text_generation_overrides: dict[str, object] | None = None,
        speech_generation_overrides: dict[str, object] | None = None,
        clear_conversation: bool = False,
    ) -> Generator[GenerationChunk | DecodeChunk | TurnDone, None, None]:
        """Run one turn and stream generation/audio events from both workers."""
        if self._gen_req is None or self._gen_evt is None:
            raise RuntimeError("stream runtime is not started")
        if self._dec_req is None:
            raise RuntimeError("stream runtime is not started")
        # Refuse to start turns when worker state is already unhealthy.
        self._raise_if_worker_dead()

        request = TurnStart(
            session_id=str(session_id),
            turn_id=str(turn_id),
            audio_bytes=bytes(audio_bytes),
            decode_audio=bool(decode_audio),
            text_generation_overrides=text_generation_overrides,
            speech_generation_overrides=speech_generation_overrides,
            clear_conversation=bool(clear_conversation),
        )
        self._gen_req.put(request)
        if bool(decode_audio):
            self._dec_req.put(DecodeTurnStart(turn_id=str(turn_id), enabled=True))

        generation_done = False
        decode_done = not bool(decode_audio)
        # Hold turn.done until decode is fully drained so audio tail is never truncated.
        pending_turn_done: TurnDone | None = None
        try:
            while not (generation_done and decode_done and pending_turn_done is None):
                # Fail fast on child process crash instead of waiting forever on queue timeouts.
                self._raise_if_worker_dead()
                # Pull one generation event with a short timeout to keep decode draining responsive.
                try:
                    event: GenerationProcessEvent | None = self._gen_evt.get(timeout=0.01)
                except Empty:
                    event = None

                if isinstance(event, WorkerError):
                    if event.turn_id is None or str(event.turn_id) == str(turn_id):
                        raise RuntimeError(f"generation worker failed: {event.message}")
                elif isinstance(event, WorkerReady):
                    # Ignore any stale startup marker that arrives during turn streaming.
                    pass
                elif isinstance(event, GenerationChunk):
                    if str(event.turn_id) != str(turn_id):
                        continue
                    if bool(decode_audio):
                        self._dec_req.put(DecodeTurnChunk(chunk=event))
                    yield event
                elif isinstance(event, TurnDone):
                    if str(event.turn_id) != str(turn_id):
                        continue
                    generation_done = True
                    pending_turn_done = event

                # Drain ready decode events each loop so decode queue never backs up.
                decode_events, decode_finished = self._drain_decode_events(turn_id=str(turn_id))
                decode_done = decode_done or bool(decode_finished)
                for decode_event in decode_events:
                    yield decode_event
                # Emit terminal turn.done only after decode has signaled completion.
                if decode_done and pending_turn_done is not None:
                    yield pending_turn_done
                    pending_turn_done = None
        finally:
            # Always close decode turn state even if streaming exits early.
            if bool(decode_audio):
                self._dec_req.put(DecodeTurnEnd(turn_id=str(turn_id)))


__all__ = ["StreamRuntime"]
