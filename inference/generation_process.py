"""Generation process with in-file generation logic and process loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty
from typing import Any, Callable, Generator

import torch

from inference.inference import (
    apply_repetition_penalty,
    load_audio_bytes,
    resolve_speech_decode_config,
    resolve_text_decode_config,
    sample_next_token,
    talker_step_with_embeds,
    talker_step_with_token,
)
from inference.loader import load_runtime
from inference.stream_protocol import (
    GenerationChunk,
    GenerationFinalState,
    GenerationProcessEvent,
    SessionClear,
    Shutdown,
    TurnDone,
    TurnCancel,
    TurnStart,
    WorkerError,
    WorkerReady,
)


@dataclass
class _HistoryTurn:
    """Store one prior user feature and assistant text for multi-turn prompts."""

    user_speech_feature: torch.Tensor
    assistant_text: str


@dataclass(frozen=True)
class _PreparedTurn:
    """Bundle one fully prepared generation turn."""

    device: torch.device
    current_user_feature: torch.Tensor
    speech_features: list[torch.Tensor]
    prompt_ids: list[int]
    control_ids: set[int]
    text_decode_cfg: Any
    speech_decode_cfg: Any
    thinker_rng: torch.Generator
    talker_rng: torch.Generator


@dataclass
class _ThinkerState:
    """Track Thinker incremental KV decode state."""

    past_key_values: Any
    last_logits: torch.Tensor
    context_len: int
    steps: int = 0
    stop_seen: bool = False


@dataclass
class _TalkerState:
    """Track Talker interleaved read/write decode state."""

    speech_offset: int
    min_speech_id: int
    max_speech_id: int
    eos_id: int
    sep_embed: torch.Tensor
    past_key_values: Any = None
    context_len: int = 0
    last_logits: torch.Tensor | None = None
    sep_added: bool = False
    eos_seen: bool = False
    stop_for_repeat: bool = False
    repeat_run: int = 0
    last_unit_id: int | None = None
    generated_lm_ids: list[int] = field(default_factory=list)
    speech_unit_ids: list[int] = field(default_factory=list)
    pending_fused_chunks: list[torch.Tensor] = field(default_factory=list)
    pending_chunk_offset: int = 0
    pending_unread_rows: int = 0
    total_fused_rows: int = 0
    total_read_rows: int = 0


def _build_multi_turn_prompt(
    tokenizer: Any,
    *,
    speech_id: int,
    assistant_history: list[str],
) -> list[int]:
    """Build prompt ids for prior assistant turns plus current user speech turn."""
    messages = [{"role": "system", "content": ""}]
    for assistant_text in assistant_history:
        messages.append({"role": "user", "content": "<speech>"})
        messages.append({"role": "assistant", "content": str(assistant_text)})
    messages.append({"role": "user", "content": "<speech>"})
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = [int(token_id) for token_id in encoded["input_ids"]]
    expected_speech_segments = len(assistant_history) + 1
    speech_count = sum(1 for token_id in prompt_ids if int(token_id) == int(speech_id))
    if speech_count != expected_speech_segments:
        raise ValueError(
            f"expected exactly {expected_speech_segments} <speech> tokens in prompt, got {speech_count}"
        )
    return prompt_ids


def _trim_projected_feature(*, projected: torch.Tensor, projected_mask: torch.Tensor | None) -> torch.Tensor:
    """Trim one projected speech row to valid frames."""
    if projected.ndim != 3 or int(projected.size(0)) != 1:
        raise RuntimeError("speech projector returned invalid shape for single-turn encode")
    row = projected[0]
    if projected_mask is None:
        valid_len = int(row.size(0))
    else:
        if projected_mask.ndim != 2 or int(projected_mask.size(0)) != 1:
            raise RuntimeError("speech projector returned invalid mask shape for single-turn encode")
        valid_len = int(projected_mask[0].sum().item())
    if valid_len <= 0:
        raise ValueError("speech encoder returned zero valid frames for user audio")
    return row[:valid_len].detach()


def _encode_user_speech_feature(*, runtime: Any, waveform: torch.Tensor, sampling_rate: int) -> torch.Tensor:
    """Encode one waveform into one projected speech feature row."""
    encoded, frame_masks = runtime.thinker.encode_speech([waveform], sampling_rate=[int(sampling_rate)])
    projected, projected_mask = runtime.thinker.project_speech_features(encoded, frame_masks)
    return _trim_projected_feature(projected=projected, projected_mask=projected_mask)


def _build_stage_rngs(*, device: torch.device, base_seed: int, turn_index: int) -> tuple[torch.Generator, torch.Generator]:
    """Build independent Thinker/Talker RNG streams for one turn."""
    thinker_rng = torch.Generator(device=device)
    talker_rng = torch.Generator(device=device)
    thinker_rng.manual_seed(int(base_seed) + int(turn_index) * 2 + 1)
    talker_rng.manual_seed(int(base_seed) + int(turn_index) * 2 + 2)
    return thinker_rng, talker_rng


class GenerationSession:
    """Run generation turns for one session id with in-memory turn history."""

    def __init__(self, *, runtime: Any, cfg: dict[str, Any], max_history_turns: int) -> None:
        """Store runtime/config and initialize per-session mutable history."""
        self.runtime = runtime
        self.cfg = cfg
        self.max_history_turns = int(max_history_turns)
        self._turn_index = 0
        self._history: list[_HistoryTurn] = []

    def clear(self) -> None:
        """Reset one session history and turn counter."""
        self._history.clear()
        self._turn_index = 0

    def _assistant_history(self) -> list[str]:
        """Return assistant text history in chronological order."""
        return [turn.assistant_text for turn in self._history]

    def _speech_feature_history(self) -> list[torch.Tensor]:
        """Return cached user speech features in chronological order."""
        return [turn.user_speech_feature for turn in self._history]

    def _append_history(self, *, user_feature: torch.Tensor, assistant_text: str) -> None:
        """Append one finished turn and enforce history window size."""
        self._history.append(
            _HistoryTurn(user_speech_feature=user_feature.detach(), assistant_text=str(assistant_text))
        )
        if len(self._history) > int(self.max_history_turns):
            self._history = self._history[-int(self.max_history_turns) :]

    def _prepare_turn(self, *, request: TurnStart) -> _PreparedTurn:
        """Prepare one turn from audio bytes plus current session history."""
        device = self.runtime.device
        text_decode_cfg = resolve_text_decode_config(self.cfg, request.text_generation_overrides)
        speech_decode_cfg = resolve_speech_decode_config(self.cfg, request.speech_generation_overrides)
        # Keep control ids explicit so only content text is fused into Talker.
        control_ids = {int(token_id) for token_id in getattr(self.runtime.tokenizer, "all_special_ids", [])}
        control_ids.add(int(self.runtime.token_contract.assistant_stop_id))

        input_wav, sampling_rate = load_audio_bytes(request.audio_bytes)
        input_wav = input_wav.to(device=device)
        current_user_feature = _encode_user_speech_feature(
            runtime=self.runtime,
            waveform=input_wav,
            sampling_rate=int(sampling_rate),
        )
        speech_features = self._speech_feature_history() + [current_user_feature]
        prompt_ids = _build_multi_turn_prompt(
            self.runtime.tokenizer,
            speech_id=int(self.runtime.token_contract.speech_id),
            assistant_history=self._assistant_history(),
        )
        thinker_rng, talker_rng = _build_stage_rngs(
            device=device,
            base_seed=int(self.cfg["runtime"]["seed"]),
            turn_index=int(self._turn_index),
        )
        return _PreparedTurn(
            device=device,
            current_user_feature=current_user_feature,
            speech_features=speech_features,
            prompt_ids=prompt_ids,
            control_ids=control_ids,
            text_decode_cfg=text_decode_cfg,
            speech_decode_cfg=speech_decode_cfg,
            thinker_rng=thinker_rng,
            talker_rng=talker_rng,
        )

    def _init_thinker_state(self, *, prepared: _PreparedTurn) -> _ThinkerState:
        """Run Thinker prefill and return incremental Thinker state."""
        input_ids = torch.tensor([prepared.prompt_ids], dtype=torch.long, device=prepared.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        outputs = self.runtime.thinker(
            input_ids=input_ids,
            attention_mask=attention_mask,
            speech_features=prepared.speech_features,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        if outputs.past_key_values is None:
            raise RuntimeError("thinker forward did not return past_key_values")
        return _ThinkerState(
            past_key_values=outputs.past_key_values,
            last_logits=outputs.logits[0, -1, :].float(),
            context_len=int(outputs.logits.shape[1]),
        )

    def _step_thinker_once(self, *, prepared: _PreparedTurn, state: _ThinkerState) -> tuple[int, torch.Tensor]:
        """Generate one Thinker token and aligned hidden row."""
        next_token_id = sample_next_token(
            logits=state.last_logits,
            temperature=float(prepared.text_decode_cfg.temperature),
            top_p=float(prepared.text_decode_cfg.top_p),
            rng=prepared.thinker_rng,
        )
        step_ids = torch.tensor([[int(next_token_id)]], dtype=torch.long, device=prepared.device)
        state.context_len += 1
        step_attention = torch.ones((1, state.context_len), dtype=torch.long, device=prepared.device)
        outputs = self.runtime.thinker(
            input_ids=step_ids,
            attention_mask=step_attention,
            past_key_values=state.past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        if outputs.past_key_values is None:
            raise RuntimeError("thinker decode step did not return past_key_values")
        state.past_key_values = outputs.past_key_values
        state.last_logits = outputs.logits[0, -1, :].float()
        state.steps += 1
        if int(next_token_id) == int(self.runtime.token_contract.assistant_stop_id):
            state.stop_seen = True
        return int(next_token_id), outputs.hidden_states[-1][0, -1, :].detach()

    def _init_talker_state(self, *, prepared: _PreparedTurn) -> _TalkerState:
        """Build Talker state and speech-token metadata."""
        speech_offset = self.runtime.talker.speech_token_offset
        speech_vocab_size = self.runtime.talker.speech_vocab_size
        if speech_offset is None or int(speech_vocab_size) <= 0:
            raise ValueError("talker speech token mapping is not configured")
        eos_id = int(self.runtime.token_contract.eos_id)
        sep_id = int(self.runtime.token_contract.sep_id)
        sep_embed = self.runtime.talker.get_input_embeddings()(
            torch.tensor([sep_id], dtype=torch.long, device=prepared.device)
        ).squeeze(0)
        return _TalkerState(
            speech_offset=int(speech_offset),
            min_speech_id=int(speech_offset),
            max_speech_id=int(speech_offset) + int(speech_vocab_size) - 1,
            eos_id=eos_id,
            sep_embed=sep_embed,
        )

    def _append_fused_conditioning(
        self,
        *,
        prepared: _PreparedTurn,
        state: _TalkerState,
        hidden_rows: list[torch.Tensor],
        content_token_ids: list[int],
    ) -> None:
        """Fuse one text chunk and enqueue Talker conditioning rows."""
        if not hidden_rows or not content_token_ids:
            return
        if len(hidden_rows) != len(content_token_ids):
            raise RuntimeError("content ids and hidden rows must stay aligned")
        aligned_hidden = torch.stack(hidden_rows, dim=0)
        content_ids = torch.tensor(content_token_ids, dtype=torch.long, device=prepared.device)
        fused_rows = self.runtime.talker.fuse(hidden_rows=[aligned_hidden], content_ids=[content_ids])
        if len(fused_rows) != 1:
            raise RuntimeError(f"expected exactly one fused row, got {len(fused_rows)}")
        fused = fused_rows[0]
        if fused.ndim != 2 or int(fused.size(0)) <= 0:
            raise ValueError("talker fuse returned invalid conditioning row")
        # Keep chunks queued so conditioning append stays linear-time.
        state.pending_fused_chunks.append(fused)
        rows = int(fused.size(0))
        state.pending_unread_rows += rows
        state.total_fused_rows += rows

    def _pop_fused_rows(self, *, state: _TalkerState, take_rows: int) -> torch.Tensor:
        """Pop exactly N unread fused rows from the queue."""
        if int(take_rows) <= 0:
            raise ValueError("take_rows must be > 0")
        if int(take_rows) > int(state.pending_unread_rows):
            raise ValueError("cannot pop more rows than pending unread conditioning")
        pieces: list[torch.Tensor] = []
        remaining = int(take_rows)
        while remaining > 0:
            if not state.pending_fused_chunks:
                raise RuntimeError("pending conditioning chunks are unexpectedly empty")
            head = state.pending_fused_chunks[0]
            start = int(state.pending_chunk_offset)
            available = int(head.size(0)) - start
            if available <= 0:
                raise RuntimeError("pending conditioning chunk offset is out of range")
            take = min(remaining, available)
            pieces.append(head[start : start + take])
            remaining -= take
            state.pending_unread_rows -= take
            if take == available:
                state.pending_fused_chunks.pop(0)
                state.pending_chunk_offset = 0
            else:
                state.pending_chunk_offset = start + take
        if len(pieces) == 1:
            return pieces[0]
        return torch.cat(pieces, dim=0)

    def _run_talker_round(self, *, prepared: _PreparedTurn, state: _TalkerState, thinker_finished: bool) -> None:
        """Run one Talker read/write round and append new units to state."""
        if state.eos_seen or state.stop_for_repeat:
            return
        if len(state.speech_unit_ids) >= int(prepared.speech_decode_cfg.max_new_tokens):
            return
        read_happened = False
        should_read = (
            int(state.pending_unread_rows) >= int(prepared.speech_decode_cfg.read_length)
            or (thinker_finished and int(state.pending_unread_rows) > 0)
        )
        if should_read:
            read_take = min(int(prepared.speech_decode_cfg.read_length), int(state.pending_unread_rows))
            read_chunk = self._pop_fused_rows(state=state, take_rows=read_take)
            state.past_key_values, state.context_len, state.last_logits = talker_step_with_embeds(
                talker=self.runtime.talker,
                chunk_embeds=read_chunk,
                past_key_values=state.past_key_values,
                context_len=state.context_len,
                device=prepared.device,
            )
            state.total_read_rows += read_take
            read_happened = True
        # Append <sep> only after all text conditioning is consumed.
        if thinker_finished and int(state.pending_unread_rows) <= 0 and not state.sep_added:
            state.past_key_values, state.context_len, state.last_logits = talker_step_with_embeds(
                talker=self.runtime.talker,
                chunk_embeds=state.sep_embed.unsqueeze(0),
                past_key_values=state.past_key_values,
                context_len=state.context_len,
                device=prepared.device,
            )
            state.sep_added = True
        # During interleaving, write only after a read; after text ends, write to completion.
        if not thinker_finished and not read_happened:
            return
        if state.last_logits is None:
            return
        for _ in range(int(prepared.speech_decode_cfg.write_length)):
            if len(state.speech_unit_ids) >= int(prepared.speech_decode_cfg.max_new_tokens):
                break
            if state.eos_seen or state.stop_for_repeat:
                break
            masked_logits = torch.full_like(state.last_logits, float("-inf"))
            masked_logits[state.min_speech_id : state.max_speech_id + 1] = state.last_logits[
                state.min_speech_id : state.max_speech_id + 1
            ]
            masked_logits[state.eos_id] = state.last_logits[state.eos_id]
            apply_repetition_penalty(
                masked_logits,
                state.generated_lm_ids,
                float(prepared.speech_decode_cfg.repetition_penalty),
            )
            next_token_id = sample_next_token(
                masked_logits,
                temperature=float(prepared.speech_decode_cfg.temperature),
                top_p=float(prepared.speech_decode_cfg.top_p),
                rng=prepared.talker_rng,
            )
            if int(next_token_id) == int(state.eos_id):
                state.eos_seen = True
                break
            if int(next_token_id) < int(state.min_speech_id) or int(next_token_id) > int(state.max_speech_id):
                raise ValueError(f"talker generated invalid token id {next_token_id}")
            unit_id = int(next_token_id) - int(state.speech_offset)
            if state.last_unit_id is not None and unit_id == state.last_unit_id:
                state.repeat_run += 1
            else:
                state.repeat_run = 1
                state.last_unit_id = unit_id
            if state.repeat_run > int(prepared.speech_decode_cfg.max_repeat_run):
                state.stop_for_repeat = True
                break
            state.speech_unit_ids.append(unit_id)
            state.generated_lm_ids.append(int(next_token_id))
            state.past_key_values, state.context_len, state.last_logits = talker_step_with_token(
                talker=self.runtime.talker,
                token_id=int(next_token_id),
                past_key_values=state.past_key_values,
                context_len=state.context_len,
                device=prepared.device,
            )

    def _thinker_finished(self, *, prepared: _PreparedTurn, state: _ThinkerState) -> bool:
        """Return whether Thinker reached a stop condition."""
        return bool(state.stop_seen or int(state.steps) >= int(prepared.text_decode_cfg.max_new_tokens))

    def _talker_finished(self, *, prepared: _PreparedTurn, state: _TalkerState) -> bool:
        """Return whether Talker reached a stop condition."""
        return bool(
            state.eos_seen
            or state.stop_for_repeat
            or len(state.speech_unit_ids) >= int(prepared.speech_decode_cfg.max_new_tokens)
        )

    def _run_generation(
        self,
        *,
        turn_id: str,
        prepared: _PreparedTurn,
    ) -> Generator[GenerationChunk, None, GenerationFinalState]:
        """Run one interleaved generation loop and emit cumulative generation chunks."""
        thinker_state = self._init_thinker_state(prepared=prepared)
        talker_state = self._init_talker_state(prepared=prepared)
        content_token_ids: list[int] = []
        pending_content_ids: list[int] = []
        pending_hidden_rows: list[torch.Tensor] = []
        emitted_text = ""
        emitted_units_count = 0
        while True:
            thinker_finished = self._thinker_finished(prepared=prepared, state=thinker_state)
            talker_finished = self._talker_finished(prepared=prepared, state=talker_state)
            if not thinker_finished:
                next_token_id, hidden_row = self._step_thinker_once(prepared=prepared, state=thinker_state)
                if int(next_token_id) not in prepared.control_ids:
                    content_token_ids.append(int(next_token_id))
                    if not talker_finished:
                        pending_content_ids.append(int(next_token_id))
                        pending_hidden_rows.append(hidden_row)
                        if len(pending_content_ids) >= int(prepared.speech_decode_cfg.read_length):
                            self._append_fused_conditioning(
                                prepared=prepared,
                                state=talker_state,
                                hidden_rows=pending_hidden_rows,
                                content_token_ids=pending_content_ids,
                            )
                            pending_content_ids.clear()
                            pending_hidden_rows.clear()
            thinker_finished = self._thinker_finished(prepared=prepared, state=thinker_state)
            talker_finished = self._talker_finished(prepared=prepared, state=talker_state)
            if thinker_finished and pending_content_ids and not talker_finished:
                self._append_fused_conditioning(
                    prepared=prepared,
                    state=talker_state,
                    hidden_rows=pending_hidden_rows,
                    content_token_ids=pending_content_ids,
                )
                pending_content_ids.clear()
                pending_hidden_rows.clear()
            if talker_finished and pending_content_ids:
                pending_content_ids.clear()
                pending_hidden_rows.clear()
            if thinker_finished and not content_token_ids:
                raise ValueError("thinker generated no plain content tokens")
            if not talker_finished:
                self._run_talker_round(prepared=prepared, state=talker_state, thinker_finished=thinker_finished)
            current_text = self.runtime.tokenizer.decode(content_token_ids, skip_special_tokens=True)
            current_units = [int(unit_id) for unit_id in talker_state.speech_unit_ids]
            if current_text != emitted_text or len(current_units) != int(emitted_units_count):
                emitted_text = current_text
                emitted_units_count = len(current_units)
                yield GenerationChunk(turn_id=turn_id, text=emitted_text, unit_ids=current_units, finalize=False)
            talker_finished = self._talker_finished(prepared=prepared, state=talker_state)
            if thinker_finished and talker_finished:
                break
        if not content_token_ids:
            raise ValueError("thinker generated no plain content tokens")
        if not talker_state.speech_unit_ids:
            raise ValueError("talker generated no speech units")
        final_text = self.runtime.tokenizer.decode(content_token_ids, skip_special_tokens=True)
        final_units = [int(unit_id) for unit_id in talker_state.speech_unit_ids]
        yield GenerationChunk(turn_id=turn_id, text=final_text, unit_ids=final_units, finalize=True)
        queued_conditioning_consumed = bool(int(talker_state.total_read_rows) >= int(talker_state.total_fused_rows))
        return GenerationFinalState(
            assistant_text_token_ids=[int(token_id) for token_id in content_token_ids],
            speech_token_ids=final_units,
            thinker_stop_seen=bool(thinker_state.stop_seen),
            talker_eos_seen=bool(talker_state.eos_seen),
            queued_conditioning_consumed=queued_conditioning_consumed,
        )

    def run_turn(
        self,
        *,
        request: TurnStart,
        is_cancelled: Callable[[], bool],
    ) -> Generator[GenerationProcessEvent, None, None]:
        """Run one turn and emit generation chunks plus terminal done event."""
        if bool(request.clear_conversation):
            self.clear()
        with torch.inference_mode():
            prepared = self._prepare_turn(request=request)
            generation_stream = self._run_generation(turn_id=request.turn_id, prepared=prepared)
            while True:
                # Exit quickly when caller requested cancellation for this turn.
                if bool(is_cancelled()):
                    return
                try:
                    chunk = next(generation_stream)
                except StopIteration as stop:
                    final_state = stop.value
                    if final_state is None:
                        raise RuntimeError("generation returned no final state")
                    assistant_text = self.runtime.tokenizer.decode(
                        final_state.assistant_text_token_ids,
                        skip_special_tokens=True,
                    ).strip()
                    self._append_history(
                        user_feature=prepared.current_user_feature,
                        assistant_text=assistant_text,
                    )
                    self._turn_index += 1
                    yield TurnDone(turn_id=str(request.turn_id), final_state=final_state)
                    return
                yield chunk


def run_generation_process(
    *,
    cfg: dict[str, Any],
    max_history_turns: int,
    request_queue: Any,
    event_queue: Any,
    control_queue: Any,
) -> None:
    """Run a blocking generation worker loop on one dedicated process."""
    # Fail fast on startup errors so parent runtime does not hang waiting forever.
    try:
        # Generation worker intentionally skips decoder assets (decode is a separate process).
        runtime = load_runtime(cfg, load_decoder=False)
    except Exception as exc:
        event_queue.put(WorkerError(stage="generation", message=f"startup failed: {exc}"))
        return
    event_queue.put(WorkerReady(stage="generation"))
    sessions: dict[str, GenerationSession] = {}

    def _drain_cancel_for_turn(*, session_id: str, turn_id: str) -> bool:
        """Drain cancel queue and return true when current turn was cancelled."""
        cancelled = False
        while True:
            try:
                control = control_queue.get_nowait()
            except Empty:
                break
            if isinstance(control, TurnCancel):
                # Match cancel strictly on session+turn so reused turn_ids cannot cross-cancel.
                if str(control.session_id) == str(session_id) and str(control.turn_id) == str(turn_id):
                    cancelled = True
        return cancelled

    while True:
        request = request_queue.get()
        if isinstance(request, Shutdown):
            return
        if isinstance(request, SessionClear):
            session = sessions.get(request.session_id)
            if session is not None:
                session.clear()
            continue
        if not isinstance(request, TurnStart):
            event_queue.put(WorkerError(stage="generation", message=f"unsupported request type: {type(request).__name__}"))
            continue
        session = sessions.get(request.session_id)
        if session is None:
            session = GenerationSession(runtime=runtime, cfg=cfg, max_history_turns=int(max_history_turns))
            sessions[request.session_id] = session

        def _is_turn_cancelled() -> bool:
            """Return true when this active turn has a pending cancel request."""
            return _drain_cancel_for_turn(
                session_id=str(request.session_id),
                turn_id=str(request.turn_id),
            )

        try:
            for event in session.run_turn(request=request, is_cancelled=_is_turn_cancelled):
                event_queue.put(event)
        except Exception as exc:
            event_queue.put(
                WorkerError(
                    stage="generation",
                    message=str(exc),
                    turn_id=str(request.turn_id),
                )
            )
__all__ = ["GenerationSession", "run_generation_process"]
