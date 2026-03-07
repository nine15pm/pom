const statusEl = document.getElementById("status");
const recordBtn = document.getElementById("record-btn");
const resetBtn = document.getElementById("reset-btn");
const turnListEl = document.getElementById("turn-list");

const state = {
  socket: null,
  socketReady: false,
  busy: false,
  recording: false,
  recordingPending: false,
  recordIntentToken: 0,
  activeTurnId: null,
  mediaStream: null,
  mediaRecorder: null,
  recordChunks: [],
  recordingStartMs: 0,
  websocketPath: "/ws",
  audioContext: null,
  audioTurnId: null,
  audioLastSeq: -1,
  audioNextTime: 0,
  turnRows: new Map(),
};

// Update the top-line app status text in one place.
function setStatus(text) {
  statusEl.textContent = `Status: ${text}`;
}

// Enable or disable controls from one state snapshot.
function syncControls() {
  recordBtn.disabled = !state.socketReady || state.busy;
  resetBtn.disabled = !state.socketReady || state.busy || state.recording || state.recordingPending;
  recordBtn.textContent = state.recording || state.recordingPending ? "Release To Send" : "Hold To Talk";
  recordBtn.classList.toggle("recording", state.recording || state.recordingPending);
}

// Build one websocket URL using the current page origin and configured path.
function websocketUrl(path) {
  const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${wsProto}//${window.location.host}${path}`;
}

// Generate one unique turn id for request/response correlation.
function makeTurnId() {
  return `${Date.now()}-${Math.floor(Math.random() * 1e9)}`;
}

// Keep one row pair per turn so streamed updates replace cleanly.
function getOrCreateTurnRows(turnId) {
  const existing = state.turnRows.get(turnId);
  if (existing) {
    return existing;
  }

  const userRow = document.createElement("li");
  userRow.textContent = "You: [recording will appear here]";
  turnListEl.appendChild(userRow);

  const assistantRow = document.createElement("li");
  assistantRow.textContent = "Pom: …";
  turnListEl.appendChild(assistantRow);

  const rows = { userRow, assistantRow };
  state.turnRows.set(turnId, rows);
  return rows;
}

// Render one user turn line for the current request.
function renderUserText(turnId, text) {
  const rows = getOrCreateTurnRows(turnId);
  rows.userRow.textContent = `You: ${text || ""}`;
}

// Render one assistant text snapshot for a turn.
function renderAssistantText(turnId, text) {
  const rows = getOrCreateTurnRows(turnId);
  rows.assistantRow.textContent = `Pom: ${text || ""}`;
}

// Clear all rendered conversation rows from the app state and DOM.
function clearTranscript() {
  state.turnRows.clear();
  turnListEl.innerHTML = "";
}

// Mark one turn as finished and unlock controls.
function finishTurn(statusText) {
  state.busy = false;
  state.activeTurnId = null;
  state.audioTurnId = null;
  state.audioLastSeq = -1;
  state.audioNextTime = 0;
  syncControls();
  setStatus(statusText);
}

// Mark one turn as the active audio stream and reset chunk ordering state.
function beginTurnAudio(turnId) {
  state.audioTurnId = turnId;
  state.audioLastSeq = -1;
  state.audioNextTime = 0;
}

// Create or resume one shared audio output context for streamed playback.
async function getOrCreateAudioContext() {
  if (!state.audioContext) {
    state.audioContext = new AudioContext();
  }
  if (state.audioContext.state === "suspended") {
    await state.audioContext.resume();
  }
  return state.audioContext;
}

// Decode one base64 PCM16 payload into normalized float samples.
function decodePcm16Base64(audioB64) {
  const binary = atob(audioB64);
  const byteLength = binary.length;
  const sampleCount = Math.floor(byteLength / 2);
  const samples = new Float32Array(sampleCount);
  for (let index = 0; index < sampleCount; index += 1) {
    const lo = binary.charCodeAt(index * 2);
    const hi = binary.charCodeAt(index * 2 + 1);
    let value = (hi << 8) | lo;
    if (value >= 0x8000) {
      value -= 0x10000;
    }
    samples[index] = value / 32768;
  }
  return samples;
}

// Queue one streamed audio chunk at the next continuous playback timestamp.
async function playAudioChunk(message) {
  if (message.turn_id !== state.audioTurnId) {
    return;
  }
  const seq = Number(message.seq);
  if (!Number.isFinite(seq) || seq <= state.audioLastSeq) {
    return;
  }
  state.audioLastSeq = seq;

  if (message.encoding !== "pcm_s16le") {
    return;
  }
  const sampleRate = Number(message.sample_rate);
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    return;
  }
  const audioB64 = typeof message.audio_b64 === "string" ? message.audio_b64 : "";
  if (!audioB64) {
    return;
  }

  const context = await getOrCreateAudioContext();
  const samples = decodePcm16Base64(audioB64);
  if (samples.length <= 0) {
    return;
  }

  const buffer = context.createBuffer(1, samples.length, sampleRate);
  buffer.copyToChannel(samples, 0);
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);

  const now = context.currentTime;
  const startAt = Math.max(now, state.audioNextTime || now);
  source.start(startAt);
  state.audioNextTime = startAt + buffer.duration;
}

// Handle one parsed server message with explicit type routing.
async function handleServerMessage(message) {
  const messageType = message.type;
  if (typeof messageType !== "string") {
    return;
  }

  if (messageType === "text.delta") {
    if (message.turn_id !== state.activeTurnId) {
      return;
    }
    renderAssistantText(message.turn_id, String(message.text || ""));
    return;
  }

  if (messageType === "turn.done") {
    if (message.turn_id !== state.activeTurnId) {
      return;
    }
    renderAssistantText(message.turn_id, String(message.assistant_text || ""));
    finishTurn(`connected (turn done: ${String(message.stop_reason || "ok")})`);
    return;
  }

  if (messageType === "audio.chunk") {
    if (message.turn_id !== state.activeTurnId) {
      return;
    }
    await playAudioChunk(message);
    return;
  }

  if (messageType === "session.cleared") {
    setStatus("connected (conversation cleared)");
    return;
  }

  if (messageType === "error") {
    const turnId = typeof message.turn_id === "string" ? message.turn_id : state.activeTurnId;
    if (turnId) {
      renderAssistantText(turnId, `[error] ${String(message.message || "unknown error")}`);
    }
    finishTurn("connected (last turn failed)");
  }
}

// Send one turn.start request over websocket for the next audio clip.
function sendTurnStart({ audioB64, userText }) {
  if (!state.socket || !state.socketReady) {
    throw new Error("socket is not connected");
  }
  if (state.busy) {
    throw new Error("turn already in progress");
  }
  if (typeof audioB64 !== "string" || audioB64.length === 0) {
    throw new Error("audioB64 must be a non-empty string");
  }

  const turnId = makeTurnId();
  state.busy = true;
  state.activeTurnId = turnId;
  beginTurnAudio(turnId);
  syncControls();
  setStatus("connected (running turn)");
  renderUserText(turnId, userText || "[voice turn]");
  renderAssistantText(turnId, "…");

  const payload = {
    type: "turn.start",
    turn_id: turnId,
    audio_b64: audioB64,
    decode_audio: true,
  };
  state.socket.send(JSON.stringify(payload));
}

// Increment and return one recording intent token for this pointer/key press.
function beginRecordIntent() {
  state.recordIntentToken += 1;
  state.recordingPending = true;
  syncControls();
  return state.recordIntentToken;
}

// Return true only when the current async flow still matches live press intent.
function isIntentActive(intentToken) {
  return intentToken === state.recordIntentToken;
}

// Pick a recorder mime type the browser advertises for speech capture.
function pickRecorderMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg",
  ];
  for (const mimeType of candidates) {
    if (MediaRecorder.isTypeSupported(mimeType)) {
      return mimeType;
    }
  }
  return "";
}

// Convert one Uint8Array byte buffer into a base64 string.
function bytesToBase64(bytes) {
  let binary = "";
  for (let index = 0; index < bytes.length; index += 1) {
    binary += String.fromCharCode(bytes[index]);
  }
  return btoa(binary);
}

// Encode one mono float32 waveform into little-endian PCM16 WAV bytes.
function encodeWavPcm16({ mono, sampleRate }) {
  const blockAlign = 2;
  const bitsPerSample = 16;
  const byteRate = sampleRate * blockAlign;
  const dataBytes = mono.length * 2;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);

  // Write ASCII text into the WAV header.
  function writeAscii(offset, text) {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index));
    }
  }

  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeAscii(36, "data");
  view.setUint32(40, dataBytes, true);

  // Quantize float32 samples into signed PCM16 payload samples.
  let offset = 44;
  for (let index = 0; index < mono.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, mono[index]));
    const value = clamped < 0 ? clamped * 32768 : clamped * 32767;
    view.setInt16(offset, Math.round(value), true);
    offset += 2;
  }
  return new Uint8Array(buffer);
}

// Downmix one AudioBuffer into a mono Float32Array.
function audioBufferToMono(buffer) {
  const mono = new Float32Array(buffer.length);
  const channels = buffer.numberOfChannels;
  for (let channel = 0; channel < channels; channel += 1) {
    const channelData = buffer.getChannelData(channel);
    for (let index = 0; index < buffer.length; index += 1) {
      mono[index] += channelData[index] / channels;
    }
  }
  return mono;
}

// Decode a recorded browser blob and return WAV payload as base64.
async function blobToWavBase64(blob) {
  const rawBytes = await blob.arrayBuffer();
  const audioContext = new AudioContext();
  try {
    const decoded = await audioContext.decodeAudioData(rawBytes.slice(0));
    const mono = audioBufferToMono(decoded);
    const wavBytes = encodeWavPcm16({ mono, sampleRate: decoded.sampleRate });
    return bytesToBase64(wavBytes);
  } finally {
    await audioContext.close();
  }
}

// Lazily request microphone access and cache the stream for later turns.
async function getOrCreateMicrophone() {
  if (state.mediaStream) {
    return state.mediaStream;
  }
  state.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  return state.mediaStream;
}

// Start one push-to-talk recording session and capture chunks in memory.
async function startRecording(intentToken) {
  if (!state.socketReady || state.busy || state.recording || !isIntentActive(intentToken)) {
    return;
  }
  // Prime audio output in a user gesture path to avoid autoplay restrictions later.
  await getOrCreateAudioContext();
  if (!isIntentActive(intentToken)) {
    state.recordingPending = false;
    syncControls();
    return;
  }
  const stream = await getOrCreateMicrophone();
  if (!isIntentActive(intentToken)) {
    state.recordingPending = false;
    syncControls();
    return;
  }
  const mimeType = pickRecorderMimeType();
  const recorder = mimeType
    ? new MediaRecorder(stream, { mimeType })
    : new MediaRecorder(stream);
  if (!isIntentActive(intentToken)) {
    state.recordingPending = false;
    syncControls();
    return;
  }

  state.recordChunks = [];
  state.mediaRecorder = recorder;
  state.recordingPending = false;
  state.recording = true;
  state.recordingStartMs = Date.now();
  syncControls();
  setStatus("connected (recording...)");

  recorder.addEventListener("dataavailable", (event) => {
    if (event.data && event.data.size > 0) {
      state.recordChunks.push(event.data);
    }
  });

  recorder.addEventListener("stop", async () => {
    // Finalize one recorded utterance and send it as turn.start.
    const durationMs = Math.max(0, Date.now() - state.recordingStartMs);
    state.mediaRecorder = null;
    setStatus("connected (preparing audio)");

    try {
      const blob = new Blob(state.recordChunks, { type: recorder.mimeType || "audio/webm" });
      if (blob.size <= 0) {
        setStatus("connected (empty recording, try again)");
        return;
      }
      const audioB64 = await blobToWavBase64(blob);
      const durationSec = (durationMs / 1000).toFixed(1);
      sendTurnStart({
        audioB64,
        userText: `[voice ${durationSec}s]`,
      });
    } catch (err) {
      setStatus(`connected (record failed: ${String(err)})`);
    } finally {
      state.recordChunks = [];
    }
  });

  recorder.start();
}

// Stop active push-to-talk recording and trigger send on recorder stop.
function stopRecording() {
  // Invalidate any in-flight async setup so release cannot start late recording.
  state.recordIntentToken += 1;
  state.recordingPending = false;
  if (!state.recording) {
    syncControls();
    return;
  }
  state.recording = false;
  syncControls();
  if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
    state.mediaRecorder.stop();
  }
}

// Open websocket and bind event handlers for one app session.
function connectWebSocket() {
  setStatus("connecting websocket...");
  const socket = new WebSocket(websocketUrl(state.websocketPath));
  state.socket = socket;

  socket.addEventListener("open", () => {
    state.socketReady = true;
    syncControls();
    setStatus("connected");
  });

  socket.addEventListener("message", (event) => {
    try {
      const parsed = JSON.parse(String(event.data));
      void handleServerMessage(parsed);
    } catch (_err) {
      setStatus("connected (received invalid message)");
    }
  });

  socket.addEventListener("close", () => {
    state.socketReady = false;
    state.socket = null;
    const wasBusy = state.busy;
    state.busy = false;
    state.activeTurnId = null;
    state.audioTurnId = null;
    state.audioLastSeq = -1;
    state.audioNextTime = 0;
    state.recording = false;
    state.recordingPending = false;
    state.recordIntentToken += 1;
    syncControls();
    setStatus(wasBusy ? "disconnected (turn interrupted)" : "disconnected");
  });

  socket.addEventListener("error", () => {
    setStatus("websocket error");
  });
}

// Keep reset explicit: apply clear at the next turn boundary.
function onResetConversation() {
  if (!state.socketReady) {
    setStatus("disconnected");
    return;
  }
  clearTranscript();
  if (state.socket) {
    state.socket.send(JSON.stringify({ type: "session.clear" }));
    setStatus("connected (clearing conversation)");
  }
}

// Load app runtime config so UI follows backend websocket path config.
async function loadAppConfig() {
  try {
    const response = await fetch("/app/config", { method: "GET" });
    if (!response.ok) {
      return;
    }
    const payload = await response.json();
    const path = typeof payload.websocket_path === "string" ? payload.websocket_path : "";
    if (path.startsWith("/")) {
      state.websocketPath = path;
    }
  } catch (_err) {
    // Keep default path when config endpoint is unavailable.
  }
}

// Bind UI interactions with push-to-talk semantics and reset behavior.
function bindUi() {
  resetBtn.addEventListener("click", onResetConversation);

  // Hold to talk with pointer events so mouse and touch follow one flow.
  recordBtn.addEventListener("pointerdown", async (event) => {
    event.preventDefault();
    try {
      recordBtn.setPointerCapture(event.pointerId);
    } catch (_err) {
      // Ignore capture failures and keep recording flow functional.
    }
    const intentToken = beginRecordIntent();
    try {
      await startRecording(intentToken);
    } catch (err) {
      setStatus(`microphone unavailable: ${String(err)}`);
      state.recording = false;
      state.recordingPending = false;
      syncControls();
    }
  });
  recordBtn.addEventListener("pointerup", () => stopRecording());
  recordBtn.addEventListener("pointercancel", () => stopRecording());
  recordBtn.addEventListener("lostpointercapture", () => stopRecording());

  // Keep keyboard accessibility: hold Enter/Space to talk, release to send.
  recordBtn.addEventListener("keydown", async (event) => {
    if ((event.key === " " || event.key === "Enter") && !event.repeat) {
      event.preventDefault();
      const intentToken = beginRecordIntent();
      try {
        await startRecording(intentToken);
      } catch (err) {
        setStatus(`microphone unavailable: ${String(err)}`);
        state.recording = false;
        state.recordingPending = false;
        syncControls();
      }
    }
  });
  recordBtn.addEventListener("keyup", (event) => {
    if (event.key === " " || event.key === "Enter") {
      event.preventDefault();
      stopRecording();
    }
  });
}

// Initialize app shell, config, websocket, and control handlers.
async function init() {
  syncControls();
  bindUi();
  await loadAppConfig();
  connectWebSocket();
  // Expose a tiny debug helper for manual WS testing from browser devtools.
  window.pomDebugSendTurnStart = sendTurnStart;
}

void init();
