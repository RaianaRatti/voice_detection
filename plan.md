# Speaker diarization — build plan

A fully offline, browser-viewable speaker timer built in Python.
No black boxes. Every stage is inspectable and replaceable.

---

## Stack summary

| Stage | Library |
|---|---|
| Audio capture | `sounddevice` |
| Voice activity detection | `py-webrtcvad` |
| Feature extraction | `librosa` |
| Speaker embeddings | `SpeechBrain` ECAPA-TDNN (pretrained, offline) |
| Clustering | `scikit-learn` AgglomerativeClustering |
| Backend | `Flask` + `flask-sock` (WebSockets) |
| Frontend | Vanilla HTML + CSS + JS |

---

## Data flow

```
Microphone
    ↓  sounddevice          → raw PCM chunks (16kHz, 16-bit, mono, 30ms frames)
    ↓  py-webrtcvad         → label each frame speech or silence; merge into utterances
    ↓  SpeechBrain          → 192-dim embedding vector per utterance
    ↓  AgglomerativeClustering → speaker ID integer (0, 1, 2 …)
    ↓  speaker_tracker.py  → {speaker_id: total_seconds_spoken}
    ↓  Flask WebSocket      → JSON push to browser every utterance
    ↓  app.js               → live speaker timer cards re-render in the browser
```

Note: librosa/MFCCs are used in the notebooks for learning but SpeechBrain handles
its own internal feature extraction at runtime. You do not need to pipe librosa
output into SpeechBrain.

---

## Key parameters (all defined in config.py)

| Parameter | Default | What it controls |
|---|---|---|
| `SAMPLE_RATE` | 16000 | Hz — WebRTC VAD requires 8000, 16000, or 32000 |
| `FRAME_MS` | 30 | VAD frame size — must be 10, 20, or 30 ms |
| `VAD_AGGRESSIVENESS` | 2 | 0–3; higher filters more aggressively |
| `MIN_SPEECH_MS` | 300 | Discard utterances shorter than this |
| `CLUSTER_THRESHOLD` | 0.75 | Agglomerative distance cutoff for new-speaker detection |
| `HISTORY_WINDOW` | 50 | Rolling count of recent embeddings kept for re-clustering |

---

## Phase 0 — Project setup

**Goal:** Create the folder skeleton, install dependencies, confirm everything imports.

### Step 0.1 — Create the directory structure

Create the following folders and empty `__init__.py` files by hand or with mkdir:

```
diarization/
├── audio/
├── features/
├── embeddings/
├── diarization/
├── server/
├── models/
├── notebooks/
└── tests/
```

Add an empty `__init__.py` inside: `audio/`, `features/`, `embeddings/`,
`diarization/`, and `server/`.

### Step 0.2 — Create `requirements.txt`

Create `diarization/requirements.txt` and add:

```
sounddevice
numpy
webrtcvad-wheels
librosa
speechbrain
scikit-learn
flask
flask-sock
torch
torchaudio
notebook
matplotlib
scipy
```

Install with: `pip install -r requirements.txt`

### Step 0.3 — Create `config.py`

Create `voice_detection/config.py`.

Define the following constants (use the defaults from the table above):
`SAMPLE_RATE`, `FRAME_MS`, `VAD_AGGRESSIVENESS`, `MIN_SPEECH_MS`,
`CLUSTER_THRESHOLD`, `HISTORY_WINDOW`, and `MODELS_DIR` (path to the `models/`
folder).

### Step 0.4 — Smoke-test imports

Open a Python REPL in the project root and import each library:
`sounddevice`, `webrtcvad`, `librosa`, `speechbrain`, `sklearn`, `flask`.
Fix any installation errors before moving on.

---

## Phase 1 — Audio capture

**Goal:** Stream microphone audio into your program as numpy arrays.
**Files edited:** `audio/capture.py`

### Step 1.1 — Write `audio/capture.py`

This module's job: open the microphone with `sounddevice` and yield a
steady stream of fixed-size audio frames.

It should expose one callable (a generator or a class with a `read()` method)
that, when called in a loop, returns a numpy array of shape `(N,)` containing
`FRAME_MS` milliseconds of 16-bit mono audio at `SAMPLE_RATE`.

Use `sounddevice.InputStream` with `dtype='int16'`, `channels=1`, and
`blocksize = SAMPLE_RATE * FRAME_MS // 1000`.

### Step 1.2 — Learn: `notebooks/01_audio_capture.ipynb`

Open the notebook. Use `audio/capture.py` to record ~5 seconds of yourself
speaking. Plot the raw waveform with `matplotlib`. Observe:
- What silence looks like vs speech (amplitude envelope)
- What happens when you change `FRAME_MS`

---

## Phase 2 — Voice activity detection

**Goal:** Label each audio frame as speech or silence, then merge consecutive
speech frames into utterances.
**Files edited:** `audio/vad.py`

### Step 2.1 — Write `audio/vad.py`

This module wraps `py-webrtcvad`. It should expose:

- A `VAD` class initialized with `VAD_AGGRESSIVENESS` from config.
- A `is_speech(frame_bytes)` method that takes one frame as raw bytes
  (not numpy — convert with `.tobytes()`) and returns `True` or `False`.
- A `merge_frames(frames)` function that takes a list of `(frame, is_speech)`
  tuples and groups consecutive speech frames into utterances, discarding
  any utterance shorter than `MIN_SPEECH_MS`. Returns a list of numpy arrays,
  one per utterance.

Important: `webrtcvad` requires raw bytes at exactly the right frame size.
The frame duration must be 10, 20, or 30 ms exactly — if it isn't, the library
raises an error. Double-check your `blocksize` from Phase 1.

### Step 2.2 — Learn: `notebooks/02_vad_exploration.ipynb`

Load the recording from notebook 01. Run VAD on each frame. Plot the waveform
with speech frames highlighted in a different color. Experiment with
`VAD_AGGRESSIVENESS` values 0, 1, 2, 3 and observe how the boundaries shift.

### Step 2.3 — Write `tests/test_vad.py`

Write two simple tests:
- Feed a frame of zeros (silence) → expect `is_speech` to return `False`.
- Feed a frame of a sine wave at 300 Hz (voice-range) → expect `True`.

Use `numpy` to generate the sine wave, convert to int16 bytes.

---

## Phase 3 — Feature extraction (learning only)

**Goal:** Understand MFCCs — the classic audio feature used in speaker recognition.
This phase is for building intuition. The MFCC module is NOT used in the live
pipeline (SpeechBrain handles its own features internally), but understanding it
will help you reason about what the embedding model is doing.
**Files edited:** `features/mfcc.py`, `notebooks/03_mfcc_exploration.ipynb`

### Step 3.1 — Write `features/mfcc.py`

This module wraps `librosa`. Expose one function:
`extract(audio_np, sample_rate) -> np.ndarray`

Internally, call `librosa.feature.mfcc(y=..., sr=..., n_mfcc=40)`.
Return the resulting matrix of shape `(40, T)` where T is the number of frames.

### Step 3.2 — Learn: `notebooks/03_mfcc_exploration.ipynb`

Take two utterances from different speakers (from notebook 01 recordings).
Run `features/mfcc.py` on each. Plot the MFCC spectrograms side by side with
`librosa.display.specshow`. Observe:
- How the coefficients differ between speakers
- What happens when you change `n_mfcc` (try 13, 20, 40)
- Why the first coefficient (energy) is often discarded in speaker recognition

---

## Phase 4 — Speaker embeddings

**Goal:** Convert each utterance into a fixed-size vector that represents the
speaker's voice, regardless of what they said.
**Files edited:** `embeddings/encoder.py`

### Step 4.1 — Write `embeddings/encoder.py`

This module loads the SpeechBrain ECAPA-TDNN model and exposes:

`encode(audio_np, sample_rate) -> np.ndarray`  (shape: `(192,)`)

Internally:
- Load `speechbrain.pretrained.EncoderClassifier` with
  `source="speechbrain/spkrec-ecapa-voxceleb"` and `savedir=MODELS_DIR`.
  This downloads weights on first call and caches them in `models/`.
- Convert the numpy array to a torch tensor (float32, shape `(1, N)`).
- Call `classifier.encode_batch(tensor)` and return `.squeeze().numpy()`.

The model runs on CPU by default. Expect ~50–200ms per utterance depending
on utterance length.

### Step 4.2 — Learn: `notebooks/04_embedding_space.ipynb`

Record 3–4 short utterances per speaker (2–3 different speakers). Compute
embeddings for each. Then:
- Use `sklearn.decomposition.PCA(n_components=2)` to project to 2D.
- Scatter-plot the points, colored by speaker.
- You should see loose clusters — same speaker's embeddings land near each other.
- Try `TSNE` from sklearn if PCA clusters don't separate well.

This is the key intuition check: if the clusters don't form, something is wrong
with your audio (too short, too noisy) before you try to cluster automatically.

### Step 4.3 — Write `tests/test_embeddings.py`

Write one test:
- Generate two numpy arrays of random float32 noise (simulating audio).
- Call `encode()` on each.
- Assert output shape is `(192,)` and dtype is float32.

(This tests the interface, not speaker accuracy.)

---

## Phase 5 — Clustering and speaker tracking

**Goal:** Assign speaker IDs to each new utterance, maintain per-speaker
speaking time, and handle new speakers appearing mid-conversation.
**Files edited:** `diarization/clustering.py`, `diarization/speaker_tracker.py`

### Step 5.1 — Write `diarization/clustering.py`

This module wraps `sklearn.cluster.AgglomerativeClustering`. Expose:

`assign_speaker(new_embedding, history) -> int`

Where `history` is a list of `(embedding, speaker_id)` tuples from recent
utterances (capped at `HISTORY_WINDOW`).

Logic:
1. If history is empty, return speaker ID 0 (first speaker).
2. Stack all embeddings from history into a matrix.
3. Compute cosine distance between `new_embedding` and each row.
4. Find the closest match. If its distance is below `CLUSTER_THRESHOLD`,
   return that speaker's ID.
5. Otherwise, return `max(existing_ids) + 1` (new speaker detected).

Do not call `AgglomerativeClustering.fit()` on every frame — that approach
does not generalize to online (streaming) use. Use the distance comparison
approach above instead. You will experiment with full clustering in the notebook.

### Step 5.2 — Learn: `notebooks/05_clustering_tuning.ipynb`

Take the full set of embeddings from notebook 04 (multiple speakers).
Fit `AgglomerativeClustering` with `distance_threshold=X, n_clusters=None,
linkage='average'` for several values of X. For each:
- Print the number of clusters found.
- Plot a dendrogram using `scipy.cluster.hierarchy.dendrogram`.
- Observe where the tree cuts when you change the threshold.

This builds intuition for what `CLUSTER_THRESHOLD` in config.py is doing.
Tune it until the clustering matches the true speaker count on your recording.

### Step 5.3 — Write `diarization/speaker_tracker.py`

This module maintains state across utterances. It should expose a
`SpeakerTracker` class with:

- `self.history`: list of `(embedding, speaker_id)`, capped at `HISTORY_WINDOW`
- `self.speak_times`: dict of `{speaker_id: float}` (total seconds)
- `self.current_speaker`: int or None

Methods:
- `update(embedding, duration_seconds) -> int`
  Calls `clustering.assign_speaker()`, updates `speak_times` and `history`,
  sets `current_speaker`, and returns the speaker ID.
- `get_state() -> dict`
  Returns `{"current": int, "times": {str: float}}` — the JSON payload
  that will be sent to the browser.

### Step 5.4 — Write `tests/test_clustering.py`

Write two tests:
- Pass two embeddings that are nearly identical (same vector + tiny noise) →
  expect the same speaker ID returned for both.
- Pass two embeddings that are very different (orthogonal vectors) →
  expect different speaker IDs.

---

## Phase 6 — Pipeline

**Goal:** Wire all stages into a single loop that runs in a background thread,
continuously reading from the mic and pushing results to a shared state object.
**Files edited:** `diarization/pipeline.py`, `server/state.py`

### Step 6.1 — Write `server/state.py`

A simple shared-memory object using `threading.Lock`. Expose:

- `update(speaker_id, times_dict)` — called from the pipeline thread
- `get()` — called from the Flask thread; returns the latest state dict

This is the only communication channel between the pipeline thread and the
web server thread.

### Step 6.2 — Write `diarization/pipeline.py`

This is the main processing loop. Expose a `run(state)` function meant to be
called in a `threading.Thread`.

The loop should:
1. Open the mic stream via `audio/capture.py`.
2. Collect frames into a buffer.
3. On each frame, call `vad.is_speech()`.
4. When VAD transitions from speech → silence (end of utterance), pass the
   buffered frames to `vad.merge_frames()`.
5. If the resulting utterance is long enough (`MIN_SPEECH_MS`), call
   `embeddings/encoder.py` to get the embedding.
6. Call `speaker_tracker.update(embedding, duration)`.
7. Call `state.update(...)` with the new speaker ID and times.
8. Clear the buffer and repeat.

---

## Phase 7 — Server

**Goal:** Serve the frontend and push live speaker updates over a WebSocket.
**Files edited:** `server/app.py`

### Step 7.1 — Write `server/app.py`

Create a Flask app with two routes:

- `GET /` — serves `frontend/index.html`
- `GET /static/<path>` — serves `frontend/style.css` and `frontend/app.js`
- WebSocket `/ws` (using `flask_sock`) — on connect, enter a loop that calls
  `state.get()` and sends the JSON result every 500ms as long as the client
  is connected.

In `if __name__ == '__main__'`:
- Create the `state.SharedState` instance.
- Start `pipeline.run(state)` in a `daemon=True` background thread.
- Start the Flask server.

---

## Phase 8 — Frontend

**Goal:** A clean browser page that shows each speaker as a labeled card with a
live running timer, and highlights the current speaker.
**Files edited:** `frontend/index.html`, `frontend/style.css`, `frontend/app.js`

### Step 8.1 — Write `frontend/index.html`

A minimal HTML page that:
- Loads `style.css` and `app.js`.
- Contains a single `<div id="speakers">` where speaker cards will be injected.
- Contains a "Connect" button that starts the WebSocket connection (so the page
  does nothing until the user clicks it — avoids errors on page load).

### Step 8.2 — Write `frontend/style.css`

Style the speaker cards. Each card should show:
- A speaker label ("Speaker 1", "Speaker 2", etc.)
- A formatted time display (MM:SS)
- A highlight or border when that speaker is currently active

Use a distinct accent color per speaker (cycling through a small palette).
Keep the layout simple — flexbox row of cards.

### Step 8.3 — Write `frontend/app.js`

On "Connect" click:
- Open `new WebSocket("ws://localhost:5000/ws")`.
- On each `message` event, parse the JSON `{current, times}` payload.
- For each speaker ID in `times`, find or create a card in `#speakers`.
- Update the time display on each card.
- Highlight only the card whose ID matches `current`.
- Format seconds as MM:SS with a small helper function.

---

## Phase 9 — End-to-end test

**Goal:** Confirm the full pipeline works with real audio before any tuning.

### Step 9.1 — First run

Run `python server/app.py`. Open `http://localhost:5000` in a browser.
Click Connect. Speak. Pause. Have someone else (or a second voice recording
played from a speaker) speak. Observe:
- Speaker cards appear as new speakers are detected.
- Timers increment while a speaker is active.
- The active card highlights correctly.

### Step 9.2 — Tune `CLUSTER_THRESHOLD` in `config.py`

If the same speaker keeps generating new cards → threshold is too low, increase it.
If two different speakers always collapse into one card → threshold is too high,
decrease it. Adjust in increments of 0.05 and re-run.

### Step 9.3 — Tune `VAD_AGGRESSIVENESS` in `config.py`

If the pipeline is slow to react (long silence before a new utterance registers)
→ the VAD is holding frames too long. Increase aggressiveness or reduce
`MIN_SPEECH_MS`. If it's too twitchy (short noises trigger a new utterance) →
decrease aggressiveness or increase `MIN_SPEECH_MS`.

---

## File edit summary

| File | Phase | What goes in it |
|---|---|---|
| `requirements.txt` | 0 | All dependencies |
| `config.py` | 0 | All tunable constants |
| `audio/capture.py` | 1 | sounddevice mic stream → numpy frames |
| `audio/vad.py` | 2 | webrtcvad wrapper + utterance merger |
| `features/mfcc.py` | 3 | librosa MFCC extraction (learning only) |
| `embeddings/encoder.py` | 4 | SpeechBrain ECAPA-TDNN inference |
| `diarization/clustering.py` | 5 | Online cosine-distance speaker assignment |
| `diarization/speaker_tracker.py` | 5 | Per-speaker time state + history |
| `diarization/pipeline.py` | 6 | Main processing loop (runs in thread) |
| `server/state.py` | 6 | Thread-safe shared state |
| `server/app.py` | 7 | Flask routes + WebSocket push |
| `frontend/index.html` | 8 | Page shell + Connect button |
| `frontend/style.css` | 8 | Speaker cards, timers, active highlight |
| `frontend/app.js` | 8 | WebSocket client + DOM updates |
| `tests/test_vad.py` | 2 | Silence + sine wave VAD tests |
| `tests/test_embeddings.py` | 4 | Embedding shape + dtype test |
| `tests/test_clustering.py` | 5 | Same/different speaker assignment tests |
| `notebooks/01_audio_capture.ipynb` | 1 | Record + plot waveform |
| `notebooks/02_vad_exploration.ipynb` | 2 | Visualize speech/silence labels |
| `notebooks/03_mfcc_exploration.ipynb` | 3 | Plot MFCC spectrograms |
| `notebooks/04_embedding_space.ipynb` | 4 | PCA/TSNE plot of speaker embeddings |
| `notebooks/05_clustering_tuning.ipynb` | 5 | Dendrogram + threshold tuning |