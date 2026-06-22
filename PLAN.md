# voice_detection — Project Plan

This document is the authoritative record of what has been designed, built, and
deliberately scoped out of this repository. It is a living document; update it
whenever the architecture changes.

---

## What this project is

`voice_detection` is a **real-time speaker-diarization web application**. It
listens to a microphone, segments the audio stream into speech utterances,
identifies which speaker produced each utterance, and streams per-speaker time
totals to a browser UI — all with sub-second latency on a local machine.

Custom VAD model training has been spun out into a **separate project** and is
not part of this repository.

---

## Architecture

```
Microphone (sounddevice)
        │  raw int16 frames  (30 ms / 480 samples @ 16 kHz)
        ▼
  audio/capture.py          — pulls frames from the OS mic via sounddevice
        │
        ▼
  audio/vad.py  (webrtcvad) — speech / silence gate → bool
        │  speech frames only
        ▼
  diarization/pipeline.py   — buffers frames into utterances
        │  utterance PCM  (≥ MIN_SPEECH_MS)
        ▼
  embeddings/encoder.py     — ECAPA-TDNN (SpeechBrain) → 192-dim vector
        │
        ▼
  diarization/clustering.py — nearest-centroid cosine similarity
        │  speaker id
        ▼
  diarization/speaker_tracker.py — accumulates speak times, history
        │
        ▼
  server/state.py           — thread-safe SharedState (lock-guarded)
        │  JSON over WebSocket  (polled every 0.5 s)
        ▼
  Browser UI (HTML/CSS/JS)  — speaker cards with live time totals
```

---

## What has been built

### Backend

| Component | File(s) | Status |
|---|---|---|
| Mic capture thread | `audio/capture.py` | ✅ Complete |
| Speech/silence gate | `audio/vad.py` | ✅ Uses `webrtcvad` |
| VAD hang-over frames | `diarization/pipeline.py` | ✅ Prevents silence-stealing mid-speech |
| `MIN_SPEECH_MS` filter | `diarization/pipeline.py` | ✅ Filters short noise bursts |
| Speaker embedding | `embeddings/encoder.py` | ✅ ECAPA-TDNN via SpeechBrain |
| Per-speaker centroid clustering | `diarization/clustering.py` | ✅ Cosine distance, `CLUSTER_THRESHOLD = 0.85` |
| Speaker history (in-memory) | `diarization/speaker_tracker.py` | ✅ Capped at `HISTORY_WINDOW = 100` |
| `NEW_SPEAKER_TIME` gate | `diarization/speaker_tracker.py` | ✅ Suppresses one-off blips |
| "Silent" pseudo-speaker | `diarization/pipeline.py` | ✅ Tracks cumulative silence |
| Thread-safe shared state | `server/state.py` | ✅ Lock-guarded SharedState |
| Flask + WebSocket server | `server/app.py` | ✅ Streams state to browser every 0.5 s |
| `/reset` HTTP endpoint | `server/app.py` | ✅ Clears backend state |
| Tunable constants | `config.py` | ✅ Single source of truth |

### Frontend

| Feature | Status |
|---|---|
| Speaker cards with live time totals | ✅ |
| Millisecond-precision silence display | ✅ |
| Reset button (clears frontend + backend) | ✅ |
| WebSocket reconnection handling | ✅ |
| Phantom-speaker-card bug fixed | ✅ |

### Key bugs resolved

- Instance variable scope errors in the diarization pipeline
- Audio buffer not clearing on reset / disconnect
- Clustering switched from nearest-neighbor embedding to per-speaker centroid
  averaging, significantly reducing false speaker splits
- `CLUSTER_THRESHOLD` raised from 0.72 → 0.85 for tighter same-speaker matching

---

## What is NOT in this repo (deliberately scoped out)

### Custom VAD model

A custom 4-class neural VAD model (`VADNet`, trained on LibriSpeech + AMI +
ESC-50) was prototyped in this repo but has been **extracted into a separate
project**. Nothing related to model training, labeling, or evaluation remains
here.

The live pipeline exclusively uses **`webrtcvad`** (Google's WebRTC VAD) for
the speech/silence gate. `audio/vad.py` calls `webrtcvad` and returns a plain
bool; there is no custom model integration in the runtime path.

### Embedding persistence

Speaker embeddings are computed live and held **in memory only**
(`SpeakerTracker.history`). They are never written to disk. A `/reset` or
process restart wipes them.

---

## Configuration reference

All tunable constants live in `config.py`:

| Constant | Default | Effect |
|---|---|---|
| `SAMPLE_RATE` | 16000 | Mic sample rate (Hz) |
| `FRAME_MS` | 30 | VAD frame duration (ms) |
| `MIN_SPEECH_MS` | — | Minimum utterance length; shorter frames are discarded |
| `CLUSTER_THRESHOLD` | 0.85 | Cosine similarity threshold; above → same speaker |
| `HISTORY_WINDOW` | 100 | Max embeddings kept per speaker for centroid computation |
| `NEW_SPEAKER_TIME` | — | Seconds a new speaker must speak before being registered |

---

## Speaker assignment algorithm (summary)

1. Encode the utterance → 192-dim embedding via ECAPA-TDNN.
2. Average all stored embeddings per speaker → one centroid per speaker.
3. Compute cosine distance from the new embedding to each centroid.
4. If the minimum distance ≤ `1 − CLUSTER_THRESHOLD`: assign that speaker.
5. Otherwise: mint a new speaker id (`max(existing) + 1`).
6. Don't register the new speaker until they've accumulated `NEW_SPEAKER_TIME`
   seconds, to suppress noise blips.

---

## Known limitations / future work

- **No persistence across sessions** — speaker identities reset on every
  restart. Long-session diarization would require saving and reloading embeddings.
- **Single-channel only** — the pipeline assumes a single mono mic input.
- **No overlap detection** — simultaneous speakers collapse into whichever
  speaker's centroid is nearest.
- **webrtcvad is agnostic to noise type** — music, fans, and keyboard sounds
  can trigger the speech gate. Integrating the custom VAD model (from the
  separate project) could suppress these.
- **Browser-only UI** — no mobile-optimized layout.