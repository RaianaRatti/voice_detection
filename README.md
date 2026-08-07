# Live Speaker Detection

Real-time speaker diarization in the browser. Listens to your microphone,
segments the audio stream into speech utterances, and tracks how long each
speaker has been talking — all running locally with sub-second latency.

## Demo Video

https://github.com/user-attachments/assets/9141443b-cc4a-46a2-acaf-5b8cc924689e

## How it works

```
Mic → webrtcvad → utterance buffer → ECAPA-TDNN → centroid clustering → WebSocket → Browser
```

1. **Capture** — `sounddevice` pulls 30 ms frames from the microphone at 16 kHz.
2. **VAD gate** — `webrtcvad` classifies each frame as speech or silence.
   Hang-over logic prevents silence from cutting into the tail of an utterance.
3. **Utterance assembly** — consecutive speech frames are buffered until silence
   is detected. Utterances shorter than `MIN_SPEECH_MS` are discarded.
4. **Speaker embedding** — SpeechBrain's pretrained ECAPA-TDNN model encodes
   each utterance into a 192-dimensional speaker vector.
5. **Speaker assignment** — the embedding is compared (cosine similarity) against
   a per-speaker centroid computed from recent history. If it falls within
   `CLUSTER_THRESHOLD` of an existing speaker, they get the credit; otherwise a
   new speaker is created.
6. **UI** — a Flask + WebSocket server streams per-speaker cumulative times to
   the browser every 0.5 s. Speaker cards update in real time.

---

## Features

- Live speaker cards with running time totals
- Tracks cumulative silence as its own "Silent" entry
- Reset button clears all state (frontend + backend)
- Per-speaker centroid clustering — more stable than nearest-neighbor matching
- `NEW_SPEAKER_TIME` guard suppresses phantom speakers from noise blips
- All configuration in a single `config.py`

---

## Requirements

- Python 3.9+
- A working microphone
- `pip install flask flask-sock sounddevice webrtcvad speechbrain torch`

> SpeechBrain will download the ECAPA-TDNN model (~80 MB) on first run and cache
> it under `models/`.

---

## Quick start

```bash
git clone https://github.com/your-username/voice_detection
cd voice_detection
pip install -r requirements.txt
python server/app.py
```

Open `http://localhost:5000` in your browser, allow microphone access, and start
talking.

To reset mid-session, click the **Reset** button in the UI or POST to
`/reset`.

---

## Configuration

All tunable constants are in `config.py`:

| Constant | Default | Description |
|---|---|---|
| `SAMPLE_RATE` | `16000` | Microphone sample rate (Hz) |
| `FRAME_MS` | `30` | VAD frame duration (ms) |
| `MIN_SPEECH_MS` | `300` | Minimum utterance length; shorter frames are discarded |
| `CLUSTER_THRESHOLD` | `0.85` | Cosine similarity threshold for same-speaker matching |
| `HISTORY_WINDOW` | `100` | Max embeddings kept per speaker for centroid computation |
| `NEW_SPEAKER_TIME` | `1.0` | Seconds a new speaker must speak before being registered |

Raising `CLUSTER_THRESHOLD` toward `1.0` makes the system more aggressive about
creating new speakers. Lowering it merges more utterances into existing speakers.

---

## Project structure

```
voice_detection/
├── audio/
│   ├── capture.py          # Mic capture thread (sounddevice)
│   └── vad.py              # webrtcvad wrapper → bool
├── diarization/
│   ├── pipeline.py         # Orchestrates capture → VAD → embed → cluster
│   ├── clustering.py       # Centroid cosine-similarity speaker assignment
│   └── speaker_tracker.py  # Per-speaker history and speak-time accumulation
├── embeddings/
│   └── encoder.py          # ECAPA-TDNN via SpeechBrain
├── visualizations/         # Visualizations for audio, vad, mfcc, embedding, clustering
├── server/
│   ├── app.py              # Flask + WebSocket server, /reset endpoint
│   |── state.py            # Thread-safe SharedState
|   |── templates/          # Frontend CSS
│   └── static/             # Frontend HTML/JS
├── config.py               # All tunable constants
└── requirements.txt
```

---

## Notes

**Embeddings are not persisted.** Speaker identities exist only for the lifetime
of the process. Restarting the server (or hitting Reset) wipes all history.

**Custom VAD model.** A neural 4-class VAD trained on LibriSpeech, AMI, and
ESC-50 was prototyped alongside this project but lives in a separate repository.
The live pipeline uses `webrtcvad` exclusively.

**Single channel only.** The pipeline assumes one mono microphone. Multi-device
or multi-channel setups are not supported.
