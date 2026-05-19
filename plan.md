# Voice Tracker App — Your ML Learning Roadmap

You want to build an app that listens to a conversation and shows a live bar for each speaker, tracking how many seconds they've talked. Silence gets its own bar too.

This is called **speaker diarization** — and it's a genuinely interesting ML problem. Here's your path from zero to working app.

---

## What You're Actually Building

```
[Microphone] → [Audio chunks] → [Who is speaking?] → [Update UI bars]
                                       ↑
                               This is the ML part
```

The app has three layers:
1. **Audio capture** — grab mic input in the browser
2. **Speaker diarization** — figure out who's talking (or if it's silence)
3. **UI** — draw and update the bars in real time

---

## Phase 1: Understand the Core Concepts (3–5 hours)

Before touching code, get comfortable with these ideas. None require math — just intuition.

### 1.1 What is a feature?
ML models don't work with raw audio (millions of numbers per second). They work with *features* — compact summaries that capture what matters.

For audio, the key feature is the **spectrogram**: a visual map of which frequencies are present at each moment in time. Think of it like sheet music — it shows what's "happening" in the sound without storing every raw sample.

**Learn:** Search "spectrogram explained" on YouTube. 3Blue1Brown's Fourier Transform video is also great background.

### 1.2 What is a neural network doing?
A neural network is a function that takes numbers in and produces numbers out, trained by showing it millions of examples until it gets good at a task.

For speaker recognition, it was trained on thousands of hours of labeled audio: "this chunk = person A, this chunk = person B." It learned what makes each voice distinctive (pitch, rhythm, timbre).

**Learn:** Watch "But what is a neural network?" by 3Blue1Brown (~20 min). Just the first video is enough for now.

### 1.3 What is an embedding?
This is the key idea behind how speaker recognition works. A neural network can be trained to turn a voice clip into a small list of numbers (e.g., 256 numbers) called an **embedding** or **voice print**. 

The magic: clips from the same speaker produce *similar* numbers. Clips from different speakers produce *different* numbers. You don't need to know who anyone is in advance — you just cluster the similar ones together.

**Learn:** Search "word embeddings explained simply" — the concept is identical, just applied to text instead of voice.

### 1.4 What is clustering?
Once you have embeddings, you need to group similar ones together. This is **clustering** — finding natural groups in data without knowing labels ahead of time.

The simplest algorithm is **k-means**: pick k centers, assign each point to the nearest center, move centers to the average of their group, repeat. For diarization, you often use a fancier version since you don't know k (number of speakers) in advance.

**Learn:** Search "k-means clustering visual explanation." There are great interactive demos online.

---

## Phase 2: Explore the Tools (2–3 hours)

You don't need to train a model from scratch. People have already trained excellent speaker recognition models and published them for free. Your job is to use them.

### 2.1 The ML library you'll use: ONNX Runtime (in the browser)

Models are usually trained in Python, but they can be exported to a format called **ONNX** and run in a browser with JavaScript. This means no server needed.

**Poke around:** https://onnxruntime.ai/

### 2.2 The model you'll use: SpeakerNet or pyannote

Two good options:

| Option | Where it runs | Complexity |
|---|---|---|
| **ONNX SpeakerNet** (NVIDIA) | Browser (JS) | Simpler — just JS |
| **pyannote-audio** | Python backend | More powerful |

**Recommended starting point:** Start with a Python backend using `pyannote-audio`. It handles diarization end-to-end and is well documented. You can add a web frontend later.

**Explore:** https://github.com/pyannote/pyannote-audio

### 2.3 Audio capture: Web Audio API

The browser has built-in tools for capturing mic input in real time. No library needed — just the Web Audio API.

**Learn:** MDN's "Using the Web Audio API" intro page.

---

## Phase 3: Build It, Step by Step

### Project Structure

Before writing a single line of code, create this folder layout. Every file below has a home here — even though it's a small project, this mirrors how real ML projects are organized at companies.

```
voice-tracker/
│
├── data/                        # Audio files used for testing
│   ├── test.wav                 # Your recorded test clip (generated, not written)
│   ├── you_clip1.wav            # Sample clips for comparing voices (generated)
│   ├── you_clip2.wav
│   └── conversation.wav         # A multi-speaker recording for Step 5 (generated)
│
├── src/                         # All source code lives here
│   ├── audio/                   # Everything related to capturing sound
│   │   └── capture.py           # Step 1 — record mic to .wav
│   │
│   ├── features/                # Everything related to analyzing sound
│   │   └── spectrogram.py       # Step 2 — compute and display spectrograms
│   │
│   ├── models/                  # Everything related to ML models
│   │   ├── embeddings.py        # Step 3 — generate voice embeddings
│   │   └── similarity.py        # Step 4 — compare voices with cosine similarity
│   │
│   ├── pipeline/                # The end-to-end ML pipeline
│   │   └── diarize.py           # Step 5 — full diarization on a file
│   │
│   ├── streaming/               # Real-time audio processing
│   │   └── stream.py            # Step 6 — live mic input in chunks
│   │
│   └── ui/                      # Display and interface
│       └── terminal.py          # Step 7 — live terminal bar display
│
├── main.py                      # Entry point — ties everything together
├── requirements.txt             # All pip dependencies listed here
└── README.md                    # How to install and run the project
```

Create this structure first with:
```bash
mkdir -p voice-tracker/data voice-tracker/src/{audio,features,models,pipeline,streaming,ui}
cd voice-tracker
touch main.py requirements.txt README.md
touch src/audio/capture.py src/features/spectrogram.py
touch src/models/embeddings.py src/models/similarity.py
touch src/pipeline/diarize.py src/streaming/stream.py src/ui/terminal.py
```

---

### Step 1 — Get mic audio into Python
**Goal:** A Python script that records 3 seconds of audio from your mic and saves it as a .wav file.

**What you'll learn:** `pyaudio` or `sounddevice` library, audio formats (sample rate, bit depth).

**File:** `src/audio/capture.py`

```python
import sounddevice as sd
import scipy.io.wavfile as wav

sample_rate = 16000
duration = 3  # seconds
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()
wav.write("data/test.wav", sample_rate, audio)
```

**Milestone:** You have `data/test.wav` — a file you recorded.

---

### Step 2 — Compute a spectrogram
**Goal:** Turn your .wav file into a visual spectrogram using Python.

**What you'll learn:** `librosa` library, the Short-Time Fourier Transform (STFT), mel spectrograms.

**File:** `src/features/spectrogram.py`

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load("data/test.wav", sr=16000)
S = librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(librosa.power_to_db(S), sr=sr)
plt.colorbar()
plt.show()
```

**Milestone:** You can see your voice as an image. This is exactly what the ML model "sees."

---

### Step 3 — Generate voice embeddings
**Goal:** Turn an audio clip into a list of numbers (the voice print).

**What you'll learn:** Loading pre-trained models, running inference, what embeddings look like.

**File:** `src/models/embeddings.py`

```python
from pyannote.audio import Model, Inference

model = Model.from_pretrained("pyannote/embedding")
inference = Inference(model, window="whole")
embedding = inference("data/test.wav")
print(embedding.shape)  # something like (1, 512)
print(embedding)         # a list of numbers — your voice's "fingerprint"
```

**Milestone:** You can see the numbers that represent your voice.

---

### Step 4 — Compare two voices
**Goal:** Show that two clips of the same person are similar, and two different people are different.

**What you'll learn:** Cosine similarity — the standard way to compare embeddings.

**File:** `src/models/similarity.py`

```python
from sklearn.metrics.pairwise import cosine_similarity
from pyannote.audio import Model, Inference

model = Model.from_pretrained("pyannote/embedding")
inference = Inference(model, window="whole")

emb1 = inference("data/you_clip1.wav")
emb2 = inference("data/you_clip2.wav")
emb3 = inference("data/someone_else.wav")

print(cosine_similarity(emb1, emb2))  # should be high (~0.8–0.99)
print(cosine_similarity(emb1, emb3))  # should be lower (~0.1–0.5)
```

**Milestone:** You've written your first ML comparison. This is the core of speaker recognition.

---

### Step 5 — Full diarization on a recording
**Goal:** Take a multi-speaker recording and have the model automatically segment it: "Speaker A: 0–12s, Speaker B: 12–18s, Silence: 18–20s..."

**What you'll learn:** The pyannote pipeline, reading diarization output.

**File:** `src/pipeline/diarize.py`

```python
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipeline("data/conversation.wav")

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{speaker}: {turn.start:.1f}s – {turn.end:.1f}s")
```

**Milestone:** You can see the conversation broken down by speaker, with timestamps. This is the ML doing real work.

---

### Step 6 — Process audio in real time (streaming)
**Goal:** Instead of a file, process the mic input in chunks, running the diarization every few seconds.

**What you'll learn:** Streaming audio buffers, overlapping windows (so you don't miss words at chunk boundaries).

This is the trickiest step. The trick is to record in overlapping chunks — e.g., process every 2 seconds, with 0.5s overlap on each side.

**File:** `src/streaming/stream.py`

```python
import sounddevice as sd
import numpy as np

CHUNK = 2.0   # seconds per chunk
OVERLAP = 0.5 # seconds of overlap
SR = 16000
buffer = np.zeros(int((CHUNK + OVERLAP) * SR))

def callback(indata, frames, time, status):
    global buffer
    new_audio = indata[:, 0]
    buffer = np.roll(buffer, -len(new_audio))
    buffer[-len(new_audio):] = new_audio
    # → run diarization on buffer here

with sd.InputStream(samplerate=SR, channels=1, callback=callback):
    input("Recording... press Enter to stop")
```

**Milestone:** The script prints speaker labels in real time as you talk.

---

### Step 7 — Add the UI
**Goal:** A terminal display that shows a bar per speaker, updating in real time.

**Options:**
- **Simplest:** Python script prints to terminal; you watch it there.
- **Better:** Python backend sends updates via WebSocket; a simple HTML page draws the bars.
- **Fanciest:** Move inference to the browser with ONNX Runtime and the Web Audio API (no Python needed).

Start simple. A terminal display using Python's `rich` library is satisfying and takes 20 lines.

**File:** `src/ui/terminal.py`

```python
from rich.live import Live
from rich.table import Table

speaker_times = {"Silence": 0.0}

def render():
    table = Table(title="Speaker Time")
    table.add_column("Speaker")
    table.add_column("Seconds")
    for name, secs in speaker_times.items():
        bar = "█" * int(secs)
        table.add_row(name, f"{bar} {secs:.1f}s")
    return table

with Live(render(), refresh_per_second=4) as live:
    # update speaker_times in your audio loop
    live.update(render())
```

**Milestone:** You have the full app working in a terminal.

---

### Step 8 — Wire it all together in main.py
**Goal:** One file that imports from all the modules above and runs the full app end-to-end.

**File:** `main.py`

```python
from src.audio.capture import record_clip
from src.streaming.stream import start_stream
from src.pipeline.diarize import run_diarization
from src.ui.terminal import start_display

if __name__ == "__main__":
    start_display(start_stream(run_diarization))
```

The functions above don't exist yet exactly like this — this is the *shape* you're working toward. As you build each module, you'll refactor it to export a clean function that `main.py` can call. This is what "good project structure" means in practice: each file does one thing, and the entry point just orchestrates them.

---

### Step 9 — Polish and explore
Once it works, you can explore:
- Move the frontend to a web page (Flask + WebSocket + Canvas)
- Try running the model in the browser with ONNX Runtime (no Python server)
- Improve silence detection (energy threshold vs. ML-based VAD — voice activity detection)
- Handle speaker names: let the user label each detected speaker

---

## Tools & Libraries Summary

| Tool | Purpose | Install |
|---|---|---|
| `sounddevice` | Record mic audio | `pip install sounddevice` |
| `librosa` | Audio analysis, spectrograms | `pip install librosa` |
| `pyannote-audio` | Speaker diarization (the ML) | `pip install pyannote-audio` |
| `sklearn` | Cosine similarity, clustering | `pip install scikit-learn` |
| `rich` | Pretty terminal UI | `pip install rich` |
| `flask` | Web server (for browser UI) | `pip install flask` |

**Note:** `pyannote-audio` requires a free Hugging Face account to download the pretrained model weights. Sign up at huggingface.co and accept the model's terms.

---

## What You'll Have Learned by the End

- What features and spectrograms are
- What a neural network does (in practical terms)
- What embeddings are and why they're powerful
- How clustering works
- How to load and use a pre-trained ML model
- How real-time audio streaming works
- How ML pipelines are structured in Python

This is a solid foundation. The same ideas — embeddings, similarity, clustering, pre-trained models — show up everywhere in modern ML: face recognition, music recommendation, semantic search, image similarity. You'll be able to read about those and understand what's happening.

Good luck!
