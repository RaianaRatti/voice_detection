import librosa
import numpy as np
import pandas as pd
import webrtcvad
from pathlib import Path
from tqdm import tqdm

SAMPLE_RATE   = 16000
FRAME_MS      = 30
FRAME_SIZE    = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples

LIBRISPEECH_DIR = "data/audio/librispeech"   # where you extracted the tar.gz
OUTPUT_CSV      = "data/labels/librispeech_labels.csv"

vad = webrtcvad.Vad(1)  # aggressiveness 1 — permissive, good for clean audio


def label_file(wav_path: Path) -> list[dict]:
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    audio_int16 = (audio * 32768).astype(np.int16)

    rows = []
    for i in range(0, len(audio_int16) - FRAME_SIZE, FRAME_SIZE):
        frame = audio_int16[i : i + FRAME_SIZE]
        if len(frame) != FRAME_SIZE:
            continue

        is_speech = vad.is_speech(frame.tobytes(), SAMPLE_RATE)
        label = "speech" if is_speech else "silence"

        rows.append({
            "filename": wav_path.name,
            "start_ms": int(i / SAMPLE_RATE * 1000),
            "end_ms":   int((i + FRAME_SIZE) / SAMPLE_RATE * 1000),
            "label":    label
        })

    return rows


def run():
    wav_files = list(Path(LIBRISPEECH_DIR).rglob("*.flac"))
    print(f"Found {len(wav_files)} audio files")

    # copy flac files flat into a single folder so dataset.py can find them
    flat_dir = Path("data/audio/librispeech_flat")
    flat_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for wav_path in tqdm(wav_files):
        # copy to flat dir as wav
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        flat_path = flat_dir / (wav_path.stem + ".wav")
        if not flat_path.exists():
            import soundfile as sf
            sf.write(flat_path, audio, SAMPLE_RATE)

        rows = label_file(flat_path)
        all_rows.extend(rows)

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)

    # balance: cap silence frames so they don't dwarf speech frames
    speech_count  = (df["label"] == "speech").sum()
    silence_count = (df["label"] == "silence").sum()
    print(f"Before balancing — speech: {speech_count}, silence: {silence_count}")

    silence_df = df[df["label"] == "silence"].sample(
        n=min(silence_count, speech_count), random_state=42
    )
    speech_df  = df[df["label"] == "speech"]
    df_balanced = pd.concat([speech_df, silence_df]).sample(frac=1, random_state=42)

    df_balanced.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df_balanced)} labeled frames → {OUTPUT_CSV}")
    print(f"After balancing — speech: {len(speech_df)}, silence: {len(silence_df)}")


if __name__ == "__main__":
    run()