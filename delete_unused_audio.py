# delete_unused_audio.py
import pandas as pd
from pathlib import Path
import shutil

df = pd.read_csv('train_data/labels/all_labels.csv')
used_files = set()

for filename in df['filename']:
    # Extract base audio directory
    parts = Path(filename).parts
    used_files.add(parts[0])  # e.g., "librispeech", "ami", "ESC-50"

print(f"Used datasets: {used_files}")

# Delete unused directories
for dataset_dir in Path("train_data/audio").iterdir():
    if dataset_dir.is_dir() and dataset_dir.name not in used_files:
        print(f"Deleting {dataset_dir.name}...")
        shutil.rmtree(dataset_dir)

print("✓ Cleanup complete")