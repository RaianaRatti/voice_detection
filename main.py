import warnings
import os
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from src.audio.capture import record_clip
from src.streaming.stream import start_stream
from src.pipeline.diarize import run_diarization
from src.ui.terminal import start_display

if __name__ == "__main__":
    start_display(start_stream(run_diarization))