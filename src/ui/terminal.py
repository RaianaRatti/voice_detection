# Terminal display with bar for each speaker, updating real time

def start_display(data_source):
    total_times = {}

    for speaker_times in data_source:
        if isinstance(speaker_times, dict):
            for speaker, secs in speaker_times.items():
                total_times[speaker] = total_times.get(speaker, 0) + secs

    if not total_times:
        print("No speakers detected.")
        return

    print("\nFinal speaker times:")
    for speaker, secs in total_times.items():
        bar = "█" * int(secs)
        print(f"  {speaker}: {bar} {secs:.1f}s")