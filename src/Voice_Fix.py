import sounddevice as sd
import soundfile as sf
from pathlib import Path

sample_rate = 16000
duration = 3.0  # seconds per recording
num_utterances = 1

out_dir = Path("Recorded Voice Dataset")
out_dir.mkdir(parents=True, exist_ok=True)

print("Recording", num_utterances, "utterances.")
print("Speak a different sentence each time.")

for i in range(1, num_utterances + 1):
    input(f"\nPress ENTER to start recording utterance {i}/{num_utterances}...")
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    filename = out_dir / f"New_varun_Recorded_{i:02d}.wav"
    sf.write(str(filename), audio, sample_rate)
    print("Saved:", filename)

print("\nDone. Your clean WAVs are in data/live/")
