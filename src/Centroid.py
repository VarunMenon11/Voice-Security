# build_centroid.py
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf

# ---- load config ----
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

LIVE_DIR      = Path(cfg["live_dir"])
MODEL_PATH    = cfg["model_path"]
CENTROID_PATH = cfg["centroid_path"]

SAMPLE_RATE = cfg["sample_rate"]
N_MELS      = cfg["n_mels"]
WIN_LENGTH  = int(cfg["win_ms"] / 1000 * SAMPLE_RATE)
HOP_LENGTH  = int(cfg["hop_ms"] / 1000 * SAMPLE_RATE)

# ---- model ----
class GE2EEncoder(nn.Module):
    def __init__(self, n_mels=N_MELS, hidden_size=256, proj_size=256, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_size, proj_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.proj(out)
        out = F.normalize(out, p=2, dim=1)
        return out

mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=512,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

def load_wav(path):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.from_numpy(data).float().unsqueeze(0)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav

def embed_utterance(path, encoder, device="cpu"):
    wav = load_wav(path)
    with torch.no_grad():
        mel = mel_spec(wav)
        mel_db = amplitude_to_db(mel).squeeze(0)
    mel_db = mel_db.transpose(0, 1).unsqueeze(0).to(device)
    encoder.eval()
    with torch.no_grad():
        emb = encoder(mel_db)
    return emb.squeeze(0).cpu().numpy()

def main():
    assert LIVE_DIR.is_dir(), f"{LIVE_DIR} not found"

    device = "cpu"
    encoder = GE2EEncoder()
    encoder.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    encoder.to(device)

    embs = []
    for wav_path in sorted(LIVE_DIR.glob("*.wav")):
        print("Processing:", wav_path)
        embs.append(embed_utterance(str(wav_path), encoder, device))

    embs = np.stack(embs, axis=0)
    centroid = embs.mean(axis=0)
    np.save(CENTROID_PATH, centroid)
    print(f"Saved centroid to {CENTROID_PATH}, shape: {centroid.shape}")

if __name__ == "__main__":
    main()
