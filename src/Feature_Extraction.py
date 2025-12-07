from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

MODEL_PATH    = cfg["model_path"]
CENTROID_PATH = cfg["centroid_path"]
LIVE_DIR      = Path(cfg["live_dir"])
REPLAY_DIR    = Path(cfg["Recorded_Voice_Dataset"])
SAMPLE_RATE   = cfg["sample_rate"]
N_MELS        = cfg["n_mels"]
WIN_LENGTH    = int(cfg["win_ms"] / 1000 * SAMPLE_RATE)
HOP_LENGTH    = int(cfg["hop_ms"] / 1000 * SAMPLE_RATE)
THRESHOLD     = cfg["threshold"]

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

encoder = GE2EEncoder()
encoder.load_state_dict(torch.load("ge2e_encoder.pth", map_location="cpu"))
encoder.eval()

X = []     
y = []     

def add_folder(folder, label, X, y):
    print("Processing folder:", folder)
    for wav_path in sorted(folder.glob("*.wav")):
        emb = embed_utterance(str(wav_path), encoder)
        X.append(emb)
        y.append(label)

add_folder(LIVE_DIR, 0, X, y)
add_folder(REPLAY_DIR, 1, X, y)

#print("LIVE_DIR:", LIVE_DIR, list(LIVE_DIR.glob("*.wav")))
#print("REPLAY_DIR:", REPLAY_DIR, list(REPLAY_DIR.glob("*.wav")))

print("len(X) =", len(X))
X = np.vstack(X)
y = np.array(y)

np.save("X_spoof.npy", X)
np.save("y_spoof.npy", y)

print("Saved features:", X.shape, y.shape)
