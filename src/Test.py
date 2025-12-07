
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import joblib

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

MODEL_PATH      = cfg["model_path"]
CENTROID_PATH   = cfg["centroid_path"]
SPOOF_MODEL_PATH = cfg["spoof_model_path"]

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

def cosine_sim(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        help="Path to test wav file (overrides config.yaml test_file)",
        default=None,
    )
    args = parser.parse_args()

    # Decide which file to test
    test_file = args.file if args.file is not None else DEFAULT_TEST_FILE
    test_path = Path(test_file)
    assert test_path.is_file(), f"Test file not found: {test_path}"

    device = "cpu"

    # Load encoder
    encoder = GE2EEncoder()
    encoder.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    encoder.to(device)

    # Load centroid
    centroid = np.load(CENTROID_PATH)

    # Load spoof classifier (SVM / RF / etc.)
    spoof_clf = joblib.load(SPOOF_MODEL_PATH)

    # Embed test file
    emb = embed_utterance(str(test_path), encoder, device=device)

    # Speaker similarity
    score = cosine_sim(emb, centroid)

    # Spoof prediction (0 = LIVE, 1 = REPLAY)
    spoof_label = spoof_clf.predict([emb])[0]
    if hasattr(spoof_clf, "predict_proba"):
        spoof_proba = spoof_clf.predict_proba([emb])[0]
    else:
        spoof_proba = None

    label_name = "LIVE" if spoof_label == 0 else "REPLAY"

    print("====================================")
    print("Test file      :", test_path)
    print("Speaker score  :", score)
    print("Threshold      :", THRESHOLD)
    print("Spoof label    :", label_name)
    if spoof_proba is not None:
        print("Spoof prob     :", spoof_proba)
    print("====================================")

    # Logic:
    # 1) If REPLAY → always deny
    # 2) Else if LIVE + score >= threshold → grant
    # 3) Else → deny

    if spoof_label == 1:
        print("→ DENIED: REPLAY attack detected")
    elif score >= THRESHOLD:
        print("→ ACCESS GRANTED: Varun & LIVE")
    else:
        print("→ DENIED: LIVE but does not sound like Varun enough")

if __name__ == "__main__":
    main()
