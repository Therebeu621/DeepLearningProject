from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from .config import SR, N_MELS, HOP, TARGET_LEN

def wav_to_logmel(path: Path, sr=SR, n_mels=N_MELS, hop_length=HOP, target_len=TARGET_LEN):
    y, sr = librosa.load(path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-6)  # normalisation
    x = torch.tensor(S_db, dtype=torch.float32)        # [n_mels, T]
    T = x.shape[1]
    if T > target_len:
        x = x[:, :target_len]
    elif T < target_len:
        x = F.pad(x, (0, target_len - T))
    return x  # [n_mels, target_len]
