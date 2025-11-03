import random
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from .config import SR, N_MELS, HOP, TARGET_LEN


def _time_shift(waveform: np.ndarray, max_shift_pct: float) -> np.ndarray:
    if max_shift_pct <= 0 or waveform.size == 0:
        return waveform
    max_shift = int(len(waveform) * max_shift_pct)
    if max_shift < 1:
        return waveform
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return waveform
    shifted = np.zeros_like(waveform)
    if shift > 0:
        shifted[: -shift] = waveform[shift:]
    else:
        shift = -shift
        shifted[shift:] = waveform[:-shift]
    return shifted


def _add_noise(waveform: np.ndarray, snr_range: Tuple[float, float]) -> np.ndarray:
    if waveform.size == 0:
        return waveform
    snr_db = np.random.uniform(*snr_range)
    signal_rms = np.sqrt(np.mean(waveform**2))
    if signal_rms == 0:
        return waveform
    noise_rms = signal_rms / (10 ** (snr_db / 20))
    noise = np.random.normal(0, noise_rms, size=waveform.shape)
    return waveform + noise


def _spec_augment(tensor: torch.Tensor, max_freq_masks: int = 2, max_time_masks: int = 2) -> torch.Tensor:
    x = tensor.clone()
    n_mels, T = x.shape

    for _ in range(random.randint(1, max_freq_masks)):
        max_band = max(1, int(n_mels * 0.15))
        band = random.randint(1, max_band)
        start = random.randint(0, max(0, n_mels - band))
        x[start:start + band, :] = 0.0

    for _ in range(random.randint(1, max_time_masks)):
        max_band = max(1, int(T * 0.15))
        band = random.randint(1, max_band)
        start = random.randint(0, max(0, T - band))
        x[:, start:start + band] = 0.0

    return x


def wav_to_logmel(
    path: Path,
    sr: int = SR,
    n_mels: int = N_MELS,
    hop_length: int = HOP,
    target_len: int = TARGET_LEN,
    augment_time: bool = False,
    augment_noise: bool = False,
    augment_spec: bool = False,
    time_shift_pct: float = 0.1,
    noise_snr_db: Tuple[float, float] = (-20.0, -10.0),
) -> torch.Tensor:
    y, sr = librosa.load(path, sr=sr, mono=True)

    if augment_time:
        y = _time_shift(y, time_shift_pct)
    if augment_noise:
        y = _add_noise(y, noise_snr_db)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-6)
    x = torch.tensor(S_db, dtype=torch.float32)
    T = x.shape[1]
    if T > target_len:
        x = x[:, :target_len]
    elif T < target_len:
        x = F.pad(x, (0, target_len - T))

    if augment_spec:
        x = _spec_augment(x)
    return x
