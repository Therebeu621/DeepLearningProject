import random
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from .config import SR, N_MELS, HOP, TARGET_LEN


def _time_shift(waveform: np.ndarray, max_shift_pct: float) -> np.ndarray:
    """
    Circularly shift the waveform by up to ±max_shift_pct of its length.
    """
    if max_shift_pct <= 0 or waveform.size == 0:
        return waveform
    max_shift = int(len(waveform) * max_shift_pct)
    if max_shift < 1:
        return waveform
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return waveform
    return np.roll(waveform, shift)


def _add_noise(waveform: np.ndarray, sigma_range: Tuple[float, float]) -> np.ndarray:
    """
    Add zero-mean Gaussian noise with std sampled from sigma_range.
    """
    if waveform.size == 0:
        return waveform
    sigma = np.random.uniform(*sigma_range)
    noise = np.random.normal(0.0, sigma, size=waveform.shape)
    return waveform + noise


def _spec_augment(tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply SpecAugment with 1-2 frequency and time masks.
    """
    x = tensor.clone()
    n_mels, T = x.shape

    for _ in range(random.randint(1, 2)):
        band = random.randint(6, 12)
        start = random.randint(0, max(0, n_mels - band))
        x[start:start + band, :] = 0.0

    for _ in range(random.randint(1, 2)):
        band = random.randint(10, 20)
        start = random.randint(0, max(0, T - band))
        x[:, start:start + band] = 0.0

    return x


def wav_to_logmel(
    path: Path,
    sr: int = SR,
    n_mels: int = N_MELS,
    hop_length: int = HOP,
    target_len: int = TARGET_LEN,
    train_mode: bool = False,
    augment_time: bool = False,
    augment_noise: bool = False,
    augment_spec: bool = False,
    time_shift_pct: float = 0.1,
    noise_sigma: Tuple[float, float] = (0.005, 0.02),
) -> torch.Tensor:
    """
    Convert waveform to a normalized log-mel spectrogram with optional augmentations.
    """
    y, sr = librosa.load(path, sr=sr, mono=True)

    if train_mode and augment_time:
        y = _time_shift(y, time_shift_pct)
    if train_mode and augment_noise:
        y = _add_noise(y, noise_sigma)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-6)
    x = torch.tensor(S_db, dtype=torch.float32)
    T = x.shape[1]
    if T > target_len:
        if train_mode:
            start = random.randint(0, T - target_len)
            x = x[:, start:start + target_len]
        else:
            x = x[:, :target_len]
    elif T < target_len:
        x = F.pad(x, (0, target_len - T))

    if train_mode and augment_spec:
        x = _spec_augment(x)
    return x
