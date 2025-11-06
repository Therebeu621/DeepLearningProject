import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch
import soundfile as sf

from .config import DATA_DIR, WEIGHTS_DIR
from .precompute_embeddings import _load_backbone

# constants used during embedding extraction
TARGET_SR = 32000      # sample rate used by PANNs CNN14
TARGET_LEN = 10.0      # clip duration in seconds (padding/truncation)


def _load_classes():
    emb_dir = DATA_DIR / "embeddings"
    with (emb_dir / "classes.json").open("r", encoding="utf-8") as f:
        return json.load(f)["classes"]


def _load_scaler_params(meta):
    mean = np.array(meta["scaler_mean"])
    scale = np.array(meta["scaler_scale"])
    return mean, scale


def _normalize(x, mean, scale):
    return (x - mean) / scale


def _extract_embedding(model, wav_path: Path):
    device = torch.device("cpu")
    target_len_samples = int(TARGET_SR * TARGET_LEN)

    waveform, sr = sf.read(wav_path, always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32, copy=False)

    if sr != TARGET_SR:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR, res_type="kaiser_best")

    if waveform.size == 0:
        waveform = np.zeros(target_len_samples, dtype=np.float32)

    if waveform.shape[0] < target_len_samples:
        pad_width = target_len_samples - waveform.shape[0]
        waveform = np.pad(waveform, (0, pad_width))
    elif waveform.shape[0] > target_len_samples:
        waveform = waveform[:target_len_samples]

    audio = torch.from_numpy(waveform).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.forward_embedding(audio)
    return emb.squeeze(0).cpu().numpy()


def _load_head(weights_path: Path):
    if weights_path.suffix == ".pkl":
        import joblib

        meta = joblib.load(weights_path)
        model = meta["model"]
        mean, scale = _load_scaler_params(meta)
        classes = meta["classes"]
        return ("logreg", model, mean, scale, classes)
    elif weights_path.suffix == ".pt":
        state = torch.load(weights_path, map_location="cpu")
        from .train_embeddings import MLPHead

        model = MLPHead(state["input_dim"], len(state["classes"]))
        model.load_state_dict(state["state_dict"])
        model.eval()
        mean, scale = np.array(state["scaler_mean"]), np.array(state["scaler_scale"])
        classes = state["classes"]
        return ("mlp", model, mean, scale, classes)
    else:
        raise ValueError(f"Format de poids inconnu: {weights_path}")


def predict(wav_path: Path, weights_path: Path):
    model_type, head, mean, scale, classes = _load_head(weights_path)
    backbone = _load_backbone()
    emb = _extract_embedding(backbone, wav_path)
    emb = _normalize(emb, mean, scale)

    if model_type == "logreg":
        probs = head.predict_proba([emb])[0]
    else:
        with torch.no_grad():
            logits = head(torch.from_numpy(emb).float().unsqueeze(0))
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    topk = min(3, len(classes))
    indices = np.argsort(probs)[::-1][:topk]
    print(f"Top-{topk} predictions for {wav_path}:")
    for rank, idx in enumerate(indices, start=1):
        print(f"  {rank}. {classes[idx]} — p={probs[idx]:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inférence via embeddings PANNs")
    parser.add_argument("--wav", required=True, help="Chemin vers le wav à prédire")
    parser.add_argument(
        "--weights",
        help="Chemin vers les poids (pkl ou pt). Par défaut cherche dans weights/linear_head.*",
    )
    args = parser.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    if args.weights:
        weights_path = Path(args.weights)
    else:
        default_pkl = WEIGHTS_DIR / "linear_head.pkl"
        default_pt = WEIGHTS_DIR / "linear_head.pt"
        if default_pkl.exists():
            weights_path = default_pkl
        elif default_pt.exists():
            weights_path = default_pt
        else:
            raise FileNotFoundError("Aucun poids trouvé (linear_head.pkl ou linear_head.pt).")

    predict(wav_path, weights_path)
