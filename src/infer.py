import argparse
from pathlib import Path
import pandas as pd
import torch

from .config import CSV_PATH, SUBSET_DIR, AUDIO_DIR
from .data_utils import resolve_audio_path
from .features import wav_to_logmel
from .model import SimpleCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_state(weights_path: Path):
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        class_to_idx = checkpoint.get("class_to_idx")
    else:
        state = checkpoint
        class_to_idx = None
    return state, class_to_idx


def _load_labels_for_checkpoint(n_classes: int, saved_mapping=None):
    """
    Essaie d'inférer l'ordre des labels (noms lisibles) depuis le checkpoint
    ou, à défaut, depuis les métadonnées.
    """
    if saved_mapping:
        idx_to_name = {idx: name for name, idx in saved_mapping.items()}
        return [idx_to_name[i] for i in range(n_classes) if i in idx_to_name]

    subset_meta = SUBSET_DIR / "subset_meta.csv"
    if subset_meta.exists():
        meta = pd.read_csv(subset_meta)
    else:
        meta = pd.read_csv(CSV_PATH)

    cat = meta["class"].astype("category").cat
    labels = list(cat.categories)
    if len(labels) != n_classes:
        return None
    return labels


def predict_one(wav_path: Path, weights: Path, labels=None):
    """
    Charge un wav et affiche la prédiction top-1 + top-k.
    """
    x = wav_to_logmel(wav_path, train_mode=False).unsqueeze(0)  # [1, n_mels, T]

    state, saved_mapping = _load_state(weights)
    n_classes = state["head.1.weight"].shape[0]

    if labels is None:
        labels = _load_labels_for_checkpoint(n_classes, saved_mapping)

    model = SimpleCNN(n_classes=n_classes).to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.no_grad():
        logits = model(x.to(DEVICE))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    idx = int(probs.argmax())
    if labels is not None and 0 <= idx < len(labels):
        print(f"Pred: {labels[idx]}  (p={probs[idx]:.2f})")
    else:
        print(f"Pred classID: {idx} (p={probs[idx]:.2f})")

    k = min(3, len(probs))
    topk = torch.topk(torch.from_numpy(probs), k)
    print("Top-k:")
    for rank, (score, class_idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist()), start=1):
        if labels is not None and 0 <= class_idx < len(labels):
            name = labels[class_idx]
        else:
            name = f"classID={class_idx}"
        print(f"  {rank}. {name} — p={score:.2f}")


def _default_example() -> Path:
    subset_dir = Path("data/subset")
    if subset_dir.exists():
        for p in subset_dir.glob("*.wav"):
            return p

    meta_candidates = [
        SUBSET_DIR / "subset_meta.csv",
        CSV_PATH,
    ]
    for meta_path in meta_candidates:
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            if not meta.empty:
                return resolve_audio_path(meta.iloc[0], AUDIO_DIR)
    raise FileNotFoundError("Aucun wav accessible. Fournis --wav PATH.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inférence sur un fichier UrbanSound.")
    parser.add_argument("--wav", type=str, help="Chemin vers un fichier .wav à prédire.")
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/urbansound_cnn.pt",
        help="Chemin vers le checkpoint torch à utiliser.",
    )
    args = parser.parse_args()

    wav_path = Path(args.wav) if args.wav else _default_example()
    if not wav_path.exists():
        raise FileNotFoundError(f"Fichier wav introuvable: {wav_path}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {weights_path}")

    predict_one(wav_path, weights=weights_path)
