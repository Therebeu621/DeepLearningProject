from pathlib import Path
import pandas as pd
import torch

from .config import CSV_PATH, SUBSET_DIR, AUDIO_DIR
from .data_utils import resolve_audio_path
from .features import wav_to_logmel
from .model import SimpleCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_state(weights_path: Path):
    # Chargement sécurisé des poids
    return torch.load(weights_path, map_location=DEVICE, weights_only=True)


def _load_labels_for_checkpoint(n_classes: int):
    """
    Essaie d'inférer l'ordre des labels (noms lisibles) depuis les métadonnées.
    Priorité :
      1) subset_meta.csv si présent (cohérent avec un entraînement subset)
      2) CSV complet UrbanSound8K
    On recode : classID = codes(class) afin d'obtenir un mapping 0..k-1
    correspondant à l'entraînement.
    Si le nombre de catégories != n_classes, on retourne None (fallback classID).
    """
    # 1) subset si dispo
    subset_meta = SUBSET_DIR / "subset_meta.csv"
    if subset_meta.exists():
        meta = pd.read_csv(subset_meta)
    else:
        # 2) sinon, CSV complet
        meta = pd.read_csv(CSV_PATH)

    # Harmonise: classID = codes(class)
    cat = meta["class"].astype("category").cat
    labels = list(cat.categories)

    # Vérifie la cohérence avec le checkpoint
    if len(labels) != n_classes:
        return None
    return labels


def predict_one(wav_path: Path, weights: str = "weights/urbansound_cnn.pt", labels=None):
    x = wav_to_logmel(wav_path).unsqueeze(0)  # [1, n_mels, T]

    # Déduire dynamiquement le nb de classes depuis le checkpoint
    state = _load_state(Path(weights))
    n_classes = state["head.1.weight"].shape[0]

    # Si labels non fournis, tentative auto
    if labels is None:
        labels = _load_labels_for_checkpoint(n_classes)

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


if __name__ == "__main__":
    # Exemple : prend un .wav du subset si présent, sinon message explicite
    example = None
    subset_dir = Path("data/subset")
    if subset_dir.exists():
        for p in subset_dir.glob("*.wav"):
            example = p
            break

    if example is None:
        meta_candidates = [
            SUBSET_DIR / "subset_meta.csv",
            CSV_PATH,
        ]
        for meta_path in meta_candidates:
            if meta_path.exists():
                meta = pd.read_csv(meta_path)
                if not meta.empty:
                    example = resolve_audio_path(meta.iloc[0], AUDIO_DIR)
                    break

    if example is None:
        raise FileNotFoundError("Aucun wav accessible. Fourni un chemin à predict_one().")

    predict_one(example)
