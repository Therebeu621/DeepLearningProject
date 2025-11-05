from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix

from .config import CSV_PATH, AUDIO_DIR, SUBSET_DIR, BATCH_SIZE
from .data_utils import resolve_audio_path
from .features import wav_to_logmel
from .model import SimpleCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class US8KDataset(Dataset):
    """
    Dataset compatible avec UrbanSound8K complet OU subset.
    - Si on évalue sur le subset, on lit les wav copiés dans data/subset/.
    - Sinon, on lit depuis AUDIO_DIR (structure fold*/ ou par classes).
    """
    def __init__(self, df: pd.DataFrame, audio_dir: Path, use_subset_audio: bool = False):
        self.df = df.reset_index(drop=True).copy()
        self.audio_dir = audio_dir
        self.use_subset_audio = use_subset_audio

        self.labels = self.df["classID"].astype(int).tolist()
        self.paths = []
        for _, row in self.df.iterrows():
            # 1) Priorité: wav copié dans data/subset/
            if self.use_subset_audio:
                subset_path = SUBSET_DIR / row["slice_file_name"]
                if subset_path.exists():
                    self.paths.append(subset_path)
                    continue
            # 2) Fallback: résolution automatique dans le dataset source
            self.paths.append(resolve_audio_path(row, self.audio_dir))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        x = wav_to_logmel(self.paths[i])  # [n_mels, T]
        return x, self.labels[i]


def _load_state(weights_path: Path):
    """
    Charge un state_dict (mode weights_only si dispo).
    """
    return torch.load(weights_path, map_location=DEVICE, weights_only=True)


def _find_weights() -> Path:
    """
    Tente plusieurs emplacements possibles pour le fichier de poids entraîné.
    - DeepLearningProject/weights/urbansound_cnn.pt
    - weights/urbansound_cnn.pt
    - chemin relatif au fichier courant (src/..)
    """
    this_file = Path(__file__).resolve()
    pkg_root = this_file.parents[1]            # .../DeepLearningProject
    repo_root = pkg_root.parent                # .../(racine du repo)

    candidates = [
        pkg_root / "weights" / "urbansound_cnn.pt",
        repo_root / "weights" / "urbansound_cnn.pt",
        Path.cwd() / "weights" / "urbansound_cnn.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    # dernier recours: emplacement historique
    fallback = Path("weights/urbansound_cnn.pt")
    return fallback


def main():
    # 1) Choix du CSV d'évaluation : subset s’il existe, sinon CSV complet
    subset_meta = SUBSET_DIR / "subset_meta.csv"
    use_subset = subset_meta.exists()
    meta_path = subset_meta if use_subset else CSV_PATH

    meta = pd.read_csv(meta_path)

    # 2) Harmonisation des labels (compact 0..k-1) quel que soit le CSV
    meta["classID"] = meta["class"].astype("category").cat.codes

    # 3) Split: on garde le fold 10 comme set d'évaluation si présent
    if "fold" in meta.columns:
        eval_df = meta[meta["fold"] == 10].copy()
        if len(eval_df) == 0:
            eval_df = meta.copy()
    else:
        eval_df = meta.copy()

    # 4) DataLoader
    dl = DataLoader(
        US8KDataset(eval_df, AUDIO_DIR, use_subset_audio=use_subset),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # 5) Localise et charge les poids + construit le modèle avec la bonne sortie
    weights = _find_weights()
    if not weights.exists():
        raise FileNotFoundError(f"Poids introuvables : {weights.resolve()}")

    state = _load_state(weights)
    # La dernière couche est head.1: Linear(... -> n_classes)
    n_classes = state["head.1.weight"].shape[0]

    model = SimpleCNN(n_classes=n_classes).to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    # 6) Inférence
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dl:
            logits = model(xb.to(DEVICE))
            y_true.extend(yb.tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())

    # 7) Noms de classes stables depuis le CSV complet si possible
    #    (pour éviter KeyError si le fold n'expose pas toutes les classes)
    id_to_name_full = {}
    try:
        full_meta = pd.read_csv(CSV_PATH)
        full_meta["classID"] = full_meta["class"].astype("category").cat.codes
        cat = full_meta["class"].astype("category").cat
        id_to_name_full = {code: cat.categories[code] for code in range(len(cat.categories))}
    except Exception:
        pass  # on tombera sur "class_{i}" si indisponible

    # On ne rapporte que les étiquettes réellement rencontrées (y_true ∪ y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    target_names = [id_to_name_full.get(i, f"class_{i}") for i in labels]

    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0))
    print(confusion_matrix(y_true, y_pred, labels=labels))


if __name__ == "__main__":
    main()
