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
    - Sinon, on lit depuis AUDIO_DIR/fold{n}/slice_file_name.
    """
    def __init__(self, df: pd.DataFrame, audio_dir: Path, use_subset_audio: bool = False):
        self.df = df.reset_index(drop=True).copy()
        self.audio_dir = audio_dir
        self.use_subset_audio = use_subset_audio
        self.labels = self.df["classID"].astype(int).tolist()
        self.paths = []
        for _, row in self.df.iterrows():
            if self.use_subset_audio:
                subset_path = SUBSET_DIR / row["slice_file_name"]
                if subset_path.exists():
                    self.paths.append(subset_path)
                    continue
            self.paths.append(resolve_audio_path(row, self.audio_dir))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        x = wav_to_logmel(self.paths[i])  # [n_mels, T]
        return x, self.labels[i]


def _load_state(weights_path: Path):
    """
    Charge un state_dict en mode 'weights_only=True' (plus sûr).
    """
    return torch.load(weights_path, map_location=DEVICE, weights_only=True)


def main():
    # 1) Choix du CSV d'évaluation : subset s’il existe, sinon CSV complet
    subset_meta = SUBSET_DIR / "subset_meta.csv"
    use_subset = subset_meta.exists()
    meta_path = subset_meta if use_subset else CSV_PATH

    meta = pd.read_csv(meta_path)

    # 2) Harmonisation des labels :
    #    on recode systématiquement classID = codes( class ) pour être
    #    aligné avec l'entraînement subset (compact 0..k-1).
    #    Sur le dataset complet (10 classes), cela génère aussi 0..9.
    meta["classID"] = meta["class"].astype("category").cat.codes

    # 3) Split: on garde le fold 10 comme set de validation/évaluation
    if "fold" in meta.columns:
        eval_df = meta[meta["fold"] == 10].copy()
        if len(eval_df) == 0:  # sécurité si jamais pas de fold dans le subset
            eval_df = meta.copy()
    else:
        eval_df = meta.copy()

    # 4) Prépare le DataLoader
    dl = DataLoader(
        US8KDataset(eval_df, AUDIO_DIR, use_subset_audio=use_subset),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # 5) Déduire n_classes depuis le checkpoint, puis construire + charger le modèle
    weights = Path("weights/urbansound_cnn.pt")
    if not weights.exists():
        raise FileNotFoundError(f"Poids introuvables : {weights.resolve()}")

    state = _load_state(weights)
    # La dernière couche est head.1: Linear(out=in_features -> n_classes)
    n_classes = state["head.1.weight"].shape[0]

    model = SimpleCNN(n_classes=n_classes).to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    # 6) Inférence + métriques
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dl:
            logits = model(xb.to(DEVICE))
            y_true.extend(yb.tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())

    # Pour des noms lisibles dans le rapport, on peut essayer de remonter les names
    # à partir des catégories utilisées pour coder classID :
    # (on reconstruit sur l’ensemble meta pour garder l’ordre stable)
    cat = meta["class"].astype("category").cat
    id_to_name = {code: cat.categories[code] for code in range(len(cat.categories))}
    target_names = [id_to_name[i] for i in sorted(set(y_true) | set(y_pred))]

    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
