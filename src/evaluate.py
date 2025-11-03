import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix

from .config import CSV_PATH, AUDIO_DIR, SUBSET_DIR, BATCH_SIZE, REPORTS_DIR
from .data_utils import resolve_audio_path
from .features import wav_to_logmel
from .model import SimpleCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class US8KDataset(Dataset):
    """
    Dataset compatible avec UrbanSound8K complet OU subset.
    - Si on évalue sur le subset, on lit les wav copiés dans data/subset/.
    - Sinon, on lit depuis AUDIO_DIR/fold{n}/slice_file_name ou structure par classe.
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
        x = wav_to_logmel(self.paths[i], train_mode=False)  # [n_mels, T]
        return x, self.labels[i]


def _load_state(weights_path: Path):
    """
    Charge le checkpoint et retourne (state_dict, class_to_idx | None).
    """
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        class_to_idx = checkpoint.get("class_to_idx")
    else:
        state = checkpoint
        class_to_idx = None
    return state, class_to_idx


def main():
    # 1) Choix du CSV d'évaluation : subset s’il existe, sinon CSV complet
    subset_meta = SUBSET_DIR / "subset_meta.csv"
    use_subset = subset_meta.exists()
    meta_path = subset_meta if use_subset else CSV_PATH

    meta = pd.read_csv(meta_path)

    # 2) Harmonisation des labels :
    #    on recode systématiquement classID = codes( class ) pour être
    #    aligné avec l'entraînement subset (compact 0..k-1).
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

    state, ckpt_mapping = _load_state(weights)
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

    # 7) Préparation des labels pour l'affichage
    labels = sorted(set(y_true) | set(y_pred))
    if ckpt_mapping:
        idx_to_name = {idx: name for name, idx in ckpt_mapping.items()}
        name_lookup = {i: idx_to_name.get(i, str(i)) for i in labels}
    else:
        cat = meta["class"].astype("category").cat
        id_to_name = {code: cat.categories[code] for code in range(len(cat.categories))}
        name_lookup = {i: id_to_name.get(i, str(i)) for i in labels}

    target_names = [name_lookup[i] for i in labels]

    report_str = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
    )
    print(report_str)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0, output_dict=True
    )

    REPORTS_DIR.mkdir(exist_ok=True)
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    cm_norm_path = REPORTS_DIR / "confusion_matrix_norm.png"
    metrics_path = REPORTS_DIR / "metrics.json"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=target_names,
        yticklabels=target_names,
        ylabel="Vérité terrain",
        xlabel="Prédiction",
        title="Matrice de confusion",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)

    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums != 0)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=target_names,
        yticklabels=target_names,
        ylabel="Vérité terrain",
        xlabel="Prédiction",
        title="Matrice de confusion (normalisée)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )

    fig.tight_layout()
    fig.savefig(cm_norm_path, dpi=200)
    plt.close(fig)

    per_class = {
        name: {
            "precision": float(report_dict[name]["precision"]),
            "recall": float(report_dict[name]["recall"]),
            "f1": float(report_dict[name]["f1-score"]),
            "support": int(report_dict[name]["support"]),
        }
        for name in target_names
        if name in report_dict
    }

    metrics_payload = {
        "accuracy": float(report_dict.get("accuracy", 0.0)),
        "macro_f1": float(report_dict.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(report_dict.get("weighted avg", {}).get("f1-score", 0.0)),
        "per_class": per_class,
        "classification_report": report_dict,
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    print("Confusion matrix saved to:", cm_path)
    print("Normalized confusion matrix saved to:", cm_norm_path)
    print("Metrics JSON saved to:", metrics_path)


if __name__ == "__main__":
    main()
