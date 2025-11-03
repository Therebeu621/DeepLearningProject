import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .config import DATA_DIR, WEIGHTS_DIR, REPORTS_DIR, SEED


torch.manual_seed(SEED)
np.random.seed(SEED)

EMB_DIR = DATA_DIR / "embeddings"


def _load_embeddings():
    X = np.load(EMB_DIR / "X.npy")
    y = np.load(EMB_DIR / "y.npy")
    fold_path = EMB_DIR / "fold.npy"
    folds = np.load(fold_path) if fold_path.exists() else np.full(len(X), -1, dtype=np.int64)
    with (EMB_DIR / "classes.json").open("r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]
    return X, y, folds, classes


def _train_val_split(folds, y):
    valid_mask = folds >= 0
    unique_folds = np.unique(folds[valid_mask])
    if valid_mask.any() and len(unique_folds) >= 2:
        val_fold = unique_folds.max()
        train_idx = np.where(folds != val_fold)[0]
        val_idx = np.where(folds == val_fold)[0]
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(skf.split(np.zeros_like(y), y))
    return train_idx, val_idx


def _plot_confusion(cm, labels, path, title, normalize=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0.0 if normalize else None, vmax=1.0 if normalize else None)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Vérité terrain",
        xlabel="Prédiction",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j] if not normalize else cm[i, j]
            text = f"{value:.2f}" if normalize else str(value)
            color = "white" if (normalize and value > 0.5) or (not normalize and value > cm.max() / 2) else "black"
            ax.text(j, i, text, ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


class MLPHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_logreg(X_train, y_train):
    model = LogisticRegression(max_iter=2000, solver="liblinear", multi_class="ovr")
    model.fit(X_train, y_train)
    return model


def train_mlp(X_train, y_train, X_val, y_val, num_classes, epochs=200, patience=20):
    device = torch.device("cpu")
    model = MLPHead(X_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb.to(device))
            loss = criterion(logits, yb.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * yb.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                loss = criterion(logits, yb.to(device))
                val_loss += loss.item() * yb.size(0)

        val_loss /= len(val_loader.dataset)
        train_loss /= len(train_loader.dataset)

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def main(model_type: str):
    X, y, folds, classes = _load_embeddings()
    train_idx, val_idx = _train_val_split(folds, y)

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    REPORTS_DIR.mkdir(exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)

    if model_type == "logreg":
        model = train_logreg(X_train, y_train)
        y_pred = model.predict(X_val)
        weights_path = WEIGHTS_DIR / "linear_head.pkl"
        joblib.dump({"model": model, "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "classes": classes}, weights_path)
    else:
        mlp = train_mlp(X_train, y_train, X_val, y_val, num_classes=len(classes))
        with torch.no_grad():
            logits = mlp(torch.from_numpy(X_val).float())
            y_pred = torch.argmax(logits, dim=1).numpy()
        weights_path = WEIGHTS_DIR / "linear_head.pt"
        torch.save(
            {
                "state_dict": mlp.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "classes": classes,
                "input_dim": X.shape[1],
            },
            weights_path,
        )

    report = classification_report(y_val, y_pred, target_names=classes, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_val, y_pred)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums != 0)
    cm_norm = np.nan_to_num(cm_norm)

    _plot_confusion(cm, classes, REPORTS_DIR / "confusion_matrix_embeddings.png", "Matrice de confusion (embeddings)")
    _plot_confusion(cm_norm, classes, REPORTS_DIR / "confusion_matrix_embeddings_norm.png", "Matrice de confusion normalisée (embeddings)", normalize=True)

    metrics_payload = {
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
        "classification_report": report,
        "weights_path": str(weights_path),
        "model_type": model_type,
    }

    with (REPORTS_DIR / "metrics_embeddings.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    print(f"Saved metrics to {REPORTS_DIR / 'metrics_embeddings.json'}")
    print(f"Saved weights to {weights_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classifier on precomputed embeddings")
    parser.add_argument("--model", choices=["logreg", "mlp"], default="logreg", help="Type de tête de classification")
    args = parser.parse_args()
    main(args.model)
