import argparse
import copy
import random
from pathlib import Path

import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm

from .config import (
    URBAN_ROOT,
    CSV_PATH,
    AUDIO_DIR,
    WEIGHTS_DIR,
    BATCH_SIZE,
    EPOCHS,
    LR,
    SEED,
)
from .data_utils import resolve_audio_path
from .features import wav_to_logmel
from .model import SimpleCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
try:
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
except AttributeError:
    pass


class US8KDataset(Dataset):
    def __init__(
        self,
        df,
        audio_dir: Path,
        train_mode: bool = False,
        augment_time: bool = False,
        augment_noise: bool = False,
        augment_spec: bool = False,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.audio_dir = audio_dir
        self.paths = [resolve_audio_path(r, self.audio_dir) for _, r in self.df.iterrows()]
        self.labels = self.df["classID"].astype(int).tolist()
        self.train_mode = train_mode
        self.augment_time = augment_time
        self.augment_noise = augment_noise
        self.augment_spec = augment_spec

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        x = wav_to_logmel(
            self.paths[i],
            train_mode=self.train_mode,
            augment_time=self.augment_time,
            augment_noise=self.augment_noise,
            augment_spec=self.augment_spec,
        )
        return x, self.labels[i]

def run_epoch(loader, model, loss_fn, opt=None):
    train = opt is not None
    model.train(train)
    tot_loss, correct, total = 0.0, 0, 0
    for xb, yb in tqdm(loader, disable=True):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        tot_loss += loss.item()*yb.size(0)
        correct += (logits.argmax(1)==yb).sum().item()
        total += yb.size(0)
    return tot_loss/total, correct/total

def main(
    use_subset: bool = True,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
    aug_time: bool = False,
    aug_noise: bool = False,
    aug_spec: bool = False,
):
    meta = pd.read_csv(CSV_PATH)
    if use_subset and (Path("data/subset/subset_meta.csv").exists()):
        meta = pd.read_csv("data/subset/subset_meta.csv")
        # Remap class -> classID (0..k-1) si subset
        meta["classID"] = meta["class"].astype('category').cat.codes
    elif "classID" not in meta.columns:
        meta["classID"] = meta["class"].astype("category").cat.codes

    meta["classID"] = meta["classID"].astype(int)

    class_mapping_df = (
        meta[["class", "classID"]]
        .drop_duplicates()
        .sort_values("classID")
    )
    class_to_idx = {row["class"]: int(row["classID"]) for _, row in class_mapping_df.iterrows()}

    train_df = meta[meta["fold"] != 10].copy()
    val_df   = meta[meta["fold"] == 10].copy()

    print("=== Training configuration ===")
    print(f"URBAN_ROOT : {URBAN_ROOT}")
    print(f"CSV_PATH   : {CSV_PATH}")
    print(f"N_CLASSES  : {len(class_to_idx)}")
    print(f"Train size : {len(train_df)} | Val size : {len(val_df)}")
    print(f"Batch size : {batch_size} | Epochs : {epochs} | LR : {lr}")
    print(
        "Augmentations : "
        f"time_shift={'ON' if aug_time else 'off'} | "
        f"noise={'ON' if aug_noise else 'off'} | "
        f"spec_aug={'ON' if aug_spec else 'off'}"
    )

    train_ds = US8KDataset(
        train_df,
        AUDIO_DIR,
        train_mode=True,
        augment_time=aug_time,
        augment_noise=aug_noise,
        augment_spec=aug_spec,
    )
    val_ds = US8KDataset(val_df, AUDIO_DIR, train_mode=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = SimpleCNN(n_classes=len(class_to_idx)).to(DEVICE)

    counts = train_df["classID"].value_counts().reindex(range(len(class_to_idx)), fill_value=1)
    weights = (1.0 / counts.values)
    weights = weights / weights.sum() * len(class_to_idx)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    print("Class weights :", weights)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    patience = 5

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(train_dl, model, loss_fn, opt)
        va_loss, va_acc = run_epoch(val_dl,   model, loss_fn, None)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"[{ep:02d}] train {tr_acc*100:5.1f}% | val {va_acc*100:5.1f}%  "
            f"(loss {va_loss:.4f}) | lr {current_lr:.2e}"
        )

        if va_loss < best_val_loss - 1e-4:
            best_val_loss = va_loss
            best_epoch = ep
            epochs_no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {ep}. Best val loss {best_val_loss:.4f} (epoch {best_epoch}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    out = WEIGHTS_DIR / "urbansound_cnn.pt"
    checkpoint = {
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
    }
    torch.save(checkpoint, out)
    print("Saved:", out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UrbanSound8K classifier.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Nombre d'époques d'entraînement.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Taille de batch.")
    parser.add_argument("--lr", type=float, default=LR, help="Taux d'apprentissage.")
    parser.add_argument(
        "--no-subset",
        action="store_false",
        dest="use_subset",
        help="Utiliser le CSV complet même si un subset est présent.",
    )
    parser.add_argument("--aug-time", action="store_true", help="Active le time-shift aléatoire.")
    parser.add_argument("--aug-noise", action="store_true", help="Ajoute un bruit gaussien léger (σ 0.005→0.02).")
    parser.add_argument("--aug-spec", action="store_true", help="Applique un SpecAugment (masques temps/fréquence).")
    args = parser.parse_args()
    main(
        use_subset=args.use_subset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        aug_time=args.aug_time,
        aug_noise=args.aug_noise,
        aug_spec=args.aug_spec,
    )
