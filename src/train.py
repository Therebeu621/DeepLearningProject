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
    def __init__(self, df, audio_dir: Path):
        self.df = df.reset_index(drop=True).copy()
        self.audio_dir = audio_dir
        self.paths = [resolve_audio_path(r, self.audio_dir) for _, r in self.df.iterrows()]
        self.labels = self.df["classID"].astype(int).tolist()
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        x = wav_to_logmel(self.paths[i])
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

def main(use_subset=True):
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

    train_dl = DataLoader(US8KDataset(train_df, AUDIO_DIR), batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(US8KDataset(val_df, AUDIO_DIR),   batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN(n_classes=len(class_to_idx)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for ep in range(1, EPOCHS+1):
        tr_loss, tr_acc = run_epoch(train_dl, model, loss_fn, opt)
        va_loss, va_acc = run_epoch(val_dl,   model, loss_fn, None)
        print(f"[{ep:02d}] train {tr_acc*100:5.1f}% | val {va_acc*100:5.1f}%  (loss {va_loss:.4f})")

    out = WEIGHTS_DIR / "urbansound_cnn.pt"
    checkpoint = {
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
    }
    torch.save(checkpoint, out)
    print("Saved:", out)

if __name__ == "__main__":
    main()
