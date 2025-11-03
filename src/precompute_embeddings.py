import json
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from panns_inference import AudioTagging
from tqdm import tqdm

from .config import CSV_PATH, SUBSET_DIR, DATA_DIR, AUDIO_DIR
from .data_utils import resolve_audio_path


TARGET_SR = 32000


def _load_metadata() -> Tuple[pd.DataFrame, dict, bool]:
    subset_meta = SUBSET_DIR / "subset_meta.csv"
    use_subset = subset_meta.exists()
    df = pd.read_csv(subset_meta if use_subset else CSV_PATH)

    if "classID" not in df.columns:
        df["classID"] = df["class"].astype("category").cat.codes
    df["classID"] = df["classID"].astype(int)

    class_mapping_df = (
        df[["class", "classID"]]
        .drop_duplicates()
        .sort_values("classID")
    )
    class_to_idx = {row["class"]: int(row["classID"]) for _, row in class_mapping_df.iterrows()}
    df["classID"] = df["class"].map(class_to_idx)

    return df.reset_index(drop=True), class_to_idx, use_subset


def _load_backbone():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AudioTagging(checkpoint_path=None, device=device)
    model.model.eval()
    return model, device


def compute_embeddings():
    df, class_to_idx, use_subset = _load_metadata()
    audio_root = SUBSET_DIR if use_subset else AUDIO_DIR

    print("Using subset csv" if use_subset else "Using full csv")
    print(f"Total files: {len(df)} | Classes: {len(class_to_idx)}")

    model, device = _load_backbone()
    print(f"Backbone loaded (CNN14) on {device}.")

    embeddings, labels, folds = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding", unit="file"):
        wav_path = resolve_audio_path(row, audio_root)
        waveform, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, embedding = model.inference(waveform_tensor)
        embeddings.append(np.squeeze(embedding))  # embedding already numpy, no need for .cpu()
        labels.append(int(row["classID"]))
        if "fold" in df.columns and not pd.isna(row["fold"]):
            fold_val = int(row["fold"])
        else:
            fold_val = -1
        folds.append(fold_val)

    embeddings = np.stack(embeddings).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)
    folds = np.array(folds, dtype=np.int64)

    out_dir = DATA_DIR / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X.npy", embeddings)
    np.save(out_dir / "y.npy", labels)
    np.save(out_dir / "fold.npy", folds)

    ordered_classes = [cls for cls, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    with (out_dir / "classes.json").open("w", encoding="utf-8") as f:
        json.dump({"classes": ordered_classes}, f, ensure_ascii=False, indent=2)

    print(f"✅ Embeddings saved: X.shape={embeddings.shape}, y.shape={labels.shape}")


if __name__ == "__main__":
    compute_embeddings()
