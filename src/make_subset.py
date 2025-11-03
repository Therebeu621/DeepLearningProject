import shutil
from pathlib import Path

import pandas as pd

from .config import CSV_PATH, AUDIO_DIR, SUBSET_DIR
from .data_utils import resolve_audio_path

# Laisse vide pour choisir automatiquement les classes les plus fréquentes.
TARGET_CLASSES = []
N_PER_CLASS = 100
MAX_CLASSES = 3


def _select_classes(df: pd.DataFrame) -> list[str]:
    unique = df["class"].unique().tolist()
    if TARGET_CLASSES:
        selected = [c for c in TARGET_CLASSES if c in unique]
        if selected:
            return selected
    return (
        df["class"]
        .value_counts()
        .head(MAX_CLASSES)
        .index.tolist()
    )


def main():
    df = pd.read_csv(CSV_PATH)
    classes = _select_classes(df)
    if not classes:
        raise ValueError("Aucune classe disponible pour créer le subset.")

    sub = (
        df[df["class"].isin(classes)]
        .groupby("class", group_keys=False)
        .apply(lambda d: d.sample(min(N_PER_CLASS, len(d)), random_state=42))
        .reset_index(drop=True)
    )

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for _, row in sub.iterrows():
        src = resolve_audio_path(row, AUDIO_DIR)
        dst = SUBSET_DIR / row["slice_file_name"]
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    sub.to_csv(SUBSET_DIR / "subset_meta.csv", index=False)
    print(
        "Subset: {rows} lignes, {files} wav copiés -> {dest}".format(
            rows=len(sub), files=copied, dest=SUBSET_DIR
        )
    )


if __name__ == "__main__":
    main()
