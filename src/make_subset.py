import argparse
import shutil
from pathlib import Path
from typing import List

import pandas as pd

from .config import CSV_PATH, AUDIO_DIR, SUBSET_DIR
from .data_utils import resolve_audio_path

# Laisse vide pour choisir automatiquement les classes les plus fréquentes.
TARGET_CLASSES: List[str] = [
    "01_gunshot",
    "03_moped_alarm",
    "04_moped",
    "05_claxon",
    "06_car_door",
    "07_loud_people",
    "08_motor_cycle",
    "09_terrace_noise",
    "10_music",
]
MAX_CLASSES = 9


def _select_classes(df: pd.DataFrame) -> List[str]:
    """Retourne les classes à inclure dans le subset."""
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


def main(n_per_class: int = 200):
    """Génère un subset équilibré en copiant les wavs nécessaires."""
    df = pd.read_csv(CSV_PATH)
    classes = _select_classes(df)
    if not classes:
        raise ValueError("Aucune classe disponible pour créer le subset.")

    sub = (
        df[df["class"].isin(classes)]
        .groupby("class", group_keys=False)
        .apply(lambda d: d.sample(min(n_per_class, len(d)), random_state=42))
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
    parser = argparse.ArgumentParser(description="Créer un subset UrbanSound.")
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=200,
        help="Nombre maximum d'échantillons par classe.",
    )
    args = parser.parse_args()
    main(n_per_class=args.n_per_class)
