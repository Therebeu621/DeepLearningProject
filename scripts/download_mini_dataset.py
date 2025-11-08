"""
Télécharge un extrait (ou la totalité) du dataset UrbanSounds/urban_sounds_small.
On peut cibler quelques classes pour des tests rapides ou tout rapatrier pour
former un jeu plus riche.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Optional

from huggingface_hub import snapshot_download

DEFAULT_CLASSES = [
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


def _build_allow_patterns(classes: Optional[Iterable[str]]) -> Optional[List[str]]:
    if classes is None:
        return None  # téléchargement complet
    classes = [c.strip() for c in classes if c.strip()]
    if not classes:
        return ["urban_sounds_small/metadata.csv"]
    patterns = ["urban_sounds_small/metadata.csv"]
    patterns.extend([f"urban_sounds_small/{cls}/*.wav" for cls in classes])
    return patterns


def main():
    parser = argparse.ArgumentParser(description="Télécharger UrbanSounds/urban_sounds_small.")
    parser.add_argument(
        "--classes",
        type=str,
        help="Liste de classes séparées par des virgules. Par défaut : 9 classes cibles.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Télécharge toutes les classes du dataset (plus lourd mais plus riche).",
    )
    args = parser.parse_args()

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    if args.full:
        allow_patterns = None
        msg = "🚀 Téléchargement COMPLET du dataset UrbanSounds/urban_sounds_small..."
    else:
        classes = DEFAULT_CLASSES if args.classes is None else args.classes.split(",")
        allow_patterns = _build_allow_patterns(classes)
        msg = (
            "🚀 Téléchargement ciblé (classes: "
            + ", ".join(classes if classes else ["metadata only"])
            + ")..."
        )

    print(msg)
    snapshot_download(
        repo_id="UrbanSounds/urban_sounds_small",
        repo_type="dataset",
        local_dir="data/urban_sounds_small",
        allow_patterns=allow_patterns,
        resume_download=True,
    )
    print("✅ Téléchargé → data/urban_sounds_small/urban_sounds_small")


if __name__ == "__main__":
    main()
