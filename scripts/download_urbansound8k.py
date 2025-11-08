#!/usr/bin/env python3
"""
Télécharge et extrait l'archive officielle UrbanSound8K (~6 Go) depuis Zenodo.
Pratique pour préparer un jeu complet sans passer par Kaggle.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ZIP_URL = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1"


def run(cmd: list[str]):
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Commande échouée ({' '.join(cmd)}): {exc}") from exc


def ensure_download(url: str, archive: Path, force: bool):
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() and not force:
        print(f"ℹ️  Archive déjà présente: {archive}")
        return
    print(f"⬇️  Téléchargement UrbanSound8K → {archive}")
    run(["curl", "-L", "-C", "-", url, "-o", str(archive)])


def extract_archive(archive: Path, dest_dir: Path, force: bool):
    if not archive.exists():
        raise SystemExit(f"Archive introuvable: {archive}. Lance d'abord le téléchargement.")

    if dest_dir.exists():
        if force:
            print(f"♻️  Suppression de {dest_dir} (option --force).")
            shutil.rmtree(dest_dir)
        else:
            print(f"ℹ️  Dossier déjà prêt: {dest_dir}")
            return

    print(f"📦 Extraction vers {dest_dir.parent} ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest_dir.parent)
    print(f"✅ UrbanSound8K disponible sous {dest_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Télécharger/extraire UrbanSound8K complet.")
    parser.add_argument(
        "--dest",
        default="data/UrbanSound8K",
        help="Dossier de destination (contiendra audio/ et metadata/).",
    )
    parser.add_argument(
        "--archive",
        default=None,
        help="Chemin personnalisé pour l'archive .tar.gz (défaut: data/UrbanSound8K.tar.gz).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Suppose que l'archive est déjà présente (désactive curl).",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Télécharge seulement l'archive sans extraction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retelecharge ou ré-extrait même si des fichiers existent déjà.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dest = Path(args.dest).expanduser().resolve()
    archive = Path(args.archive).expanduser().resolve() if args.archive else dest.parent / "UrbanSound8K.tar.gz"

    if not args.skip_download:
        ensure_download(ZIP_URL, archive, args.force)
    else:
        if not archive.exists():
            raise SystemExit(f"Archive manquante: {archive}")
        print("ℹ️  Téléchargement sauté (option --skip-download).")

    if args.skip-extract:
        print("ℹ️  Extraction sautée (option --skip-extract).")
        return

    extract_archive(archive, dest, args.force)


if __name__ == "__main__":
    if shutil.which("curl") is None:
        raise SystemExit("curl est requis pour ce script.")
    main()
