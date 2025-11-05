"""
Génère un fichier CSV compatible UrbanSound8K à partir du mini-dataset HuggingFace.

Colonnes : slice_file_name, class, classID, fold
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path("data/urban_sounds_small/urban_sounds_small")
META = ROOT / "metadata.csv"

if not META.exists():
    raise FileNotFoundError(f"❌ Fichier introuvable : {META}. Télécharge d'abord le dataset.")

print("📄 Lecture de metadata.csv ...")
df = pd.read_csv(META)  # colonnes: audio, text

# Vérifie quels fichiers audio existent réellement
present = []
for _, r in df.iterrows():
    wav = ROOT / r["text"] / r["audio"]
    if wav.exists():
        present.append({"slice_file_name": r["audio"], "class": r["text"]})
df = pd.DataFrame(present)

if df.empty:
    raise ValueError("❌ Aucun fichier audio valide trouvé dans le dataset.")

# Gère le cas où une classe a <10 fichiers
counts = df["class"].value_counts()
n_splits = min(10, counts.min()) if not counts.empty else 1

# Génère les identifiants de classe
df["classID"] = df["class"].astype("category").cat.codes

# Attribution des folds (ou fold=1 si peu d'échantillons)
if n_splits < 2:
    df["fold"] = 1
else:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = [None] * len(df)
    for f, (_, idx_val) in enumerate(skf.split(df, df["classID"]), start=1):
        for i in idx_val:
            folds[i] = f
    df["fold"] = folds

# Sauvegarde
out = ROOT / "UrbanSound8K_like.csv"
df.to_csv(out, index=False)

print(f"✅ CSV généré : {out}")
print(f"   → {len(df)} lignes, {df['class'].nunique()} classes, folds = {sorted(df['fold'].unique())}")
