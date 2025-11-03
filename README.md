# DeepLearningProject – UrbanSound8K
## Setup (Ubuntu/WSL)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

## Dataset
- Par défaut, le code cherche un dataset UrbanSound8K dans l'ordre suivant :
  - variable d'environnement `URBAN_SOUND_ROOT` (ou `URBAN_ROOT`)
  - `data/UrbanSound8K`
  - `data/urban_sounds_small/urban_sounds_small` (mini dataset HuggingFace, voir Option A ci-dessous)
  - ancien chemin `/mnt/c/Users/.../UrbanSound8K`
- Les fichiers audio peuvent être organisés par folds (`audio/fold1/...`) ou par classes (`10_music/...`). Les scripts détectent automatiquement la bonne structure.

### Mini-dataset HuggingFace (recommandé pour tester)
1. Télécharge les wav + métadonnées légères (4 classes).
    ```bash
    pip install --no-cache-dir huggingface_hub

    python - <<'PY'
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="UrbanSounds/urban_sounds_small",
        repo_type="dataset",
        local_dir="data/urban_sounds_small",
        allow_patterns=[
            "urban_sounds_small/metadata.csv",
            "urban_sounds_small/01_gunshot/*.wav",
            "urban_sounds_small/05_claxon/*.wav",
            "urban_sounds_small/07_loud_people/*.wav",
            "urban_sounds_small/10_music/*.wav",
        ],
        local_dir_use_symlinks=False,
    )
    print("✅ Téléchargé → data/urban_sounds_small/urban_sounds_small")
    PY
    ```
2. Génère un CSV compatible UrbanSound8K (colonnes `slice_file_name`, `class`, `classID`, `fold`).
    ```bash
    python - <<'PY'
    from pathlib import Path
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold

    ROOT = Path("data/urban_sounds_small/urban_sounds_small")
    df = pd.read_csv(ROOT / "metadata.csv")  # colonnes: audio, text

    present = []
    for _, r in df.iterrows():
        wav = ROOT / r["text"] / r["audio"]
        if wav.exists():
            present.append({"slice_file_name": r["audio"], "class": r["text"]})
    df = pd.DataFrame(present)

    df["classID"] = df["class"].astype("category").cat.codes
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    folds = [None] * len(df)
    for f, (_, idx_val) in enumerate(skf.split(df, df["classID"]), start=1):
        for i in idx_val:
            folds[i] = f
    df["fold"] = folds

    out = ROOT / "UrbanSound8K_like.csv"
    df.to_csv(out, index=False)
    print(f"✅ CSV généré : {out} ({len(df)} lignes, {df['class'].nunique()} classes)")
    PY
    ```

## Run
python -m src.check_data
python -m src.make_subset   # optionnel, crée un mini-jeu data/subset/
python -m src.train
python -m src.evaluate
python -m src.infer
