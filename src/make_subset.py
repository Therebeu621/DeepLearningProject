from pathlib import Path
import shutil
import pandas as pd
from .config import CSV_PATH, AUDIO_DIR, SUBSET_DIR

CLASSES = ["dog_bark", "siren", "car_horn"]
N_PER_CLASS = 100

def main():
    df = pd.read_csv(CSV_PATH)
    sub = (df[df["class"].isin(CLASSES)]
           .groupby("class", group_keys=False)
           .apply(lambda d: d.sample(min(N_PER_CLASS, len(d)), random_state=42))
           .reset_index(drop=True))

    # copie des wav vers data/subset/
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for _, r in sub.iterrows():
        fold = f"fold{int(r['fold'])}"
        src = AUDIO_DIR / fold / r["slice_file_name"]
        dst = SUBSET_DIR / r["slice_file_name"]
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    sub.to_csv(SUBSET_DIR / "subset_meta.csv", index=False)
    print(f"Subset: {len(sub)} lignes, {copied} fichiers copiés -> {SUBSET_DIR}")

if __name__ == "__main__":
    main()
