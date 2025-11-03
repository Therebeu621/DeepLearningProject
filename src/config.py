# src/config.py
from pathlib import Path
import os

# --- Dossiers projet ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
SUBSET_DIR   = DATA_DIR / "subset"
WEIGHTS_DIR  = PROJECT_ROOT / "weights"
REPORTS_DIR  = PROJECT_ROOT / "reports"
WEIGHTS_DIR.mkdir(exist_ok=True)
SUBSET_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# --- Détection de la racine dataset ---
def _candidate_roots():
    # Priorité aux variables d'env
    env = os.getenv("URBAN_SOUND_ROOT") or os.getenv("URBAN_ROOT")
    if env:
        yield Path(env).expanduser()

    # Dossiers standards du repo (mini-dataset HF puis US8K)
    yield DATA_DIR / "urban_sounds_small" / "urban_sounds_small"
    yield DATA_DIR / "UrbanSound8K"
    # (AUCUN chemin personnel ici)

URBAN_ROOT = None
for candidate in _candidate_roots():
    if candidate.exists():
        URBAN_ROOT = candidate
        break
if URBAN_ROOT is None:
    raise FileNotFoundError(
        "Aucune racine de dataset trouvée. "
        "Définis URBAN_SOUND_ROOT ou place le dataset dans data/urban_sounds_small/urban_sounds_small "
        "ou data/UrbanSound8K."
    )

# --- Sélection du CSV ---
CSV_CANDIDATES = (
    URBAN_ROOT / "UrbanSound8K_like.csv",           # CSV généré pour mini-dataset
    URBAN_ROOT / "metadata.csv",                     # mini-dataset HF
    URBAN_ROOT / "UrbanSound8K.csv",                 # CSV à la racine
    URBAN_ROOT / "metadata" / "UrbanSound8K.csv",    # CSV dans metadata/
)
CSV_PATH = None
for p in CSV_CANDIDATES:
    if p.exists():
        CSV_PATH = p
        break
if CSV_PATH is None:
    raise FileNotFoundError(
        f"Impossible de trouver un CSV (candidats: {', '.join(str(p) for p in CSV_CANDIDATES)})"
    )

# --- Dossier audio ---
AUDIO_DIR = URBAN_ROOT / "audio" if (URBAN_ROOT / "audio").exists() else URBAN_ROOT

# --- Hyperparams audio / features ---
SR         = 22050
N_MELS     = 64
HOP        = 512
TARGET_LEN = 173  # ~4s à 22.05 kHz avec hop 512 (ajuste si besoin)

# --- Hyperparams train ---
BATCH_SIZE = 64
EPOCHS     = 8
LR         = 5e-4
SEED       = 42

# --- Nombre de classes : inféré depuis le CSV (override possible par env) ---
def _infer_num_classes(csv_path: Path) -> int:
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if "classID" in df.columns:
            return int(df["classID"].nunique())
        elif "class" in df.columns:
            return int(df["class"].astype("category").cat.codes.nunique())
        else:
            raise ValueError("CSV sans colonnes 'classID' ni 'class'.")
    except Exception as e:
        # Fallback (évite de planter au tout début) : 10 par défaut
        # mais logiquement tu as pandas/scikit-learn installés maintenant.
        print(f"[config] Avertissement: impossible d'inférer N_CLASSES ({e}). Fallback=10.")
        return 10

N_CLASSES = int(os.getenv("URBAN_N_CLASSES", _infer_num_classes(CSV_PATH)))
