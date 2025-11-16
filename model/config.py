# model/config.py
from pathlib import Path
from typing import Optional
import os
import warnings

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
    # 1) Variables d'env prioritaires
    env = os.getenv("URBAN_SOUND_ROOT") or os.getenv("URBAN_ROOT")
    if env:
        yield Path(env).expanduser()
    # 2) Dossiers standard du repo
    yield DATA_DIR / "urban_sounds_small" / "urban_sounds_small"
    yield DATA_DIR / "UrbanSound8K"

_CANDIDATE_ROOTS = list(_candidate_roots())
if not _CANDIDATE_ROOTS:
    _CANDIDATE_ROOTS = [DATA_DIR / "UrbanSound8K"]

URBAN_ROOT = next((c for c in _CANDIDATE_ROOTS if c.exists()), _CANDIDATE_ROOTS[-1])
HAS_DATASET = URBAN_ROOT.exists()

if not HAS_DATASET:
    warnings.warn(
        "Dataset UrbanSound introuvable. "
        "Certaines commandes (train/evaluate) nécessitent URBAN_SOUND_ROOT.",
        RuntimeWarning,
    )

# --- Sélection du CSV ---
CSV_CANDIDATES = (
    URBAN_ROOT / "UrbanSound8K_like.csv",           # CSV généré mini-dataset
    URBAN_ROOT / "metadata.csv",                     # mini-dataset HF
    URBAN_ROOT / "UrbanSound8K.csv",                 # CSV à la racine
    URBAN_ROOT / "metadata" / "UrbanSound8K.csv",    # CSV dans metadata/
)
CSV_PATH = next((p for p in CSV_CANDIDATES if p.exists()), None) if HAS_DATASET else None
if HAS_DATASET and CSV_PATH is None:
    raise FileNotFoundError(
        f"Impossible de trouver un CSV (candidats: {', '.join(str(p) for p in CSV_CANDIDATES)})"
    )

# --- Dossier audio ---
AUDIO_DIR = URBAN_ROOT / "audio" if (URBAN_ROOT / "audio").exists() else URBAN_ROOT

# =========================
# 🔊 Audio / features
# =========================
SR         = int(os.getenv("AUDIO_SR", 22050))
N_MELS     = int(os.getenv("FEAT_N_MELS", 64))
HOP        = int(os.getenv("FEAT_HOP", 512))
FMIN       = int(os.getenv("FEAT_FMIN", 0))
FMAX       = int(os.getenv("FEAT_FMAX", 8000))   # utile pour mieux capter les sons aigus (claxon)
TARGET_LEN = int(os.getenv("FEAT_TARGET_LEN", 173))  # frames (~4s @22.05kHz, hop=512)

# =========================
# 🏋️ Entraînement
# =========================
BATCH_SIZE = int(os.getenv("TRAIN_BATCH", 16))
EPOCHS     = int(os.getenv("TRAIN_EPOCHS", 30))       # un peu plus haut, on garde l’early stopping
LR         = float(os.getenv("TRAIN_LR", 1e-3))
SEED       = int(os.getenv("TRAIN_SEED", 42))
VAL_SPLIT  = float(os.getenv("TRAIN_VAL_SPLIT", 0.1))  # 10% validation sur mini-jeu

# Early stopping & scheduler
ES_PATIENCE   = int(os.getenv("ES_PATIENCE", 5))
ES_MIN_DELTA  = float(os.getenv("ES_MIN_DELTA", 1e-4))
SCHED_FACTOR  = float(os.getenv("SCHED_FACTOR", 0.5))
SCHED_PATIENCE= int(os.getenv("SCHED_PATIENCE", 2))

# Augmentations (activables par env)
AUG_ENABLED        = os.getenv("AUG_ENABLED", "1") == "1"
AUG_TIME_SHIFT_MS  = int(os.getenv("AUG_TIME_SHIFT_MS", 50))   # 0 pour désactiver
AUG_ADD_NOISE_DB   = int(os.getenv("AUG_ADD_NOISE_DB", 10))    # 0 pour désactiver
AUG_SPEC_FREQ_MASKS= int(os.getenv("AUG_SPEC_FREQ_MASKS", 1))
AUG_SPEC_TIME_MASKS= int(os.getenv("AUG_SPEC_TIME_MASKS", 1))

# Limiter l’échantillon (dev rapide)
SUBSET_LIMIT = int(os.getenv("SUBSET_LIMIT", 0))  # 0 = pas de limite

# =========================
# 🧠 Nombre de classes
# =========================
def _infer_num_classes(csv_path: Optional[Path]) -> int:
    if not csv_path or not csv_path.exists():
        return 10  # par défaut UrbanSound8K
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
        print(f"[config] Avertissement: impossible d'inférer N_CLASSES ({e}). Fallback=10.")
        return 10

N_CLASSES = int(os.getenv("URBAN_N_CLASSES", _infer_num_classes(CSV_PATH)))


def ensure_dataset_available():
    """
    À appeler depuis les scripts qui nécessitent la présence physique du dataset.
    """
    if not HAS_DATASET:
        raise FileNotFoundError(
            "Dataset UrbanSound introuvable. "
            "Définis URBAN_SOUND_ROOT ou place le dataset dans "
            "data/urban_sounds_small/urban_sounds_small ou data/UrbanSound8K."
        )
    if CSV_PATH is None:
        raise FileNotFoundError(
            f"Impossible de trouver un CSV (candidats: {', '.join(str(p) for p in CSV_CANDIDATES)})"
        )

# =========================
# 🔎 Inférence
# =========================
CONF_THRESHOLD = float(os.getenv("INFER_CONF_THRESHOLD", 0.60))  # <60% → “pas sûr”
TOPK           = int(os.getenv("INFER_TOPK", 3))
