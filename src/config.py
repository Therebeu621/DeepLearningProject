from pathlib import Path

URBAN_ROOT = Path("/mnt/c/Users/aniss/Desktop/VraiDLP/UrbanSound8K")

CSV_PATH_A = URBAN_ROOT / "UrbanSound8K.csv"
CSV_PATH_B = URBAN_ROOT / "metadata" / "UrbanSound8K.csv"
CSV_PATH = CSV_PATH_A if CSV_PATH_A.exists() else CSV_PATH_B

AUDIO_DIR_A = URBAN_ROOT / "audio"
AUDIO_DIR_B = URBAN_ROOT           # tes fold1..fold10 sont à la racine → ça choisira B
AUDIO_DIR = AUDIO_DIR_A if AUDIO_DIR_A.exists() else AUDIO_DIR_B
# Audio / features
SR = 22050
N_MELS = 64
HOP = 512
TARGET_LEN = 173  # ~4s à 22kHz avec hop 512

# Train
BATCH_SIZE = 64
EPOCHS = 8
LR = 1e-3
SEED = 42
N_CLASSES = 10

# Dossiers projet (générés côté repo)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SUBSET_DIR = DATA_DIR / "subset"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)
SUBSET_DIR.mkdir(parents=True, exist_ok=True)
