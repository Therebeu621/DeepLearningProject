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
  - `data/urban_sounds_small/urban_sounds_small` (mini dataset HuggingFace)
  - ancien chemin `/mnt/c/Users/.../UrbanSound8K`
- Les fichiers audio peuvent être organisés par folds (`audio/fold1/...`) ou par classes (`10_music/...`). Les scripts détectent automatiquement la bonne structure.

## Run
python -m src.check_data
python -m src.make_subset   # optionnel, crée un mini-jeu data/subset/
python -m src.train
python -m src.evaluate
python -m src.infer
