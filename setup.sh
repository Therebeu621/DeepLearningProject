#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Initialisation du projet DeepLearningProject – UrbanSound8K"

# ───────────────────────────────────────────────
# 0) Trouver une version Python compatible (3.8–3.12)
# ───────────────────────────────────────────────
choose_python() {
  local preferred=("python3.12" "python3.11" "python3.10" "python3.9" "python3.8")
  local bin=""
  for c in "${preferred[@]}"; do
    if command -v "$c" >/dev/null 2>&1; then bin="$(command -v "$c")"; break; fi
  done
  if [ -z "$bin" ] && command -v python3 >/dev/null 2>&1; then
    local ver
    ver="$(python3 -c 'import sys;print(sys.version_info[:2])' 2>/dev/null || true)"
    if [[ "$ver" =~ ^\([[:space:]]*3,[[:space:]]*([0-9]+)\)[[:space:]]*$ ]]; then
      local minor="${BASH_REMATCH[1]}"
      if [ "$minor" -ge 8 ] && [ "$minor" -le 12 ]; then
        bin="$(command -v python3)"
      fi
    fi
  fi
  echo "$bin"
}

PYTHON_BIN="$(choose_python)"
if [ -z "$PYTHON_BIN" ]; then
  echo "❌ Aucun Python compatible trouvé (besoin 3.8–3.12)."
  echo "👉 macOS : brew install python@3.12"
  echo "👉 Ubuntu : sudo apt-get install python3.12 python3.12-venv"
  exit 1
fi

echo "🐍 Python choisi : $("$PYTHON_BIN" -V)"

# ───────────────────────────────────────────────
# 1) Créer ou recréer le venv si incompatible
# ───────────────────────────────────────────────
if [ -d ".venv" ]; then
  venv_ver="$(.venv/bin/python -c 'import sys;print(".".join(map(str,sys.version_info[:2])))' 2>/dev/null || echo "")"
  if [ -n "$venv_ver" ]; then
    minor="${venv_ver#*.}"
    if [ "${venv_ver%%.*}" -eq 3 ] && [ "$minor" -ge 8 ] && [ "$minor" -le 12 ]; then
      echo "✅ Venv existant compatible (Python $venv_ver) – réutilisation."
    else
      echo "♻️  Venv incompatible (Python $venv_ver) – recréation."
      rm -rf .venv
      "$PYTHON_BIN" -m venv .venv
    fi
  else
    echo "♻️  Venv corrompu – recréation."
    rm -rf .venv
    "$PYTHON_BIN" -m venv .venv
  fi
else
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# ───────────────────────────────────────────────
# 2) Installation + Dataset + Exécution du projet
# ───────────────────────────────────────────────
echo "⬆️  Dépendances..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "🎧 Téléchargement du mini-dataset HuggingFace..."
pip install --no-cache-dir --upgrade huggingface_hub hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1
python scripts/download_mini_dataset.py

echo "🗂️  Génération du CSV..."
python scripts/generate_csv.py

# Assure que les modules src soient trouvables
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
[ -f src/__init__.py ] || : > src/__init__.py

# Variable pour le dataset
export URBAN_SOUND_ROOT="$PWD/data/urban_sounds_small/urban_sounds_small"

echo "🧪 Vérification des données..."
python -m src.check_data

echo "✂️  Création d'un subset..."
python -m src.make_subset || true

echo "🏋️  Entraînement..."
python -m src.train

# Crée un lien symbolique vers les poids
mkdir -p weights
ln -sfn "$PWD/weights" "$PWD/DeepLearningProject/weights" || true

echo "🧮 Évaluation..."
python -m src.evaluate

echo "🔎 Inférence..."
python -m src.infer

echo "✅ Installation et exécution terminées avec succès !"
