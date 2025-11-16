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

# Assure que les modules du package soient trouvables
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

WEIGHTS_PATH="$PWD/weights/urbansound_cnn.pt"
if [ ! -f "$WEIGHTS_PATH" ]; then
  echo "❌ Poids pré-entraînés introuvables : $WEIGHTS_PATH"
  echo "👉 Ajoute ton checkpoint (ex: weights/urbansound_cnn.pt) avant de lancer ce script."
  exit 1
fi

echo "⚙️  Démarrage du serveur MCP..."
uvicorn mcp_server.server:app --host 127.0.0.1 --port 8000
