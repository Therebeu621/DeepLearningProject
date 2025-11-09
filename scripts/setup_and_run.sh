#!/usr/bin/env bash
set -euo pipefail

echo "🚀 SonicWatch – setup & run"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 1) Python detection (>=3.12)
PY_BIN="${PY_BIN:-}"
for candidate in python3.12 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PY_BIN="$(command -v "$candidate")"
    break
  fi
done
if [ -z "$PY_BIN" ]; then
  echo "❌ Python 3.12+ introuvable. Installe python3.12 et relance." >&2
  exit 1
fi
echo "✅ Python sélectionné : $($PY_BIN -V)"

# 2) Créer/activer venv
if [ ! -d ".venv" ]; then
  "$PY_BIN" -m venv .venv
fi
source .venv/bin/activate

# 3) Dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4) Vérifier dataset
export URBAN_SOUND_ROOT="$PWD/data/UrbanSound8K"
META="$URBAN_SOUND_ROOT/metadata/UrbanSound8K.csv"
if [ ! -f "$META" ]; then
  echo "⚠️ Dataset UrbanSound8K incomplet ou introuvable. Téléchargement automatique…"
  python scripts/download_urbansound8k.py
fi
if [ ! -f "$META" ] || [ ! -d "$URBAN_SOUND_ROOT/audio" ]; then
  echo "❌ Dataset toujours incomplet (fichiers audio/metadata manquants)." >&2
  echo "Vérifie manuellement que data/UrbanSound8K/audio/ et metadata/UrbanSound8K.csv existent." >&2
  exit 1
fi
echo "✅ UrbanSound8K détecté (metadata + audio)."

# 5) Entraînement
python -m model.train --epochs 15 --batch-size 32 \
  --no-subset --use-sampler --aug-spec \
  --lr 3e-4

# 6) Évaluation
python -m model.evaluate --no-subset

# 7) Lancer MCP (background)
echo "🔌 Démarrage du serveur MCP…"
pkill -f "uvicorn mcp_server.server:app" >/dev/null 2>&1 || true
uvicorn mcp_server.server:app --host 127.0.0.1 --port 8000 \
  > /tmp/mcp_server.log 2>&1 &
MCP_PID=$!
echo "MCP PID: $MCP_PID (logs : /tmp/mcp_server.log)"

# 8) Tests rapides
echo "🔎 Tests de santé API"
curl -s http://127.0.0.1:8000/health || true
curl -s http://127.0.0.1:8000/metrics?variant=cnn || true

cat <<'EOF'

📌 Commandes utiles :
  python -m model.infer --wav data/subset/shot556_29_ch01_180718_162104_16_.wav --topk 3
  python chatbot/orchestrator.py infer data/subset/shot556_29_ch01_180718_162104_16_.wav --topk 3
  curl -s http://127.0.0.1:8000/metrics?variant=cnn | jq

Arrêt du serveur MCP :
  kill -9 '"$MCP_PID"'

EOF
