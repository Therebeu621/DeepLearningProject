# MCP Server – UrbanSound Specialist

Ce micro-serveur FastAPI sert de passerelle MCP/HTTP entre le LLM local et le modèle spécialiste.

## Installation

```bash
source .venv/bin/activate
pip install -r requirements.txt          # ajoute fastapi + uvicorn
```

## Lancement

```bash
# dataset complet : s'assurer que URBAN_SOUND_ROOT pointe vers data/UrbanSound8K
export URBAN_SOUND_ROOT="$PWD/data/UrbanSound8K"
uvicorn mcp_server.server:app --reload --port 8000
```

Endpoints :

| Méthode | Route      | Description |
|---------|------------|-------------|
| GET     | `/health`  | Statut du serveur & device |
| GET     | `/metrics?variant=cnn|embeddings` | Retourne `reports/metrics*.json` |
| GET     | `/reports` | Liste les fichiers disponibles dans `reports/` |
| POST    | `/infer`   | Lance une inférence (top‑k, confiance, seuil) |

### Exemple d’inférence

```bash
curl -X POST http://localhost:8000/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "wav_path": "data/subset/shot556_29_ch01_180718_162104_16_.wav",
    "topk": 3,
    "use_subset": true
  }'
```

Réponse :
```json
{
  "wav_path": "data/subset/shot556_29_ch01_180718_162104_16_.wav",
  "weights": "weights/urbansound_cnn.pt",
  "use_subset": true,
  "predicted_label": "gun_shot",
  "confidence": 0.92,
  "threshold": 0.6,
  "is_confident": true,
  "top_k": [
    {"rank": 1, "label": "gun_shot", "confidence": 0.92},
    {"rank": 2, "label": "car_horn", "confidence": 0.04},
    {"rank": 3, "label": "siren", "confidence": 0.02}
  ]
}
```

## Utilisation par le LLM (MCP)
- `POST /infer` fournit la classe prédite + top‑k + indicateur `is_confident`.
- `GET /metrics?variant=cnn` donne les scores actuels pour contextualiser la réponse.
- `GET /metrics?variant=embeddings` compare avec la baseline PANNs.
- `GET /reports` permet d’indiquer à l’utilisateur quels graphiques (matrices de confusion, JSON) sont disponibles.

Ces appels sont décrits dans `chatbot/prompt.md` pour guider ton LLM local.
