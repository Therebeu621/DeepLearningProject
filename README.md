# SonicWatch – UrbanSound8K

# SonicWatch – UrbanSound8K

> **Démarrage rapide**
> 1. `bash scripts/setup_and_run.sh` (installe, entraîne, évalue, lance le serveur MCP).  
> 2. Ouvre LM Studio, charge ton modèle (ex. `mistralai/mistral-7b-instruct-v0.3`) et démarre le serveur local (http://127.0.0.1:1234).  
> 3. `python chatbot/orchestrator_lmstudio.py` pour discuter avec SonicWatch via le LLM.  
> 4. Tester : `Écoute le son data/subset/shot556_29_ch01_180718_162104_16_.wav et dis-moi ce que c’est`

## Evaluer le modele 
python -m model.evaluate --no-subset


## 1) Introduction
SonicWatch est un chatbot audio spécialisé qui identifie les bruits urbains à partir d’un fichier WAV.  
Architecture : un CNN PyTorch entraîné sur UrbanSound8K + un LLM local (LM Studio) connecté via un serveur MCP FastAPI qui effectue l’inférence et renvoie les top‑k probabilités.

## 2) Jeu de données
- **Dataset** : UrbanSound8K (10 classes — air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music).  
- **Organisation attendue** :
  ```
  data/
    UrbanSound8K/
      audio/fold1/*.wav … fold10/*.wav
      metadata/UrbanSound8K.csv
  ```
- **Variable d’environnement** :
  ```bash
  export URBAN_SOUND_ROOT="$PWD/data/UrbanSound8K"
  ```

## 3) Modèle & Features
- Extraction : mel-spectrogrammes `librosa` (64 mels, SR 22.05 kHz).  
- Modèle : CNN léger (PyTorch) avec SpecAugment léger (1× TimeMask ≤16 frames + 1× FreqMask ≤8 bins, appliqué uniquement pendant l’entraînement).  
- Batches équilibrés grâce à `WeightedRandomSampler`.

## 4) Installation & Environnement
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt    # torch, torchaudio, librosa, scikit-learn, fastapi, uvicorn, numpy, requests…
```

## 5) Entraînement (repro officielle)
```bash
export URBAN_SOUND_ROOT="$PWD/data/UrbanSound8K"
python -m model.train --epochs 15 --batch-size 32 \
  --no-subset --use-sampler --aug-spec \
  --lr 3e-4
```
- `--no-subset` : utilise les 7 895 échantillons d’entraînement (fold1–9) et 837 de validation (fold10).  
- `--use-sampler` : équilibre les batches.  
- `--aug-spec` : active SpecAugment.  
- Early stopping surveille la perte validation.

## 6) Évaluation
```bash
python -m model.evaluate --no-subset
```
Exemple actuel :
- Accuracy globale ≈ **75 %**
- Macro-F1 ≈ **0.73**
- dog_bark F1 ≈ 0.80 • jackhammer ≈ 0.96 • street_music ≈ 0.55 • children_playing ≈ 0.57  
Artefacts produits : `reports/confusion_matrix.png`, `reports/confusion_matrix_norm.png`, `reports/metrics.json`.

## 7) Serveur MCP (FastAPI)
```bash
uvicorn mcp_server.server:app --host 127.0.0.1 --port 8000
```
Endpoints :
- `GET /health` – statut + device + chemin des poids
- `GET /metrics?variant=cnn|embeddings`
- `GET /reports`
- `POST /infer` – upload WAV multipart → top-k, probas, `is_confident` (seuil 60 %).  
Variable à exporter : `URBAN_SOUND_ROOT="$PWD/data/UrbanSound8K"`.

## 8) Orchestrateurs
- **CLI de test** :
  ```bash
  python chatbot/orchestrator.py infer data/...wav --topk 3
  ```
- **LLM local (LM Studio)** :
  1. Activer le Local LLM Server (port 1234).  
  2. Vérifier : `curl http://localhost:1234/v1/models`.  
  3. Charger le prompt système `chatbot/prompt.md` (persona “SonicWatch”).  
  4. Configurer MCP pour cibler `http://127.0.0.1:8000`.

## 9) Exemples d’usage
```bash
# Inférence directe
python -m model.infer --wav data/subset/shot556_29_ch01_180718_162104_16_.wav --topk 3

# Orchestrateur CLI
python chatbot/orchestrator.py infer data/subset/...wav --topk 3

# API
curl -s http://127.0.0.1:8000/health
```

## 10) Résultats & Discussion
- Excellente précision sur les classes percussives (gun_shot, jackhammer).  
- Street_music & children_playing restent plus difficiles à cause de la variabilité de scène.  
- SpecAugment léger + sampler équilibré stabilisent les performances globales.  
- Pistes : ajouter BatchNorm + label smoothing (0.05), `ReduceLROnPlateau`, pitch/tempo augmentations ciblées.

## 11) Reproductibilité
- Seeds fixés dans `model/train.py` (PyTorch / NumPy / random).  
- Ajouter `URBAN_SOUND_ROOT`, versions Python 3.12 + libs listées dans `requirements.txt`.  
- Arborescence type :
  ```
  DeepLearningProject/
    model/         # config, train, evaluate, features…
    mcp_server/    # FastAPI
    chatbot/       # prompt + orchestrateurs
    reports/       # métriques & matrices
    weights/       # checkpoints
  ```

## 12) Licence & Auteurs
- UrbanSound8K : © Maguire et al., disponible via https://urbansounddataset.weebly.com/urbansound8k.html  
- Projet SonicWatch, usage pédagogique.
