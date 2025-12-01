# SonicWatch – UrbanSound8K

> **Démarrage rapide**
> 1. `bash scripts/setup_and_run.sh` (installe, entraîne, évalue, lance le serveur MCP).  
> 2. Ouvre LM Studio, charge ton modèle (ex. `mistralai/mistral-7b-instruct-v0.3`) et démarre le serveur local (http://127.0.0.1:1234).  
> 3. `python chatbot/orchestrator_lmstudio.py` pour discuter avec SonicWatch via le LLM.  
> 4. Tester : `Écoute le son data/subset/shot556_29_ch01_180718_162104_16_.wav et dis-moi ce que c’est`

## Evaluer le modele 
python -m model.evaluate --no-subset


## Pour tester avec interface

Prérequis : Avoir installé les dépendances (`pip install -r requirements.txt`)

Terminal 1 : Serveur MCP (Modèle Spécialiste)

Charge le modèle PyTorch et le sert via l'API.
Bash

    source .venv/bin/activate
    uvicorn mcp_server.server:app --port 8000

Terminal 2 : Serveur LLM (LM Studio)

Charge le modèle de langue pour le dialogue.

    Ouvrez LM Studio.

    Chargez un modèle (ex: Mistral 7B).

    Allez à l'onglet Serveur (</>) et cliquez sur "Start Server".

 Terminal 3 : Interface Web (Gradio)

Lance l'application web qui connecte les deux serveurs.
Bash

    source .venv/bin/activate
    python app_gradio.py


Ouvrez votre navigateur et allez à l'URL affichée dans le Terminal 3 :

    http://127.0.0.1:7860
 

## 1) Introduction
SonicWatch est un chatbot audio spécialisé qui identifie les bruits urbains à partir d'un fichier WAV.
Architecture : un CNN PyTorch entraîné sur UrbanSound8K + un LLM local (LM Studio) connecté via un serveur MCP FastAPI qui effectue l'inférence et renvoie les top‑k probabilités.

## 2) Architecture des services (pourquoi plusieurs terminaux ?)

SonicWatch adopte une **architecture micro-services modulaire** où chaque composant tourne dans son propre processus :

1. **Serveur MCP (FastAPI)** – `uvicorn mcp_server.server:app --port 8000`
   - Charge le modèle PyTorch CNN et effectue les inférences audio
   - Expose une API REST (`/infer`, `/metrics`, `/reports`)
   - Permet de remplacer le modèle sans toucher au reste du système

2. **Serveur LLM (LM Studio)** – Interface graphique ou `http://localhost:1234`
   - Héberge le modèle de langage (Mistral 7B, Llama, etc.)
   - Gère le dialogue en langage naturel avec l'utilisateur
   - Peut être remplacé par Ollama, vLLM ou tout serveur compatible OpenAI API

3. **Interface utilisateur** – `python app_gradio.py` ou `python chatbot/orchestrator_lmstudio.py`
   - Gradio : interface web moderne avec upload de fichiers
   - Orchestrateur CLI : chatbot en ligne de commande
   - Connecte les deux services précédents (MCP + LLM)

4. **(Optionnel) Terminal d'entraînement** – `python -m model.train` ou `scripts/setup_and_run.sh`
   - Entraînement du modèle CNN
   - Évaluation et génération des rapports
   - Peut tourner indépendamment des services

**Pourquoi cette séparation ?**
- **Modularité** : Chaque service peut être développé, déployé et mis à jour indépendamment
- **Flexibilité** : Remplacer LM Studio par Ollama ne nécessite que de changer l'URL de l'API
- **Scalabilité** : Les services peuvent être déployés sur des machines différentes (ex: GPU distant pour le LLM)
- **Robustesse** : Un crash d'un service n'affecte pas les autres

**Simplification** : Le script `scripts/setup_and_run.sh` lance automatiquement le serveur MCP en arrière-plan, réduisant à 2-3 terminaux au lieu de 4.

## 3) Carte du projet (structure des fichiers)

```
DeepLearningProject/
├── model/                          # Module de deep learning
│   ├── config.py                   # Configuration (paths, hyperparamètres, constantes)
│   ├── model.py                    # Architecture CNN (SimpleCNN)
│   ├── features.py                 # Extraction mel-spectrogrammes + augmentations
│   ├── train.py                    # Entraînement du CNN avec early stopping
│   ├── evaluate.py                 # Évaluation + matrices de confusion
│   ├── infer.py                    # Inférence standalone sur un fichier WAV
│   ├── data_utils.py               # Utilitaires pour charger le dataset
│   ├── make_subset.py              # Création d'un subset de test rapide
│   ├── precompute_embeddings.py    # Extraction embeddings PANNs (baseline)
│   └── train_embeddings.py         # Entraînement baseline avec embeddings
│
├── mcp_server/                     # Serveur API FastAPI
│   ├── server.py                   # Endpoints REST (/infer, /metrics, /health)
│   └── mcp.json                    # Configuration LM Studio pour le serveur MCP
│
├── chatbot/                        # Orchestrateurs LLM
│   ├── orchestrator_lmstudio.py    # Chatbot CLI avec LM Studio (mode terminal)
│   ├── orchestrator.py             # Orchestrateur de test (sans LLM)
│   └── prompt.md                   # Prompt système pour le persona "SonicWatch"
│
├── scripts/                        # Scripts d'installation et setup
│   ├── setup_and_run.sh            # Setup complet + lancement auto
│   ├── download_urbansound8k.py    # Téléchargement du dataset complet
│   └── generate_csv.py             # Génération de metadata CSV
│
├── data/                           # Datasets audio
│   ├── UrbanSound8K/               # Dataset complet (8732 fichiers WAV)
│   │   ├── audio/fold1..fold10/    # Fichiers WAV organisés par fold
│   │   └── metadata/UrbanSound8K.csv
│   └── subset/                     # Subset léger pour tests rapides
│
├── weights/                        # Checkpoints entraînés
│   └── urbansound_cnn.pt           # Poids du CNN (state_dict + class_mapping)
│
├── reports/                        # Métriques et visualisations
│   ├── metrics.json                # Accuracy, F1, precision/recall par classe
│   ├── confusion_matrix.png        # Matrice de confusion (valeurs absolues)
│   └── confusion_matrix_norm.png   # Matrice de confusion normalisée
│
├── app_gradio.py                   # Interface web Gradio (glassmorphism UI)
├── requirements.txt                # Dépendances Python (torch, librosa, fastapi, etc.)
└── README.md                       # Ce fichier
```

**Fichiers clés pour démarrer** :
- `scripts/setup_and_run.sh` : Installation et lancement automatique
- `model/config.py` : Tous les hyperparamètres et chemins configurables
- `mcp_server/server.py` : API principale pour l'inférence
- `chatbot/prompt.md` : Personnalisation du comportement du chatbot

## 4) Jeu de données
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

## 5) Modèle & Features
- Extraction : mel-spectrogrammes `librosa` (64 mels, SR 22.05 kHz).  
- Modèle : CNN léger (PyTorch) avec SpecAugment léger (1× TimeMask ≤16 frames + 1× FreqMask ≤8 bins, appliqué uniquement pendant l’entraînement).  
- Batches équilibrés grâce à `WeightedRandomSampler`.

## 6) Installation & Environnement
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt    # torch, torchaudio, librosa, scikit-learn, fastapi, uvicorn, numpy, requests…
```

## 7) Entraînement (repro officielle)
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

## 8) Évaluation
```bash
python -m model.evaluate --no-subset
```
Exemple actuel :
- Accuracy globale ≈ **75 %**
- Macro-F1 ≈ **0.73**
- dog_bark F1 ≈ 0.80 • jackhammer ≈ 0.96 • street_music ≈ 0.55 • children_playing ≈ 0.57  
Artefacts produits : `reports/confusion_matrix.png`, `reports/confusion_matrix_norm.png`, `reports/metrics.json`.

## 9) Serveur MCP (FastAPI)
```bash
uvicorn mcp_server.server:app --host 127.0.0.1 --port 8000
```
Endpoints :
- `GET /health` – statut + device + chemin des poids
- `GET /metrics?variant=cnn|embeddings`
- `GET /reports`
- `POST /infer` – upload WAV multipart → top-k, probas, `is_confident` (seuil 60 %).  
Variable à exporter : `URBAN_SOUND_ROOT="$PWD/data/UrbanSound8K"`.

## 10) Orchestrateurs
- **CLI de test** :
  ```bash
  python chatbot/orchestrator.py infer data/...wav --topk 3
  ```
- **LLM local (LM Studio)** :
  1. Activer le Local LLM Server (port 1234).  
  2. Vérifier : `curl http://localhost:1234/v1/models`.  
  3. Charger le prompt système `chatbot/prompt.md` (persona “SonicWatch”).  
  4. Configurer MCP pour cibler `http://127.0.0.1:8000`.

## 11) Exemples d'usage
```bash
# Inférence directe
python -m model.infer --wav data/subset/shot556_29_ch01_180718_162104_16_.wav --topk 3

# Orchestrateur CLI
python chatbot/orchestrator.py infer data/subset/...wav --topk 3

# API
curl -s http://127.0.0.1:8000/health
```

## 12) Résultats & Discussion
- Excellente précision sur les classes percussives (gun_shot, jackhammer).  
- Street_music & children_playing restent plus difficiles à cause de la variabilité de scène.  
- SpecAugment léger + sampler équilibré stabilisent les performances globales.  
- Pistes : ajouter BatchNorm + label smoothing (0.05), `ReduceLROnPlateau`, pitch/tempo augmentations ciblées.

## 13) Reproductibilité
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

## 14) TROUBLESHOOTING

### Chemins de fichiers
- **Chemins absolus requis** : Le serveur MCP nécessite des chemins absolus pour les fichiers WAV. Exemple : `/home/user/DeepLearningProject/data/subset/shot556_29_ch01_180718_162104_16_.wav`
- **Chemins relatifs** : Depuis la racine du projet, vous pouvez utiliser : `data/subset/nom_fichier.wav`
- Par défaut, l'interface Gradio et le chatbot cherchent dans `data/subset/` si `use_subset=True`

### Setup LM Studio détaillé
1. Télécharger et installer [LM Studio](https://lmstudio.ai/)
2. Dans LM Studio, aller dans l'onglet "Search" et télécharger un modèle compatible (ex: `mistralai/mistral-7b-instruct-v0.3`)
3. Aller dans l'onglet "Local Server" (icône `</>`)
4. Sélectionner le modèle téléchargé
5. Cliquer sur "Start Server" (port par défaut : 1234)
6. Vérifier que le serveur fonctionne : `curl http://localhost:1234/v1/models`

### Alternative : Ollama + FastMCP
Si vous préférez une approche plus légère sans interface graphique :
```bash
# Installer Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Télécharger un modèle
ollama pull mistral

# Démarrer le serveur
ollama serve
```

Modifier ensuite `chatbot/orchestrator_lmstudio.py` pour pointer vers `http://localhost:11434` au lieu de `http://localhost:1234`.

**Pourquoi LM Studio ?** Choisi pour sa simplicité d'utilisation (GUI intuitive) et sa compatibilité OpenAI API. Pour un déploiement production, Ollama + FastMCP serait plus adapté.

### Pourquoi 4 terminaux ?

L'architecture distribuée nécessite plusieurs processus indépendants (voir section **2) Architecture des services** pour les détails) :

1. **Terminal 1 : Serveur MCP** – `uvicorn mcp_server.server:app --port 8000`
   - Lance le serveur FastAPI qui héberge le modèle CNN
   - Effectue les inférences audio et expose l'API REST
   - Doit rester actif pendant toute la session

2. **Terminal 2 : Serveur LLM** – LM Studio (GUI) ou Ollama
   - Héberge le modèle de langage pour le dialogue
   - Compatible OpenAI API sur `http://localhost:1234`
   - Peut être remplacé par `ollama serve` ou vLLM

3. **Terminal 3 : Interface utilisateur** – `python app_gradio.py` ou CLI
   - Gradio : interface web sur `http://localhost:7860`
   - CLI : `python chatbot/orchestrator_lmstudio.py`
   - Connecte MCP + LLM pour former le chatbot complet

4. **(Optionnel) Terminal 4 : Entraînement** – `python -m model.train`
   - Uniquement nécessaire pour réentraîner le modèle
   - Peut tourner en parallèle des services (entraînement long)
   - Non requis pour l'utilisation normale

**Pourquoi cette architecture ?**
- **Séparation des responsabilités** : Chaque service a un rôle unique (audio, langage, interface)
- **Flexibilité** : Remplacer LM Studio par Ollama ne nécessite qu'un changement d'URL
- **Débogage** : Les logs de chaque service sont isolés dans leur terminal
- **Production** : Les services peuvent être déployés sur des machines séparées (ex: GPU distant)
- **Micro-services** : Architecture standard dans l'industrie (cf. Docker Compose, Kubernetes)

**Simplification pour développement** :
- Le script `scripts/setup_and_run.sh` lance le serveur MCP en arrière-plan
- Réduit à **2-3 terminaux** au lieu de 4 (MCP automatique + LM Studio + Interface)
- Pour production : utiliser `systemd`, `supervisor` ou Docker

### Erreurs courantes

**`libfuse` manquant (Linux)** :
```bash
sudo apt-get install libfuse2  # Ubuntu/Debian
```

**Conflits de versions Gradio** :
```bash
pip install --upgrade gradio>=4.0,<5
```

**OpenAI API incompatible** :
Vérifier que LM Studio est bien démarré et que l'URL de base est correcte dans les variables d'environnement.

## 15) Liens
- UrbanSound8K :  https://urbansounddataset.weebly.com/urbansound8k.html

