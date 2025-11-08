# 🚀 Installation & exécution automatique
```bash
bash setup.sh
```

# 🎵 DeepLearningProject – Spécialiste UrbanSound8K

## 🎯 Spécialisation du chatbot
Le futur chatbot agit comme interface pour un classifieur audio spécialisé dans la détection de nuisances urbaines (gunshots, sirènes, foules, musique forte, etc.).  
Le LLM devra simplement router la requête vers ce modèle PyTorch et présenter :
- la classe prédite + confiance ;
- les 3 meilleures alternatives pour gérer l’incertitude ;
- un lien vers les métriques `reports/metrics.json` pour justifier les performances.

## 📦 Jeu de données & motivation
- **Dataset** : UrbanSound8K (ou mini jeu `UrbanSounds/urban_sounds_small` pour prototypage rapide).
- **Justification** : classes clairement annotées, split officiel 10-fold → parfait pour benchmarker un spécialiste audio.
- **Téléchargements** :
  1. **Mini jeu HF** `python scripts/download_mini_dataset.py [--classes ... | --full]` (par défaut 9 classes cibles, ~100 fichiers).  
  2. **Dataset complet** `python scripts/download_urbansound8k.py` (≈6 Go) puis `export URBAN_SOUND_ROOT="$PWD/data/UrbanSound8K"`.
- **Création d’un subset équilibré** (pour réduire les temps de calcul) :
  ```bash
  python -m model.make_subset \
    --n-per-class 80 \
    --classes air_conditioner,car_horn,children_playing,dog_bark,drilling,engine_idling,gun_shot,jackhammer,siren,street_music
  ```
  Cela génère `data/subset/subset_meta.csv` (ici 800 extraits / 10 classes), copiant les WAV nécessaires pour des tests rapides.
- **Variables utiles** :
  - `SKIP_MINI_DATASET=1` → évite de recharger le mini jeu dans `setup.sh`.
  - `USE_FULL_URBAN=1` → force `setup.sh` à pointer directement sur UrbanSound8K complet.
  - `USE_STRONG_AUG=1` → applique `--aug-time --aug-noise --aug-spec` pendant l’entraînement.

## 🧩 Installation manuelle (si vous ne lancez pas `setup.sh`)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Mini jeu HuggingFace (facultatif mais pratique)
pip install --no-cache-dir --upgrade huggingface_hub hf-transfer
python scripts/download_mini_dataset.py
python scripts/generate_csv.py
```

## 🏋️ Pipeline du modèle spécialiste
1. **Vérifier les données**
   ```bash
   python -m model.check_data
   ```
2. **Créer un subset équilibré (développement rapide)**
   ```bash
   python -m model.make_subset --n-per-class 120   # optionnel
   ```
3. **Entraîner la CNN**
   ```bash
   # subset (rapide)
   python -m model.train --epochs 30 --batch-size 64 --aug-time

   # dataset complet (fold10=val)
   URBAN_SOUND_ROOT=data/UrbanSound8K \
   python -m model.train --epochs 12 --batch-size 64 --no-subset --aug-time --aug-noise --aug-spec
   ```
   - Sauvegarde : `weights/urbansound_cnn.pt` (contient les poids + mapping `class_to_idx`).
   - Journal : les logs affichent la meilleure époque (early stopping inclus).
4. **Évaluer & générer les rapports**
   ```bash
   python -m model.evaluate
   ```
   - Exporte `reports/metrics.json`, `reports/confusion_matrix.png`, `reports/confusion_matrix_norm.png`.
   - Utiliser ces fichiers dans le rapport pour justifier les performances (accuracy, macro/weighted F1, par-classe).

## 📈 Baseline “modèle existant” (embeddings PANNs)
Comparaison demandée dans l’énoncé :
```bash
# subset : rien à préciser ; dataset complet : ajouter --no-subset
python -m model.precompute_embeddings [--no-subset]

# Régression logistique (référence rapide)
python -m model.train_embeddings --model logreg --logreg-max-iter 4000

# Tête MLP un peu plus riche (fine-tuning léger)
python -m model.train_embeddings \
  --model mlp \
  --hidden-dim 512 \
  --dropout 0.3 \
  --mlp-epochs 200 \
  --mlp-patience 30
```
Résultats produits :
- `weights/linear_head.pkl|pt`
- `reports/metrics_embeddings.json`, `reports/confusion_matrix_embeddings*.png`

Comparez ensuite `reports/metrics.json` (CNN end-to-end) VS `reports/metrics_embeddings.json` (baseline) et discutez des écarts dans le rapport.

## 🔍 Inférence
### CNN end-to-end
```bash
python -m model.infer --wav data/subset/shot556_150_ch01_180718_164053_99_.wav
# ignorer le subset et évaluer sur UrbanSound8K complet
python -m model.infer --wav data/UrbanSound8K/audio/fold5/100032-3-0-0.wav --no-subset
```
Affiche la prédiction top-1 + top-3.

### Variante embeddings (tête logistique/MLP)
```bash
python -m model.infer_embeddings --wav path/to/audio.wav \
  --weights weights/linear_head.pkl   # ou linear_head.pt
```

## 🔌 MCP Server & Chatbot prompt
- Serveur FastAPI : `mcp_server/server.py` (doc dans `mcp_server/README.md`). Lancer `uvicorn mcp_server.server:app --port 8000` puis appeler :
  - `POST /infer` → classe prédite, top‑k, indicateur `is_confident` (seuil 60 %).
  - `GET /metrics?variant=cnn|embeddings` → scores issus de `reports/metrics*.json`.
  - `GET /reports` → liste des fichiers (matrices de confusion, JSON, etc.).
- Prompt LLM : `chatbot/prompt.md` décrit les règles du bot “SonicWatch” (transparence, mention des top‑k, rappel des métriques, suggestions d’actions). À intégrer comme prompt système dans ton petit LLM local.
- Exemple d’appel HTTP :
  ```bash
  curl -X POST http://localhost:8000/infer \
    -H 'Content-Type: application/json' \
    -d '{"wav_path":"data/subset/shot556_29_ch01_180718_162104_16_.wav","topk":3}'
  ```
  Réponse → JSON avec `predicted_label`, `confidence`, `top_k`, `threshold`, `reports`.

### Brancher ton LLM local / orchestrateur
1. **Prompt système** : charge le contenu de `chatbot/prompt.md` (persona SonicWatch + règles d’explication) comme prompt système de ton LLM (ex. dans LM Studio, Oobabooga, Text-Generation-WebUI, etc.).
2. **Déclaration MCP / outils** : ajoute à ton orchestrateur un outil HTTP pointant vers `http://127.0.0.1:8000` avec les opérations suivantes :
   - `GET /health`
   - `GET /metrics?variant=cnn` et `GET /metrics?variant=embeddings`
   - `GET /reports`
   - `POST /infer` avec payload `{"wav_path": "...", "topk": 3, "use_subset": true}`
3. **Boucle d’utilisation** : le LLM doit, à chaque demande utilisateur, appeler `/infer`, vérifier `is_confident`, puis éventuellement compléter avec `/metrics` et `/reports` pour contextualiser la réponse.

Une fois ces routes exposées à ton LLM, l’intégration MCP est complète : le chatbot peut consommer les API et répondre aux utilisateurs en citant la classe détectée, la confiance et les métriques globales.

> ✅ Pour tester sans LLM, utilise le mini-orchestrateur CLI fourni :
> ```bash
> source .venv/bin/activate
> python chatbot/orchestrator.py health
> python chatbot/orchestrator.py metrics --variant cnn
> python chatbot/orchestrator.py infer data/subset/shot556_29_ch01_180718_162104_16_.wav --topk 3
> ```
> Il applique la mise en forme décrite dans `chatbot/prompt.md` (top‑k, confiance, contexte). Idéal pour déboguer avant de brancher un vrai LLM.

Pour une interaction “bateau de chat” :
```bash
python chatbot/chat_loop.py
```
Tu obtiens un prompt `you>` et peux écrire en langage naturel ou utiliser les commandes explicites (`infer`, `metrics`, `reports`, etc.). Ce script appelle `/infer`, `/metrics`, `/reports` automatiquement et formate la réponse façon SonicWatch.

## 📊 Résultats actuels
### Subset UrbanSound8K (10 classes × 80 clips = 800 fichiers, fold10=val → 79 fichiers)
- `reports/metrics.json` (CNN end-to-end, `--aug-time`, batch 64) → **accuracy 64.6 %**, **macro-F1 64.1 %**.  
  - Classes solides : `gun_shot` (F1=1.0), `jackhammer` (0.78), `drilling` (0.71).  
  - Faiblesses : `children_playing` (0.31) et `street_music` (0.43).  
  - Confusions visibles dans `reports/confusion_matrix*.png`.
- `reports/metrics_embeddings.json` (PANNs CNN14 + MLP tête 512 neurones, dropout 0.3) → **accuracy 83.5 %**, **macro-F1 82.8 %**.  
  - Améliorations notables sur `children_playing`, `street_music` et `jackhammer`.

### Mini-dataset HuggingFace (≈9 classes, 223 clips)
- CNN simple (`--aug-time`) → **accuracy 54.5 %**, **macro-F1 51.8 %**.  
- Baseline PANNs + logreg → **accuracy 77.3 %**, **macro-F1 75.2 %**.

### Analyse critique & pistes
1. **Quantité de données** : même avec 800 échantillons, certaines classes restent difficiles. Pour dépasser 65 % d’accuracy, viser UrbanSound8K complet (8732 clips) ou activer `--no-subset` + `USE_FULL_URBAN=1` dans `setup.sh`.
2. **Modèle léger** : la CNN maison reste limitée. La tête MLP sur embeddings montre qu’un fine-tuning de backbone pré-entraînée est la prochaine étape naturelle.
3. **Augmentations progressives** : `--aug-time` suffit pour le subset, mais `--aug-noise --aug-spec` redeviennent pertinentes une fois le dataset élargi.
4. **Seuils/UX** : exposez les top-k + la confiance pour éviter de sur-interpréter les classes encore fragiles (street_music, children_playing).

## 🚀 Pistes d’amélioration (post-deadline)
- **UrbanSound8K complet** : télécharge l’archive officielle (~6 Go), pointe `URBAN_SOUND_ROOT` vers le dossier extrait (`UrbanSound8K/` contenant `audio/` et `metadata/UrbanSound8K.csv`) puis relance `model.train / model.evaluate`. On obtient ~8 k clips → gros gain de généralisation.
- **Fine-tuning via embeddings** : réutilise les features PANNs (`model.precompute_embeddings`) mais entraîne une tête un peu plus riche qu’une simple régression linéaire (ex. MLPHead avec dropout ou léger dégel des couches finales). Comparer les métriques dans `reports/metrics_embeddings.json`.
- **Augmentations fortes** : une fois que la base de données est suffisamment grande, activer `--aug-noise` et `--aug-spec` ou jouer sur `FEAT_TARGET_LEN`, `FEAT_N_MELS` (`model/config.py`) peut accroître la robustesse.

## 🧪 Points à documenter dans le rapport
- Architecture audio : melspec 64 bandes, SR 22.05 kHz, normalisation, SpecAugment optionnel.
- Hyperparamètres clés : LR 1e-3, batch 16/32, scheduler cosine, pondération inverse des classes.
- Métriques à reporter : accuracy, macro/weighted F1, per-class (issues du JSON).
- Comparaison baseline vs CNN + pistes d’amélioration (plus de données, fine-tuning backbone, augmentation ciblée).

Ces éléments couvrent la partie “modèle spécialiste” et fournissent tout le nécessaire pour le futur serveur MCP et la validation par les enseignants.
