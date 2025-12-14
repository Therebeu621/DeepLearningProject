# Rapport V1 – SonicWatch

## Introduction
La démocratisation des grands modèles de langage (LLM) permet désormais aux entreprises de créer des assistants spécialisés capables de converser dans un langage naturel tout en s’appuyant sur des briques métiers précises. Cette combinaison LLM + outils métiers est particulièrement intéressante pour les cas d’usage où l’interprétation et la pédagogie sont aussi importantes que le calcul lui‑même : maintenance industrielle, cybersécurité, ou, comme dans notre cas, analyse acoustique. Les chatbots professionnels ont donc besoin d’interfaces qui guident l’utilisateur, sécurisent les appels aux modèles, et offrent un rendu intelligible des résultats.

SonicWatch s’inscrit dans ce contexte : c’est un assistant audio capable d’expliquer les résultats d’un classifieur de sons urbains. Le modèle spécialiste est entraîné sur UrbanSound8K, un dataset de 10 classes (air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music). L’utilisateur fournit un fichier WAV, le chatbot déclenche l’inférence puis commente les résultats en français.

Afin de concilier robustesse et flexibilité, nous séparons clairement les rôles : un LLM généraliste sert la couche conversationnelle (LM Studio + modèle Mistral), un modèle spécialiste PyTorch calcule les probabilités sur UrbanSound8K, un serveur MCP (FastAPI) fait l’intermédiation et expose les métriques, et deux orchestrateurs (CLI et Gradio) assurent l’expérience utilisateur.

## Installation et mode d’emploi

Avant de lancer le projet, quelques pré-requis sont nécessaires :

- Python 3.12 installé sur la machine ;
- LM Studio installé (version bureau) pour exécuter un petit LLM local ;
- une connexion Internet temporaire pour télécharger le dataset UrbanSound8K lors du premier lancement.

Dans le rapport, on suppose que le projet est fourni sous forme d’archive (fichier .zip).  
Après avoir décompressé l’archive, placez-vous simplement à la racine du dossier du projet dans un terminal :

```bash
cd DeepLearningProject   # ou nom du dossier décompressé
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.1. Démarrage rapide (recommandé)

Pour lancer une première expérimentation de bout en bout, nous avons fourni un script tout-en-un :

```bash
bash scripts/setup_and_run.sh
```

Ce script :

vérifie la présence de l’environnement virtuel et des dépendances ;

télécharge UrbanSound8K si nécessaire et prépare l’arborescence data/UrbanSound8K ;

entraîne le modèle CNN sur le dataset complet (optionnel, désactivé par défaut) ;

lance le serveur MCP (FastAPI) en arrière-plan avec les poids pré-entraînés.

> **Note** : Le script utilise désormais les poids pré-entraînés. L'entraînement et l'évaluation sont optionnels (voir README).

Une fois cette étape terminée :

il suffit d’ouvrir LM Studio, d’installer ou de charger un modèle de langue (par exemple `mistralai/mistral-7b-instruct-v0.3`) et de démarrer le serveur local ;  
pour plus de détails sur la configuration de LM Studio et le choix du modèle, nous renvoyons à la section dédiée du fichier `README.md`.

puis de lancer le chatbot en ligne de commande :

```bash
python chatbot/orchestrator_lmstudio.py
```

ou bien l’interface Web :

```bash
python app_gradio.py
```

L’interface Gradio est alors accessible via l’URL indiquée dans le terminal (par défaut http://127.0.0.1:7860).

Pour les options avancées (réentraînement manuel, variantes embeddings, appels directs à l’API), nous renvoyons au fichier README.md du projet.

### 2.2. Mode manuel (services séparés)

Pour un lancement plus fin des différents services, on peut démarrer séparément :

Serveur MCP (modèle spécialiste) :

```bash
uvicorn mcp_server.server:app --host 127.0.0.1 --port 8000
```

Serveur LLM (LM Studio) :

- ouvrir LM Studio ;
- charger un modèle Instruct (par exemple Mistral 7B) ;
- démarrer le serveur local (par défaut http://127.0.0.1:1234).

Orchestrateurs :

en mode terminal :

```bash
python chatbot/orchestrator_lmstudio.py
```

en mode interface Web :

```bash
python app_gradio.py
```

Dans tous les cas, l'utilisateur peut soit taper une consigne en français (par exemple « Écoute le son data/UrbanSound8K/audio/fold5/100032-3-0-0.wav et dis-moi ce que c'est »), soit importer un fichier WAV via l'interface Gradio.

## Méthodologie

### LLM et prompt
Nous utilisons LM Studio comme serveur local compatible OpenAI. Le modèle recommandé est `mistralai/mistral-7b-instruct-v0.3`, suffisamment léger pour tourner sur un PC et déjà orienté instructions. Le prompt système (`chatbot/prompt.md`) définit SonicWatch : persona scientifique, ton factuel, réponses en français, structure attendue (résultat, métriques, recommandations). Dans `chatbot/orchestrator_lmstudio.py` et `app_gradio.py`, nous instancions `openai.OpenAI` en précisant `base_url=f"{LMSTUDIO_BASE_URL}/v1"` et une API key factice (`lm-studio`). Les messages sont construits avec les rôles `system`, `user`, `assistant`, puis envoyés à `client.chat.completions.create(...)`. Le LLM n’a jamais accès directement aux WAV : il reçoit un résumé JSON (CLI) ou un contexte textuel (UI) provenant du serveur MCP.

### Modèle spécialiste
Le dataset UrbanSound8K contient ~8 732 extraits répartis en 10 classes et 10 folds, avec un CSV metadata décrivant les labels. Dans `model/`, nous avons :
- `train.py` : pipeline principal (mel-spectrogrammes via `features.py`, CNN PyTorch, SpecAugment optionnel, `WeightedRandomSampler`, checkpointing).
- `precompute_embeddings.py` et `train_embeddings.py` : pipeline alternatif basé sur les embeddings PANNs (AudioSet) suivis d’une tête linéaire (poids `weights/linear_head.pkl`).
- `evaluate.py`, `infer.py`, `make_subset.py` pour reproduire les expériences, fabriquer un subset léger et exécuter des inférences unitaires.

Deux variantes sont maintenues :
1. **CNN log-mel** : modèle convolutif léger entraîné from scratch, exporté dans `weights/urbansound_cnn.pt`. Il offre **71.3 % d'accuracy** et **72.5 % de macro-F1** sur le fold 10 (résultats reproductibles dans `reports/metrics.json`). Les classes percussives comme `gun_shot` (86.5% F1, 100% recall) et `jackhammer` obtiennent d'excellentes performances, tandis que `children_playing` et `street_music` restent plus difficiles en raison de leur grande variabilité spectrale.
2. **Baseline embeddings** : on projette chaque WAV via PANNs (Pretrained Audio Neural Networks, Kong et al. 2020), un réseau pré-entraîné sur AudioSet contenant 527 classes audio. PANNs génère des embeddings de dimension 2048 représentant des caractéristiques audio de haut niveau. Nous entraînons ensuite un classifieur linéaire ou MLP sur ces embeddings. Ce chemin sert de point de comparaison et alimente les rapports "embeddings".

#### Justification : CNN from scratch vs Transformers
Contrairement aux approches récentes privilégiant les Transformers (Audio Spectrogram Transformer, Wav2Vec2), nous avons choisi un **CNN simple entraîné from scratch** pour plusieurs raisons :

1. **Taille du dataset** : Avec seulement 7 895 échantillons d'entraînement (folds 1-9), les Transformers risquent fortement le surapprentissage sans pré-entraînement massif. Les CNN ont démontré leur efficacité sur des datasets de taille modeste.

2. **Data augmentation efficace** : Nous compensons la petite taille du dataset par :
   - SpecAugment (masques temps/fréquence)
   - Time-shift aléatoire (±10% de la durée)
   - Ajout de bruit gaussien léger (σ ∈ [0.005, 0.02])
   - WeightedRandomSampler pour équilibrer les classes déséquilibrées

3. **Inductive bias adapté** : Les spectrogrammes audio présentent des motifs locaux (harmoniques, transitoires) naturellement capturés par les convolutions. Les Transformers nécessitent plus de données pour apprendre ces invariances.

4. **Efficacité computationnelle** : Notre CNN (~217k paramètres) s'entraîne en ~15 minutes sur CPU et infère en <100ms par fichier. Un Transformer comparable nécessiterait GPU et temps d'entraînement ×10.

Les métriques (accuracy, macro-F1, rapports détaillés par classe) sont exportées dans `reports/metrics*.json` ainsi que des matrices de confusion PNG (normalisée ou non). Cela permet de recharger les performances sans relancer l'entraînement, tout en tenant les poids prêts pour le serveur MCP.

### MCP (Model Context Protocol)
`mcp_server/server.py` est une API FastAPI dédiée. Elle charge les poids au démarrage, cache les modèles en mémoire et expose des endpoints :
- `GET /health` : statut, device (CPU/GPU), checkpoint utilisé.
- `POST /infer` : prend `wav_path`, `topk`, `use_subset`, renvoie un JSON avec `predicted_label`, `confidence`, `threshold`, `is_confident`, `top_k` détaillé.
- `GET /metrics?variant=cnn|embeddings` et `GET /reports` : renvoient les JSON ou la liste des fichiers pour que l’interface puisse afficher des synthèses.

L’orchestrateur appelle d’abord /infer (ou /metrics /reports selon le besoin), construit un bloc textuel formaté (classe, confiance, top‑k, statistiques globales) puis le transmet au LLM pour verbaliser la réponse. Cette séquence garantit que les informations chiffrées proviennent toujours du modèle spécialiste, et que le LLM n’invente pas de nouveaux résultats.

### Interface utilisateur
`app_gradio.py` fournit l'UI Web. Elle s'appuie sur `gr.Blocks` avec un layout en deux colonnes (chat à gauche, upload à droite) et ajoute :
- Un en‑tête glassmorphism (gradient, typographie, icônes).
- Un sélecteur de prompts exemples (métriques, rapports, analyses prédéfinies).
- Un module d'upload WAV (`gr.File`) avec statut visuel et un bouton "Analyser ce fichier".
- Une carte d'information rappelant les formats supportés, le nombre de classes et le modèle utilisé.
Le CSS personnalisé apporte transparence, ombres, hover effects et rend l'interface lisible même sur petits écrans.

## Développement : difficultés et solutions
- **Mise en place de l’environnement** : UrbanSound8K est volumineux (plusieurs Go) et PyTorch doit rester en mode CPU pour respecter les contraintes de certaines machines. Nous avons scripté `scripts/setup_and_run.sh`, limité Torch à la version CPU (`--extra-index-url https://download.pytorch.org/whl/cpu`) et utilisé un subset `data/subset/` pour les tests rapides. Cela nous permet d’itérer sans réimporter l’intégralité du dataset.

- **Conception du modèle** : équilibrer les classes et éviter l’overfitting s’est avéré délicat. Nous avons combiné `WeightedRandomSampler`, SpecAugment léger et un early stopping basé sur la perte de validation. La baseline PANNs sert de garde-fou : si une régression apparaît sur le CNN, nous pouvons comparer rapidement avec les embeddings.

- **Intégration MCP** : exposer des WAV stockés localement nécessite une bonne gestion des chemins et des erreurs (fichier manquant, timeout). Nous avons centralisé les appels dans `post_infer`, `get_metrics`, `get_reports` côté orchestrateur, ajouté des `raise_for_status()` et remonté les messages d’erreur au chat pour aider l’utilisateur (“Erreur MCP, impossible d’analyser ce WAV…”).

- **Intégration LLM** : LM Studio n’accepte que les rôles `user/assistant`. Nous avons donc converti les messages système en pseudo-messages utilisateur lors de l’envoi (helper `prepare_messages_for_lmstudio`). Nous avons également dû reformuler certaines réponses côté Python (résumés top‑k, listes de rapports) pour éviter que le LLM ne se lance dans de longs paragraphes hors sujet.

- **Interface Gradio** : au départ, la zone d’upload occupait la moitié de l’écran. Nous avons réduit les marges, remplacé le composant `Audio` par un `UploadButton` plus compact et ajouté des exemples préremplis pour guider les tests. Le CSS glassmorphism améliore le contraste et rend visibles les états (boutons, hover, statuts). L’utilisateur peut maintenant soit taper un chemin, soit glisser-déposer un fichier.

## Conclusion
Cette première version de SonicWatch livre un pipeline complet : modèles CNN et embeddings entraînés, métriques exportées, serveur MCP opérationnel, orchestrateur CLI et chatbot Gradio connectés à un LLM local. Nous savons désormais manipuler des données audio en PyTorch, entraîner des modèles convolutionnels, exploiter un LLM via une API compatible OpenAI, concevoir un protocole MCP simplifié et bâtir une interface moderne.

Pour la suite, nous envisageons plusieurs améliorations :

1. Raffiner l’architecture et les hyperparamètres du CNN (exploration d’autres schémas de convolution, réglage plus fin des augmentations et de la régularisation).
2. Améliorer encore l’interface utilisateur et le prompt du chatbot (messages plus pédagogiques, meilleure gestion des cas ambigus, nouveaux exemples guidés).
3. Simplifier le déploiement en fournissant un empaquetage plus intégré (scripts supplémentaires, voire une conteneurisation type Docker ou une alternative à LM Studio si nécessaire).

Avec ces fondations, nous disposons d’un socle solide pour livrer une version 2 plus performante et mieux outillée pour l’industrie.
