# 🚀 Installation et exécution en une commande
bash setup.sh

# 🎵 DeepLearningProject – UrbanSound8K

## 🧩 Installation (Ubuntu / WSL)

### 1. Créer et activer l’environnement virtuel

python3 -m venv .venv
source .venv/bin/activate


### 2. Installer les dépendances

pip install --upgrade pip
pip install -r requirements.txt

🔹 Mini-dataset HuggingFace (recommandé pour tester rapidement)

1. Télécharger les fichiers audio et métadonnées

pip install --no-cache-dir --upgrade huggingface_hub hf-transfer
python scripts/download_mini_dataset.py

2. Générer un CSV compatible UrbanSound8K

python scripts/generate_csv.py

- Exécution des scripts

python -m src.check_data
python -m src.make_subset   # optionnel, crée un mini-jeu data/subset/
python -m src.train
python -m src.evaluate
python -m src.infer


