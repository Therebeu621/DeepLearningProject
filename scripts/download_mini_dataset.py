"""
Télécharge le mini-dataset UrbanSounds depuis Hugging Face.
Utilisé pour tester rapidement le projet sans télécharger UrbanSound8K complet.
"""

from huggingface_hub import snapshot_download
import os

# Active le téléchargeur rapide (optionnel mais utile)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

print("🚀 Téléchargement du mini-dataset UrbanSounds/urban_sounds_small...")

snapshot_download(
    repo_id="UrbanSounds/urban_sounds_small",
    repo_type="dataset",
    local_dir="data/urban_sounds_small",
    allow_patterns=[
        "urban_sounds_small/metadata.csv",
        "urban_sounds_small/01_gunshot/*.wav",
        "urban_sounds_small/05_claxon/*.wav",
        "urban_sounds_small/07_loud_people/*.wav",
        "urban_sounds_small/10_music/*.wav",
    ],
    resume_download=True,
)

print("✅ Téléchargé → data/urban_sounds_small/urban_sounds_small")
