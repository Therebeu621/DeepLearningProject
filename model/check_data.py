from .config import CSV_PATH, AUDIO_DIR
assert CSV_PATH.exists(), f"CSV introuvable: {CSV_PATH}"
assert AUDIO_DIR.exists(), f"Dossier audio introuvable: {AUDIO_DIR}"
print("OK: CSV et audio trouvés.")
