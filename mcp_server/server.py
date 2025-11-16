from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model.config import (
    AUDIO_DIR,
    CSV_PATH,
    SUBSET_DIR,
    CONF_THRESHOLD,
)
from model.features import wav_to_logmel
from model.model import SimpleCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_WEIGHTS = Path("weights/urbansound_cnn.pt")
DEFAULT_TOPK = 3

app = FastAPI(title="UrbanSound MCP Server", version="1.0.0")


class InferenceRequest(BaseModel):
    wav_path: str = Field(..., description="Chemin vers un fichier .wav local")
    topk: int = Field(DEFAULT_TOPK, ge=1, le=10)
    use_subset: bool = Field(True, description="Utiliser data/subset si disponible")
    weights_path: Optional[str] = Field(None, description="Chemin alternatif vers un checkpoint torch")


def _load_checkpoint(weights_path: Path):
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        mapping = checkpoint.get("class_to_idx")
    else:
        state = checkpoint
        mapping = None
    return state, mapping


def _load_label_names(use_subset: bool) -> List[str]:
    subset_meta = SUBSET_DIR / "subset_meta.csv"
    if use_subset and subset_meta.exists():
        meta = pd.read_csv(subset_meta)
    elif CSV_PATH and CSV_PATH.exists():
        meta = pd.read_csv(CSV_PATH)
    else:
        return []
    cat = meta["class"].astype("category").cat
    return list(cat.categories)


class ModelService:
    def __init__(self, weights_path: Path, use_subset: bool):
        state, saved_mapping = _load_checkpoint(weights_path)
        n_classes = state["head.1.weight"].shape[0]
        self.model = SimpleCNN(n_classes=n_classes).to(DEVICE)
        self.model.load_state_dict(state)
        self.model.eval()
        self.labels = None
        if saved_mapping:
            idx_to_name = {idx: name for name, idx in saved_mapping.items()}
            self.labels = [idx_to_name.get(i, f"class_{i}") for i in range(n_classes)]
        if self.labels is None:
            names = _load_label_names(use_subset)
            if len(names) == n_classes:
                self.labels = names
            else:
                self.labels = [f"class_{i}" for i in range(n_classes)]

    def predict(self, wav_path: Path, topk: int):
        if not wav_path.exists():
            raise FileNotFoundError(f"Fichier wav introuvable: {wav_path}")
        with torch.no_grad():
            x = wav_to_logmel(wav_path, train_mode=False).unsqueeze(0)
            logits = self.model(x.to(DEVICE))
            probs = torch.softmax(logits, dim=1).cpu().squeeze(0)
        k = min(topk, probs.shape[0])
        top_values, top_indices = torch.topk(probs, k)
        results = []
        for rank, (score, idx) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
            label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
            results.append({"rank": rank, "label": label, "confidence": score})
        best = results[0]
        return {
            "predicted_label": best["label"],
            "confidence": best["confidence"],
            "threshold": CONF_THRESHOLD,
            "is_confident": best["confidence"] >= CONF_THRESHOLD,
            "top_k": results,
        }


class ModelCache:
    def __init__(self):
        self._cache: Dict[tuple[Path, bool], ModelService] = {}

    def get(self, weights_path: Path, use_subset: bool) -> ModelService:
        key = (weights_path.resolve(), use_subset)
        if key not in self._cache:
            self._cache[key] = ModelService(weights_path, use_subset)
        return self._cache[key]


model_cache = ModelCache()


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "default_weights": str(DEFAULT_WEIGHTS)}


@app.get("/metrics")
def metrics(variant: str = "cnn"):
    if variant not in {"cnn", "embeddings"}:
        raise HTTPException(400, detail="variant doit être 'cnn' ou 'embeddings'")
    path = Path("reports/metrics.json") if variant == "cnn" else Path("reports/metrics_embeddings.json")
    if not path.exists():
        raise HTTPException(404, detail=f"Fichier de métriques introuvable: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["source"] = str(path)
    return data


@app.post("/infer")
def infer(request: InferenceRequest):
    weights = Path(request.weights_path) if request.weights_path else DEFAULT_WEIGHTS
    try:
        service = model_cache.get(weights, request.use_subset)
        result = service.predict(Path(request.wav_path), topk=request.topk)
    except FileNotFoundError as exc:
        raise HTTPException(404, detail=str(exc)) from exc
    return {
        "wav_path": request.wav_path,
        "weights": str(weights),
        "use_subset": request.use_subset,
        **result,
    }


@app.get("/reports")
def list_reports():
    report_dir = Path("reports")
    if not report_dir.exists():
        raise HTTPException(404, detail="Le dossier reports/ est introuvable.")
    files = sorted(p.name for p in report_dir.iterdir() if p.is_file())
    return {"reports": files}
