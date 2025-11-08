"""Boucle de chat interactive avec appels HTTP (/infer, /metrics, /reports).
Simule un LLM simple: l'utilisateur tape une intention, nous détectons
le besoin (infer/metrics/reports) et interrogeons le serveur MCP.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Dict

import requests

DEFAULT_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000")
PROMPT_PATH = Path(__file__).with_name("prompt.md")


class MCPClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _get(self, endpoint: str, **params) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}{endpoint}", params=params or None, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}{endpoint}", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        return self._get("/health")

    def metrics(self, variant: str = "cnn") -> Dict[str, Any]:
        return self._get("/metrics", variant=variant)

    def reports(self) -> Dict[str, Any]:
        return self._get("/reports")

    def infer(self, wav_path: str, topk: int = 3, use_subset: bool = True) -> Dict[str, Any]:
        payload = {"wav_path": wav_path, "topk": topk, "use_subset": use_subset}
        return self._post("/infer", payload)


HELP_TEXT = """
Commandes disponibles :
  infer <chemin_wav> [--topk int] [--no-subset]
  metrics [cnn|embeddings]
  reports
  health
  prompt          # affiche le prompt SonicWatch
  help            # rappelle cette liste
  exit / quit
"""


def format_infer(infer_data: Dict[str, Any], metrics: Dict[str, Any], reports: Dict[str, Any]) -> str:
    topk = " • ".join(
        f"{item['rank']}) {item['label']} ({item['confidence']:.2%})"
        for item in infer_data.get("top_k", [])
    )
    status = "✅" if infer_data.get("is_confident") else "⚠️"  # seuil 60%
    reports_list = reports.get("reports", [])
    reports_str = ", ".join(reports_list) if reports_list else "aucun fichier"
    return (
        "🎧 Résultat\n"
        f"- Fichier : {infer_data.get('wav_path')}\n"
        f"- Classe : {infer_data.get('predicted_label')}\n"
        f"- Confiance : {infer_data.get('confidence', 0):.2%} (seuil {infer_data.get('threshold', 0.6):.0%}) → {status}\n"
        f"- Top-k : {topk or 'N/A'}\n\n"
        "📊 Contexte modèle\n"
        f"- Accuracy globale : {metrics.get('accuracy', 0):.2%}\n"
        f"- Macro-F1 : {metrics.get('macro_f1', 0):.2%}\n"
        f"- Rapports : {reports_str}\n"
    )


def parse_infer_args(line: str):
    tokens = shlex.split(line)
    wav_path = tokens[1]
    topk = 3
    use_subset = True
    i = 2
    while i < len(tokens):
        if tokens[i] == "--topk" and i + 1 < len(tokens):
            topk = int(tokens[i + 1])
            i += 2
        elif tokens[i] == "--no-subset":
            use_subset = False
            i += 1
        else:
            raise ValueError(f"Option inconnue: {tokens[i]}")
    return wav_path, topk, use_subset


def run_chat_loop(client: MCPClient) -> None:
    print("=== SonicWatch CLI ===")
    print("Tape 'help' pour voir les commandes.")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Fin de session.")
            break
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            print("👋 Bye")
            break
        if line.lower() == "help":
            print(HELP_TEXT)
            continue
        if line.lower() == "prompt":
            if PROMPT_PATH.exists():
                print(PROMPT_PATH.read_text(encoding="utf-8"))
            else:
                print("prompt.md introuvable")
            continue
        if line.lower() == "health":
            data = client.health()
            print(f"Status={data.get('status')} device={data.get('device')} weights={data.get('default_weights')}")
            continue
        if line.startswith("metrics"):
            variant = "cnn"
            parts = line.split()
            if len(parts) > 1:
                variant = parts[1]
            data = client.metrics(variant=variant)
            print(f"📊 Metrics ({variant}) : accuracy {data.get('accuracy'):.2%} | macro-F1 {data.get('macro_f1'):.2%}")
            continue
        if line == "reports":
            data = client.reports()
            print("Rapports disponibles :", ", ".join(data.get("reports", [])) or "aucun")
            continue
        if line.startswith("infer "):
            try:
                wav_path, topk, use_subset = parse_infer_args(line)
            except Exception as exc:
                print("❌", exc)
                continue
            try:
                infer_data = client.infer(wav_path, topk=topk, use_subset=use_subset)
                metrics = client.metrics(variant="cnn")
                reports = client.reports()
                print(format_infer(infer_data, metrics, reports))
            except requests.HTTPError as exc:
                print("❌ Erreur API:", exc)
            except Exception as exc:
                print("❌", exc)
            continue
        # fallback: simple heuristique -> si la phrase contient 'rapport', 'metrics', etc.
        lowered = line.lower()
        if "rapport" in lowered or "report" in lowered:
            data = client.reports()
            print("Rapports: ", ", ".join(data.get("reports", [])))
        elif re.search(r"(metric|score|performance)", lowered):
            data = client.metrics(variant="cnn")
            print(f"Accuracy {data.get('accuracy'):.2%} | Macro-F1 {data.get('macro_f1'):.2%}")
        elif "analyser" in lowered or lowered.startswith("test "):
            print("Utilise la commande `infer <wav>` pour lancer une prédiction.")
        else:
            print("Je ne comprends pas. Tape `help` pour la liste des commandes.")


def main():
    parser = argparse.ArgumentParser(description="Boucle de chat interactive (SonicWatch)")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL du serveur MCP (défaut http://127.0.0.1:8000)")
    args = parser.parse_args()
    client = MCPClient(args.url)
    run_chat_loop(client)


if __name__ == "__main__":
    main()
