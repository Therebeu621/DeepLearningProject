"""Simple CLI orchestrator bridging the MCP server and a human user.
Simulates the behaviour described in chatbot/prompt.md: fetch metrics,
run inference, and print a friendly summary.
"""

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path
from typing import Any, Dict

import requests

DEFAULT_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000")
PROMPT_PATH = Path(__file__).with_name("prompt.md")


def _get(url: str, endpoint: str, **kwargs) -> Dict[str, Any]:
    resp = requests.get(f"{url}{endpoint}", timeout=10, params=kwargs or None)
    resp.raise_for_status()
    return resp.json()


def _post(url: str, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{url}{endpoint}", json=payload, timeout=20)
    resp.raise_for_status()
    return resp.json()


def cmd_health(args: argparse.Namespace) -> None:
    data = _get(args.url, "/health")
    print(textwrap.dedent(
        f"""
        ✅ Serveur joignable
        - Status : {data.get('status')}
        - Device : {data.get('device')}
        - Poids par défaut : {data.get('default_weights')}
        """
    ).strip())


def cmd_metrics(args: argparse.Namespace) -> None:
    data = _get(args.url, "/metrics", variant=args.variant)
    print(f"📊 Metrics ({args.variant}) — accuracy {data.get('accuracy'):.2%}, macro-F1 {data.get('macro_f1'):.2%}")
    print("Source :", data.get("source"))


def _format_topk(topk: Any) -> str:
    chunks = []
    for item in topk:
        chunks.append(f"{item['rank']}) {item['label']} ({item['confidence']:.2%})")
    return " • ".join(chunks)


def cmd_infer(args: argparse.Namespace) -> None:
    payload = {
        "wav_path": args.wav,
        "topk": args.topk,
        "use_subset": not args.no_subset,
    }
    if args.weights:
        payload["weights_path"] = args.weights
    result = _post(args.url, "/infer", payload)
    metrics = _get(args.url, "/metrics", variant="cnn")

    conf = result["confidence"]
    status = "✅" if result["is_confident"] else "⚠️"
    print("🎧 Résultat")
    print(f"- Classe : {result['predicted_label']}")
    print(f"- Confiance : {conf:.2%} (seuil {result['threshold']:.0%}) → {status}")
    print("- Top-k :", _format_topk(result["top_k"]))
    print("\n📊 Contexte modèle")
    print(f"- Accuracy globale : {metrics['accuracy']:.2%}")
    print(f"- Macro-F1 : {metrics['macro_f1']:.2%}")
    reports = _get(args.url, "/reports").get("reports", [])
    print("- Rapports dispo :", ", ".join(reports) if reports else "aucun")


def cmd_prompt(_: argparse.Namespace) -> None:
    if PROMPT_PATH.exists():
        print(PROMPT_PATH.read_text(encoding="utf-8"))
    else:
        print("prompt.md introuvable.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Client CLI pour le serveur MCP UrbanSound")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL du serveur (défaut http://127.0.0.1:8000)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_health = sub.add_parser("health", help="Vérifier l'état du serveur")
    p_health.set_defaults(func=cmd_health)

    p_metrics = sub.add_parser("metrics", help="Afficher les métriques cnn/embeddings")
    p_metrics.add_argument("--variant", choices=["cnn", "embeddings"], default="cnn")
    p_metrics.set_defaults(func=cmd_metrics)

    p_infer = sub.add_parser("infer", help="Lancer une inférence")
    p_infer.add_argument("wav", help="Chemin vers un WAV")
    p_infer.add_argument("--topk", type=int, default=3)
    p_infer.add_argument("--weights", help="Checkpoint alternatif")
    p_infer.add_argument("--no-subset", action="store_true", help="Forcer l'usage du dataset complet")
    p_infer.set_defaults(func=cmd_infer)

    p_prompt = sub.add_parser("prompt", help="Afficher le prompt système SonicWatch")
    p_prompt.set_defaults(func=cmd_prompt)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
