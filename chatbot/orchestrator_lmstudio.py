import json
import os
import re
import sys
from pathlib import Path

import requests
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234")
LMSTUDIO_BASE = os.getenv("LMSTUDIO_BASE", f"{LMSTUDIO_BASE_URL}/v1")
LMSTUDIO_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
MODEL = os.getenv("LMSTUDIO_MODEL", "mistralai/mistral-7b-instruct-v0.3")
API_BASE = os.getenv("MCP_BASE_URL", "http://127.0.0.1:8000")
PROMPT_PATH = Path("chatbot/prompt.md")
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()

client = OpenAI(base_url=f"{LMSTUDIO_BASE_URL}/v1", api_key="not-needed")


def post_infer(wav_path: str, topk: int = 3, use_subset: bool = True):
    payload = {"wav_path": wav_path, "topk": topk, "use_subset": use_subset}
    r = requests.post(f"{API_BASE}/infer", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def get_metrics(variant: str = "cnn"):
    r = requests.get(f"{API_BASE}/metrics", params={"variant": variant}, timeout=10)
    r.raise_for_status()
    return r.json()


def get_reports():
    r = requests.get(f"{API_BASE}/reports", timeout=10)
    r.raise_for_status()
    return r.json()


def detect_wav_path(text: str):
    match = re.search(r"([\w./-]+\.wav)", text)
    if match:
        return match.group(1)
    return None


def _format_percent(value, decimals=2, default="N/A"):
    try:
        return f"{float(value):.{decimals}%}"
    except (TypeError, ValueError):
        return default


def _format_topk(topk):
    if not topk:
        return "aucun"
    chunks = []
    for item in topk:
        rank = item.get("rank")
        label = item.get("label", "?")
        conf = _format_percent(item.get("confidence"), 2)
        if rank is not None:
            chunks.append(f"{rank}) {label} ({conf})")
        else:
            chunks.append(f"{label} ({conf})")
    return " • ".join(chunks)


def build_summary_text(infer_data, metrics, reports):
    conf = infer_data.get("confidence")
    threshold = infer_data.get("threshold")
    status = "✅" if infer_data.get("is_confident") else "⚠️"
    topk_str = _format_topk(infer_data.get("top_k") or [])
    reports_list = []
    if isinstance(reports, dict):
        reports_list = reports.get("reports") or []
    elif isinstance(reports, list):
        reports_list = reports
    reports_text = ", ".join(reports_list) if reports_list else "aucun"
    return (
        "🎧 Résultat\n"
        f"- Classe : {infer_data.get('predicted_label', 'inconnu')}\n"
        f"- Confiance : {_format_percent(conf, 2)} (seuil {_format_percent(threshold, 0)}) → {status}\n"
        f"- Top-k : {topk_str}\n"
        "\n📊 Contexte modèle\n"
        f"- Accuracy globale : {_format_percent(metrics.get('accuracy'), 2)}\n"
        f"- Macro-F1 : {_format_percent(metrics.get('macro_f1'), 2)}\n"
        f"- Rapports dispo : {reports_text}"
    )


def prepare_messages_for_lmstudio(history):
    """LM Studio n'accepte que les roles user/assistant, on transforme les messages systeme."""
    prepared = []
    for msg in history:
        if msg.get("role") == "system":
            prepared.append({
                "role": "user",
                "content": "[SYSTEM]\n" + msg.get("content", ""),
            })
        else:
            prepared.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })
    return prepared


def run_chat():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    print("🧠 LM Studio orchestrateur prêt. Tape 'exit' pour quitter.\n")
    while True:
        user = input("Vous: ")
        if user.strip().lower() in {"exit", "quit"}:
            print("👋 Fin de session.")
            break
        if not user.strip():
            continue

        messages.append({"role": "user", "content": user})
        context_chunks = []
        has_context = False
        summary_text = None

        wav_path = detect_wav_path(user)
        if wav_path:
            print(f"🔁 Appel MCP pour le fichier : {wav_path}")
            try:
                infer_data = post_infer(wav_path)
                metrics = get_metrics("cnn")
                reports = get_reports()
                summary_text = build_summary_text(infer_data, metrics, reports)
                print("\n" + summary_text + "\n")
                context_chunks.append(
                    "INFERENCE_JSON\n" + json.dumps(infer_data, ensure_ascii=False, indent=2)
                )
                context_chunks.append(
                    "METRICS_JSON\n" + json.dumps(metrics, ensure_ascii=False, indent=2)
                )
                context_chunks.append(
                    "REPORTS\n" + json.dumps(reports, ensure_ascii=False)
                )
            except Exception as exc:
                print(f"❌ Erreur MCP, impossible d'analyser ce WAV : {exc}")
                continue
        elif "metric" in user.lower() or "score" in user.lower():
            try:
                metrics = get_metrics("cnn")
                context_chunks.append(
                    "METRICS_JSON\n" + json.dumps(metrics, ensure_ascii=False, indent=2)
                )
            except Exception as exc:
                print(f"❌ Erreur MCP lors de la récupération des métriques : {exc}")
                continue
        elif "rapport" in user.lower() or "report" in user.lower():
            try:
                reports = get_reports()
                context_chunks.append("REPORTS\n" + json.dumps(reports, ensure_ascii=False))
            except Exception as exc:
                print(f"❌ Erreur MCP lors de la récupération des rapports : {exc}")
                continue

        if context_chunks:
            has_context = True
            context = "\n\n".join(context_chunks)
            messages.append({
                "role": "user",
                "content": (
                    "Voici les données MCP obtenues. Utilise-les pour répondre en suivant tes règles.\n"
                    + context
                ),
            })
        if has_context:
            print("📨 Contexte MCP injecté dans le prompt (INFERENCE_JSON + METRICS_JSON + REPORTS)")

        if summary_text:
            messages.append({
                "role": "user",
                "content": (
                    "Voici le résumé déjà formaté de la prédiction du modèle CNN. "
                    "Explique ce résultat en français clair, sans réécrire le gabarit, "
                    "et sans répéter ton mode d'emploi.\n\n"
                    + summary_text
                ),
            })

        lm_messages = prepare_messages_for_lmstudio(messages)
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=lm_messages,
                temperature=0.3,
            )
        except Exception as exc:
            print(f"❌ Erreur LLM: {exc}")
            continue
        reply = resp.choices[0].message.content
        print("\n🤖 LLM:", reply, "\n")
        messages.append({"role": "assistant", "content": reply})


def main():
    run_chat()


if __name__ == "__main__":
    main()
