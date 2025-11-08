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

LMSTUDIO_BASE = os.getenv("LMSTUDIO_BASE", "http://localhost:1234/v1")
LMSTUDIO_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
MODEL = os.getenv("LMSTUDIO_MODEL", "mistralai/mistral-7b-instruct-v0.3")
API_BASE = os.getenv("MCP_BASE_URL", "http://127.0.0.1:8000")
PROMPT_PATH = Path("chatbot/prompt.md")
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()

client = OpenAI(base_url=LMSTUDIO_BASE, api_key=LMSTUDIO_KEY)


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


def run_chat():
    messages = [
        {"role": "user", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Compris. J'appliquerai ces règles."},
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

        wav_path = detect_wav_path(user)
        if wav_path:
            try:
                infer_data = post_infer(wav_path)
                metrics = get_metrics("cnn")
                reports = get_reports()
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
                print("❌ Erreur API:", exc)
                messages.append({"role": "assistant", "content": f"Erreur MCP: {exc}"})
                continue
        elif "metric" in user.lower() or "score" in user.lower():
            try:
                metrics = get_metrics("cnn")
                context_chunks.append(
                    "METRICS_JSON\n" + json.dumps(metrics, ensure_ascii=False, indent=2)
                )
            except Exception as exc:
                print("❌", exc)
                messages.append({"role": "assistant", "content": f"Erreur MCP: {exc}"})
                continue
        elif "rapport" in user.lower() or "report" in user.lower():
            try:
                reports = get_reports()
                context_chunks.append("REPORTS\n" + json.dumps(reports, ensure_ascii=False))
            except Exception as exc:
                print("❌", exc)
                messages.append({"role": "assistant", "content": f"Erreur MCP: {exc}"})
                continue

        if context_chunks:
            context = "\n\n".join(context_chunks)
            messages.append({
                "role": "user",
                "content": (
                    "Voici les données MCP obtenues. Utilise-les pour répondre en suivant tes règles.\n"
                    + context
                ),
            })

        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
        )
        reply = resp.choices[0].message.content
        print("\n🤖 LLM:", reply, "\n")
        messages.append({"role": "assistant", "content": reply})


def main():
    run_chat()


if __name__ == "__main__":
    main()
