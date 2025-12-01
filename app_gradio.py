"""
Gradio UI for SonicWatch: chat + WAV upload + MCP/LM Studio calls.
Version fusionnée : logique récente + style glassmorphism.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LMSTUDIO_BASE = os.getenv("LMSTUDIO_BASE", "http://localhost:1234/v1")
LMSTUDIO_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
MODEL = os.getenv("LMSTUDIO_MODEL", "mistralai/mistral-7b-instruct-v0.3")
API_BASE = os.getenv("MCP_BASE_URL", "http://127.0.0.1:8000")
PROMPT_PATH = Path("chatbot/prompt.md")


def load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    return (
        "Tu es un assistant sonore. Reste concis, explique en francais, et utilise "
        "les donnees MCP (inference, metriques, rapports) si elles sont fournies."
    )


SYSTEM_PROMPT = load_system_prompt()

# ---------------------------------------------------------------------------
# Gradio client patch: tolerate boolean schemas (fixes TypeError in gradio_client)
# ---------------------------------------------------------------------------
import gradio_client.utils as gc_utils  # type: ignore

_orig_json_schema_to_python_type = gc_utils.json_schema_to_python_type


def _safe_json_schema_to_python_type(schema, *args, **kwargs):
    if isinstance(schema, bool):
        return "boolean" if schema else "false"
    try:
        return _orig_json_schema_to_python_type(schema, *args, **kwargs)
    except TypeError:
        return "unknown"


gc_utils.json_schema_to_python_type = _safe_json_schema_to_python_type

# ---------------------------------------------------------------------------
# Styling (glassmorphism)
# ---------------------------------------------------------------------------
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

body, .gradio-container {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1d35 50%, #0f1629 100%) !important;
}

/* Header avec effet glassmorphism */
.app-header {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    text-align: center;
}

.app-header h1 {
    font-size: 2.5rem !important;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0 !important;
}

.app-header p {
    color: rgba(255, 255, 255, 0.7);
    font-size: 1rem;
    margin: 0 !important;
}

/* Chatbot moderne */
.chatbot-container {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
}

/* Upload section avec drag & drop style */
#upload-zone {
    background: rgba(102, 126, 234, 0.05);
    border: 2px dashed rgba(102, 126, 234, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.3s ease;
    text-align: center;
}

#upload-zone:hover {
    background: rgba(102, 126, 234, 0.1);
    border-color: rgba(102, 126, 234, 0.5);
    transform: translateY(-2px);
}

#upload-status {
    color: rgba(255, 255, 255, 0.7);
    font-size: 0.9rem;
    margin: 0.5rem 0;
}

.file-ready {
    color: #4ade80 !important;
    font-weight: 600;
}

/* Boutons modernes */
.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.5rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s ease !important;
}

.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

/* Examples cards */
.example-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
    margin: 0.3rem 0;
}

.example-card:hover {
    background: rgba(255, 255, 255, 0.1);
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

/* Input field */
textarea, input {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: white !important;
    backdrop-filter: blur(10px) !important;
}

textarea:focus, input:focus {
    border-color: rgba(102, 126, 234, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
}

::-webkit-scrollbar-thumb {
    background: rgba(102, 126, 234, 0.5);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(102, 126, 234, 0.7);
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.4s ease;
}
"""

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
client = OpenAI(base_url=LMSTUDIO_BASE, api_key=LMSTUDIO_KEY)

# ---------------------------------------------------------------------------
# MCP helpers
# ---------------------------------------------------------------------------
def post_infer(wav_path: str, topk: int = 3, use_subset: bool = True):
    payload = {"wav_path": wav_path, "topk": topk, "use_subset": use_subset}
    try:
        r = requests.post(f"{API_BASE}/infer", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Erreur serveur MCP: {e}"}


def get_metrics(variant: str = "cnn"):
    try:
        r = requests.get(f"{API_BASE}/metrics", params={"variant": variant}, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Erreur serveur MCP: {e}"}


def get_reports():
    try:
        r = requests.get(f"{API_BASE}/reports", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Erreur serveur MCP: {e}"}


def detect_wav_path(text: str) -> Optional[str]:
    match = re.search(r"([\w./-]+\.wav)", text)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------
def build_messages(message: str, history: List[Tuple[str, str]]) -> List[dict]:
    """
    Construit une liste de messages compatible avec LM Studio :
    - uniquement roles "user" et "assistant"
    - le "system prompt" est injecté au début du dernier message user.
    """
    messages: List[dict] = []

    # 1) Historique de la conversation (user / assistant uniquement)
    for u, a in history or []:
        messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})

    # 2) Contexte MCP (inference / métriques / rapports)
    context_chunks: List[str] = []
    wav_path = detect_wav_path(message)

    if wav_path:
        infer_data = post_infer(wav_path)
        metrics = get_metrics("cnn")
        reports = get_reports()
        context_chunks.append("INFERENCE_JSON\n" + json.dumps(infer_data, ensure_ascii=False, indent=2))
        context_chunks.append("METRICS_JSON\n" + json.dumps(metrics, ensure_ascii=False, indent=2))
        context_chunks.append("REPORTS\n" + json.dumps(reports, ensure_ascii=False))
    elif any(k in message.lower() for k in ["metric", "metrique", "score"]):
        metrics = get_metrics("cnn")
        context_chunks.append("METRICS_JSON\n" + json.dumps(metrics, ensure_ascii=False, indent=2))
    elif any(k in message.lower() for k in ["rapport", "report"]):
        reports = get_reports()
        context_chunks.append("REPORTS\n" + json.dumps(reports, ensure_ascii=False))

    # 3) Gros message user final
    user_parts: List[str] = [SYSTEM_PROMPT]

    if context_chunks:
        user_parts.append(
            "Voici des donnees MCP (inference / metriques / rapports). "
            "Utilise-les pour repondre clairement :\n\n" + "\n\n".join(context_chunks)
        )

    user_parts.append("Question utilisateur :\n" + message)

    messages.append({"role": "user", "content": "\n\n".join(user_parts)})

    return messages


def respond(message: str, history: List[Tuple[str, str]]):
    history = history or []
    messages = build_messages(message, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
        )
        reply = completion.choices[0].message.content
    except Exception as exc:
        reply = f"Erreur LLM: {exc}"
    new_history = history + [(message, reply)]
    return new_history, new_history


def analyze_uploaded_audio(audio_filepath: Optional[str], history: List[Tuple[str, str]]):
    history = history or []
    if not audio_filepath:
        return history, history
    message = f"🎵 Analyse ce fichier: {audio_filepath}"
    return respond(message, history)


def handle_audio_upload(uploaded_file: Optional[str]):
    if not uploaded_file:
        return None, '<div id="upload-status">📁 Aucun fichier sélectionné</div>'
    filepath = uploaded_file
    filename = Path(filepath).name
    html = f'<div id="upload-status" class="file-ready">✅ Fichier prêt: {filename}</div>'
    return filepath, html


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
EXAMPLES = [
    "📊 Quelles sont les metriques du modele CNN ?",
    "📈 Liste les rapports visuels disponibles.",
    # À adapter avec un vrai chemin WAV chez toi (subset ou UrbanSound8K/audio/...)
    "🎵 Analyse: data/subset/shot556_29_ch01_180718_162104_16_.wav",
    "🎵 Analyse: data/subset/O-AS-roos.002.200120.141547.39.wav",
]

if __name__ == "__main__":
    with gr.Blocks(css=APP_CSS, theme=gr.themes.Soft(), analytics_enabled=False) as demo:
        chat_state = gr.State([])
        selected_audio = gr.State(None)

        # Header stylé
        gr.HTML(
            """
        <div class="app-header fade-in">
            <h1>🎧 SonicWatch</h1>
            <p>Analyse intelligente des nuisances sonores urbaines</p>
        </div>
        """
        )

        with gr.Row():
            # Colonne chat
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=500,
                    elem_classes="chatbot-container",
                    type="tuples",
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="💬 Pose ta question ou colle un chemin .wav",
                        show_label=False,
                        scale=8,
                    )
                    send_btn = gr.Button("📤 Envoyer", elem_classes="btn-primary", scale=2)
                clear_btn = gr.Button("🗑️ Effacer l'historique")

                gr.Markdown("### 💡 Exemples de requêtes")
                for ex in EXAMPLES:
                    gr.Button(ex, elem_classes="example-card").click(
                        fn=lambda h, ex=ex: respond(ex, h),
                        inputs=chat_state,
                        outputs=[chatbot, chat_state],
                    )

            # Colonne upload
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload WAV")
                with gr.Group(elem_id="upload-zone"):
                    upload_status = gr.HTML('<div id="upload-status">🎵 Glisse un fichier WAV ici</div>')
                    audio_input = gr.File(label="", file_types=[".wav"], type="filepath")
                analyze_btn = gr.Button("🔍 Analyser ce fichier", elem_classes="btn-primary")
                gr.HTML(
                    """
                <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.03); border-radius: 12px;">
                    <h4 style="color: white; margin-bottom: 0.5rem;">ℹ️ Info</h4>
                    <p style="color: rgba(255,255,255,0.7); font-size: 0.85rem; margin: 0;">
                        Formats acceptes: WAV<br>
                        Classes: 10 types de sons urbains<br>
                        Modele: CNN + tete lineaire
                    </p>
                </div>
                """
                )

        # Events
        msg_input.submit(respond, inputs=[msg_input, chat_state], outputs=[chatbot, chat_state]).then(
            lambda: "", outputs=msg_input
        )
        send_btn.click(respond, inputs=[msg_input, chat_state], outputs=[chatbot, chat_state]).then(
            lambda: "", outputs=msg_input
        )
        clear_btn.click(lambda: ([], []), outputs=[chatbot, chat_state])

        audio_input.change(handle_audio_upload, inputs=audio_input, outputs=[selected_audio, upload_status])
        analyze_btn.click(analyze_uploaded_audio, inputs=[selected_audio, chat_state], outputs=[chatbot, chat_state])

    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, inbrowser=False)
