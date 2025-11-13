import gradio as gr
import json
import os
import re
import requests
from openai import OpenAI
from pathlib import Path

# --- Configuration (Copied from your orchestrator) ---
LMSTUDIO_BASE = os.getenv("LMSTUDIO_BASE", "http://localhost:1234/v1")
LMSTUDIO_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
MODEL = os.getenv("LMSTUDIO_MODEL", "mistralai/mistral-7b-instruct-v0.3")
API_BASE = os.getenv("MCP_BASE_URL", "http://127.0.0.1:8000")
PROMPT_PATH = Path("chatbot/prompt.md")
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()

# --- OpenAI Client Setup ---
client = OpenAI(base_url=LMSTUDIO_BASE, api_key=LMSTUDIO_KEY)

# --- MCP Server Helper Functions (Copied from your orchestrator) ---
def post_infer(wav_path: str, topk: int = 3, use_subset: bool = True):
    payload = {"wav_path": wav_path, "topk": topk, "use_subset": use_subset}
    # Note: Added error handling for robustness in a web app
    try:
        r = requests.post(f"{API_BASE}/infer", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to MCP server: {e}"}

def get_metrics(variant: str = "cnn"):
    try:
        r = requests.get(f"{API_BASE}/metrics", params={"variant": variant}, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to MCP server: {e}"}

def get_reports():
    try:
        r = requests.get(f"{API_BASE}/reports", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to MCP server: {e}"}

def detect_wav_path(text: str):
    match = re.search(r"([\w./-]+\.wav)", text)
    if match:
        return match.group(1)
    return None

# --- The Main Chatbot Logic ---
# This function is what Gradio calls every time you hit "Submit"
def respond(message, chat_history):
    # 1. Format the history for the LLM
    # The system prompt is the first "user" message, and the assistant agrees.
    messages = [
        {"role": "user", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Compris. Je suis prêt à vous aider."}
    ]
    # Add the rest of the conversation
    for user_msg, assistant_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    # Add the new user message
    messages.append({"role": "user", "content": message})

    # 2. Check for tools/context (Your orchestrator logic)
    context_chunks = []
    
    wav_path = detect_wav_path(message)
    if wav_path:
        infer_data = post_infer(wav_path)
        metrics = get_metrics("cnn") # Or your best model, e.g., "embeddings"
        reports = get_reports()
        
        context_chunks.append("INFERENCE_JSON\n" + json.dumps(infer_data, ensure_ascii=False, indent=2))
        context_chunks.append("METRICS_JSON\n" + json.dumps(metrics, ensure_ascii=False, indent=2))
        context_chunks.append("REPORTS\n" + json.dumps(reports, ensure_ascii=False))

    elif "metric" in message.lower() or "score" in message.lower():
        metrics = get_metrics("cnn") # Or your best model
        context_chunks.append("METRICS_JSON\n" + json.dumps(metrics, ensure_ascii=False, indent=2))

    elif "rapport" in message.lower() or "report" in message.lower():
        reports = get_reports()
        context_chunks.append("REPORTS\n" + json.dumps(reports, ensure_ascii=False))

    # 3. Add context to the LLM message
    if context_chunks:
        context = "\n\n".join(context_chunks)
        context_message = (
            "Voici les données MCP obtenues. Utilise-les pour répondre en suivant tes règles.\n"
            + context
        )
        messages.append({"role": "user", "content": context_message})

    # 4. Call the LLM (LM Studio)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
        )
        reply = resp.choices[0].message.content
        return reply
    except Exception as e:
        return f"Erreur de connexion au LLM (LM Studio): {e}"


def analyze_uploaded_audio(audio_filepath):
    if not audio_filepath:
        return gr.update(), gr.update()
    message = f"Analyse ce fichier: {audio_filepath}"
    reply = respond(message, [])
    history = [(message, reply)]
    return history, history


def handle_audio_upload(uploaded_files):
    if not uploaded_files:
        return None, "Aucun fichier sélectionné."
    file_obj = (
        uploaded_files[0]
        if isinstance(uploaded_files, (list, tuple))
        else uploaded_files
    )
    filepath = getattr(file_obj, "name", None) or (
        file_obj if isinstance(file_obj, str) else None
    )
    if not filepath:
        return None, "Aucun fichier sélectionné."
    filename = Path(filepath).name
    return filepath, f"Fichier prêt : {filename}"

# --- Launch the Web Interface ---
if __name__ == "__main__":
    with gr.Blocks() as demo:
        selected_audio = gr.State(None)

        chat = gr.ChatInterface(
            fn=respond,
            title="SonicWatch 🎧",
            description="Posez-moi des questions sur les sons. Pour analyser un fichier, donnez-moi son chemin (ex: data/subset/...).",
            examples=[
                ["Quelles sont les métriques du modèle ?"],
                ["Donne-moi la liste des rapports."],
                ["Analyse ce fichier: data/subset/shot556_29_ch01_180718_162104_16_.wav"]
            ]
        )

        gr.Markdown("### 🔊 Analyser un fichier audio (upload)")

        with gr.Row():
            with gr.Column(scale=3):
                upload_status = gr.Markdown(
                    "Aucun fichier sélectionné.",
                    elem_id="upload-status",
                )
                audio_input = gr.UploadButton(
                    "Cliquez pour uploader un fichier audio",
                    file_types=["audio"],
                    size="md",
                    scale=1,
                    min_width=0,
                )
            analyze_btn = gr.Button("Analyser ce fichier", scale=1)

        audio_input.upload(
            fn=handle_audio_upload,
            inputs=audio_input,
            outputs=[selected_audio, upload_status],
        )

        analyze_btn.click(
            fn=analyze_uploaded_audio,
            inputs=selected_audio,
            outputs=[chat.chatbot, chat.chatbot_state]
        )

    demo.launch(share=False) # Set share=True to get a public link
