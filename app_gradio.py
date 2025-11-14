import gradio as gr
import json
import os
import re
import requests
from openai import OpenAI
from pathlib import Path

# --- Configuration ---
LMSTUDIO_BASE = os.getenv("LMSTUDIO_BASE", "http://localhost:1234/v1")
LMSTUDIO_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
MODEL = os.getenv("LMSTUDIO_MODEL", "mistralai/mistral-7b-instruct-v0.3")
API_BASE = os.getenv("MCP_BASE_URL", "http://127.0.0.1:8000")
PROMPT_PATH = Path("chatbot/prompt.md")
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()

# --- Modern CSS with Glassmorphism ---
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

/* Messages utilisateur */
.user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 0.8rem 1.2rem !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
}

/* Messages assistant */
.bot-message {
    background: rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 0.8rem 1.2rem !important;
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
#examples-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.example-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
}

.example-card:hover {
    background: rgba(255, 255, 255, 0.1);
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
}

/* Results card avec couleurs */
.result-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-left: 4px solid;
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.confidence-high {
    border-color: #4ade80;
    background: rgba(74, 222, 128, 0.1);
}

.confidence-medium {
    border-color: #fbbf24;
    background: rgba(251, 191, 36, 0.1);
}

.confidence-low {
    border-color: #f87171;
    background: rgba(248, 113, 113, 0.1);
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

/* Icons mapping */
.icon-air_conditioner::before { content: "❄️ "; }
.icon-car::before { content: "🚗 "; }
.icon-children::before { content: "👶 "; }
.icon-dog::before { content: "🐕 "; }
.icon-drilling::before { content: "🔨 "; }
.icon-engine_idling::before { content: "🚙 "; }
.icon-gun_shot::before { content: "💥 "; }
.icon-jackhammer::before { content: "⚒️ "; }
.icon-siren::before { content: "🚨 "; }
.icon-street_music::before { content: "🎵 "; }
"""

# --- OpenAI Client ---
client = OpenAI(base_url=LMSTUDIO_BASE, api_key=LMSTUDIO_KEY)

# --- MCP Functions ---
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

def detect_wav_path(text: str):
    match = re.search(r"([\w./-]+\.wav)", text)
    return match.group(1) if match else None

# --- Format results with visual cards ---
def format_inference_result(data):
    if "error" in data:
        return f"❌ {data['error']}"
    
    result = data.get("result", {})
    top_class = result.get("top_class", "inconnu")
    confidence = result.get("confidence", 0)
    top_k = result.get("top_k", [])
    
    # Icon mapping
    icon_map = {
        "air_conditioner": "❄️", "car": "🚗", "children": "👶",
        "dog": "🐕", "drilling": "🔨", "engine_idling": "🚙",
        "gun_shot": "💥", "jackhammer": "⚒️", "siren": "🚨",
        "street_music": "🎵"
    }
    
    # Confidence level
    if confidence >= 0.6:
        conf_class = "high"
        conf_icon = "✅"
        conf_text = "Haute"
    elif confidence >= 0.4:
        conf_class = "medium"
        conf_icon = "⚠️"
        conf_text = "Moyenne"
    else:
        conf_class = "low"
        conf_icon = "❌"
        conf_text = "Faible"
    
    icon = icon_map.get(top_class, "🔊")
    
    output = f"""
### 🎧 Résultat d'analyse

<div class="result-card confidence-{conf_class}">
    <h4>{icon} Classe détectée: <strong>{top_class}</strong></h4>
    <p>{conf_icon} Confiance: <strong>{confidence*100:.1f}%</strong> ({conf_text})</p>
</div>

#### 📊 Top-{len(top_k)} prédictions:
"""
    
    for i, (cls, prob) in enumerate(top_k, 1):
        cls_icon = icon_map.get(cls, "•")
        bar_width = int(prob * 30)
        bar = "█" * bar_width + "░" * (30 - bar_width)
        output += f"\n{i}. {cls_icon} **{cls}** ({prob*100:.1f}%)\n   `{bar}`\n"
    
    return output

# --- Main Chatbot ---
def respond(message, chat_history):
    messages = [
        {"role": "user", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Compris. Je suis prêt à vous aider."}
    ]
    
    for user_msg, assistant_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": message})
    
    context_chunks = []
    wav_path = detect_wav_path(message)
    
    if wav_path:
        infer_data = post_infer(wav_path)
        metrics = get_metrics("cnn")
        reports = get_reports()
        
        context_chunks.append("INFERENCE_JSON\n" + json.dumps(infer_data, ensure_ascii=False, indent=2))
        context_chunks.append("METRICS_JSON\n" + json.dumps(metrics, ensure_ascii=False, indent=2))
        context_chunks.append("REPORTS\n" + json.dumps(reports, ensure_ascii=False))
    elif "metric" in message.lower() or "score" in message.lower():
        metrics = get_metrics("cnn")
        context_chunks.append("METRICS_JSON\n" + json.dumps(metrics, ensure_ascii=False, indent=2))
    elif "rapport" in message.lower() or "report" in message.lower():
        reports = get_reports()
        context_chunks.append("REPORTS\n" + json.dumps(reports, ensure_ascii=False))
    
    if context_chunks:
        context = "\n\n".join(context_chunks)
        context_message = "Voici les données MCP obtenues. Utilise-les pour répondre en suivant tes règles.\n" + context
        messages.append({"role": "user", "content": context_message})
    
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Erreur LLM: {e}"

def analyze_uploaded_audio(audio_filepath):
    if not audio_filepath:
        return gr.update(), gr.update()
    message = f"Analyse ce fichier: {audio_filepath}"
    reply = respond(message, [])
    history = [(message, reply)]
    return history, history

def handle_audio_upload(uploaded_files):
    if not uploaded_files:
        return None, "📁 Aucun fichier sélectionné"
    
    file_obj = uploaded_files[0] if isinstance(uploaded_files, (list, tuple)) else uploaded_files
    filepath = getattr(file_obj, "name", None) or (file_obj if isinstance(file_obj, str) else None)
    
    if not filepath:
        return None, "📁 Aucun fichier sélectionné"
    
    filename = Path(filepath).name
    return filepath, f'<span class="file-ready">✅ Fichier prêt: {filename}</span>'

# --- Launch ---
if __name__ == "__main__":
    with gr.Blocks(css=APP_CSS, theme=gr.themes.Soft()) as demo:
        selected_audio = gr.State(None)
        
        # Header
        gr.HTML("""
        <div class="app-header fade-in">
            <h1>🎧 SonicWatch</h1>
            <p>Analyse intelligente des nuisances sonores urbaines</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chatbot
                chat = gr.ChatInterface(
                    fn=respond,
                    examples=[
                        ["📊 Quelles sont les métriques du modèle CNN ?"],
                        ["⚖️ Compare les performances CNN vs embeddings"],
                        ["📈 Liste les rapports visuels disponibles"],
                        ["🎵 Analyse: data/subset/shot556_29_ch01_180718_162104_16_.wav"],
                    ],
                    chatbot=gr.Chatbot(height=450, elem_classes="chatbot-container"),
                    textbox=gr.Textbox(placeholder="💬 Posez votre question...", scale=7)
                )
            
            with gr.Column(scale=1):
                # Upload Section
                gr.HTML('<h3 style="color: white; margin-bottom: 1rem;">📤 Upload Audio</h3>')
                
                with gr.Group(elem_id="upload-zone"):
                    upload_status = gr.HTML(
                        '<div id="upload-status">🎵 Glissez un fichier WAV ici</div>'
                    )
                    audio_input = gr.UploadButton(
                        "📂 Choisir un fichier",
                        file_types=["audio"],
                        elem_classes="btn-primary"
                    )
                
                analyze_btn = gr.Button("🔍 Analyser", elem_classes="btn-primary", size="lg")
                
                # Quick info
                gr.HTML("""
                <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.03); border-radius: 12px;">
                    <h4 style="color: white; margin-bottom: 0.5rem;">ℹ️ Info</h4>
                    <p style="color: rgba(255,255,255,0.7); font-size: 0.85rem; margin: 0;">
                        Formats acceptés: WAV<br>
                        Classes: 10 types de sons urbains<br>
                        Modèle: CNN (71.33% accuracy)
                    </p>
                </div>
                """)
        
        # Events
        audio_input.upload(
            fn=handle_audio_upload,
            inputs=audio_input,
            outputs=[selected_audio, upload_status]
        )
        
        analyze_btn.click(
            fn=analyze_uploaded_audio,
            inputs=selected_audio,
            outputs=[chat.chatbot, chat.chatbot_state]
        )
    
    demo.launch(share=False)