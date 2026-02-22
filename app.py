import streamlit as st
import os, time, subprocess, sys
from rag import (
    build_knowledge_base, 
    generate_proposal, 
    extract_rfp_text, 
    research_competitors, 
    extract_emails, 
    get_client_name, 
    generate_email_body, 
    send_real_email
)
from utils import save_to_word

# Hotfix
try: from duckduckgo_search import DDGS
except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", "duckduckgo-search==6.3.0"])

TMP_KB, TMP_RFP, TMP_OUT = "/tmp/knowledge_base", "/tmp/rfp_input", "/tmp/output"
for p in [TMP_KB, TMP_RFP, TMP_OUT]:
    if not os.path.exists(p): os.makedirs(p)

st.set_page_config(page_title="Proposera AI | Neural Midnight", layout="wide", page_icon="💠")

# --- UI CSS ---
st.markdown("""
    <style>
        .stApp { background-color: #0A101E; background-image: linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px); background-size: 50px 50px; color: #E6F1FF; }
        .form-card { background-color: #121E33; border: 1px solid #1F2A44; padding: 40px; border-radius: 14px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5); max-width: 850px; margin: 0 auto; }
        h1 { color: #00E0FF; text-align: center; font-weight: 800; text-shadow: 0 0 15px rgba(0,224,255,0.4); margin-bottom: 0px; }
        .subtitle { color: #9CB3D1; text-align: center; font-size: 14px; margin-bottom: 30px; }
        .stButton>button { background: linear-gradient(90deg, #007BFF, #00E0FF) !important; color: white !important; font-weight: 700; padding: 14px; border: none; border-radius: 8px; width: 100%; transition: 0.3s; }
        .preview-box { height: 450px; overflow-y: scroll; background-color: #0E1627; padding: 25px; border-radius: 10px; border: 1px solid #1F2F4A; color: #E6F1FF; font-family: 'Inter', sans-serif; line-height: 1.6; }
        [data-testid="stSidebar"] { background-color: #0A101E; border-right: 1px solid #1F2A44; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 📬 Dispatch Control")
    sender_mail = st.text_input("Gmail Address", value="jeevamissvmins34@gmail.com")
    app_pass = st.text_input("App Password", type="password")
    st.divider()
    st.markdown("## 🧠 Reasoning Brain")
    provider = st.selectbox("Provider", ["Google Gemini", "OpenAI GPT", "Anthropic Claude"])
    
    if provider == "Google Gemini":
        model_list = [
            "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-001",
            "gemini-2.0-flash-exp-image-generation", "gemini-2.0-flash-lite-001", "gemini-2.0-flash-lite",
            "gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts", "gemma-3-1b-it",
            "gemma-3-4b-it", "gemini-flash-latest", "gemini-flash-lite-latest", "gemini-pro-latest",
            "gemini-2.5-flash-lite", "gemini-2.5-flash-image", "gemini-2.5-flash-lite-preview-09-2025",
            "gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-3.1-pro-preview",
            "gemini-3.1-pro-preview-customtools", "gemini-3-pro-image-preview", "nano-banana-pro-preview",
            "gemini-robotics-er-1.5-preview", "gemini-2.5-computer-use-preview-10-2025", "deep-research-pro-preview-12-2025"
        ]
    elif provider == "OpenAI GPT": model_list = ["gpt-4o", "gpt-4o-mini", "o1-preview"]
    else: model_list = ["claude-3-5-sonnet-latest"]

    selected_model = st.selectbox("Model", model_list)
    user_api_key = st.text_input(f"{provider} Key (BYOK)", type="password")
    temp = st.slider("Neural Creativity", 0.0, 1.0, 0.3)

# --- MAIN UI ---
st.markdown("<h1>Proposera <span style='color:white'>AI</span></h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Autonomous Enterprise Proposal Engineering • Neural Midnight Engine</p>", unsafe_allow_html=True)

_, col_center, _ = st.columns([1, 2, 1])

with col_center:
    with st.expander("💎 System Configuration"):
        kb_files = st.file_uploader("Train Brain (PDF)", accept_multiple_files=True, type="pdf")
        if st.button("Optimize Memory"):
            if kb_files:
                for f in kb_files:
                    with open(os.path.join(TMP_KB, f.name), "wb") as out: out.write(f.getbuffer())
                st.session_state['vectorstore'] = build_knowledge_base(user_api_key)
                st.success("✅ Neural Brain Ready.")

    st.markdown("<div class='form-card'>", unsafe_allow_html=True)
    rfp_files = st.file_uploader("Autonomous Mission Intake (Batch PDFs)", type="pdf", accept_multiple_files=True)
    
    if st.button("⚡ EXECUTE NEURAL SEQUENCE"):
        if rfp_files and 'vectorstore' in st.session_state:
            with st.status("🚀 Processing Pipeline...", expanded=True) as status:
                results = []
                for f in rfp_files:
                    status.write(f"📂 Analyzing: {f.name}")
                    path = os.path.join(TMP_RFP, f.name)
                    with open(path, "wb") as out: out.write(f.getbuffer())
                    text = extract_rfp_text(path)
                    try:
                        client = get_client_name(text, selected_model, user_api_key)
                        web_data = research_competitors(text)
                        prop = generate_proposal(text, st.session_state['vectorstore'], web_data, selected_model, temp, client, user_api_key)
                        email_body = generate_email_body(prop, selected_model, client, user_api_key)
                        doc_p = os.path.join(TMP_OUT, f"Proposal_{f.name}.docx")
                        save_to_word(prop, doc_p)
                        results.append({"file": f.name, "client": client, "email": extract_emails(text), "text": prop, "email_body": email_body, "doc": doc_p})
                    except Exception as e: st.error(f"Error: {str(e)}")
                status.update(label="✅ Sequence Complete", state="complete", expanded=False)
            st.session_state['batch_results'] = results

    if 'batch_results' in st.session_state:
        st.divider()
        for res in st.session_state['batch_results']:
            with st.expander(f"✨ Result: {res['client']}", expanded=True):
                ca, cb = st.columns(2)
                with ca: 
                    with open(res['doc'], "rb") as f: st.download_button("📥 Download .DOCX", f, file_name=f"Proposal_{res['file']}.docx")
                with cb:
                    if st.button(f"🚀 Dispatch to {res['email']}", key=f"btn_{res['file']}"):
                        if not app_pass: st.error("Enter App Password in Sidebar!")
                        else:
                            with st.spinner("Dispatching..."):
                                msg = send_real_email(res['email'], f"Proposal for {res['client']}", res['email_body'], res['doc'], sender_mail, app_pass)
                                if msg is True: st.success("✅ Dispatched!")
                                else: st.error(msg)
                st.markdown(f"<div class='preview-box'>{res['text']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
