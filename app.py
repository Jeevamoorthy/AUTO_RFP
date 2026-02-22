import streamlit as st
import os, time, subprocess, sys
from rag import build_knowledge_base, generate_proposal, extract_rfp_text, research_competitors, extract_emails, get_client_name, generate_email_body, send_real_email
from utils import save_to_word

# Hotfix for Streamlit Cloud
try:
    from duckduckgo_search import DDGS
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "duckduckgo-search==6.3.0"])

# --- PAGE CONFIG ---
st.set_page_config(page_title="Proposera AI | Neural Midnight", layout="wide", page_icon="💠")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        .stApp { background-color: #0A101E; background-image: linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px); background-size: 50px 50px; color: #E6F1FF; }
        .form-card { background-color: #121E33; border: 1px solid #1F2A44; padding: 40px; border-radius: 14px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5); max-width: 850px; margin: 0 auto; }
        h1 { color: #00E0FF; text-align: center; font-weight: 800; text-shadow: 0 0 15px rgba(0,224,255,0.4); }
        .stButton>button { background: linear-gradient(90deg, #007BFF, #00E0FF) !important; color: white !important; font-weight: 700; padding: 14px; border: none; border-radius: 8px; width: 100%; transition: 0.3s; }
        .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(0, 200, 255, 0.5); }
        .preview-box { height: 450px; overflow-y: scroll; background-color: #0E1627; padding: 25px; border-radius: 10px; border: 1px solid #1F2F4A; color: #E6F1FF; font-family: 'Inter', sans-serif; }
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
        model_list = ["gemini-2.5-flash", "gemini-2.0-flash-lite", "deep-research-pro-preview-12-2025","gemini-2.5-pro","gemini-2.0-flash","gemini-pro-latest"]
        key_help = "https://aistudio.google.com/app/apikey"
    elif provider == "OpenAI GPT":
        model_list = ["gpt-4o", "gpt-4o-mini"]
        key_help = "https://platform.openai.com/api-keys"
    else:
        model_list = ["claude-3-5-sonnet-latest", "claude-3-haiku-20240307"]
        key_help = "https://console.anthropic.com/settings/keys"

    selected_model = st.selectbox("Intelligence Model", model_list)
    user_api_key = st.text_input(f"{provider} API Key (Override)", type="password", help=f"Get key here: {key_help}")
    temp = st.slider("Neural Creativity", 0.0, 1.0, 0.3)

# --- MAIN UI ---
st.markdown("<h1>Proposera <span style='color:white'>AI</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#9CB3D1;'>Autonomous Enterprise Proposal Engineering • Neural Midnight Engine</p>", unsafe_allow_html=True)

_, col_center, _ = st.columns([1, 2, 1])

with col_center:
    with st.expander("💎 System Configuration (Enterprise Brain)"):
        kb_files = st.file_uploader("Upload Company PDFs", accept_multiple_files=True, type="pdf")
        if st.button("Optimize Neural Brain"):
            if kb_files:
                os.makedirs("data/knowledge_base", exist_ok=True)
                for f in kb_files:
                    with open(os.path.join("data/knowledge_base", f.name), "wb") as file: file.write(f.getbuffer())
                st.session_state['vectorstore'] = build_knowledge_base(); st.success("✅ Neural Brain Ready.")

    st.markdown("<div class='form-card'>", unsafe_allow_html=True)
    rfp_files = st.file_uploader("Inbound Client RFPs (Multiple OK)", type="pdf", accept_multiple_files=True)
    
    if st.button("⚡ EXECUTE NEURAL SEQUENCE"):
        if rfp_files and 'vectorstore' in st.session_state:
            with st.status("🚀 Processing Autonomous Pipeline...", expanded=True) as status:
                results = []
                for f in rfp_files:
                    status.write(f"📂 **Analyzing:** {f.name}")
                    path = os.path.join("data/rfp_input", f.name); os.makedirs("data/rfp_input", exist_ok=True)
                    with open(path, "wb") as out: out.write(f.getbuffer())
                    text = extract_rfp_text(path)
                    client = get_client_name(text, selected_model, user_api_key)
                    web_data = research_competitors(text)
                    prop = generate_proposal(text, st.session_state['vectorstore'], web_data, selected_model, temp, client, user_api_key)
                    email_body = generate_email_body(prop, selected_model, client, user_api_key)
                    results.append({"file": f.name, "client": client, "email": extract_emails(text), "text": prop, "email_body": email_body, "doc": save_to_word(prop, f"output/Proposal_{f.name}.docx")})
                status.update(label="✅ Sequence Complete", state="complete", expanded=False)
            st.session_state['batch_results'] = results

    if 'batch_results' in st.session_state:
        for res in st.session_state['batch_results']:
            with st.expander(f"✨ {res['client']} | Protocol Complete", expanded=True):
                ca, cb = st.columns(2)
                with ca: 
                    with open(res['doc'], "rb") as f: st.download_button("📥 Download .DOCX", f, file_name=f"Proposal_{res['file']}.docx")
                with cb:
                    if st.button(f"🚀 Dispatch to {res['email']}", key=f"btn_{res['file']}"):
                        if not app_pass: st.error("Missing Gmail Credentials!")
                        else:
                            with st.spinner("Dispatching..."):
                                msg = send_real_email(res['email'], f"Proposal for {res['client']}", res['email_body'], res['doc'], sender_mail, app_pass)
                                if msg is True: st.success("✅ Dispatched!")
                                else: st.error(msg)
                st.markdown(f"<div class='preview-box'>{res['text']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
